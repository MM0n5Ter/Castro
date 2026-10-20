#include <Radiation.H>

#include <AMReX_ParmParse.H>

#include <HypreSolvers.H>
#include <elliptic_face_ops.H>
#include <radiation_helpers.H>
#include <AMReX_LO_BCTYPES.H>

#include <_hypre_sstruct_mv.h>
#include <HYPRE_krylov.h>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace amrex;

static int ispow2(int i)
{
  return (i == 1) ? 1 : (((i <= 0) || (i & 1)) ? 0 : ispow2(i / 2));
}

static int query_struct_method_id ()
{
    ParmParse pp("hmabec");
#if (AMREX_SPACEDIM == 1)
    int struct_method_id = 0;
#else
    int struct_method_id = 1;
#endif
    pp.query("struct_method_id", struct_method_id);
    return struct_method_id;
}

Real MultiLevelEllipticSolver::flux_scale = 1.0;

#if (AMREX_SPACEDIM == 1)
int MultiLevelEllipticSolver::vl[2] = { 0, 0 };
int MultiLevelEllipticSolver::vh[2] = { 0, 0 };
#endif

MultiLevelEllipticSolver::MultiLevelEllipticSolver(int _coarse_lvl, int _fine_lvl,
                               int _solver_choice)
  : crse_level(_coarse_lvl), fine_level(_fine_lvl),
    solver_choice(_solver_choice),
    geom(fine_level+1),
    grids(fine_level+1),
    dmap(fine_level+1),
    fine_ratio(fine_level+1),
    bd(fine_level+1),
    subgrids(fine_level+1),
    acoefs(fine_level+1),
    bcoefs(fine_level+1),
    SPa(fine_level+1),
    coarse_interp(fine_level+1),
    normal_grad_store(fine_level+1),
    c_interp(fine_level+1),
    c_extrap_deriv(fine_level+1),
    c_connections(fine_level+1),
    hgrid(NULL), stencil(NULL), graph(NULL),
    A(NULL), A0(NULL), b(NULL), x(NULL),
    solver_handle(NULL), solver(NULL), precond(NULL)
{
  ParmParse pp("hmabec");

  verbose = 1;       pp.query("v", verbose); pp.query("verbose", verbose);
  log_threshold = 0; pp.query("log_threshold", log_threshold);
  bho = 0;           pp.query("bho", bho);
  subgrid_mode = 0;  pp.query("subgrid_mode", subgrid_mode);

  static int first = 1;
  if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
    first = 0;
    std::cout << "hmabec.bho               = " << bho << std::endl;
    std::cout << "hmabec.subgrid_mode      = " << subgrid_mode << std::endl;
    std::cout << "hmabec.verbose           = " << verbose << std::endl;
    std::cout << "hmabec.log_threshold = " << log_threshold << std::endl;
  }

  if (solver_choice == 100 || solver_choice == 102 || solver_choice == 104 ||
      solver_choice == 105 ||
      solver_choice == 150 || solver_choice == 151 || solver_choice == 153 ||
      solver_choice == 1002) {
    ObjectType = HYPRE_PARCSR;
  }
  else if (solver_choice == 101 || solver_choice == 103 ||
           solver_choice == 106 || solver_choice == 107 ||
           solver_choice == 152 ||
           solver_choice == 1003) {
    ObjectType = HYPRE_SSTRUCT;
  }
  else if (solver_choice == 108 || solver_choice == 109) {
    ObjectType = HYPRE_STRUCT;
  }
  else {
    std::cout << "MultiLevelEllipticSolver: no such solver" << std::endl;
    exit(1);
  }

  int nparts = fine_level - crse_level + 1;

#if (AMREX_SPACEDIM == 1)

  HYPRE_SStructGridCreate(MPI_COMM_WORLD, 2, nparts, &hgrid);

#else

  HYPRE_SStructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, nparts, &hgrid);

#endif
}

MultiLevelEllipticSolver::~MultiLevelEllipticSolver()
{
  HYPRE_SStructVectorDestroy(b);
  HYPRE_SStructVectorDestroy(x);

  HYPRE_SStructMatrixDestroy(A);
  HYPRE_SStructMatrixDestroy(A0);

  HYPRE_SStructGraphDestroy(graph);
  HYPRE_SStructStencilDestroy(stencil);
  HYPRE_SStructGridDestroy(hgrid);
}

void MultiLevelEllipticSolver::attachLevel(int             level,
                              const Geometry& _geom,
                              const BoxArray& _grids,
                              const DistributionMapping& _dmap,
                              IntVect         _refinement)
{
  int part = level - crse_level;

  geom[level]  = _geom;
  grids[level] = _grids;
  dmap[level]  = _dmap;
  fine_ratio[level] = _refinement;

#if (AMREX_SPACEDIM == 1)

  if (geom[level].isAnyPeriodic()) {
    BL_ASSERT(geom[level].isPeriodic(0));
    BL_ASSERT(geom[level].Domain().smallEnd(0) == 0);

    int is_periodic[2];
    is_periodic[0] = geom[level].period(0);
    is_periodic[1] = 0;
    BL_ASSERT(ispow2(is_periodic[0]));

    HYPRE_SStructGridSetPeriodic(hgrid, part, is_periodic);
  }

#else

  if (geom[level].isAnyPeriodic()) {
    int is_periodic[AMREX_SPACEDIM];
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      is_periodic[i] = 0;
      if (geom[level].isPeriodic(i)) {
        is_periodic[i] = geom[level].period(i);
        BL_ASSERT(ispow2(is_periodic[i]));
        BL_ASSERT(geom[level].Domain().smallEnd(i) == 0);
      }
    }
    HYPRE_SStructGridSetPeriodic(hgrid, part, is_periodic);
  }

#endif

  int myid = ParallelDescriptor::MyProc();

  subgrids[level].resize(grids[level].size());

  if (!subgrid_mode || level == fine_level || grids[level+1].size() == 0) {
    for (int i = 0; i < grids[level].size(); i++) {
      if (dmap[level][i] == myid) {
        HYPRE_SStructGridSetExtents(hgrid, part,
                                    loV(grids[level][i]),
                                    hiV(grids[level][i]));
      }
    }
  }
  else {
    BoxArray mask = grids[level+1];
    mask.coarsen(fine_ratio[level]);
    for (int i = 0; i < grids[level].size(); i++) {
      if (dmap[level][i] == myid) {
        subgrids[level][i] = amrex::complementIn(grids[level][i], mask);
#if 0
        std::cout << "Coarse level, grid " << i << " subgrids are "
             << subgrids[level][i] << std::endl;
        std::cout << "Box numPts = " << grids[level][i].numPts()
             << "Subgrids numPts = " << subgrids[level][i].numPts() << std::endl;
#endif
        for (int j = 0; j < subgrids[level][i].size(); j++) {
          HYPRE_SStructGridSetExtents(hgrid, part,
                                      loV(subgrids[level][i][j]),
                                      hiV(subgrids[level][i][j]));
        }
      }
    }
  }

  HYPRE_SStructVariable vars[1] = {HYPRE_SSTRUCT_VARIABLE_CELL};
  HYPRE_SStructGridSetVariables(hgrid, part, 1, vars);
}

static void
interpolate_transverse(BoxAuxStorage& coarse_interp, const Mask& msk,
                      const Box& reg, const Box& coarse_box,
                      AMREX_D_DECL(const IntVect& rat, const IntVect& vj1, const IntVect& vk1),
                      AMREX_D_DECL(const IntVect& ve,  const IntVect& vjr, const IntVect& vkr),
                      AMREX_D_DECL(int idir, int jdir, int kdir),
                      int coarse_level_index)
{
  for (IntVect vc = coarse_box.smallEnd(); vc <= coarse_box.bigEnd(); coarse_box.next(vc)) {
    IntVect vf = rat * vc;
    vf[idir] = reg.smallEnd(idir);
    Box face(vf, vf + ve);
    if (msk(vf) == RadBoundaryData::not_covered) {
#if (0)
      for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
        coarse_interp(v).push(coarse_level_index, vc,     1.0);
      }
#elif (AMREX_SPACEDIM == 1)
      coarse_interp(vf).push(coarse_level_index, vc, 1.0);
#elif (AMREX_SPACEDIM == 2)
      if (msk(vf-vj1) != RadBoundaryData::not_covered &&
          msk(vf+vjr) == RadBoundaryData::not_covered) {
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real xx = (v[jdir] - vf[jdir] - 0.5 * ve[jdir]) / rat[jdir];
          coarse_interp(v).push(coarse_level_index, vc+vj1, xx);
          coarse_interp(v).push(coarse_level_index, vc,     1.0 - xx);
        }
      }
      else if (msk(vf-vj1) == RadBoundaryData::not_covered &&
               msk(vf+vjr) == RadBoundaryData::not_covered) {
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real xx = (v[jdir] - vf[jdir] - 0.5 * ve[jdir]) / rat[jdir];
          coarse_interp(v).push(coarse_level_index, vc+vj1, 0.5*xx*(xx+1));
          coarse_interp(v).push(coarse_level_index, vc,     1.0-xx*xx);
          coarse_interp(v).push(coarse_level_index, vc-vj1, 0.5*xx*(xx-1));
        }
      }
      else if (msk(vf-vj1) == RadBoundaryData::not_covered &&
               msk(vf+vjr) != RadBoundaryData::not_covered) {
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real xx = (v[jdir] - vf[jdir] - 0.5 * ve[jdir]) / rat[jdir];
          coarse_interp(v).push(coarse_level_index, vc,     1.0 + xx);
          coarse_interp(v).push(coarse_level_index, vc-vj1, -xx);
        }
      }
      else {
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          coarse_interp(v).push(coarse_level_index, vc,     1.0);
        }
      }
#elif (AMREX_SPACEDIM == 3)

      if (msk(vf-vj1) != RadBoundaryData::not_covered &&
          msk(vf+vjr) == RadBoundaryData::not_covered) {
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real xx = (v[jdir] - vf[jdir] - 0.5 * ve[jdir]) / rat[jdir];
          coarse_interp(v).push(coarse_level_index, vc+vj1, xx);
          coarse_interp(v).push(coarse_level_index, vc,     1.0 - xx);
        }
      }
      else if (msk(vf-vj1) == RadBoundaryData::not_covered &&
               msk(vf+vjr) == RadBoundaryData::not_covered) {
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real xx = (v[jdir] - vf[jdir] - 0.5 * ve[jdir]) / rat[jdir];
          coarse_interp(v).push(coarse_level_index, vc+vj1, 0.5*xx*(xx+1));
          coarse_interp(v).push(coarse_level_index, vc,     1.0-xx*xx);
          coarse_interp(v).push(coarse_level_index, vc-vj1, 0.5*xx*(xx-1));
        }
      }
      else if (msk(vf-vj1) == RadBoundaryData::not_covered &&
               msk(vf+vjr) != RadBoundaryData::not_covered) {
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real xx = (v[jdir] - vf[jdir] - 0.5 * ve[jdir]) / rat[jdir];
          coarse_interp(v).push(coarse_level_index, vc,     1.0 + xx);
          coarse_interp(v).push(coarse_level_index, vc-vj1, -xx);
        }
      }
      else {
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          coarse_interp(v).push(coarse_level_index, vc,     1.0);
        }
      }

      if (msk(vf-vk1) != RadBoundaryData::not_covered &&
          msk(vf+vkr) == RadBoundaryData::not_covered) {
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real yy = (v[kdir] - vf[kdir] - 0.5 * ve[kdir]) / rat[kdir];
          coarse_interp(v).push(coarse_level_index, vc+vk1, yy);
          coarse_interp(v).push(coarse_level_index, vc,    -yy);
        }
      }
      else if (msk(vf-vk1) == RadBoundaryData::not_covered &&
               msk(vf+vkr) == RadBoundaryData::not_covered) {
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real yy = (v[kdir] - vf[kdir] - 0.5 * ve[kdir]) / rat[kdir];
          coarse_interp(v).push(coarse_level_index, vc+vk1, 0.5*yy*(yy+1));
          coarse_interp(v).push(coarse_level_index, vc,     -yy*yy);
          coarse_interp(v).push(coarse_level_index, vc-vk1, 0.5*yy*(yy-1));
        }
      }
      else if (msk(vf-vk1) == RadBoundaryData::not_covered &&
               msk(vf+vkr) != RadBoundaryData::not_covered) {
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real yy = (v[kdir] - vf[kdir] - 0.5 * ve[kdir]) / rat[kdir];
          coarse_interp(v).push(coarse_level_index, vc,      yy);
          coarse_interp(v).push(coarse_level_index, vc-vk1, -yy);
        }
      }
      else {
      }

      if (msk(vf-vj1-vk1) == RadBoundaryData::not_covered &&
          msk(vf-vj1+vkr) == RadBoundaryData::not_covered &&
          msk(vf+vjr-vk1) == RadBoundaryData::not_covered &&
          msk(vf+vjr+vkr) == RadBoundaryData::not_covered) {
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real xx = (v[jdir] - vf[jdir] - 0.5 * ve[jdir]) / rat[jdir];
          Real yy = (v[kdir] - vf[kdir] - 0.5 * ve[kdir]) / rat[kdir];
          coarse_interp(v).push(coarse_level_index, vc-vj1-vk1,  0.25*xx*yy);
          coarse_interp(v).push(coarse_level_index, vc-vj1+vk1, -0.25*xx*yy);
          coarse_interp(v).push(coarse_level_index, vc+vj1-vk1, -0.25*xx*yy);
          coarse_interp(v).push(coarse_level_index, vc+vj1+vk1,  0.25*xx*yy);
        }
      }
#endif
    }
  }
}

static void
compute_normal_gradient(BoxAuxStorage& normal_grad_store, BoxAuxStorage& coarse_interp,
                 const Mask& msk, const Box& reg,
                 const IntVect& vin, Real h, int r, int bho, int fine_lvl)
{
  if (bho == 1) {
    Real face_weight_interp = 8.0 / (h * (1 + r) * (3 + r));
    Real face_weight_near = (r - 3) / (h * (1 + r));
    Real face_weight_far = (1 - r) / (h * (3 + r));
    IntVect vi2 = 2 * vin;
    for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
      if (msk(v) == RadBoundaryData::not_covered) {
        normal_grad_store(v).push(&coarse_interp(v),    face_weight_interp);
        normal_grad_store(v).push(fine_lvl, v-vin, face_weight_near);
        normal_grad_store(v).push(fine_lvl, v-vi2, face_weight_far);
      }
    }
  }
  else {
    Real normal_grad_scale = 2.0 / (h * (1 + r));
    for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
      if (msk(v) == RadBoundaryData::not_covered) {
        normal_grad_store(v).push(&coarse_interp(v),     normal_grad_scale);
        normal_grad_store(v).push(fine_lvl, v-vin, -normal_grad_scale);
      }
    }
  }
}

void MultiLevelEllipticSolver::makeMatrixLayout()
{

  for (int level = crse_level; level <= fine_level; level++) {
    int ncomp=1;
    int ngrow=0;
    acoefs[level].reset(new MultiFab(grids[level], dmap[level], ncomp, ngrow));
    acoefs[level]->setVal(0.0);

    bcoefs[level].reset(new Array<MultiFab, AMREX_SPACEDIM>);

    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      BoxArray edge_boxes(grids[level]);
      edge_boxes.surroundingNodes(i);
      (*bcoefs[level])[i].define(edge_boxes, dmap[level], ncomp, ngrow);
      (*bcoefs[level])[i].setVal(0.0);
    }
  }

  HYPRE_SStructGridAssemble(hgrid);

#if (AMREX_SPACEDIM == 1)
  int offsets[3][2] = {{ 0,  0},
                       {-1,  0},
                       { 1,  0}};
#elif (AMREX_SPACEDIM == 2)
  int offsets[5][2] = {{ 0,  0},
                       {-1,  0},
                       { 1,  0},
                       { 0, -1},
                       { 0,  1}};
#elif (AMREX_SPACEDIM == 3)
  int offsets[7][3] = {{ 0,  0,  0},
                       {-1,  0,  0},
                       { 1,  0,  0},
                       { 0, -1,  0},
                       { 0,  1,  0},
                       { 0,  0, -1},
                       { 0,  0,  1}};
#endif

  BL_ASSERT(stencil == NULL);
#if (AMREX_SPACEDIM == 1)
  HYPRE_SStructStencilCreate(2, 3, &stencil);
#else
  HYPRE_SStructStencilCreate(AMREX_SPACEDIM, 2 * AMREX_SPACEDIM + 1, &stencil);
#endif

  for (int i = 0; i < 2 * AMREX_SPACEDIM + 1; i++) {
    HYPRE_SStructStencilSetEntry(stencil, i, offsets[i], 0);
  }

  BL_ASSERT(graph == NULL);
  HYPRE_SStructGraphCreate(MPI_COMM_WORLD, hgrid, &graph);
  HYPRE_SStructGraphSetObjectType(graph, ObjectType);

  for (int level = crse_level; level <= fine_level; level++) {
    int part = level - crse_level;
    HYPRE_SStructGraphSetStencil(graph, part, 0, stencil);
  }

  for (int level = crse_level + 1; level <= fine_level; level++) {
    int part = level - crse_level;
    coarse_interp[level].reset(new BoundaryAuxStore(grids[level], dmap[level], BoundaryAuxStore::HALO));
    normal_grad_store[level].reset(new BoundaryAuxStore(grids[level], dmap[level],  BoundaryAuxStore::HALO));
    BoundaryAuxStore entry(grids[level], dmap[level], BoundaryAuxStore::INTERIOR);
    IntVect rat = fine_ratio[level-1];

    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      IntVect vin = amrex::BASISV(idir);
      vin = (ori.isLow() ? -vin : vin);
      Real h = geom[level].CellSize(idir);
      IntVect ve;
#if (AMREX_SPACEDIM >= 2)
      int jdir = (idir + 1) % AMREX_SPACEDIM;
      IntVect vj1 = amrex::BASISV(jdir);
      IntVect vjr = rat * vj1;
      ve += (vjr - vj1);
#endif
#if (AMREX_SPACEDIM == 3)
      int kdir = (idir + 2) % 3;
      IntVect vk1 = amrex::BASISV(kdir);
      IntVect vkr = rat * vk1;
      ve += (vkr - vk1);
#endif
      for (int i = coarse_interp[level]->firstOwnedIndex(); coarse_interp[level]->isValid(i);
           i = coarse_interp[level]->nextOwnedIndex(i)) {
        Box reg = amrex::adjCell(grids[level][i], ori);
        Box coarse_box = amrex::coarsen(reg, rat);
        const Mask &msk = bd[level]->bndryMasks(ori,i);

        interpolate_transverse((*coarse_interp[level])(ori,i), msk, reg, coarse_box,
                              AMREX_D_DECL(rat,  vj1,  vk1),
                              AMREX_D_DECL(ve,   vjr,  vkr),
                              AMREX_D_DECL(idir, jdir, kdir),
                              level-1);

        compute_normal_gradient((*normal_grad_store[level])(ori,i),
                         (*coarse_interp[level])(ori,i),
                         msk, reg, vin, h, rat[idir], bho, level);

        reg.shift(-vin);
        for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
          if (msk(v+vin) == RadBoundaryData::not_covered) {
              entry(ori,i)(v).push(&(*normal_grad_store[level])(ori,i)(v+vin), 1.0);
          }
        }
      }
    }

    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      IntVect vin = amrex::BASISV(idir);
      vin = (ori.isLow() ? -vin : vin);
      for (int i = entry.firstOwnedIndex(); entry.isValid(i);
           i = entry.nextOwnedIndex(i)) {
        Box reg = amrex::adjCell(grids[level][i], ori);
        reg.shift(-vin);
        for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
#if (0 && !defined(NDEBUG))
          if (msk(v+vin) == RadBoundaryData::not_covered &&
              entry(ori,i)(v).secondary()) {
            std::cout << v << " is secondary in orientation " << ori
                 << " on processor " << ParallelDescriptor::MyProc()
                 << std::endl;
          }
#endif
          if (!entry(ori,i)(v).empty() &&
              !entry(ori,i)(v).secondary()) {
            entry(ori,i)(v).collapse();
            Vector<int> levels;
            Vector<IntVect> cells;
            int retval = entry(ori,i)(v).read_stencil_sites(levels, cells);
            BL_ASSERT(retval == 0);
            for (int j = 0; j < levels.size(); j++) {
              int needs_separate_entry = 1;
              if (levels[j] == level) {
                IntVect d = cells[j] - v;
                for (int k = 0; k < 2 * AMREX_SPACEDIM + 1; k++) {
                  if (d == IntVect(offsets[k])) {
                    needs_separate_entry = 0;
                  }
                }
              }
              if (needs_separate_entry) {
                int to_part = levels[j] - crse_level;
                HYPRE_SStructGraphAddEntries(graph,
                                             part,    cellIndex1(v),        0,
                                             to_part, cellIndex2(cells[j]), 0);
              }
            }
          }
        }
      }
    }

    const BoxArray& ff_boxes(grids[level]);
    const BoxArray& cc_boxes(grids[level-1]);
    BoxArray fc_boxes(cc_boxes);
    fc_boxes.refine(rat);
    BoxArray cf_boxes(ff_boxes);
    cf_boxes.coarsen(rat);

    c_interp[level].reset(new CoarseBoundaryAuxStore(fc_boxes, dmap[level-1],
                                              ff_boxes, BoundaryAuxStore::HALO));
    c_extrap_deriv[level].reset(new CoarseBoundaryAuxStore(*c_interp[level],
                                            BoundaryAuxStore::HALO));
    c_connections[level].reset (new CoarseBoundaryAuxStore(cc_boxes, cf_boxes,
                                            *c_interp[level],
                                            BoundaryAuxStore::OUTSIDE_GRID));

    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      IntVect vin = amrex::BASISV(idir);
      vin = (ori.isLow() ? -vin : vin);
      Real h = geom[level].CellSize(idir);
      IntVect ve;
#if (AMREX_SPACEDIM >= 2)
      int jdir = (idir + 1) % AMREX_SPACEDIM;
      IntVect vj1 = amrex::BASISV(jdir);
      IntVect vjr = rat * vj1;
      ve += (vjr - vj1);
#endif
#if (AMREX_SPACEDIM == 3)
      int kdir = (idir + 2) % 3;
      IntVect vk1 = amrex::BASISV(kdir);
      IntVect vkr = rat * vk1;
      ve += (vkr - vk1);
#endif
      for (int i = c_interp[level]->firstOwnedIndex(); c_interp[level]->isValid(i);
         i = c_interp[level]->nextOwnedIndex(i)) {
         for (int j = 0; j < (*c_interp[level]).size(ori,i); j++) {
           const Box& reg = (*c_interp[level])(ori,i,j).box();
           const Box& coarse_box = (*c_connections[level])(ori,i,j).box();
           const Mask &msk = c_interp[level]->mask(ori,i,j);

         interpolate_transverse((*c_interp[level])(ori,i,j), msk, reg, coarse_box,
                                AMREX_D_DECL(rat,  vj1,  vk1),
                                AMREX_D_DECL(ve,   vjr,  vkr),
                                AMREX_D_DECL(idir, jdir, kdir),
                                level-1);

         compute_normal_gradient((*c_extrap_deriv[level])(ori,i,j),
                          (*c_interp[level])(ori,i,j),
                          msk, reg, vin, h, rat[idir], bho, level);

          for (IntVect vc = coarse_box.smallEnd(); vc <= coarse_box.bigEnd(); coarse_box.next(vc)) {
            IntVect vf = rat * vc;
            vf[idir] = reg.smallEnd(idir);
            Box face(vf, vf + ve);
            if (msk(vf) == RadBoundaryData::not_covered) {
              for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
                  (*c_connections[level])(ori,i,j)(vc)
                      .push(&(*c_extrap_deriv[level])(ori,i,j)(v), 1.0);
              }
            }
          }
        }
      }
    }

    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      for (int i = c_interp[level]->firstOwnedIndex(); c_interp[level]->isValid(i);
           i = c_interp[level]->nextOwnedIndex(i)) {
        for (int j = 0; j < (*c_interp[level]).size(ori,i); j++) {
          const Box& reg = (*c_interp[level])(ori,i,j).box();
          const Box& coarse_box = (*c_connections[level])(ori,i,j).box();
          const Mask &msk = c_interp[level]->mask(ori,i,j);
          for (IntVect vc = coarse_box.smallEnd(); vc <= coarse_box.bigEnd(); coarse_box.next(vc)) {
            IntVect vf = rat * vc;
            vf[idir] = reg.smallEnd(idir);
            if (msk(vf) == RadBoundaryData::not_covered &&
                !(*c_connections[level])(ori,i,j)(vc).secondary()) {
              (*c_connections[level])(ori,i,j)(vc).collapse();
              Vector<int> levels;
              Vector<IntVect> cells;
              int retval = (*c_connections[level])(ori,i,j)(vc)
                .read_stencil_sites(levels, cells);
              BL_ASSERT(retval == 0);
              for (int jj = 0; jj < levels.size(); jj++) {
                int needs_separate_entry = 1;
                if (levels[jj] == level-1) {
                  IntVect d = cells[jj] - vc;
                  for (int k = 0; k < 2 * AMREX_SPACEDIM + 1; k++) {
                    if (d == IntVect(offsets[k])) {
                      needs_separate_entry = 0;
                    }
                  }
                }
                if (needs_separate_entry) {
                  int to_part = levels[jj] - crse_level;
                  HYPRE_SStructGraphAddEntries(graph,
                                               part-1,  cellIndex1(vc),        0,
                                               to_part, cellIndex2(cells[jj]), 0);
                }
              }
            }
          }
        }
      }
    }
  }

  HYPRE_SStructGraphAssemble(graph);

  BL_ASSERT(A == NULL);
  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);
  HYPRE_SStructMatrixSetObjectType(A, ObjectType);
  HYPRE_SStructMatrixInitialize(A);

  BL_ASSERT(A0 == NULL);
  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A0);
  HYPRE_SStructMatrixSetObjectType(A0, ObjectType);
  HYPRE_SStructMatrixInitialize(A0);

  BL_ASSERT(b == NULL);
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, hgrid, &b);
  HYPRE_SStructVectorSetObjectType(b, ObjectType);

  BL_ASSERT(x == NULL);
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, hgrid, &x);
  HYPRE_SStructVectorSetObjectType(x, ObjectType);

  HYPRE_SStructVectorInitialize(b);
  HYPRE_SStructVectorInitialize(x);

  HYPRE_SStructVectorAssemble(b);
  HYPRE_SStructVectorAssemble(x);
}

void MultiLevelEllipticSolver::configureSolver(Real _rel_tolerance, Real _abs_tolerance, int maxiter)
{
  BL_PROFILE("MultiLevelEllipticSolver::configureSolver");

  reltol = _rel_tolerance;
  abstol = _abs_tolerance;

  BL_ASSERT(solver_handle == NULL);
  BL_ASSERT(solver         == NULL);
  BL_ASSERT(precond        == NULL);

  if (solver_choice == 100) {
    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_SStructMatrixGetObject(A, (void**) &par_A);
    HYPRE_SStructVectorGetObject(b, (void**) &par_b);
    HYPRE_SStructVectorGetObject(x, (void**) &par_x);

    HYPRE_BoomerAMGCreate(&solver);
    HYPRE_BoomerAMGSetMinIter(solver, 1);
    HYPRE_BoomerAMGSetMaxIter(solver, maxiter);
    HYPRE_BoomerAMGSetTol(solver, reltol);
#if 0
    HYPRE_BoomerAMGSetInterpType(solver, 6);
    HYPRE_BoomerAMGSetCoarsenType(solver, 8);
    HYPRE_BoomerAMGSetStrongThreshold(solver, 0.25);
    HYPRE_BoomerAMGSetTruncFactor(solver, 0.);
#endif
    HYPRE_BoomerAMGSetup(solver, par_A, par_b, par_x);
#if 0
    int num_sweeps[4] = {2, 2, 2, 1};
    HYPRE_BoomerAMGSetNumGridSweeps(solver, num_sweeps);
#endif
  }
  else if (solver_choice == 101) {
    int nparts = fine_level - crse_level + 1;
    int *plevels = new int[nparts];
    int (*prefinements)[3] = new int[nparts][3];
    for (int i = 0; i < nparts; i++) {
      plevels[i] = i;
      if (i == 0) {
        prefinements[i][0] = 1;
        prefinements[i][1] = 1;
        prefinements[i][2] = 1;
      }
      else {
        prefinements[i][0] = fine_ratio[crse_level+i-1][0];
#if (AMREX_SPACEDIM == 1)
        prefinements[i][1] = 1;
        prefinements[i][2] = 1;
#elif (AMREX_SPACEDIM == 2)
        prefinements[i][1] = fine_ratio[crse_level+i-1][1];
        prefinements[i][2] = 1;
#elif (AMREX_SPACEDIM == 3)
        prefinements[i][1] = fine_ratio[crse_level+i-1][1];
        prefinements[i][2] = fine_ratio[crse_level+i-1][2];
#endif
      }
    }
    int pre_relax_sweeps  = prefinements[nparts-1][0] - 1;
    int post_relax_sweeps = prefinements[nparts-1][0] - 1;

    pre_relax_sweeps  = 3;
    post_relax_sweeps = 3;

    HYPRE_SStructFACCreate(MPI_COMM_WORLD, &solver_handle);
    HYPRE_SStructFACSetMaxLevels(solver_handle, nparts);
    HYPRE_SStructFACSetMaxIter(solver_handle, maxiter);
    HYPRE_SStructFACSetTol(solver_handle, reltol);
    HYPRE_SStructFACSetPLevels(solver_handle, nparts, plevels);
    HYPRE_SStructFACSetPRefinements(solver_handle, nparts, prefinements);
    HYPRE_SStructFACSetRelChange(solver_handle, 0);
    HYPRE_SStructFACSetRelaxType(solver_handle, 1);
    HYPRE_SStructFACSetRelaxType(solver_handle, 2);
    HYPRE_SStructFACSetNumPreRelax(solver_handle, pre_relax_sweeps);
    HYPRE_SStructFACSetNumPostRelax(solver_handle, post_relax_sweeps);
    HYPRE_SStructFACSetCoarseSolverType(solver_handle, 2);
    HYPRE_SStructFACSetLogging(solver_handle, 1);

    for (int i = 1; i < nparts; i++) {
      HYPRE_SStructFACZeroCFSten(A, hgrid, i, prefinements[i]);
      HYPRE_SStructFACZeroFCSten(A, hgrid, i);
    }
    for (int i = 0; i < nparts-1; i++) {
      HYPRE_SStructFACZeroAMRMatrixData(A, i, prefinements[i+1]);
    }
    HYPRE_SStructFACZeroAMRVectorData(b, plevels, prefinements);
    commitMatrix();
    commitVectors();

    HYPRE_SStructFACSetup2(solver_handle, A, b, x);

    delete[] plevels;
    delete[] prefinements;
  }
  else if (solver_choice == 102) {
    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_SStructMatrixGetObject(A, (void**) &par_A);
    HYPRE_SStructVectorGetObject(b, (void**) &par_b);
    HYPRE_SStructVectorGetObject(x, (void**) &par_x);

    ParmParse pp("hmabec");
    int gmres_restart = 5; pp.query("gmres_restart", gmres_restart);
    static int first = 1;
    if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
      first = 0;
      std::cout << "hmabec.gmres_restart            = " << gmres_restart << std::endl;
    }

    HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver);
    HYPRE_ParCSRGMRESSetKDim(solver, gmres_restart);
    HYPRE_ParCSRGMRESSetMaxIter(solver, maxiter);
    HYPRE_ParCSRGMRESSetTol(solver, reltol);

    HYPRE_ParCSRGMRESSetup(solver, par_A, par_b, par_x);
  }
  else if (solver_choice == 103) {
    ParmParse pp("hmabec");
    int gmres_restart = 5; pp.query("gmres_restart", gmres_restart);
    static int first = 1;
    if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
      first = 0;
      std::cout << "hmabec.gmres_restart            = " << gmres_restart << std::endl;
    }

    HYPRE_SStructGMRESCreate(MPI_COMM_WORLD, &solver_handle);
    HYPRE_SStructGMRESSetKDim(solver_handle, gmres_restart);
    HYPRE_SStructGMRESSetMaxIter(solver_handle, maxiter);
    HYPRE_SStructGMRESSetTol(solver_handle, reltol);

    HYPRE_SStructGMRESSetup(solver_handle, A, b, x);
  }
  else if (solver_choice == 1002) {
    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_SStructMatrixGetObject(A, (void**) &par_A);
    HYPRE_SStructVectorGetObject(b, (void**) &par_b);
    HYPRE_SStructVectorGetObject(x, (void**) &par_x);

    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_ParCSRPCGSetMaxIter(solver, maxiter);
    HYPRE_ParCSRPCGSetTol(solver, reltol);

    HYPRE_ParCSRPCGSetup(solver, par_A, par_b, par_x);
  }
  else if (solver_choice == 1003) {
    HYPRE_SStructPCGCreate(MPI_COMM_WORLD, &solver_handle);
    HYPRE_SStructPCGSetMaxIter(solver_handle, maxiter);
    HYPRE_SStructPCGSetTol(solver_handle, reltol);

    HYPRE_SStructPCGSetup(solver_handle, A, b, x);
  }
  else if (solver_choice == 104) {
    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_SStructMatrixGetObject(A, (void**) &par_A);
    HYPRE_SStructVectorGetObject(b, (void**) &par_b);
    HYPRE_SStructVectorGetObject(x, (void**) &par_x);

    ParmParse pp("hmabec");
    int gmres_restart = 5; pp.query("gmres_restart", gmres_restart);
    static int first = 1;
    if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
      first = 0;
      std::cout << "hmabec.gmres_restart            = " << gmres_restart << std::endl;
    }

    HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver);
    HYPRE_ParCSRGMRESSetKDim(solver, gmres_restart);
    HYPRE_ParCSRGMRESSetMaxIter(solver, maxiter);
    HYPRE_ParCSRGMRESSetTol(solver, reltol);

    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetMaxIter(precond, 1);
    HYPRE_BoomerAMGSetTol(precond, reltol);
    HYPRE_ParCSRGMRESSetPrecond(solver,
                                (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSolve,
                                (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSetup,
                                precond);

    HYPRE_ParCSRGMRESSetup(solver, par_A, par_b, par_x);
  }
  else if (solver_choice == 105) {
    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_SStructMatrixGetObject(A, (void**) &par_A);
    HYPRE_SStructVectorGetObject(b, (void**) &par_b);
    HYPRE_SStructVectorGetObject(x, (void**) &par_x);

    ParmParse pp("hmabec");
    int gmres_restart    = 5;    pp.query("gmres_restart", gmres_restart);
    int interp  = 6;    pp.query("interp", interp);
    int coarsen = 8;    pp.query("coarsen", coarsen);
    int relax   = 6;    pp.query("relax", relax);
    Real strong = 0.25; pp.query("strong", strong);
    Real trunc  = 0.0;  pp.query("trunc", trunc);
    int p_max_elmts    = -1; pp.query("p_max_elmts", p_max_elmts);
    int agg_num_levels = -1; pp.query("agg_num_levels", agg_num_levels);
    int num_paths      = -1; pp.query("num_paths", num_paths);
    int print_level    = 0;  pp.query("print_level", print_level);
    static int first = 1;
    if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
      first = 0;
      std::cout << "hmabec.gmres_restart            = " << gmres_restart << std::endl;
      std::cout << "hmabec.interp          = " << interp << std::endl;
      std::cout << "hmabec.coarsen         = " << coarsen << std::endl;
      std::cout << "hmabec.relax           = " << relax << std::endl;
      std::cout << "hmabec.strong          = " << strong << std::endl;
      std::cout << "hmabec.trunc           = " << trunc << std::endl;
      if (p_max_elmts >= 0) {
        std::cout << "hmabec.p_max_elmts     = " << p_max_elmts << std::endl;
      }
      if (agg_num_levels >= 0) {
        std::cout << "hmabec.agg_num_levels  = " << agg_num_levels << std::endl;
      }
      if (num_paths >= 0) {
        std::cout << "hmabec.num_paths       = " << num_paths << std::endl;
      }
    }

    HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver);
    HYPRE_ParCSRGMRESSetKDim(solver, gmres_restart);
    HYPRE_ParCSRGMRESSetMaxIter(solver, maxiter);
    HYPRE_ParCSRGMRESSetTol(solver, reltol);

    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetMaxIter(precond, 1);
    HYPRE_BoomerAMGSetTol(precond, reltol);

    HYPRE_BoomerAMGSetInterpType(precond, interp);
    HYPRE_BoomerAMGSetCoarsenType(precond, coarsen);
    HYPRE_BoomerAMGSetRelaxType(precond, relax);
    HYPRE_BoomerAMGSetStrongThreshold(precond, strong);
    HYPRE_BoomerAMGSetTruncFactor(precond, trunc);
    if (p_max_elmts >= 0) {
      HYPRE_BoomerAMGSetPMaxElmts(precond, p_max_elmts);
    }
    if (agg_num_levels >= 0) {
      HYPRE_BoomerAMGSetAggNumLevels(precond, agg_num_levels);
    }
    if (num_paths >= 0) {
      HYPRE_BoomerAMGSetNumPaths(precond, num_paths);
    }
    HYPRE_BoomerAMGSetPrintLevel(precond, print_level);
    HYPRE_ParCSRGMRESSetPrecond(solver,
                                (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSolve,
                                (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSetup,
                                precond);

    HYPRE_ParCSRGMRESSetup(solver, par_A, par_b, par_x);
  }
  else if (solver_choice == 106) {
    ParmParse pp("hmabec");
    int struct_solve_iters = 1; pp.query("struct_solve_iters", struct_solve_iters);
    int struct_method_id = query_struct_method_id();
    static int first = 1;
    if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
      first = 0;
      std::cout << "hmabec.struct_solve_iters     = " << struct_solve_iters << std::endl;
      if (struct_method_id == 0) {
        std::cout << "hmabec.struct_method_id     = HYPRE_SMG" << std::endl;
      }
      else if (!(struct_method_id == 1)) {
        std::cout << "hmabec.struct_method_id     = HYPRE_Jacobi" << std::endl;
      } else {
        std::cout << "hmabec.struct_method_id     = HYPRE_PFMG" << std::endl;
      }
    }

    HYPRE_SStructSplitCreate(MPI_COMM_WORLD, &solver_handle);
    HYPRE_SStructSplitSetMaxIter(solver_handle, maxiter);
#if 0
    HYPRE_SStructSplitSetStructIterLimit(solver_handle, struct_solve_iters);
#endif
    HYPRE_SStructSplitSetTol(solver_handle, reltol);
    if (struct_method_id == 0) {
      HYPRE_SStructSplitSetStructSolver(solver_handle, HYPRE_SMG);
    }
    else if (struct_method_id == 1) {
      HYPRE_SStructSplitSetStructSolver(solver_handle, HYPRE_PFMG);
    }
    else {
#if 0
      HYPRE_SStructSplitSetStructSolver(solver_handle, HYPRE_Jacobi);
#endif
    }

    HYPRE_SStructSplitSetup(solver_handle, A, b, x);
  }
  else if (solver_choice == 107) {
    ParmParse pp("hmabec");
    int gmres_restart = 5; pp.query("gmres_restart", gmres_restart);
    int split_precond_iters  = 1; pp.query("split_precond_iters",  split_precond_iters);
    int struct_solve_iters = 1; pp.query("struct_solve_iters", struct_solve_iters);
    int struct_method_id = query_struct_method_id();
    static int first = 1;
    if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
      first = 0;
      std::cout << "hmabec.gmres_restart            = " << gmres_restart << std::endl;
      std::cout << "hmabec.split_precond_iters      = " << split_precond_iters << std::endl;
      std::cout << "hmabec.struct_solve_iters     = " << struct_solve_iters << std::endl;
      if (struct_method_id == 0) {
        std::cout << "hmabec.struct_method_id     = HYPRE_SMG" << std::endl;
      }
      else if (struct_method_id == 1) {
        std::cout << "hmabec.struct_method_id     = HYPRE_PFMG" << std::endl;
      }
      else {
        std::cout << "hmabec.struct_method_id     = HYPRE_Jacobi" << std::endl;
      }
    }

    HYPRE_SStructGMRESCreate(MPI_COMM_WORLD, &solver_handle);
    HYPRE_SStructGMRESSetKDim(solver_handle, gmres_restart);
    HYPRE_SStructGMRESSetMaxIter(solver_handle, maxiter);
    HYPRE_SStructGMRESSetTol(solver_handle, reltol);

    HYPRE_SStructSplitCreate(MPI_COMM_WORLD, &split_preconditioner);
    HYPRE_SStructSplitSetMaxIter(split_preconditioner, split_precond_iters);
#if 0
    HYPRE_SStructSplitSetStructIterLimit(split_preconditioner, struct_solve_iters);
#endif
    HYPRE_SStructSplitSetTol(split_preconditioner, 0.0);
    if (struct_method_id == 0) {
      HYPRE_SStructSplitSetStructSolver(split_preconditioner, HYPRE_SMG);
    }
    else if (struct_method_id == 1) {
      HYPRE_SStructSplitSetStructSolver(split_preconditioner, HYPRE_PFMG);
    }
    else {
#if 0
      HYPRE_SStructSplitSetStructSolver(split_preconditioner, HYPRE_Jacobi);
#endif
    }

    HYPRE_SStructGMRESSetPrecond(solver_handle,
                        (HYPRE_PtrToSStructSolverFcn) HYPRE_SStructSplitSolve,
                        (HYPRE_PtrToSStructSolverFcn) HYPRE_SStructSplitSetup,
                        split_preconditioner);
    HYPRE_SStructGMRESSetup(solver_handle, A, b, x);
  }
  else if (solver_choice == 108) {
    HYPRE_StructMatrix s_A;
    HYPRE_StructVector s_b;
    HYPRE_StructVector s_x;
    HYPRE_SStructMatrixGetObject(A, (void**) &s_A);
    HYPRE_SStructVectorGetObject(b, (void**) &s_b);
    HYPRE_SStructVectorGetObject(x, (void**) &s_x);
    HYPRE_StructSolver& struct_solve_handle = *(HYPRE_StructSolver*)&solver;

    ParmParse pp("hmabec");
    int struct_method_id = query_struct_method_id();
    static int first = 1;
    if (struct_method_id == 0) {
      if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
        first = 0;
        std::cout << "hmabec.struct_method_id     = HYPRE_SMG" << std::endl;
      }
      HYPRE_StructSMGCreate(MPI_COMM_WORLD, &struct_solve_handle);
      HYPRE_StructSMGSetMemoryUse(struct_solve_handle, 0);
      HYPRE_StructSMGSetMaxIter(struct_solve_handle, maxiter);
      HYPRE_StructSMGSetRelChange(struct_solve_handle, 0);
      HYPRE_StructSMGSetTol(struct_solve_handle, reltol);
      HYPRE_StructSMGSetNumPreRelax(struct_solve_handle, 1);
      HYPRE_StructSMGSetNumPostRelax(struct_solve_handle, 1);
      HYPRE_StructSMGSetLogging(struct_solve_handle, 1);
      HYPRE_StructSMGSetup(struct_solve_handle, s_A, s_b, s_x);
    }
    else if (struct_method_id == 1) {
      int pfmg_smoother_type = 1; pp.query("pfmg_smoother_type", pfmg_smoother_type);
      if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
        first = 0;
        std::cout << "hmabec.struct_method_id     = HYPRE_PFMG" << std::endl;
        std::cout << "hmabec.pfmg_smoother_type = " << pfmg_smoother_type << std::endl;
      }
      HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &struct_solve_handle);
      HYPRE_StructPFMGSetSkipRelax(struct_solve_handle, 0);
      HYPRE_StructPFMGSetMaxIter(struct_solve_handle, maxiter);
      HYPRE_StructPFMGSetRelChange(struct_solve_handle, 0);
      HYPRE_StructPFMGSetTol(struct_solve_handle, reltol);
      HYPRE_StructPFMGSetRelaxType(struct_solve_handle, pfmg_smoother_type);
      HYPRE_StructPFMGSetNumPreRelax(struct_solve_handle, 1);
      HYPRE_StructPFMGSetNumPostRelax(struct_solve_handle, 1);
      HYPRE_StructPFMGSetLogging(struct_solve_handle, 1);
      HYPRE_StructPFMGSetup(struct_solve_handle, s_A, s_b, s_x);
    }
  }
  else if (solver_choice == 109) {
    HYPRE_StructMatrix s_A;
    HYPRE_StructVector s_b;
    HYPRE_StructVector s_x;
    HYPRE_SStructMatrixGetObject(A, (void**) &s_A);
    HYPRE_SStructVectorGetObject(b, (void**) &s_b);
    HYPRE_SStructVectorGetObject(x, (void**) &s_x);
    HYPRE_StructSolver& struct_pc_handle = *(HYPRE_StructSolver*)&precond;
    HYPRE_StructSolver& struct_solve_handle  = *(HYPRE_StructSolver*)&solver;

    ParmParse pp("hmabec");
    int gmres_restart = 5; pp.query("gmres_restart", gmres_restart);
    int struct_method_id = query_struct_method_id();

    HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &struct_solve_handle);
    HYPRE_GMRESSetKDim((HYPRE_Solver) struct_solve_handle, gmres_restart);
    HYPRE_StructGMRESSetMaxIter(struct_solve_handle, maxiter);
    HYPRE_StructGMRESSetTol(struct_solve_handle, reltol);

    static int first = 1;
    if (struct_method_id == 0) {
      if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
        first = 0;
        std::cout << "hmabec.gmres_restart            = " << gmres_restart << std::endl;
        std::cout << "hmabec.struct_method_id     = HYPRE_SMG" << std::endl;
      }
      HYPRE_StructSMGCreate(MPI_COMM_WORLD, &struct_pc_handle);
      HYPRE_StructSMGSetMemoryUse(struct_pc_handle, 0);
      HYPRE_StructSMGSetMaxIter(struct_pc_handle, 1);
      HYPRE_StructSMGSetRelChange(struct_pc_handle, 0);
      HYPRE_StructSMGSetTol(struct_pc_handle, reltol);
      HYPRE_StructSMGSetNumPreRelax(struct_pc_handle, 1);
      HYPRE_StructSMGSetNumPostRelax(struct_pc_handle, 1);
      HYPRE_StructSMGSetLogging(struct_pc_handle, 1);
      HYPRE_StructGMRESSetPrecond(struct_solve_handle,
                       (HYPRE_PtrToStructSolverFcn) HYPRE_StructSMGSolve,
                       (HYPRE_PtrToStructSolverFcn) HYPRE_StructSMGSetup,
                       struct_pc_handle);
    }
    else if (struct_method_id == 1) {
      int pfmg_smoother_type = 1; pp.query("pfmg_smoother_type", pfmg_smoother_type);
      int pfmg_pre_smooth  = 1; pp.query("pfmg_pre_smooth",  pfmg_pre_smooth);
      int pfmg_post_smooth = 1; pp.query("pfmg_post_smooth", pfmg_post_smooth);
      if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
        first = 0;
        std::cout << "hmabec.gmres_restart            = " << gmres_restart << std::endl;
        std::cout << "hmabec.struct_method_id     = HYPRE_PFMG" << std::endl;
        std::cout << "hmabec.pfmg_smoother_type = " << pfmg_smoother_type << std::endl;
        std::cout << "hmabec.pfmg_pre_smooth  = " << pfmg_pre_smooth << std::endl;
        std::cout << "hmabec.pfmg_post_smooth = " << pfmg_post_smooth << std::endl;
      }
      HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &struct_pc_handle);
      HYPRE_StructPFMGSetSkipRelax(struct_pc_handle, 0);
      HYPRE_StructPFMGSetMaxIter(struct_pc_handle, 1);
      HYPRE_StructPFMGSetRelChange(struct_pc_handle, 0);
      HYPRE_StructPFMGSetTol(struct_pc_handle, reltol);
      HYPRE_StructPFMGSetRelaxType(struct_pc_handle, pfmg_smoother_type);
      HYPRE_StructPFMGSetNumPreRelax(struct_pc_handle, pfmg_pre_smooth);
      HYPRE_StructPFMGSetNumPostRelax(struct_pc_handle, pfmg_post_smooth);
      HYPRE_StructPFMGSetLogging(struct_pc_handle, 1);
      HYPRE_StructGMRESSetPrecond(struct_solve_handle,
                       (HYPRE_PtrToStructSolverFcn) HYPRE_StructPFMGSolve,
                       (HYPRE_PtrToStructSolverFcn) HYPRE_StructPFMGSetup,
                       struct_pc_handle);
    }

    HYPRE_StructGMRESSetup(struct_solve_handle, s_A, s_b, s_x);
  }

  else if (solver_choice == 150) {
    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_SStructMatrixGetObject(A, (void**) &par_A);
    HYPRE_SStructVectorGetObject(b, (void**) &par_b);
    HYPRE_SStructVectorGetObject(x, (void**) &par_x);
#if 0
    HYPRE_SStructMatrixPrint("A", A, 0);
    HYPRE_SStructVectorPrint("B", b, 0);
    std::cout << "MultiLevelEllipticSolver: creating solver" << std::endl;
#endif
    HYPRE_BoomerAMGCreate(&solver);
    HYPRE_BoomerAMGSetMinIter(solver, 1);
    HYPRE_BoomerAMGSetMaxIter(solver, maxiter);
    HYPRE_BoomerAMGSetTol(solver, reltol);
#if 1
    HYPRE_BoomerAMGSetInterpType(solver, 6);
    HYPRE_BoomerAMGSetCoarsenType(solver, 8);
    HYPRE_BoomerAMGSetStrongThreshold(solver, 0.25);
    HYPRE_BoomerAMGSetTruncFactor(solver, 0.);
#endif
    HYPRE_BoomerAMGSetup(solver, par_A, par_b, par_x);
#if 0
    int num_sweeps[4] = {2,2,2,1};
    HYPRE_BoomerAMGSetNumGridSweeps(solver, num_sweeps);
#endif
  }
  else if (solver_choice == 151 || solver_choice == 153) {
    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_SStructMatrixGetObject(A, (void**) &par_A);
    HYPRE_SStructVectorGetObject(b, (void**) &par_b);
    HYPRE_SStructVectorGetObject(x, (void**) &par_x);

    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_PCGSetMaxIter(solver, maxiter);
    HYPRE_PCGSetTol(solver, reltol);
    HYPRE_PCGSetTwoNorm(solver, 1 );
    HYPRE_PCGSetRelChange(solver, 0 );

    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetTol(precond, 0.0);
    HYPRE_BoomerAMGSetMaxIter(precond, 1);
    HYPRE_BoomerAMGSetCoarsenType(precond, 8);
    HYPRE_BoomerAMGSetInterpType(precond, 6);
    HYPRE_BoomerAMGSetStrongThreshold(precond, 0.25);
    HYPRE_BoomerAMGSetTruncFactor(precond, 0.);
    if (solver_choice == 151) {
       HYPRE_BoomerAMGSetAggNumLevels(precond, 1);
       HYPRE_BoomerAMGSetNumPaths(precond, 2);
    }
    HYPRE_PCGSetPrecond(solver,
                        (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                        (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                        precond );

    HYPRE_PCGSetup(solver, (HYPRE_Matrix)par_A,
                   (HYPRE_Vector)par_b, (HYPRE_Vector)par_x);
  }

  else if (solver_choice == 152) {
    HYPRE_SStructPCGCreate(MPI_COMM_WORLD, (HYPRE_SStructSolver *) &solver);
    HYPRE_PCGSetMaxIter(solver, maxiter);
    HYPRE_PCGSetTol(solver, reltol);
    HYPRE_PCGSetTwoNorm(solver, 1 );
    HYPRE_PCGSetRelChange(solver, 0 );

    HYPRE_SStructSplitCreate(MPI_COMM_WORLD, (HYPRE_SStructSolver *) &precond);
    HYPRE_SStructSplitSetMaxIter((HYPRE_SStructSolver) precond, 1);
    HYPRE_SStructSplitSetTol((HYPRE_SStructSolver) precond, 0.0);
    HYPRE_SStructSplitSetZeroGuess((HYPRE_SStructSolver) precond);
    HYPRE_SStructSplitSetStructSolver((HYPRE_SStructSolver) precond, HYPRE_PFMG);
    HYPRE_PCGSetPrecond(solver,
                        (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSolve,
                        (HYPRE_PtrToSolverFcn) HYPRE_SStructSplitSetup,
                        precond);

    HYPRE_PCGSetup(solver, (HYPRE_Matrix) A,
                   (HYPRE_Vector) b, (HYPRE_Vector) x);

  }
  else {
    std::cout << "MultiLevelEllipticSolver: no such solver" << std::endl;
    exit(1);
  }
}

void MultiLevelEllipticSolver::releaseSolver()
{
  BL_PROFILE("MultiLevelEllipticSolver::releaseSolver");

  if (solver_choice == 100) {
    HYPRE_BoomerAMGDestroy(solver);
  }
  else if (solver_choice == 101) {
    HYPRE_SStructFACDestroy2(solver_handle);
  }
  else if (solver_choice == 102) {
    HYPRE_ParCSRGMRESDestroy(solver);
  }
  else if (solver_choice == 103) {
    HYPRE_SStructGMRESDestroy(solver_handle);
  }
  else if (solver_choice == 1002) {
    HYPRE_ParCSRPCGDestroy(solver);
  }
  else if (solver_choice == 1003) {
    HYPRE_SStructPCGDestroy(solver_handle);
  }
  else if (solver_choice == 104 || solver_choice == 105) {
    HYPRE_ParCSRGMRESDestroy(solver);
    HYPRE_BoomerAMGDestroy(precond);
  }
  else if (solver_choice == 106) {
    HYPRE_SStructSplitDestroy(solver_handle);
  }
  else if (solver_choice == 107) {
    HYPRE_SStructGMRESDestroy(solver_handle);
    HYPRE_SStructSplitDestroy(split_preconditioner);
  }
  else if (solver_choice == 108) {
    int struct_method_id = query_struct_method_id();
    if (struct_method_id == 0) {
      HYPRE_StructSMGDestroy((HYPRE_StructSolver) solver);
    }
    else {
      HYPRE_StructPFMGDestroy((HYPRE_StructSolver) solver);
    }
  }
  else if (solver_choice == 109) {
    int struct_method_id = query_struct_method_id();
    HYPRE_StructGMRESDestroy((HYPRE_StructSolver) solver);
    if (!(struct_method_id == 0)) {
      HYPRE_StructPFMGDestroy((HYPRE_StructSolver) precond);
    } else {
      HYPRE_StructSMGDestroy((HYPRE_StructSolver) precond);
    }
  }

  if (solver_choice == 150) {
    HYPRE_BoomerAMGDestroy(solver);
  }
  else if (solver_choice == 151 || solver_choice == 153) {
    HYPRE_ParCSRPCGDestroy(solver);
    HYPRE_BoomerAMGDestroy(precond);
  }
  else if (solver_choice == 152) {
    HYPRE_SStructPCGDestroy((HYPRE_SStructSolver) solver);
    HYPRE_SStructSplitDestroy((HYPRE_SStructSolver) precond);
  }

  solver_handle = NULL;
  solver         = NULL;
  precond        = NULL;
}

void MultiLevelEllipticSolver::solve()
{
  BL_PROFILE("MultiLevelEllipticSolver::solve");

  if (abstol > 0.0) {
    Real bnorm;
    hypre_SStructInnerProd((hypre_SStructVector *) b,
                           (hypre_SStructVector *) b,
                           &bnorm);
    bnorm = std::sqrt(bnorm);

    Real volume = 0.0;
    for (int level = crse_level; level <= fine_level; level++) {
      for (int i = 0; i < grids[level].size(); i++) {
        volume += grids[level][i].numPts();
      }
    }

    Real reltol_new = ((!(bnorm > 0.0) ? (reltol) : (abstol / bnorm * std::sqrt(volume))));

    if (reltol_new > reltol) {
      if (solver_choice == 100) {
        HYPRE_BoomerAMGSetTol(solver, reltol_new);
      }
      else if(solver_choice == 101) {
        HYPRE_SStructFACSetTol(solver_handle, reltol_new);
      }
      else if(solver_choice == 102) {
        HYPRE_ParCSRGMRESSetTol(solver, reltol_new);
      }
      else if (solver_choice == 103 || solver_choice == 107) {
        HYPRE_SStructGMRESSetTol(solver_handle, reltol_new);
      }
      else if(solver_choice == 1002) {
        HYPRE_ParCSRPCGSetTol(solver, reltol_new);
      }
      else if (solver_choice == 1003) {
        HYPRE_SStructPCGSetTol(solver_handle, reltol_new);
      }
      else if (solver_choice == 104 || solver_choice == 105) {
        HYPRE_ParCSRGMRESSetTol(solver, reltol_new);
        HYPRE_BoomerAMGSetTol(precond, reltol_new);
      }
      else if (solver_choice == 106) {
        HYPRE_SStructSplitSetTol(solver_handle, reltol_new);
      }
      else if (solver_choice == 108) {
        int struct_method_id = query_struct_method_id();
        if (!(struct_method_id == 0)) {
          HYPRE_StructPFMGSetTol((HYPRE_StructSolver) solver, reltol_new);
        } else {
          HYPRE_StructSMGSetTol((HYPRE_StructSolver) solver, reltol_new);
        }
      }
      else if (solver_choice == 109) {
        HYPRE_StructGMRESSetTol((HYPRE_StructSolver) solver, reltol_new);
      }
      else if (solver_choice == 150) {
        HYPRE_BoomerAMGSetTol(solver, reltol_new);
      }
      else if(solver_choice == 151 || solver_choice == 153) {
        HYPRE_PCGSetTol(solver, reltol_new);
      }
      else if(solver_choice == 152) {
        HYPRE_PCGSetTol(solver, reltol_new);
      }
    }
  }

  if (solver_choice == 100) {
    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_SStructMatrixGetObject(A, (void**) &par_A);
    HYPRE_SStructVectorGetObject(b, (void**) &par_b);
    HYPRE_SStructVectorGetObject(x, (void**) &par_x);
    HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);
  }
  else if (solver_choice == 101) {
    HYPRE_SStructFACSolve3(solver_handle, A, b, x);
  }
  else if (solver_choice == 102) {
    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_SStructMatrixGetObject(A, (void**) &par_A);
    HYPRE_SStructVectorGetObject(b, (void**) &par_b);
    HYPRE_SStructVectorGetObject(x, (void**) &par_x);

    HYPRE_ParCSRGMRESSolve(solver, par_A, par_b, par_x);
  }
  else if (solver_choice == 103 || solver_choice == 107) {
    HYPRE_SStructGMRESSolve(solver_handle, A, b, x);
  }
  else if (solver_choice == 1002) {
    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_SStructMatrixGetObject(A, (void**) &par_A);
    HYPRE_SStructVectorGetObject(b, (void**) &par_b);
    HYPRE_SStructVectorGetObject(x, (void**) &par_x);

    HYPRE_ParCSRPCGSolve(solver, par_A, par_b, par_x);
  }
  else if (solver_choice == 1003) {
    HYPRE_SStructPCGSolve(solver_handle, A, b, x);
  }
  else if (solver_choice == 104 || solver_choice == 105) {
    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_SStructMatrixGetObject(A, (void**) &par_A);
    HYPRE_SStructVectorGetObject(b, (void**) &par_b);
    HYPRE_SStructVectorGetObject(x, (void**) &par_x);

    HYPRE_ParCSRGMRESSolve(solver, par_A, par_b, par_x);
  }
  else if (solver_choice == 106) {
    HYPRE_SStructSplitSolve(solver_handle, A, b, x);
  }
  else if (solver_choice == 108) {
    HYPRE_StructMatrix s_A;
    HYPRE_StructVector s_b;
    HYPRE_StructVector s_x;
    HYPRE_SStructMatrixGetObject(A, (void**) &s_A);
    HYPRE_SStructVectorGetObject(b, (void**) &s_b);
    HYPRE_SStructVectorGetObject(x, (void**) &s_x);
    int struct_method_id = query_struct_method_id();
    if (!(struct_method_id == 0)) {
      HYPRE_StructPFMGSolve((HYPRE_StructSolver) solver, s_A, s_b, s_x);
    } else {
      HYPRE_StructSMGSolve((HYPRE_StructSolver) solver, s_A, s_b, s_x);
    }
  }
  else if (solver_choice == 109) {
    HYPRE_StructMatrix s_A;
    HYPRE_StructVector s_b;
    HYPRE_StructVector s_x;
    HYPRE_SStructMatrixGetObject(A, (void**) &s_A);
    HYPRE_SStructVectorGetObject(b, (void**) &s_b);
    HYPRE_SStructVectorGetObject(x, (void**) &s_x);
    HYPRE_StructGMRESSolve((HYPRE_StructSolver) solver, s_A, s_b, s_x);
  }
  else if (solver_choice == 150) {
    int num_iterations, myid;
    Real res;

    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_SStructMatrixGetObject(A, (void**) &par_A);
    HYPRE_SStructVectorGetObject(b, (void**) &par_b);
    HYPRE_SStructVectorGetObject(x, (void**) &par_x);
    HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);
    HYPRE_SStructVectorGather(x);

    HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
    HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &res);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid );
    if (!myid)
    {
       std::cout << num_iterations << " Hypre Multigrid Iterations_inside, Relative Residual "
            << res << std::endl;
    }
  }
  else if (solver_choice == 151 || solver_choice == 153) {
    int num_iterations, myid;
    Real res;

    HYPRE_ParCSRMatrix par_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_SStructMatrixGetObject(A, (void**) &par_A);
    HYPRE_SStructVectorGetObject(b, (void**) &par_b);
    HYPRE_SStructVectorGetObject(x, (void**) &par_x);
    HYPRE_PCGSolve(solver, (HYPRE_Matrix)par_A,
                   (HYPRE_Vector)par_b, (HYPRE_Vector)par_x);
    HYPRE_SStructVectorGather(x);

    HYPRE_PCGGetNumIterations(solver, &num_iterations);
    HYPRE_PCGGetFinalRelativeResidualNorm(solver, &res);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid );
    if (!myid)
    {
       std::cout << num_iterations << " Hypre Multigrid Iterations_inside, Relative Residual "
            << res << std::endl;
    }
  }
  else if (solver_choice == 152) {
    int num_iterations, myid;
    Real res;

    HYPRE_PCGSolve(solver, (HYPRE_Matrix) A,
                   (HYPRE_Vector) b, (HYPRE_Vector) x);
    HYPRE_SStructVectorGather(x);

    HYPRE_PCGGetNumIterations(solver, &num_iterations);
    HYPRE_PCGGetFinalRelativeResidualNorm(solver, &res);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid );
    if (!myid)
    {
       std::cout << num_iterations << " Hypre Multigrid Iterations_inside, Relative Residual "
            << res << std::endl;
    }
  }

  HYPRE_SStructVectorGather(x);

  if (verbose >= 2 && ParallelDescriptor::IOProcessor()) {
    int num_iterations;
    Real res;
    if (solver_choice == 100) {
      HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &res);
    }
    else if (solver_choice == 101) {
      HYPRE_SStructFACGetNumIterations(solver_handle, &num_iterations);
      HYPRE_SStructFACGetFinalRelativeResidualNorm(solver_handle, &res);
    }
    else if (solver_choice == 102) {
      HYPRE_ParCSRGMRESGetNumIterations(solver, &num_iterations);
      HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(solver, &res);
    }
    else if (solver_choice == 103 || solver_choice == 107) {
      HYPRE_SStructGMRESGetNumIterations(solver_handle, &num_iterations);
      HYPRE_SStructGMRESGetFinalRelativeResidualNorm(solver_handle, &res);
    }
    else if (solver_choice == 1002) {
      HYPRE_ParCSRPCGGetNumIterations(solver, &num_iterations);
      HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(solver, &res);
    }
    else if (solver_choice == 1003) {
      HYPRE_SStructPCGGetNumIterations(solver_handle, &num_iterations);
      HYPRE_SStructPCGGetFinalRelativeResidualNorm(solver_handle, &res);
    }
    else if (solver_choice == 104 || solver_choice == 105) {
      HYPRE_ParCSRGMRESGetNumIterations(solver, &num_iterations);
      HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(solver, &res);
    }
    else if (solver_choice == 106) {
      HYPRE_SStructSplitGetNumIterations(solver_handle, &num_iterations);
      HYPRE_SStructSplitGetFinalRelativeResidualNorm(solver_handle, &res);
    }
    else if (solver_choice == 108) {
      int struct_method_id = query_struct_method_id();
      HYPRE_StructSolver& struct_solve_handle = *(HYPRE_StructSolver*)&solver;
      if (!(struct_method_id == 0)) {
        HYPRE_StructPFMGGetNumIterations(struct_solve_handle, &num_iterations);
        HYPRE_StructPFMGGetFinalRelativeResidualNorm(struct_solve_handle, &res);
      } else {
        HYPRE_StructSMGGetNumIterations(struct_solve_handle, &num_iterations);
        HYPRE_StructSMGGetFinalRelativeResidualNorm(struct_solve_handle, &res);
      }
    }
    else if (solver_choice == 109) {
      HYPRE_StructSolver& struct_solve_handle = *(HYPRE_StructSolver*)&solver;
      HYPRE_StructGMRESGetNumIterations(struct_solve_handle, &num_iterations);
      HYPRE_StructGMRESGetFinalRelativeResidualNorm(struct_solve_handle, &res);
    }
    else if (solver_choice == 150) {
      HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &res);
    }
    else if (solver_choice == 151 || solver_choice == 152 || solver_choice == 153) {
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &res);
    }

    if (num_iterations >= log_threshold) {
      int oldprec = std::cout.precision(20);
      if (Radiation::active_group_id >= 0) {
        std::cout << Radiation::group_label << " Group "
             << Radiation::active_group_id << ": ";
      }
      std::cout << num_iterations
           << " Hypre Multigrid Iterations, Relative Residual "
           << res << std::endl;
      std::cout.precision(oldprec);
    }

  }
}

void MultiLevelEllipticSolver::getSolution(int level, MultiFab& dest, int icomp)
{
  BL_PROFILE("MultiLevelEllipticSolver::getSolution");

  int part = level - crse_level;

  FArrayBox scratch_fab;
  for (MFIter mfi(dest); mfi.isValid(); ++mfi) {
    int i = mfi.index();
    const Box &reg = grids[level][i];

    FArrayBox *f;
    int fcomp;
    if (!(dest.nGrow() == 0)) {
      f = &scratch_fab;
      f->resize(reg);

      fcomp = 0;
    } else {
      f = &dest[mfi];
      fcomp = icomp;
    }
    Elixir f_hold = scratch_fab.elixir();

    readVectorBox(x, part, reg, subgrids[level][i], *f, fcomp);

    if (dest.nGrow() != 0) {
        dest[mfi].copy<RunOn::Device>(*f, 0, icomp, 1);
    }
  }
}

void MultiLevelEllipticSolver::getProduct(int level, MultiFab& product)
{
  BL_PROFILE("MultiLevelEllipticSolver::getProduct");

  int part = level - crse_level;

  for (MFIter mfi(product); mfi.isValid(); ++mfi) {
    int i = mfi.index();
    const Box &reg = grids[level][i];

    readVectorBox(b, part, reg, subgrids[level][i], product[i], 0);
  }
}

Real MultiLevelEllipticSolver::currentResidualNorm()
{
  BL_PROFILE("MultiLevelEllipticSolver::currentResidualNorm");

  Real bnorm;
  hypre_SStructInnerProd((hypre_SStructVector *) b,
                         (hypre_SStructVector *) b,
                         &bnorm);
  bnorm = std::sqrt(bnorm);

  Real res;
  if (solver_choice == 100) {
    HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &res);
  }
  else if (solver_choice == 101) {
    HYPRE_SStructFACGetFinalRelativeResidualNorm(solver_handle, &res);
  }
  else if (solver_choice == 102) {
    HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(solver, &res);
  }
  else if (solver_choice == 103 || solver_choice == 107) {
    HYPRE_SStructGMRESGetFinalRelativeResidualNorm(solver_handle, &res);
  }
  else if (solver_choice == 1002) {
    HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(solver, &res);
  }
  else if (solver_choice == 1003) {
    HYPRE_SStructPCGGetFinalRelativeResidualNorm(solver_handle, &res);
  }
  else if (solver_choice == 104 || solver_choice == 105) {
    HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(solver, &res);
  }
  else if (solver_choice == 106) {
    HYPRE_SStructSplitGetFinalRelativeResidualNorm(solver_handle, &res);
  }
  else if (solver_choice == 108) {
    int struct_method_id = query_struct_method_id();
    HYPRE_StructSolver& struct_solve_handle = *(HYPRE_StructSolver*)&solver;
    if (!(struct_method_id == 0)) {
      HYPRE_StructPFMGGetFinalRelativeResidualNorm(struct_solve_handle, &res);
    } else {
      HYPRE_StructSMGGetFinalRelativeResidualNorm(struct_solve_handle, &res);
    }
  }
  else if (solver_choice == 109) {
    HYPRE_StructSolver& struct_solve_handle = *(HYPRE_StructSolver*)&solver;
    HYPRE_StructGMRESGetFinalRelativeResidualNorm(struct_solve_handle, &res);
  }
  else if (solver_choice == 150) {
    HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &res);
  }
  else if (solver_choice == 151 || solver_choice == 152 || solver_choice == 153) {
    HYPRE_PCGGetFinalRelativeResidualNorm(solver, &res);
  }

  Real volume = 0.0;
  for (int level = crse_level; level <= fine_level; level++) {
    for (int i = 0; i < grids[level].size(); i++) {
      volume += grids[level][i].numPts();
    }
  }

  return bnorm * res / std::sqrt(volume);
}

void MultiLevelEllipticSolver::setScalars(Real Alpha, Real Beta)
{
  alpha = Alpha;
  beta  = Beta;
}

void MultiLevelEllipticSolver::setMassCoefficients(int level, const MultiFab &a)
{
  BL_PROFILE("MultiLevelEllipticSolver::setMassCoefficients");

  BL_ASSERT( a.ok() );
  BL_ASSERT( a.boxArray() == acoefs[level]->boxArray() );
  MultiFab::Copy(*acoefs[level], a, 0, 0, 1, 0);
}

void MultiLevelEllipticSolver::setFaceCoefficients(int level, const MultiFab &b, int dir)
{
  BL_PROFILE("MultiLevelEllipticSolver::setFaceCoefficients");

  BL_ASSERT( b.ok() );
  BL_ASSERT( b.boxArray() == (*bcoefs[level])[dir].boxArray() );
  MultiFab::Copy((*bcoefs[level])[dir], b, 0, 0, 1, 0);
}

void MultiLevelEllipticSolver::setBcAlphaCoef(int level, const MultiFab& a)
{
  BL_ASSERT( a.ok() );
  if (! SPa[level]) {
    SPa[level].reset(new MultiFab(grids[level],dmap[level],1,0));
    BL_ASSERT( a.boxArray() == SPa[level]->boxArray() );
    BL_ASSERT( a.DistributionMap() == SPa[level]->DistributionMap() );
  }
  MultiFab::Copy(*SPa[level], a, 0, 0, 1, 0);
}

void MultiLevelEllipticSolver::hmac (const Box& bx,
                           Array4<GpuArray<Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
                           Array4<Real const> const& a,
                           Real alpha)
{
    BL_PROFILE("MultiLevelEllipticSolver::hmac");

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        if (alpha == 0.e0_rt) {
            mat(i,j,k)[0] = 0.e0_rt;
        }
        else {
            mat(i,j,k)[0] = alpha * a(i,j,k);
        }
    });

    Gpu::synchronize();
}

void MultiLevelEllipticSolver::faceFluxAtBoundary(int level,
                                  MultiFab* Flux,
                                  MultiFab& sol_field,
                                  int icomp,
                                  BoundaryMode bc_variant)
{
    BL_PROFILE("MultiLevelEllipticSolver::faceFluxAtBoundary");

    const Box& domain = bd[level]->getDomain();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        for (MFIter mfi(sol_field); mfi.isValid(); ++mfi) {
            int i = mfi.index();
            const Box &reg = grids[level][i];
            for (OrientationIter oitr; oitr; oitr++) {
                int cdir(oitr());
                int idim = oitr().coordDir();
                const RadBoundaryCondition &bct = bd[level]->bndryConds(oitr())[i];
                const Real      &bcl = bd[level]->bndryLocs(oitr())[i];
                const FArrayBox       &fs  = bd[level]->bndryValues(oitr())[mfi];
                const Mask      &msk = bd[level]->bndryMasks(oitr(), i);
                const Box &fbox = Flux[idim][mfi].box();
                const Box &sbox = sol_field[mfi].box();
                const Box &msb  = msk.box();
                const Box &bbox = (*bcoefs[level])[idim][mfi].box();
                if (reg[oitr()] == domain[oitr()]) {
                    int bctype = bct;
                    Array4<int const> bc_type_arr;
                    if (bd[level]->isMixedFace(oitr())) {
                        const BaseFab<int> &tf = *(bd[level]->boundaryTypes(oitr())[i]);
                        bc_type_arr = tf.array();
                        bctype = -1;
                    }
                    Array4<Real const> bc_alpha_arr;
                    if (SPa[level]) {
                        bc_alpha_arr = (*SPa[level])[mfi].array();
                    }

                    ELLIPTIC_OPS::face_flux_3d(Flux[idim][mfi].array(),
                                  sol_field[mfi].array(icomp),
                                  reg,
                                  cdir, bctype,
                                  bc_type_arr,
                                  bho, bcl,
                                  fs.array(bc_component),
                                  msk.array(),
                                  (*bcoefs[level])[idim][mfi].array(),
                                  beta, geom[level].CellSize(),
                                  flux_scale, oitr(),
                                  geom[level].data(), bc_variant,
                                  bc_alpha_arr);
                }
                else {
                    ELLIPTIC_OPS::face_flux(Flux[idim][mfi].array(),
                                 sol_field[mfi].array(icomp),
                                 reg,
                                 cdir, bct, bho, bcl,
                                 fs.array(bc_component),
                                 msk.array(),
                                 (*bcoefs[level])[idim][mfi].array(),
                                 beta, geom[level].CellSize(), bc_variant);
                }
            }
        }
    }
}

void MultiLevelEllipticSolver::writeVectorBox(HYPRE_SStructVector x,
                                        int part,
                                        const Box& reg,
                                        const BoxArray& sgr,
                                        Real *vec)
{
  BL_PROFILE("MultiLevelEllipticSolver::writeVectorBox");

  if (sgr.size() > 0) {
    FArrayBox vec_values_fab;
    for (int j = 0; j < sgr.size(); j++) {
      const Box& sreg = sgr[j];
      vec_values_fab.resize(sreg);
      Real *vec_values = vec_values_fab.dataPtr();
      for (IntVect v = sreg.smallEnd(); v <= sreg.bigEnd(); sreg.next(v)) {
        int is = sreg.index(v);
        int ir =  reg.index(v);
        vec_values[is] = vec[ir];
      }
      HYPRE_SStructVectorSetBoxValues(x, part, loV(sreg), hiV(sreg), 0, vec_values);
    }
  }
  else {
    HYPRE_SStructVectorSetBoxValues(x, part, loV(reg), hiV(reg), 0, vec);
  }
}

void MultiLevelEllipticSolver::readVectorBox(HYPRE_SStructVector x,
                                        int part,
                                        const Box& reg,
                                        const BoxArray& sgr,
                                        FArrayBox& f, int fcomp)
{
  BL_PROFILE("MultiLevelEllipticSolver::readVectorBox");

  BL_ASSERT(f.box() == reg);
  Real* vec = f.dataPtr(fcomp);
  if (sgr.size() > 0) {
    f.setVal<RunOn::Host>(0.0, fcomp);
    FArrayBox vec_values_fab;
    for (int j = 0; j < sgr.size(); j++) {
      const Box& sreg = sgr[j];
      vec_values_fab.resize(sreg);
      Real *vec_values = vec_values_fab.dataPtr();
      HYPRE_SStructVectorGetBoxValues(x, part, loV(sreg), hiV(sreg), 0, vec_values);
      for (IntVect v = sreg.smallEnd(); v <= sreg.bigEnd(); sreg.next(v)) {
        int is = sreg.index(v);
        int ir =  reg.index(v);
        vec[ir] = vec_values[is];
      }
    }
  }
  else {
    HYPRE_SStructVectorGetBoxValues(x, part, loV(reg), hiV(reg), 0, vec);
  }
}
