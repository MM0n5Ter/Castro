#include <Radiation.H>

#include <AMReX_ParmParse.H>

#include <MultiLevelEllipticSolver.H>
#include <ELLIPTIC_OPS.H>
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

Real MultiLevelEllipticSolver::flux_scale = 1.0;

#if (AMREX_SPACEDIM == 1)
int MultiLevelEllipticSolver::vl[2] = { 0, 0 };
int MultiLevelEllipticSolver::vh[2] = { 0, 0 };
#endif

void AuxVar::collapse()
{
  // "Flattens" the dependency list.  Any entry that points to another
  // AuxVar is replaced by a copy of the dependency list from that AuxVar.
  // The values in the inserted elements are multiplied by the value from
  // the entry being replaced.  AuxVar pointers that were in the dependency
  // list are removed from it, but the AuxVars they point to are not
  // themselves modified.  Entries pointing to the same cell are then
  // combined.

  // The loops in this function are tricky, since we're modifying
  // a list at the same time we're looping through it.  for statements
  // are used in somewhat unusual ways.  Erasing an entry invalidates
  // its iterator, so before we can do that we need to have another
  // iterator pointing to an adjacent entry.  While "past-the-end" is
  // a valid target for an iterator, we assume there is no corresponding
  // "before-the-beginning" position.

  // There is one property that can be useful in dealing with the hypre
  // semi-structured interface:  Call two AuxVar objects or two AuxLink
  // lists "similar" if they point to the same locations in the same order.
  // If two AuxVar objects are built whose AuxLink lists are similar up to
  // a point, and all the latter entries point only to locations already
  // referenced in the similar initial sections, then after being
  // collapsed the two AuxVars will be similar.  If one AuxVar is then
  // used for establishing the graph entries, and the other is used for
  // initializing the matrix coefficients, then it is ok for the second
  // AuxVar to include additional entries so long as they don't point
  // to any new locations not already established in the graph.

  for (std::list<AuxLink>::iterator it = a.begin(); it != a.end(); ) {
    AuxVar *o = it->other;
    if (o) {
      std::list<AuxLink>::iterator kt = it;
      a.insert(++kt, o->a.begin(), o->a.end());

      // multiply inserted entries by it->var here:
      for (std::list<AuxLink>::iterator jt = it; ++jt != kt; ) {
        jt->val *= it->val;
      }

      kt = it; // entry to be erased
      ++it;
      a.erase(kt);
    }
    else {
      ++it;
    }
  }

  // Combine entries that point to same cell:

  for (std::list<AuxLink>::iterator it = a.begin(); it != a.end(); ++it) {
    for (std::list<AuxLink>::iterator jt = it; ++jt != a.end(); ) {
      if (it->same_destination(*jt)) {
        it->val += jt->val;
        std::list<AuxLink>::iterator kt = jt;
        --jt;
        a.erase(kt);
      }
    }
  }
}

void AuxVar::clear()
{
  a.clear();
  secondary_pushed = 0;
}

int AuxVar::read_stencil_sites(Vector<int>& levels, Vector<IntVect>& cells)
{
  if (secondary()) {
    return 1; // failure
  }

  int n = a.size();
  levels.resize(n);
  cells.resize(n);

  int i = 0;
  for (std::list<AuxLink>::iterator it = a.begin(); it != a.end(); ++it) {
    if (it->other != NULL) {
      return 2; // failure
    }
    levels[i] = it->level;
    cells[i]  = it->index;
    i++;
  }

  return 0; // success
}

int AuxVar::read_stencil_values(Vector<Real>& values)
{
  if (secondary()) {
    return 1; // failure
  }

  int n = a.size();
  values.resize(n);

  int i = 0;
  for (std::list<AuxLink>::iterator it = a.begin(); it != a.end(); ++it) {
    if (it->other != NULL) {
      return 2; // failure
    }
    values[i] = it->val;
    i++;
  }

  return 0; // success
}

int BoundaryAuxBase::firstOwnedIndex()
{
  const int MyProc = ParallelDescriptor::MyProc();
  int i = 0;
  while (distributionMap[i] != MyProc) {
    i++;
    if (i >= distributionMap.size()) {
        i = -1;
        break;
    }
  }
  return i;
}

int BoundaryAuxBase::nextOwnedIndex(int i)
{
  const int MyProc = ParallelDescriptor::MyProc();
  do {
    i++;
    if (i >= distributionMap.size()) {
        i = -1;
        break;
    }
  } while (distributionMap[i] != MyProc);
  return i;
}

BoundaryAuxStore::BoundaryAuxStore(const BoxArray& _grids,
                         const DistributionMapping& _dmap,
                         Location loc)
    : BoundaryAuxBase(_dmap), grids(_grids)
{
  // For Location type OUTSIDE_GRID use CoarseBoundaryAuxStore instead:
  BL_ASSERT(loc == INTERIOR || loc == HALO);

  int normal_side_index = (loc == INTERIOR) ? 0 : 1;

  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();
    aux[ori].resize(grids.size());
    int ishift = ori.isLow() ? 1-normal_side_index : normal_side_index-1;
    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      Box b = amrex::adjCell(grids[i], ori);
      aux[ori][i].reset(new BoxAuxStorage(b.shift(ori.coordDir(), ishift)));
    }
  }

  // Make primary-secondary connections:

  if (loc == INTERIOR) {
    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
#if 1
      // This is the current default implementation:
      for (OrientationIter outer_orient_iter; outer_orient_iter; ++outer_orient_iter) {
        Orientation om = outer_orient_iter();
        const Box& bm = aux[om][i]->box();
        for (OrientationIter inner_orient_iter; inner_orient_iter; ++inner_orient_iter) {
          Orientation os = inner_orient_iter();
          if (os.coordDir() > om.coordDir()) {
            const Box& bs = aux[os][i]->box();
            if (bm.intersects(bs)) {
              Box reg = (bm & bs);
              for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
                  (*aux[om][i])(v).add_secondary_link(&(*aux[os][i])(v));
              }
            }
          }
        }
      }
#elif 0
      // This was the original implementation, 2D only:
      Orientation face_x_low(0, Orientation::low);
      Orientation face_x_high(0, Orientation::high);
      Orientation face_y_low(1, Orientation::low);
      Orientation face_y_high(1, Orientation::high);
      IntVect p = aux[face_x_low][i]->box().smallEnd();
      (*aux[face_x_low][i])(p).add_secondary_link(&(*aux[face_y_low][i])(p));
      p = aux[face_x_low][i]->box().bigEnd();
      (*aux[face_x_low][i])(p).add_secondary_link(&(*aux[face_y_high][i])(p));
      p = aux[face_x_high][i]->box().smallEnd();
      (*aux[face_x_high][i])(p).add_secondary_link(&(*aux[face_y_low][i])(p));
      p = aux[face_x_high][i]->box().bigEnd();
      (*aux[face_x_high][i])(p).add_secondary_link(&(*aux[face_y_high][i])(p));
#elif 0
      // This version is like the new default, except that
      // it loops through orientations in a different order.
      // Some primary/secondary pairs are therefore flipped, and in
      // the end the solvers return slightly different numbers.
      // (Results should be the same within the solver tolerance.)
      for (OrientationIter outer_orient_iter; outer_orient_iter; ++outer_orient_iter) {
        Orientation om = outer_orient_iter();
        const Box& bm = aux[om][i]->box();
        for (OrientationIter inner_orient_iter(om); ++inner_orient_iter; ) {
          Orientation os = inner_orient_iter();
          const Box& bs = aux[os][i]->box();
          if (bm.intersects(bs)) {
            Box reg = (bm & bs);
            for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
                (*aux[om][i])(v).add_secondary_link(&(*aux[os][i])(v));
            }
          }
        }
      }
#endif
    }
  }
}

CoarseBoundaryAuxStore::CoarseBoundaryAuxStore(const BoxArray& _coarse_boxes,
                                 const DistributionMapping& _coarse_dmap,
                                 const BoxArray& _fine_boxes, Location loc)
    : BoundaryAuxBase(_coarse_dmap), coarse_boxes(_coarse_boxes), fgrids(_fine_boxes)
{
  // For Location type INTERIOR use BoundaryAuxStore instead:
  BL_ASSERT(loc == OUTSIDE_GRID || loc == HALO);

  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();

    aux[ori].resize(coarse_boxes.size());
    msk[ori].resize(coarse_boxes.size());

    fine_map[ori].resize(coarse_boxes.size());

    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      std::list<Box> bl;
      std::list<int> fi;
      for (int j = 0; j < fgrids.size(); j++) {
        Box face = amrex::adjCell(fgrids[j], ori);
        if (coarse_boxes[i].intersects(face)) {
          bl.push_back(face & coarse_boxes[i]);
          fi.push_back(j);
        }
      }

      // bl now has every fine grid face in it, even those entirely
      // covered by other fine grids.  We might optimize by removing
      // the covered ones here.

      int n = bl.size();
      aux[ori][i].resize(n);
      msk[ori][i].resize(n);

      fine_map[ori][i].resize(n);

      int j = 0;
      for (std::list<int>::iterator it = fi.begin(); it != fi.end(); ++it) {
        fine_map[ori][i][j++] = *it;
      }

      j = 0;
      for (std::list<Box>::iterator it = bl.begin(); it != bl.end(); ++it) {
        aux[ori][i][j].reset(new BoxAuxStorage(*it));
        Box mask_region = *it;
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
          if (dir == ori.coordDir())
            continue;
          mask_region.grow(dir,1);
        }
        msk[ori][i][j].reset(new Mask(mask_region));

        msk[ori][i][j]->setVal<RunOn::Host>(RadBoundaryData::outside_domain);

        // If we had the Geometry, we would have the problem domain
        // and would not have to loop over coarse grids below.  Also,
        // if we ever care to do this right for periodic domains we
        // will need the Geometry.  (See RadBoundaryData.cpp)

        for (int k = 0; k < coarse_boxes.size(); k++) {
          if (coarse_boxes[k].intersects(mask_region)) {
              msk[ori][i][j]->setVal<RunOn::Host>(RadBoundaryData::not_covered,
                                                  (coarse_boxes[k] & mask_region), 0);
          }
        }

        for (int k = 0; k < fgrids.size(); k++) {
          if (fgrids[k].intersects(mask_region)) {
              msk[ori][i][j]->setVal<RunOn::Host>(RadBoundaryData::covered,
                                                  (fgrids[k] & mask_region), 0);
          }
        }

        j++;
      }
    }
  }

  link_secondary_entries(loc);
}

CoarseBoundaryAuxStore::CoarseBoundaryAuxStore(const CoarseBoundaryAuxStore& other, Location loc)
    : BoundaryAuxBase(other.distributionMap), coarse_boxes(other.coarse_boxes), fgrids(other.fgrids)
{
  // For Location type INTERIOR use BoundaryAuxStore instead:
  BL_ASSERT(loc == OUTSIDE_GRID || loc == HALO);

  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();

    aux[ori].resize(coarse_boxes.size());
    msk[ori].resize(coarse_boxes.size());

    fine_map[ori].resize(coarse_boxes.size());

    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      int n = other.aux[ori][i].size();

      aux[ori][i].resize(n);
      msk[ori][i].resize(n);

      fine_map[ori][i].resize(n);

      for (int j = 0; j < n; j++) {
        fine_map[ori][i][j] = other.fine_map[ori][i][j];

        aux[ori][i][j].reset(new BoxAuxStorage(other.aux[ori][i][j]->box()));

        msk[ori][i][j].reset(new Mask(other.msk[ori][i][j]->box()));
        msk[ori][i][j]->copy<RunOn::Host>(*other.msk[ori][i][j]);
      }
    }
  }

  link_secondary_entries(loc);
}

CoarseBoundaryAuxStore::CoarseBoundaryAuxStore(const BoxArray& _coarse_boxes,
                                 const BoxArray& _fine_boxes,
                                 const CoarseBoundaryAuxStore& other, Location loc)
    : BoundaryAuxBase(other.distributionMap), coarse_boxes(_coarse_boxes), fgrids(_fine_boxes)
{
  // For Location type INTERIOR use BoundaryAuxStore instead:
  BL_ASSERT(loc == OUTSIDE_GRID || loc == HALO);

  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();

    aux[ori].resize(coarse_boxes.size());
    msk[ori].resize(coarse_boxes.size());

    fine_map[ori].resize(coarse_boxes.size());

    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      int n = other.aux[ori][i].size();

      aux[ori][i].resize(n);
      msk[ori][i].resize(n);

      fine_map[ori][i].resize(n);

      for (int j = 0; j < n; j++) {
        fine_map[ori][i][j] = other.fine_map[ori][i][j];

        Box face = amrex::adjCell(fgrids[fine_map[ori][i][j]], ori);
        aux[ori][i][j].reset(new BoxAuxStorage(face & coarse_boxes[i]));

        Box mask_region = aux[ori][i][j]->box();
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
          if (dir == ori.coordDir())
            continue;
          mask_region.grow(dir,1);
        }
        msk[ori][i][j].reset(new Mask(mask_region));

        msk[ori][i][j]->setVal<RunOn::Host>(RadBoundaryData::outside_domain);

        // If we had the Geometry, we would have the problem domain
        // and would not have to loop over coarse grids below.  Also,
        // if we ever care to do this right for periodic domains we
        // will need the Geometry.  (See RadBoundaryData.cpp)

        for (int k = 0; k < coarse_boxes.size(); k++) {
          if (coarse_boxes[k].intersects(mask_region)) {
              msk[ori][i][j]->setVal<RunOn::Host>(RadBoundaryData::not_covered,
                                                  (coarse_boxes[k] & mask_region), 0);
          }
        }

        for (int k = 0; k < fgrids.size(); k++) {
          if (fgrids[k].intersects(mask_region)) {
              msk[ori][i][j]->setVal<RunOn::Host>(RadBoundaryData::covered,
                                                  (fgrids[k] & mask_region), 0);
          }
        }

      }
    }
  }

  link_secondary_entries(loc);
}

void CoarseBoundaryAuxStore::rebuild_links(Location loc)
{
  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();
    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      int n = aux[ori][i].size();
      for (int j = 0; j < n; j++) {
        const Box& reg = aux[ori][i][j]->box();
        for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
            (*aux[ori][i][j])(v).clear();
        }
      }
    }
  }

  link_secondary_entries(loc);
}

void CoarseBoundaryAuxStore::link_secondary_entries(Location loc)
{
  // Make primary-secondary connections:

  if (loc == OUTSIDE_GRID) {
    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      for (OrientationIter outer_orient_iter; outer_orient_iter; ++outer_orient_iter) {
        Orientation om = outer_orient_iter();
        for (OrientationIter inner_orient_iter(om); ++inner_orient_iter; ) {
          Orientation os = inner_orient_iter();

          for (int jm = 0; jm < aux[om][i].size(); jm++) {
            const Box& bm = aux[om][i][jm]->box();
            for (int js = 0; js < aux[os][i].size(); js++) {
                const Box& bs = aux[os][i][js]->box();

              if (bm.intersects(bs)) {
                Box reg = (bm & bs);
                for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
                    (*aux[om][i][jm])(v).add_secondary_link(&(*aux[os][i][js])(v));
                }
              }

            }
          }

        }
      }
    }
  }
}

void CoarseBoundaryAuxStore::makeFaceStorage(IntVect& rat, int ncomp)
{
  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();

    face_fabs[ori].resize(coarse_boxes.size());

    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      int n = aux[ori][i].size();
      face_fabs[ori][i].resize(n);
      for (int j = 0; j < n; j++) {
          Box face_box = aux[ori][i][j]->box();
        // face_box is known to be "adjacent cell", convert to face:
        face_box.shiftHalf(ori.coordDir(), (ori.isLow() ? 1 : -1));
        face_box.refine(rat);
        face_fabs[ori][i][j].reset(new FArrayBox(face_box, ncomp));
      }
    }
  }
}

void CoarseBoundaryAuxStore::resetFaceStorage(IntVect& rat, int ncomp)
{
  for (OrientationIter oitr; oitr; ++oitr) {
    Orientation ori = oitr();

    for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
      int n = face_fabs[ori][i].size();
      for (int j = 0; j < n; j++) {
        const Box& face_box = face_fabs[ori][i][j]->box();
        face_fabs[ori][i][j]->resize(face_box, ncomp);
      }
    }
  }
}

void CoarseBoundaryAuxStore::storeFaceCoeffs(const Orientation ori,
                                   MultiFab& src,
                                   int srccomp,
                                   int destcomp,
                                   int numcomp)
{
  MultiFabCopyDescriptor face_copier;

  MultiFabId mfid = face_copier.RegisterMultiFab(&src);

  Vector< Vector<FillBoxId> > fill_box_ids(face_fabs[ori].size());
  for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
    int n = face_fabs[ori][i].size();
    fill_box_ids[i].resize(n);
    for (int j = 0; j < n; j++) {
      fill_box_ids[i][j] = face_copier.AddBox(mfid, face_fabs[ori][i][j]->box(), NULL,
                               fine_map[ori][i][j],
                               srccomp, destcomp, numcomp);
    }
  }

  face_copier.CollectData();

  for (int i = firstOwnedIndex(); isValid(i); i = nextOwnedIndex(i)) {
    int n = face_fabs[ori][i].size();
    for (int j = 0; j < n; j++) {
      face_copier.FillFab(mfid, fill_box_ids[i][j], *face_fabs[ori][i][j]);
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

  // Hypre doesn't support 1D directly, so we use 2D Hypre with
  // the second dimension collapsed.
  // (SMG reduces to cyclic reduction in this case, so it's an exact solve.)
  // (PFMG will not work.)

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

  // All variables are cell-centered
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
    vf[idir] = reg.smallEnd(idir); // same as bigEnd(idir)
    Box face(vf, vf + ve);
    if (msk(vf) == RadBoundaryData::not_covered) {
#if (0)
      // force piecewise constant interpolation for debugging:
      for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
        coarse_interp(v).push(coarse_level_index, vc,     1.0);
      }
#elif (AMREX_SPACEDIM == 1)
      coarse_interp(vf).push(coarse_level_index, vc, 1.0);
#elif (AMREX_SPACEDIM == 2)
      if (msk(vf-vj1) != RadBoundaryData::not_covered &&
          msk(vf+vjr) == RadBoundaryData::not_covered) {
        // low direction not available, use linear interp upwards:
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real xx = (v[jdir] - vf[jdir] - 0.5 * ve[jdir]) / rat[jdir];
          coarse_interp(v).push(coarse_level_index, vc+vj1, xx);
          coarse_interp(v).push(coarse_level_index, vc,     1.0 - xx);
        }
      }
      else if (msk(vf-vj1) == RadBoundaryData::not_covered &&
               msk(vf+vjr) == RadBoundaryData::not_covered) {
        // use piecewise quadratic interpolation whenever possible:
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real xx = (v[jdir] - vf[jdir] - 0.5 * ve[jdir]) / rat[jdir];
          coarse_interp(v).push(coarse_level_index, vc+vj1, 0.5*xx*(xx+1));
          coarse_interp(v).push(coarse_level_index, vc,     1.0-xx*xx);
          coarse_interp(v).push(coarse_level_index, vc-vj1, 0.5*xx*(xx-1));
        }
      }
      else if (msk(vf-vj1) == RadBoundaryData::not_covered &&
               msk(vf+vjr) != RadBoundaryData::not_covered) {
        // high direction not available, use linear interp downwards:
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real xx = (v[jdir] - vf[jdir] - 0.5 * ve[jdir]) / rat[jdir];
          coarse_interp(v).push(coarse_level_index, vc,     1.0 + xx);
          coarse_interp(v).push(coarse_level_index, vc-vj1, -xx);
        }
      }
      else {
        // neither direction available, drop back to piecewise const:
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          coarse_interp(v).push(coarse_level_index, vc,     1.0);
        }
        //amrex::Error("Case not implemented");
      }
#elif (AMREX_SPACEDIM == 3)

      // First do the jdir direction, including piecewise-constant term:

      if (msk(vf-vj1) != RadBoundaryData::not_covered &&
          msk(vf+vjr) == RadBoundaryData::not_covered) {
        // low direction not available, use linear interp upwards:
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real xx = (v[jdir] - vf[jdir] - 0.5 * ve[jdir]) / rat[jdir];
          coarse_interp(v).push(coarse_level_index, vc+vj1, xx);
          coarse_interp(v).push(coarse_level_index, vc,     1.0 - xx);
        }
      }
      else if (msk(vf-vj1) == RadBoundaryData::not_covered &&
               msk(vf+vjr) == RadBoundaryData::not_covered) {
        // use piecewise quadratic interpolation whenever possible:
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real xx = (v[jdir] - vf[jdir] - 0.5 * ve[jdir]) / rat[jdir];
          coarse_interp(v).push(coarse_level_index, vc+vj1, 0.5*xx*(xx+1));
          coarse_interp(v).push(coarse_level_index, vc,     1.0-xx*xx);
          coarse_interp(v).push(coarse_level_index, vc-vj1, 0.5*xx*(xx-1));
        }
      }
      else if (msk(vf-vj1) == RadBoundaryData::not_covered &&
               msk(vf+vjr) != RadBoundaryData::not_covered) {
        // high direction not available, use linear interp downwards:
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real xx = (v[jdir] - vf[jdir] - 0.5 * ve[jdir]) / rat[jdir];
          coarse_interp(v).push(coarse_level_index, vc,     1.0 + xx);
          coarse_interp(v).push(coarse_level_index, vc-vj1, -xx);
        }
      }
      else {
        // neither direction available, drop back to piecewise const:
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          coarse_interp(v).push(coarse_level_index, vc,     1.0);
        }
      }

      // Then add on contributions from the kdir direction:

      if (msk(vf-vk1) != RadBoundaryData::not_covered &&
          msk(vf+vkr) == RadBoundaryData::not_covered) {
        // low direction not available, use linear interp upwards:
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real yy = (v[kdir] - vf[kdir] - 0.5 * ve[kdir]) / rat[kdir];
          coarse_interp(v).push(coarse_level_index, vc+vk1, yy);
          coarse_interp(v).push(coarse_level_index, vc,    -yy);
        }
      }
      else if (msk(vf-vk1) == RadBoundaryData::not_covered &&
               msk(vf+vkr) == RadBoundaryData::not_covered) {
        // use piecewise quadratic interpolation whenever possible:
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real yy = (v[kdir] - vf[kdir] - 0.5 * ve[kdir]) / rat[kdir];
          coarse_interp(v).push(coarse_level_index, vc+vk1, 0.5*yy*(yy+1));
          coarse_interp(v).push(coarse_level_index, vc,     -yy*yy);
          coarse_interp(v).push(coarse_level_index, vc-vk1, 0.5*yy*(yy-1));
        }
      }
      else if (msk(vf-vk1) == RadBoundaryData::not_covered &&
               msk(vf+vkr) != RadBoundaryData::not_covered) {
        // high direction not available, use linear interp downwards:
        for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
          Real yy = (v[kdir] - vf[kdir] - 0.5 * ve[kdir]) / rat[kdir];
          coarse_interp(v).push(coarse_level_index, vc,      yy);
          coarse_interp(v).push(coarse_level_index, vc-vk1, -yy);
        }
      }
      else {
        // neither direction available, drop back to piecewise const:
        // (do nothing, no need to explicitly add a zero contribution here)
      }

      // Finally add in cross derivative terms:

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
    Real normal_grad_scale = 2.0 / (h * (1 + r)); // normal derivative factor
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
  // Build MultiFabs first, so that distribution maps are available
  // when building the graph below:

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

  // This can be done now that attachLevel has been called for each level:

  HYPRE_SStructGridAssemble(hgrid);

  // Setup stencils:

#if (AMREX_SPACEDIM == 1)
  // if we were really 1D:
/*
  int offsets[3][1] = {{ 0}
                       {-1},
                       { 1}};
*/
  // fake 1D as a 2D problem:
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

  // Add non-stencil entries to the graph here:

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
      vin = (ori.isLow() ? -vin : vin); // outward normal unit vector
      Real h = geom[level].CellSize(idir); // normal fine grid spacing
      IntVect ve; // default constructor initializes to zero
#if (AMREX_SPACEDIM >= 2)
      int jdir = (idir + 1) % AMREX_SPACEDIM;
      IntVect vj1 = amrex::BASISV(jdir); // tangential unit vector
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
        Box coarse_box = amrex::coarsen(reg, rat); // coarse adjacent cells
        const Mask &msk = bd[level]->bndryMasks(ori,i);

        interpolate_transverse((*coarse_interp[level])(ori,i), msk, reg, coarse_box,
                              AMREX_D_DECL(rat,  vj1,  vk1),
                              AMREX_D_DECL(ve,   vjr,  vkr),
                              AMREX_D_DECL(idir, jdir, kdir),
                              level-1);

        compute_normal_gradient((*normal_grad_store[level])(ori,i),
                         (*coarse_interp[level])(ori,i),
                         msk, reg, vin, h, rat[idir], bho, level);

        // coarse_interp and normal_grad_store are now complete.  entry will be done in
        // draft form to establish the graph connections, but must be
        // done again later when the edge coefficients are available.

        reg.shift(-vin); // fine interior cells
        for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
          if (msk(v+vin) == RadBoundaryData::not_covered) {
            // value not important, since coefficient not known.
              entry(ori,i)(v).push(&(*normal_grad_store[level])(ori,i)(v+vin), 1.0);
          }
        }
      }
    }

    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      IntVect vin = amrex::BASISV(idir);
      vin = (ori.isLow() ? -vin : vin); // outward normal unit vector
      for (int i = entry.firstOwnedIndex(); entry.isValid(i);
           i = entry.nextOwnedIndex(i)) {
        Box reg = amrex::adjCell(grids[level][i], ori);
        reg.shift(-vin); // fine interior cells
//        const Mask &msk = bd[level]->bndryMasks(ori,i);
        for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
#if (0 && !defined(NDEBUG))
          if (msk(v+vin) == RadBoundaryData::not_covered &&
              entry(ori,i)(v).secondary()) {
            std::cout << v << " is secondary in orientation " << ori
                 << " on processor " << ParallelDescriptor::MyProc()
                 << std::endl;
          }
#endif
          // Even if this entry is covered, it could have a
          // not_covered secondary:
          if (!entry(ori,i)(v).empty() &&
              !entry(ori,i)(v).secondary()) {
            entry(ori,i)(v).collapse();
            Vector<int> levels;
            Vector<IntVect> cells;
            int retval = entry(ori,i)(v).read_stencil_sites(levels, cells);
            BL_ASSERT(retval == 0);
            for (int j = 0; j < levels.size(); j++) {
              // eliminate stencil-like connections:
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

    // Now add the graph entries seen by the coarse cells adjacent
    // to the coarse-fine interface.  These are averages of normal_grad_store
    // over the fine faces making up each coarse face.  Since we
    // have to do this from the processor owning the coarse grid,
    // we recompute information using CoarseBoundaryAuxStore

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
      vin = (ori.isLow() ? -vin : vin); // outward normal unit vector
      Real h = geom[level].CellSize(idir); // normal fine grid spacing
      IntVect ve; // default constructor initializes to zero
#if (AMREX_SPACEDIM >= 2)
      int jdir = (idir + 1) % AMREX_SPACEDIM;
      IntVect vj1 = amrex::BASISV(jdir); // tangential unit vector
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
           const Box& reg = (*c_interp[level])(ori,i,j).box(); // adjacent cells
           const Box& coarse_box = (*c_connections[level])(ori,i,j).box(); // adjacent cells
           const Mask &msk = c_interp[level]->mask(ori,i,j); // fine mask

         interpolate_transverse((*c_interp[level])(ori,i,j), msk, reg, coarse_box,
                                AMREX_D_DECL(rat,  vj1,  vk1),
                                AMREX_D_DECL(ve,   vjr,  vkr),
                                AMREX_D_DECL(idir, jdir, kdir),
                                level-1);

         compute_normal_gradient((*c_extrap_deriv[level])(ori,i,j),
                          (*c_interp[level])(ori,i,j),
                          msk, reg, vin, h, rat[idir], bho, level);

          // c_interp and c_extrap_deriv are now complete.  c_connections will be done
          // in draft form to establish the graph connections, but must be
          // done again later when the edge coefficients are available.

          for (IntVect vc = coarse_box.smallEnd(); vc <= coarse_box.bigEnd(); coarse_box.next(vc)) {
            IntVect vf = rat * vc;
            vf[idir] = reg.smallEnd(idir); // same as bigEnd(idir)
            Box face(vf, vf + ve);
            if (msk(vf) == RadBoundaryData::not_covered) {
              for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
                // value not important, since coefficient not known.
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
          const Box& reg = (*c_interp[level])(ori,i,j).box(); // adjacent cells
          const Box& coarse_box = (*c_connections[level])(ori,i,j).box(); // adjacent cells
          const Mask &msk = c_interp[level]->mask(ori,i,j); // fine mask
          for (IntVect vc = coarse_box.smallEnd(); vc <= coarse_box.bigEnd(); coarse_box.next(vc)) {
            IntVect vf = rat * vc;
            vf[idir] = reg.smallEnd(idir); // same as bigEnd(idir)
            // Unlike fine entry, it should not be possible for this
            // entry to be covered but have a not_covered secondary:
            if (msk(vf) == RadBoundaryData::not_covered &&
                !(*c_connections[level])(ori,i,j)(vc).secondary()) {
              (*c_connections[level])(ori,i,j)(vc).collapse();
              Vector<int> levels;
              Vector<IntVect> cells;
              int retval = (*c_connections[level])(ori,i,j)(vc)
                .read_stencil_sites(levels, cells);
              BL_ASSERT(retval == 0);
              for (int jj = 0; jj < levels.size(); jj++) {
                // eliminate stencil-like connections:
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
  //HYPRE_StructMatrixSetSymmetric(A, 1);
  //HYPRE_StructMatrixSetNumGhost(A, matrix_ghost_widths);
  HYPRE_SStructMatrixInitialize(A);

  BL_ASSERT(A0 == NULL);
  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A0);
  HYPRE_SStructMatrixSetObjectType(A0, ObjectType);
  //HYPRE_StructMatrixSetSymmetric(A0, 1);
  //HYPRE_StructMatrixSetNumGhost(A0, matrix_ghost_widths);
  HYPRE_SStructMatrixInitialize(A0);

  BL_ASSERT(b == NULL);
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, hgrid, &b);
  HYPRE_SStructVectorSetObjectType(b, ObjectType);

  BL_ASSERT(x == NULL);
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, hgrid, &x);
  HYPRE_SStructVectorSetObjectType(x, ObjectType);

  HYPRE_SStructVectorInitialize(b);
  HYPRE_SStructVectorInitialize(x);

  // the following is necessary in some cases before
  // we call the solver setup routine (and we may do that before loading
  // the vectors with data):

  HYPRE_SStructVectorAssemble(b);
  HYPRE_SStructVectorAssemble(x);
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

void MultiLevelEllipticSolver::fillBoundaryCoeffs (const Box& bx,
                           Array4<GpuArray<Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
                           Array4<Real const> const& b,
                           Real beta, const Real* dx, int n)
{
    BL_PROFILE("MultiLevelEllipticSolver::fillBoundaryCoeffs");

    if (n == 0) {

        const Real fac = beta / (dx[0] * dx[0]);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            mat(i,j,k)[0] += fac * (b(i,j,k) + b(i+1,j,k));
            mat(i,j,k)[1] = -fac * b(i,j,k);
            mat(i,j,k)[2] = -fac * b(i+1,j,k);
        });

    }
    else if (n == 1) {

        const Real fac = beta / (dx[1] * dx[1]);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            mat(i,j,k)[0] += fac * (b(i,j,k) + b(i,j+1,k));
            mat(i,j,k)[3] = -fac * b(i,j,k);
            mat(i,j,k)[4] = -fac * b(i,j+1,k);
        });

    }
    else {

        const Real fac = beta / (dx[2] * dx[2]);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            mat(i,j,k)[0] += fac * (b(i,j,k) + b(i,j,k+1));
            mat(i,j,k)[5] = -fac * b(i,j,k);
            mat(i,j,k)[6] = -fac * b(i,j,k+1);
        });

    }

    Gpu::synchronize();
}

void
MultiLevelEllipticSolver::fillLevelMat (const Box& bx,
                       Array4<GpuArray<Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
                       int cdir, int bct, int bho, Real bcl,
                       Array4<int const> const& mask,
                       Array4<Real const> const& b,
                       Real beta, const Real* dx)
{
    bool xlo = false;
    bool ylo = false;
    bool zlo = false;

    bool xhi = false;
    bool yhi = false;
    bool zhi = false;

    Real h;

    if (AMREX_SPACEDIM == 1) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            xhi = true;
            h = dx[0];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else if (AMREX_SPACEDIM == 2) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 2) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 3) {
            yhi = true;
            h = dx[1];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 3) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 4) {
            yhi = true;
            h = dx[1];
        }
        else if (cdir == 2) {
            zlo = true;
            h = dx[2];
        }
        else if (cdir == 5) {
            zhi = true;
            h = dx[2];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }

    const Real fac = beta / (h * h);

    Real bfm, bfv;
    Real stencil_weight_second, h2, th2;

    if (bct == AMREX_LO_DIRICHLET) {

        if (bho >= 1) {

            h2 = 0.5e0_rt * h;
            th2 = 3.e0_rt * h2;
            bfm = fac * (th2 - bcl) / (bcl + h2) - fac;
            stencil_weight_second = fac * (bcl - h2) / (bcl + th2);

        }
        else {

            bfv = (beta / h) / (0.5e0_rt * h + bcl);
            bfm = bfv - fac;

        }

    }
    else if (bct == AMREX_LO_NEUMANN) {

        bfm = -fac;
        stencil_weight_second = 0.e0_rt;

    }
    else {

        amrex::Error("fillLevelMat: unsupported boundary type");

    }

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        if (mask.contains(i-1,j,k)) {

            if (xlo && mask(i-1,j,k) > 0) {

                mat(i,j,k)[0] += bfm * b(i,j,k);
                mat(i,j,k)[1] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[2] += stencil_weight_second * b(i,j,k);
                }

            }

        }
        else if (mask.contains(i+1,j,k)) {

            if (xhi && mask(i+1,j,k) > 0) {

                mat(i,j,k)[0] += bfm * b(i+1,j,k);
                mat(i,j,k)[2] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[1] += stencil_weight_second * b(i+1,j,k);
                }

            }

        }
        else if (mask.contains(i,j-1,k)) {

            if (ylo && mask(i,j-1,k) > 0) {

                mat(i,j,k)[0] += bfm * b(i,j,k);
                mat(i,j,k)[3] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[4] += stencil_weight_second * b(i,j,k);
                }

            }

        }
        else if (mask.contains(i,j+1,k)) {

            if (yhi && mask(i,j+1,k) > 0) {

                mat(i,j,k)[0] += bfm * b(i,j+1,k);
                mat(i,j,k)[4] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[3] += stencil_weight_second * b(i,j+1,k);
                }

            }

        }
        else if (mask.contains(i,j,k-1)) {

            if (zlo && mask(i,j,k-1) > 0) {

                mat(i,j,k)[0] += bfm * b(i,j,k);
                mat(i,j,k)[5] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[6] += stencil_weight_second * b(i,j,k);
                }

            }

        }
        else if (mask.contains(i,j,k+1)) {

            if (zhi && mask(i,j,k+1) > 0) {

                mat(i,j,k)[0] += bfm * b(i,j,k+1);
                mat(i,j,k)[6] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[5] += stencil_weight_second * b(i,j,k+1);
                }

            }

        }
    });

    Gpu::synchronize();
}

void
MultiLevelEllipticSolver::fillLevelMat3 (const Box& bx,
                        int side_is_low, int idir,
                        Array4<GpuArray<Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
                        int cdir, int bctype,
                        Array4<int const> const& tf,
                        int bho, Real bcl,
                        Array4<int const> const& mask,
                        Array4<Real const> const& b,
                        Real beta, const Real* dx,
                        const GeometryData& geomdata, Real c,
                        Array4<Real const> const& spa)
{
    BL_PROFILE("MultiLevelEllipticSolver::fillLevelMat3");

    bool xlo = false;
    bool ylo = false;
    bool zlo = false;

    bool xhi = false;
    bool yhi = false;
    bool zhi = false;

    Real h;

    if (AMREX_SPACEDIM == 1) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            xhi = true;
            h = dx[0];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else if (AMREX_SPACEDIM == 2) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 2) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 3) {
            yhi = true;
            h = dx[1];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 3) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 4) {
            yhi = true;
            h = dx[1];
        }
        else if (cdir == 2) {
            zlo = true;
            h = dx[2];
        }
        else if (cdir == 5) {
            zhi = true;
            h = dx[2];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }

    const Real fac = beta / (h * h);

    // The -fac * b(i,j,k) term applied to the matrix diagonal is the contribution
    // from the interior stencil which must be removed at the boundary.

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        Real r;
        face_geometry_factors(i, j, k, bx.loVect()[0], bx.hiVect()[0], geomdata, idir, side_is_low, r);

        Real bfm, bfv;
        Real stencil_weight_second, h2, th2;
        int bct;

        if (mask.contains(i-1,j,k)) {

            if (xlo && mask(i-1,j,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i-1,j,k);
                }
                else {
                    bct = bctype;
                }

                if (bct == AMREX_LO_DIRICHLET) {

                    if (bho >= 1) {
                        h2 = 0.5e0_rt * h;
                        th2 = 3.e0_rt * h2;
                        bfm = fac * (th2 - bcl) / (bcl + h2)  * b(i,j,k);
                        stencil_weight_second = fac * (bcl - h2) / (bcl + th2) * b(i,j,k);
                    }
                    else  {
                        bfv = (beta / h) / (0.5e0_rt * h + bcl);
                        bfm = bfv * b(i,j,k);
                    }

                }
                else if (bct == AMREX_LO_NEUMANN) {

                    bfm  = 0.e0_rt;
                    stencil_weight_second = 0.e0_rt;

                }
                else if (bct == AMREX_LO_MARSHAK) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  0.375e0_rt * bfv;
                        stencil_weight_second = -0.125e0_rt * bfv;
                    }
                    else {
                        bfm = 0.25e0_rt * bfv;
                    }

                }
                else if (bct == AMREX_LO_SANCHEZ_POMRANING) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  1.5e0_rt * spa(i,j,k) * bfv;
                        stencil_weight_second = -0.5e0_rt * spa(i,j,k) * bfv;
                    }
                    else {
                        bfm = spa(i,j,k) * bfv;
                    }

                }
#ifndef AMREX_USE_GPU
                else {

                    amrex::Error("fillLevelMat3: unsupported boundary type");

                }
#endif

                mat(i,j,k)[0] += bfm - fac * b(i,j,k);
                mat(i,j,k)[1] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[2] += stencil_weight_second;
                }

            }

        }
        else if (mask.contains(i+1,j,k)) {

            if (xhi && mask(i+1,j,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i+1,j,k);
                }
                else {
                    bct = bctype;
                }

                if (bct == AMREX_LO_DIRICHLET) {

                    if (bho >= 1) {
                        h2 = 0.5e0_rt * h;
                        th2 = 3.e0_rt * h2;
                        bfm = fac * (th2 - bcl) / (bcl + h2)  * b(i+1,j,k);
                        stencil_weight_second = fac * (bcl - h2) / (bcl + th2) * b(i+1,j,k);
                    }
                    else {
                        bfv = (beta / h) / (0.5e0_rt * h + bcl);
                        bfm = bfv * b(i+1,j,k);
                    }

                }
                else if (bct == AMREX_LO_NEUMANN) {

                    bfm  = 0.e0_rt;
                    stencil_weight_second = 0.e0_rt;

                }
                else if (bct == AMREX_LO_MARSHAK) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  0.375e0_rt * bfv;
                        stencil_weight_second = -0.125e0_rt * bfv;
                    }
                    else {
                        bfm = 0.25e0_rt * bfv;
                    }

                }
                else if (bct == AMREX_LO_SANCHEZ_POMRANING) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  1.5e0_rt * spa(i,j,k) * bfv;
                        stencil_weight_second = -0.5e0_rt * spa(i,j,k) * bfv;
                    }
                    else {
                        bfm = spa(i,j,k) * bfv;
                    }

                }
#ifndef AMREX_USE_GPU
                else {

                    amrex::Error("fillLevelMat3: unsupported boundary type");

                }
#endif

                mat(i,j,k)[0] += bfm - fac * b(i+1,j,k);
                mat(i,j,k)[2] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[1] += stencil_weight_second;
                }

            }

        }
        else if (mask.contains(i,j-1,k)) {

            if (ylo && mask(i,j-1,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j-1,k);
                }
                else {
                    bct = bctype;
                }

                if (bct == AMREX_LO_DIRICHLET) {

                    if (bho >= 1) {
                        h2 = 0.5e0_rt * h;
                        th2 = 3.e0_rt * h2;
                        bfm = fac * (th2 - bcl) / (bcl + h2)  * b(i,j,k);
                        stencil_weight_second = fac * (bcl - h2) / (bcl + th2) * b(i,j,k);
                    }
                    else {
                        bfv = (beta / h) / (0.5e0_rt * h + bcl);
                        bfm = bfv * b(i,j,k);
                    }

                }
                else if (bct == AMREX_LO_NEUMANN) {

                    bfm  = 0.e0_rt;
                    stencil_weight_second = 0.e0_rt;

                }
                else if (bct == AMREX_LO_MARSHAK) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  0.375e0_rt * bfv;
                        stencil_weight_second = -0.125e0_rt * bfv;
                    }
                    else {
                        bfm = 0.25e0_rt * bfv;
                    }

                }
                else if (bct == AMREX_LO_SANCHEZ_POMRANING) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  1.5e0_rt * spa(i,j,k) * bfv;
                        stencil_weight_second = -0.5e0_rt * spa(i,j,k) * bfv;
                    }
                    else {
                        bfm = spa(i,j,k) * bfv;
                    }

                }
#ifndef AMREX_USE_GPU
                else {

                    amrex::Error("fillLevelMat3: unsupported boundary type");

                }
#endif

                mat(i,j,k)[0] += bfm - fac * b(i,j,k);
                mat(i,j,k)[3] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[4] += stencil_weight_second;
                }

            }

        }
        else if (mask.contains(i,j+1,k)) {

            if (yhi && mask(i,j+1,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j+1,k);
                }
                else {
                    bct = bctype;
                }

                if (bct == AMREX_LO_DIRICHLET) {

                    if (bho >= 1) {
                        h2 = 0.5e0_rt * h;
                        th2 = 3.e0_rt * h2;
                        bfm = fac * (th2 - bcl) / (bcl + h2)  * b(i,j+1,k);
                        stencil_weight_second = fac * (bcl - h2) / (bcl + th2) * b(i,j+1,k);
                    }
                    else {
                        bfv = (beta / h) / (0.5e0_rt * h + bcl);
                        bfm = bfv * b(i,j+1,k);
                    }

                }
                else if (bct == AMREX_LO_NEUMANN) {

                    bfm  = 0.e0_rt;
                    stencil_weight_second = 0.e0_rt;

                }
                else if (bct == AMREX_LO_MARSHAK) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  0.375e0_rt * bfv;
                        stencil_weight_second = -0.125e0_rt * bfv;
                    }
                    else {
                        bfm = 0.25e0_rt * bfv;
                    }

                }
                else if (bct == AMREX_LO_SANCHEZ_POMRANING) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  1.5e0_rt * spa(i,j,k) * bfv;
                        stencil_weight_second = -0.5e0_rt * spa(i,j,k) * bfv;
                    }
                    else {
                        bfm = spa(i,j,k) * bfv;
                    }

                }
#ifndef AMREX_USE_GPU
                else {

                    amrex::Error("fillLevelMat3: unsupported boundary type");

                }
#endif

                mat(i,j,k)[0] += bfm - fac * b(i,j+1,k);
                mat(i,j,k)[4] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[3] += stencil_weight_second;
                }

            }

        }
        else if (mask.contains(i,j,k-1)) {

            if (zlo && mask(i,j,k-1) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j,k-1);
                }
                else {
                    bct = bctype;
                }

                if (bct == AMREX_LO_DIRICHLET) {

                    if (bho >= 1) {
                        h2 = 0.5e0_rt * h;
                        th2 = 3.e0_rt * h2;
                        bfm = fac * (th2 - bcl) / (bcl + h2)  * b(i,j,k);
                        stencil_weight_second = fac * (bcl - h2) / (bcl + th2) * b(i,j,k);
                    }
                    else {
                        bfv = (beta / h) / (0.5e0_rt * h + bcl);
                        bfm = bfv * b(i,j,k);
                    }

                }
                else if (bct == AMREX_LO_NEUMANN) {

                    bfm  = 0.e0_rt;
                    stencil_weight_second = 0.e0_rt;

                }
                else if (bct == AMREX_LO_MARSHAK) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  0.375e0_rt * bfv;
                        stencil_weight_second = -0.125e0_rt * bfv;
                    }
                    else {
                        bfm = 0.25e0_rt * bfv;
                    }

                }
                else if (bct == AMREX_LO_SANCHEZ_POMRANING) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  1.5e0_rt * spa(i,j,k) * bfv;
                        stencil_weight_second = -0.5e0_rt * spa(i,j,k) * bfv;
                    }
                    else {
                        bfm = spa(i,j,k) * bfv;
                    }

                }
#ifndef AMREX_USE_GPU
                else {

                    amrex::Error("fillLevelMat3: unsupported boundary type");

                }
#endif

                mat(i,j,k)[0] += bfm - fac * b(i,j,k);
                mat(i,j,k)[5] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[6] += stencil_weight_second;
                }

            }

        }
        else if (mask.contains(i,j,k+1)) {

            if (zhi && mask(i,j,k+1) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j,k+1);
                }
                else {
                    bct = bctype;
                }

                if (bct == AMREX_LO_DIRICHLET) {

                    if (bho >= 1) {
                        h2 = 0.5e0_rt * h;
                        th2 = 3.e0_rt * h2;
                        bfm = fac * (th2 - bcl) / (bcl + h2)  * b(i,j,k+1);
                        stencil_weight_second = fac * (bcl - h2) / (bcl + th2) * b(i,j,k+1);
                    }
                    else {
                        bfv = (beta / h) / (0.5e0_rt * h + bcl);
                        bfm = bfv * b(i,j,k+1);
                    }

                }
                else if (bct == AMREX_LO_NEUMANN) {

                    bfm  = 0.e0_rt;
                    stencil_weight_second = 0.e0_rt;

                }
                else if (bct == AMREX_LO_MARSHAK) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  0.375e0_rt * bfv;
                        stencil_weight_second = -0.125e0_rt * bfv;
                    }
                    else {
                        bfm = 0.25e0_rt * bfv;
                    }

                }
                else if (bct == AMREX_LO_SANCHEZ_POMRANING) {

                    bfv = 2.e0_rt * c * beta * r / h;

                    if (bho >= 1) {
                        bfm  =  1.5e0_rt * spa(i,j,k) * bfv;
                        stencil_weight_second = -0.5e0_rt * spa(i,j,k) * bfv;
                    }
                    else {
                        bfm = spa(i,j,k) * bfv;
                    }

                }
#ifndef AMREX_USE_GPU
                else {

                    amrex::Error("fillLevelMat3: unsupported boundary type");

                }
#endif

                mat(i,j,k)[0] += bfm - fac * b(i,j,k+1);
                mat(i,j,k)[6] = 0.e0_rt;
                if (bho >= 1) {
                    mat(i,j,k)[5] += stencil_weight_second;
                }

            }

        }
    });

    Gpu::synchronize();
}

void MultiLevelEllipticSolver::assembleMatrix()
{
  BL_PROFILE("MultiLevelEllipticSolver::assembleMatrix");

#if (AMREX_SPACEDIM == 1)
  // if we were really 1D:
/*
  int offsets[3][1] = {{ 0}
                       {-1},
                       { 1}};
*/
  // fake 1D as a 2D problem:
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

  const int size = 2 * AMREX_SPACEDIM + 1;
  int i, idim;

  int stencil_indices[size];

  for (i = 0; i < size; i++) {
    stencil_indices[i] = i;
  }

  BaseFab<GpuArray<Real, size>> matfab; // AoS indexing
  FArrayBox sub_matrix_buf;
  for (int level = crse_level; level <= fine_level; level++) {
    int part = level - crse_level;

    for (MFIter mfi(*acoefs[level]); mfi.isValid(); ++mfi) {
      i = mfi.index();
      const Box &reg = grids[level][i];

      matfab.resize(reg);
      Real* mat = (Real*) matfab.dataPtr();
      Elixir mat_hold = matfab.elixir();

      // build matrix interior

      hmac(reg, matfab.array(), (*acoefs[level])[mfi].array(), alpha);

      for (idim = 0; idim < AMREX_SPACEDIM; idim++) {
          fillBoundaryCoeffs(reg, matfab.array(), (*bcoefs[level])[idim][mfi].array(),
               beta, geom[level].CellSize(), idim);
      }

      // add b.c.'s to matrix diagonal, and
      // zero out offdiag values at domain boundaries

      const Box& domain = bd[level]->getDomain();
      for (OrientationIter oitr; oitr; oitr++) {
        int cdir(oitr());
        idim = oitr().coordDir();
        const RadBoundaryCondition &bct = bd[level]->bndryConds(oitr())[i];
        const Real      &bcl = bd[level]->bndryLocs(oitr())[i];
        const Mask      &msk = bd[level]->bndryMasks(oitr(),i);
        const Box &bbox = (*bcoefs[level])[idim][mfi].box();
        const Box &msb  = msk.box();
        if (reg[oitr()] == domain[oitr()] || level == crse_level) {

          // Treat an exposed grid edge here as a boundary condition
          // for the linear solver:

          if (reg[oitr()] == domain[oitr()]) {
            Array4<const int> tfp{};
            int bctype = bct;
            if (bd[level]->isMixedFace(oitr())) {
              const BaseFab<int> &tf = *(bd[level]->boundaryTypes(oitr())[i]);
              tfp = tf.array();
              bctype = -1;
            }
            const FArrayBox &fs = bd[level]->bndryValues(oitr())[mfi];
            Array4<const Real> opt_bc_alpha_arr{};
            if (SPa[level]) {
                opt_bc_alpha_arr = (*SPa[level])[mfi].array();
            }
            fillLevelMat3(reg, oitr().isLow(), idim, matfab.array(), cdir, bctype,
                   tfp, bho, bcl, msk.array(), (*bcoefs[level])[idim][mfi].array(),
                   beta, geom[level].CellSize(), geom[level].data(),
                   flux_scale, opt_bc_alpha_arr);
          }
          else {
              fillLevelMat(reg, matfab.array(), cdir, bct, bho, bcl,
                    msk.array(), (*bcoefs[level])[idim][mfi].array(),
                    beta, geom[level].CellSize());
          }
        }
        else {

          // An exposed grid edge here actually borders the next coarser
          // level in the current linear system.  Zero out the interior
          // stencil using Neumann BC:

          const RadBoundaryCondition coarse_boundary_type = AMREX_LO_NEUMANN;
          fillLevelMat(reg, matfab.array(), cdir, coarse_boundary_type, bho, bcl,
                msk.array(), (*bcoefs[level])[idim][mfi].array(),
                beta, geom[level].CellSize());
        }
      }

      // initialize matrix

      if (subgrids[level][i].size() > 0) {
        for (int j = 0; j < subgrids[level][i].size(); j++) {
          const Box& sreg = subgrids[level][i][j];
          sub_matrix_buf.resize(sreg,size);
          Real* smat = sub_matrix_buf.dataPtr();
          for (IntVect v = sreg.smallEnd(); v <= sreg.bigEnd(); sreg.next(v)) {
            int is = sreg.index(v);
            int ir =  reg.index(v);
            for (int s = 0; s < size; s++) {
              smat[is * size + s] = mat[ir * size + s];
            }
          }
          HYPRE_SStructMatrixSetBoxValues(A, part, loV(sreg), hiV(sreg), 0,
                                          size, stencil_indices, smat);
        }
      }
      else {
        HYPRE_SStructMatrixSetBoxValues(A, part, loV(reg), hiV(reg), 0,
                                        size, stencil_indices, mat);
      }
    }

    // Add coarse-fine interface entries to the matrix here:
    if (level == crse_level) {
      continue;
    }

    // First we do the entries as seen by the fine cells adjacent
    // to the interface, working on the fine processor:

    BoundaryAuxStore entry(grids[level], dmap[level], BoundaryAuxStore::INTERIOR);
    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      IntVect vin = amrex::BASISV(idir), ves;
      vin = (ori.isLow() ? -vin : vin);    // outward normal unit vector
      ves = (ori.isLow() ?  ves : vin);    // edge shift vector
      Real h = geom[level].CellSize(idir); // normal fine grid spacing
      Real fine_div_factor = (-beta / h);             // divergence factor
      for (int i = coarse_interp[level]->firstOwnedIndex(); coarse_interp[level]->isValid(i);
           i = coarse_interp[level]->nextOwnedIndex(i)) {
        Box reg = amrex::adjCell(grids[level][i], ori);
        reg.shift(-vin); // fine interior cells
        const Mask &msk = bd[level]->bndryMasks(ori,i);
        for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
          if (msk(v+vin) == RadBoundaryData::not_covered) {
              entry(ori,i)(v).push(&(*normal_grad_store[level])(ori,i)(v+vin),
                                    fine_div_factor * (*bcoefs[level])[idir][i](v+ves));
          }
        }
      }
    }

    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      IntVect vin = amrex::BASISV(idir);
      vin = (ori.isLow() ? -vin : vin);    // outward normal unit vector
      for (int i = coarse_interp[level]->firstOwnedIndex(); coarse_interp[level]->isValid(i);
           i = coarse_interp[level]->nextOwnedIndex(i)) {
        Box reg = amrex::adjCell(grids[level][i], ori);
        reg.shift(-vin); // fine interior cells
//        const Mask &msk = bd[level]->bndryMasks(ori,i);
        for (IntVect v = reg.smallEnd(); v <= reg.bigEnd(); reg.next(v)) {
          if (!entry(ori,i)(v).empty() &&
              !entry(ori,i)(v).secondary()) {
            entry(ori,i)(v).collapse();
            Vector<int> levels;
            Vector<IntVect> cells;
            int retval = entry(ori,i)(v).read_stencil_sites(levels, cells);
            BL_ASSERT(retval == 0);
            Vector<Real> values;
            retval = entry(ori,i)(v).read_stencil_values(values);
            BL_ASSERT(retval == 0);
            int ientry = 2 * AMREX_SPACEDIM + 1;
            for (int j = 0; j < levels.size(); j++) {
              // identify stencil-like connections for separate treatment:
              int needs_separate_entry = 1;
              if (levels[j] == level) {
                IntVect d = cells[j] - v;
                for (int k = 0; k < 2 * AMREX_SPACEDIM + 1; k++) {
                  if (d == IntVect(offsets[k])) {
                    needs_separate_entry = 0;
                    HYPRE_SStructMatrixAddToValues(A, part, cellIndex1(v), 0,
                                                   1, &k, &values[j]);
                  }
                }
              }
              if (needs_separate_entry) {
                HYPRE_SStructMatrixSetValues(A, part, cellIndex1(v), 0,
                                             1, &ientry, &values[j]);
                ientry++;
              }
            }
          }
        }
      }
    }

    // Now add the matrix values seen by the coarse cells adjacent
    // to the coarse-fine interface.  These are averages of normal_grad_store
    // over the fine faces making up each coarse face.  We use
    // CoarseBoundaryAuxStore now because we need to work on the coarse
    // processor.

    IntVect rat = fine_ratio[level-1];
    const BoxArray& ff_boxes(grids[level]);
    const BoxArray& cc_boxes(grids[level-1]);
    BoxArray fc_boxes(cc_boxes);
    fc_boxes.refine(rat);
    BoxArray cf_boxes(ff_boxes);
    cf_boxes.coarsen(rat);
    //CoarseBoundaryAuxStore c_connections(cc_boxes, cf_boxes, BoundaryAuxStore::OUTSIDE_GRID);
    c_connections[level]->rebuild_links(BoundaryAuxStore::OUTSIDE_GRID);
    c_connections[level]->makeFaceStorage(rat);

    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      c_connections[level]->storeFaceCoeffs(ori, (*bcoefs[level])[idir], 0, 0, 1);
      IntVect vin = amrex::BASISV(idir), ves;
      vin = (ori.isLow() ? -vin : vin); // outward normal unit vector
      ves = (ori.isLow() ? -vin : ves); // edge shift vector (diff from above)
      Real hc = geom[level-1].CellSize(idir); // normal coarse grid spacing
      Real coarse_div_factor = (beta / hc);                // divergence factor
      //coarse_div_factor = 0.0;
      Real covered_div_factor = (-coarse_div_factor / hc);               // factor for covered cell
      //coarse_div_factor = 0.0;
      IntVect ve; // default constructor initializes to zero
#if (AMREX_SPACEDIM >= 2)
      int jdir = (idir + 1) % AMREX_SPACEDIM;
      ve += (rat[jdir] - 1) * amrex::BASISV(jdir);
      coarse_div_factor /= rat[jdir]; // will average over fine cells in tangential dir
#endif
#if (AMREX_SPACEDIM == 3)
      int kdir = (idir + 2) % 3;
      ve += (rat[kdir] - 1) * amrex::BASISV(kdir);
      coarse_div_factor /= rat[kdir]; // will average over fine cells in tangential dir
#endif
      for (int i = c_connections[level]->firstOwnedIndex(); c_connections[level]->isValid(i);
           i = c_connections[level]->nextOwnedIndex(i)) {
        // parallel loop is tied to coarse grids
          for (int j = 0; j < (*c_connections[level]).size(ori,i); j++) {
          const Box& reg = (*c_interp[level])(ori,i,j).box(); // adjacent cells
          const Box& coarse_box = (*c_connections[level])(ori,i,j).box(); // adjacent cells
          const Mask& msk = c_interp[level]->mask(ori,i,j); // fine mask
          const FArrayBox& face_coef_data = c_connections[level]->faceFab(ori,i,j);
          for (IntVect vc = coarse_box.smallEnd(); vc <= coarse_box.bigEnd(); coarse_box.next(vc)) {
            IntVect vf = rat * vc;
            vf[idir] = reg.smallEnd(idir); // same as bigEnd(idir)
            Box face(vf, vf + ve);
            if (msk(vf) == RadBoundaryData::not_covered) {
              // Zero out connection to covered coarse cell:
              (*c_connections[level])(ori,i,j)(vc).push(-1, vc-vin, 0.0);
              (*c_connections[level])(ori,i,j)(vc).push(level-1, vc,
                                                       covered_div_factor * (*bcoefs[level-1])[idir][i](vc+ves));
              // Add fine fluxes over face of coarse cell:
              for (IntVect v = vf; v <= face.bigEnd(); face.next(v)) {
                  (*c_connections[level])(ori,i,j)(vc)
                      .push(&(*c_extrap_deriv[level])(ori,i,j)(v), coarse_div_factor * face_coef_data(v+ves));
              }
            }
          }
        }
      }
    }

    for (OrientationIter oitr; oitr; ++oitr) {
      Orientation ori = oitr();
      int idir = ori.coordDir();
      for (int i = c_connections[level]->firstOwnedIndex(); c_connections[level]->isValid(i);
           i = c_connections[level]->nextOwnedIndex(i)) {
        // parallel loop is tied to coarse grids
          for (int j = 0; j < (*c_connections[level]).size(ori,i); j++) {
          const Box& reg = (*c_interp[level])(ori,i,j).box(); // adjacent cells
          const Box& coarse_box = (*c_connections[level])(ori,i,j).box(); // adjacent cells
          const Mask &msk = c_interp[level]->mask(ori,i,j); // fine mask
          for (IntVect vc = coarse_box.smallEnd(); vc <= coarse_box.bigEnd(); coarse_box.next(vc)) {
            IntVect vf = rat * vc;
            vf[idir] = reg.smallEnd(idir); // same as bigEnd(idir)
            if (msk(vf) == RadBoundaryData::not_covered &&
                !(*c_connections[level])(ori,i,j)(vc).secondary()) {
              (*c_connections[level])(ori,i,j)(vc).collapse();
              Vector<int> levels;
              Vector<IntVect> cells;
              int retval = (*c_connections[level])(ori,i,j)(vc)
                .read_stencil_sites(levels, cells);
              BL_ASSERT(retval == 0);
              Vector<Real> values;
              retval = (*c_connections[level])(ori,i,j)(vc).read_stencil_values(values);
              BL_ASSERT(retval == 0);
              int ientry = 2 * AMREX_SPACEDIM + 1;
              for (int jj = 0; jj < levels.size(); jj++) {
                // identify stencil-like connections for separate treatment:
                int needs_separate_entry = 1;
                if (levels[jj] == -1) {
                  // connection to covered coarse cell to be zeroed out:
                  IntVect d = cells[jj] - vc;
                  for (int k = 0; k < 2 * AMREX_SPACEDIM + 1; k++) {
                    if (d == IntVect(offsets[k])) {
                      needs_separate_entry = 0;
                      HYPRE_SStructMatrixSetValues(A, part-1, cellIndex1(vc), 0,
                                                   1, &k, &values[jj]);
                      BL_ASSERT(values[jj] == 0.0);
                    }
                  }
                  BL_ASSERT(needs_separate_entry == 0);
                }
                if (levels[jj] == level-1) {
                  // other coarse-level entry, may or may not be in stencil:
                  IntVect d = cells[jj] - vc;
                  for (int k = 0; k < 2 * AMREX_SPACEDIM + 1; k++) {
                    if (d == IntVect(offsets[k])) {
                      needs_separate_entry = 0;
                      HYPRE_SStructMatrixAddToValues(A, part-1, cellIndex1(vc), 0,
                                                     1, &k, &values[jj]);
                    }
                  }
                }
                if (needs_separate_entry) {
                  HYPRE_SStructMatrixSetValues(A, part-1, cellIndex1(vc), 0,
                                               1, &ientry, &values[jj]);
                  ientry++;
                }
              }
            }
          }
        }
      }
    }
  }
}

void MultiLevelEllipticSolver::commitMatrix()
{
  BL_PROFILE("MultiLevelEllipticSolver::commitMatrix");

  HYPRE_SStructMatrixAssemble(A);
}

void MultiLevelEllipticSolver::fillLevelVectors(int level,
                                      MultiFab& dest,
                                      int icomp,
                                      MultiFab& rhs,
                                      BoundaryMode bc_variant)
{
  BL_PROFILE("MultiLevelEllipticSolver::fillLevelVectors");

  int part = level - crse_level;

  Real *vec;
  FArrayBox scratch_fab;
  for (MFIter mfi(dest); mfi.isValid(); ++mfi) {
    int i = mfi.index();
    const Box &reg = grids[level][i];

    // initialize dest, since we will reuse the space to set up rhs below:

    FArrayBox *f;
    int fcomp;
    if (dest.nGrow() == 0) { // need a temporary if dest is the wrong size
      f = &dest[mfi];
      fcomp = icomp;
    }
    else {
      f = &scratch_fab;
      f->resize(reg);
      f->copy<RunOn::Device>(dest[mfi], icomp, 0, 1);
      fcomp = 0;
    }
    Elixir f_hold = scratch_fab.elixir();

    vec = f->dataPtr(fcomp); // sharing space, dest will be overwritten below

    writeVectorBox(x, part, reg, subgrids[level][i], vec);

    f->copy<RunOn::Device>(rhs[mfi], 0, fcomp, 1);

    // add b.c.'s to rhs

    if (bc_variant) {
      const Box& domain = bd[level]->getDomain();
      for (OrientationIter oitr; oitr; oitr++) {
        int cdir(oitr());
        int idim = oitr().coordDir();
        const RadBoundaryCondition &bct = bd[level]->bndryConds(oitr())[i];
        const Real      &bcl = bd[level]->bndryLocs(oitr())[i];
        const FArrayBox       &fs  = bd[level]->bndryValues(oitr())[mfi];
        const Mask      &msk = bd[level]->bndryMasks(oitr(), i);
        const Box &bbox = (*bcoefs[level])[idim][mfi].box();

        if (reg[oitr()] == domain[oitr()] || level == crse_level) {

          // Treat an exposed grid edge here as a boundary condition
          // for the linear solver:

          if (reg[oitr()] == domain[oitr()]) {
            Array4<const int> tfp{};
            int bctype = bct;
            if (bd[level]->isMixedFace(oitr())) {
              const BaseFab<int> &tf = *(bd[level]->boundaryTypes(oitr())[i]);
              tfp = tf.array();
              bctype = -1;
            }
            AlphaBetaSolver::fillBcVec3(reg,
                              oitr().isLow(), idim,
                              f->array(fcomp),
                              cdir, bctype,
                              tfp,
                              bho, bcl,
                              fs.array(bc_component),
                              msk.array(),
                              (*bcoefs[level])[idim][mfi].array(),
                              beta, geom[level].data());
          }
          else {
              AlphaBetaSolver::fillBcVec(reg, f->array(fcomp),
                               cdir, bct, bho, bcl,
                               fs.array(bc_component), msk.array(),
                               (*bcoefs[level])[idim][mfi].array(),
                               beta, geom[level].CellSize());
          }
        }
        // There is no else here, since we would then be at an
        // interior edge and would not need to add anything to vec.
      }
    }

    // initialize rhs

    writeVectorBox(b, part, reg, subgrids[level][i], vec);
  }
}

void MultiLevelEllipticSolver::loadLevelSolution(int level,
                                      MultiFab& dest,
                                      int icomp)
{
  BL_PROFILE("MultiLevelEllipticSolver::loadLevelSolution");

  int part = level - crse_level;

  FArrayBox scratch_fab;
  for (MFIter mfi(dest); mfi.isValid(); ++mfi) {
    int i = mfi.index();
    const Box &reg = grids[level][i];

    FArrayBox *f;
    int fcomp;
    if (dest.nGrow() == 0) { // need a temporary if dest is the wrong size
      f = &dest[mfi];
      fcomp = icomp;
    }
    else {
      f = &scratch_fab;
      f->resize(reg);
      f->copy<RunOn::Device>(dest[mfi], icomp, 0, 1);
      fcomp = 0;
    }
    Elixir f_hold = scratch_fab.elixir();

    Real* vec = f->dataPtr(fcomp);

    writeVectorBox(x, part, reg, subgrids[level][i], vec);
  }
}

void MultiLevelEllipticSolver::fillLevelRhs(int level,
                                      MultiFab& rhs, // will be altered
                                      BoundaryMode bc_variant)
{
  BL_PROFILE("MultiLevelEllipticSolver::fillLevelRhs");

  int part = level - crse_level;

  FArrayBox scratch_fab;
  for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
    int i = mfi.index();
    const Box &reg = grids[level][i];

    FArrayBox *f;
    if (rhs.nGrow() == 0) { // need a temporary if rhs is the wrong size
      f = &rhs[mfi];
    }
    else {
      f = &scratch_fab;
      f->resize(reg);
      f->copy<RunOn::Device>(rhs[mfi]);
    }
    Elixir f_hold = scratch_fab.elixir();

    Real* vec = f->dataPtr();

    // add b.c.'s to rhs

    if (bc_variant) {
      const Box& domain = bd[level]->getDomain();
      for (OrientationIter oitr; oitr; oitr++) {
        int cdir(oitr());
        int idim = oitr().coordDir();
        const RadBoundaryCondition &bct = bd[level]->bndryConds(oitr())[i];
        const Real      &bcl = bd[level]->bndryLocs(oitr())[i];
        const FArrayBox       &fs  = bd[level]->bndryValues(oitr())[mfi];
        const Mask      &msk = bd[level]->bndryMasks(oitr(), i);
        const Box &bbox = (*bcoefs[level])[idim][mfi].box();
        const Box &msb  = msk.box();
        if (reg[oitr()] == domain[oitr()] || level == crse_level) {

          // Treat an exposed grid edge here as a boundary condition
          // for the linear solver:

          if (reg[oitr()] == domain[oitr()]) {
            Array4<const int> tfp{};
            int bctype = bct;
            if (bd[level]->isMixedFace(oitr())) {
              const BaseFab<int> &tf = *(bd[level]->boundaryTypes(oitr())[i]);
              tfp = tf.array();
              bctype = -1;
            }
            AlphaBetaSolver::fillBcVec3(reg,
                              oitr().isLow(), idim,
                              f->array(),
                              cdir, bctype,
                              tfp,
                              bho, bcl,
                              fs.array(bc_component),
                              msk.array(),
                              (*bcoefs[level])[idim][mfi].array(),
                              beta, geom[level].data());
          }
          else {
              AlphaBetaSolver::fillBcVec(reg, f->array(),
                               cdir, bct, bho, bcl,
                               fs.array(bc_component), msk.array(),
                               (*bcoefs[level])[idim][mfi].array(),
                               beta, geom[level].CellSize());
          }
        }
        // There is no else here, since we would then be at an
        // interior edge and would not need to add anything to vec.
      }
    }

    // initialize rhs

    writeVectorBox(b, part, reg, subgrids[level][i], vec);
  }
}

void MultiLevelEllipticSolver::commitVectors()
{
  BL_PROFILE("MultiLevelEllipticSolver::commitVectors");

  HYPRE_SStructVectorAssemble(b);
  HYPRE_SStructVectorAssemble(x);
}

void MultiLevelEllipticSolver::configureSolver(Real _rel_tolerance, Real _abs_tolerance, int maxiter)
{
  BL_PROFILE("MultiLevelEllipticSolver::configureSolver");

  reltol = _rel_tolerance;
  abstol = _abs_tolerance; // may be used to change tolerance for solve

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
    //HYPRE_BoomerAMGSetMaxIter(solver, 1);
    HYPRE_BoomerAMGSetTol(solver, reltol);
#if 0
    // Barry used these four settings, at least for level solves:
    HYPRE_BoomerAMGSetInterpType(solver, 6);
    HYPRE_BoomerAMGSetCoarsenType(solver, 8);
    HYPRE_BoomerAMGSetStrongThreshold(solver, 0.25);
    HYPRE_BoomerAMGSetTruncFactor(solver, 0.);
#endif
    //HYPRE_BoomerAMGSetStrongThreshold(solver, 0.6);
    //HYPRE_BoomerAMGSetPrintLevel(solver, 2);
    //HYPRE_BoomerAMGSetTruncFactor(solver, 0.5);
    //HYPRE_BoomerAMGSetCoarsenType(solver, 6);
    //HYPRE_BoomerAMGSetLogging(solver, 2);
    HYPRE_BoomerAMGSetup(solver, par_A, par_b, par_x);
#if 0
    // The default num_sweeps seems to work best.  Note that changing
    // the default can't be done with a temp array on the stack, as the
    // array will not be copied.  The following code will leak since
    // num_sweeps is never deleted, but at least it will work:
    int num_sweeps[4] = {2, 2, 2, 1};
    HYPRE_BoomerAMGSetNumGridSweeps(solver, num_sweeps);
#endif
    //HYPRE_BoomerAMGSetIOutDat(solver, 2);
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
    //HYPRE_SStructFACSetRelaxType(solver_handle, 2);
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
    //HYPRE_ParCSRGMRESSetLogging(solver, 1);

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
    //HYPRE_SStructGMRESSetLogging(solver_handle, 1);

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
    //HYPRE_ParCSRPCGSetLogging(solver, 1);

    HYPRE_ParCSRPCGSetup(solver, par_A, par_b, par_x);
  }
  else if (solver_choice == 1003) {
    HYPRE_SStructPCGCreate(MPI_COMM_WORLD, &solver_handle);
    HYPRE_SStructPCGSetMaxIter(solver_handle, maxiter);
    HYPRE_SStructPCGSetTol(solver_handle, reltol);
    //HYPRE_SStructPCGSetLogging(solver_handle, 1);

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
    //HYPRE_ParCSRGMRESSetLogging(solver, 1);

    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetMaxIter(precond, 1);
    HYPRE_BoomerAMGSetTol(precond, reltol);
    //HYPRE_BoomerAMGSetStrongThreshold(precond, 0.6);
    //HYPRE_BoomerAMGSetPrintLevel(precond, 2);
    //HYPRE_BoomerAMGSetTruncFactor(precond, 0.5);
    //HYPRE_BoomerAMGSetCoarsenType(precond, 6);
    //HYPRE_BoomerAMGSetLogging(precond, 2);
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
    //HYPRE_ParCSRGMRESSetLogging(solver, 1);

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
    //HYPRE_BoomerAMGSetLogging(precond, 2);
    HYPRE_ParCSRGMRESSetPrecond(solver,
                                (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSolve,
                                (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSetup,
                                precond);

    HYPRE_ParCSRGMRESSetup(solver, par_A, par_b, par_x);
  }
  else if (solver_choice == 106) {
    // split solver
    ParmParse pp("hmabec");
    int struct_solve_iters = 1; pp.query("struct_solve_iters", struct_solve_iters);
#if (AMREX_SPACEDIM == 1)
    int struct_method_id = 0;
#else
    int struct_method_id = 1;
#endif
    pp.query("struct_method_id", struct_method_id);
    static int first = 1;
    if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
      first = 0;
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

    HYPRE_SStructSplitCreate(MPI_COMM_WORLD, &solver_handle);
    HYPRE_SStructSplitSetMaxIter(solver_handle, maxiter);
#if 0
    HYPRE_SStructSplitSetStructIterLimit(solver_handle, struct_solve_iters);
#endif
    HYPRE_SStructSplitSetTol(solver_handle, reltol);
    //HYPRE_SStructSplitSetZeroGuess(solver_handle);
    //HYPRE_SStructSplitSetLogging(solver_handle, 1);
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
    // split solver as preconditioner to GMRES
    ParmParse pp("hmabec");
    int gmres_restart = 5; pp.query("gmres_restart", gmres_restart);
    int split_precond_iters  = 1; pp.query("split_precond_iters",  split_precond_iters);
    int struct_solve_iters = 1; pp.query("struct_solve_iters", struct_solve_iters);
#if (AMREX_SPACEDIM == 1)
    int struct_method_id = 0;
#else
    int struct_method_id = 1;
#endif
    pp.query("struct_method_id", struct_method_id);
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
    //HYPRE_SStructSplitSetMaxIter(split_preconditioner, 1);
    //HYPRE_SStructSplitSetTol(split_preconditioner, reltol);
    //HYPRE_SStructSplitSetLogging(split_preconditioner, 1);
    //HYPRE_SStructSplitSetZeroGuess(split_preconditioner);
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
    // SMG or PFMG (one level only)
    HYPRE_StructMatrix s_A;
    HYPRE_StructVector s_b;
    HYPRE_StructVector s_x;
    HYPRE_SStructMatrixGetObject(A, (void**) &s_A);
    HYPRE_SStructVectorGetObject(b, (void**) &s_b);
    HYPRE_SStructVectorGetObject(x, (void**) &s_x);
    HYPRE_StructSolver& struct_solve_handle = *(HYPRE_StructSolver*)&solver;

    ParmParse pp("hmabec");
#if (AMREX_SPACEDIM == 1)
    int struct_method_id = 0;
#else
    int struct_method_id = 1;
#endif
    pp.query("struct_method_id", struct_method_id);
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
      //HYPRE_StructPFMGSetMemoryUse(struct_solve_handle, 0);
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
    // SMG or PFMG as preconditioner to GMRES (one level only)
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
#if (AMREX_SPACEDIM == 1)
    int struct_method_id = 0;
#else
    int struct_method_id = 1;
#endif
    pp.query("struct_method_id", struct_method_id);

    HYPRE_StructGMRESCreate(MPI_COMM_WORLD, &struct_solve_handle);
    //HYPRE_StructGMRESSetKDim(struct_solve_handle, gmres_restart);
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
      //HYPRE_StructPFMGSetMemoryUse(struct_pc_handle, 0);
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

  // The next one is AMG again with Barry's different settings.
  // The three after that added by Barry, appear to be for symmetric matrices.

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
    //HYPRE_BoomerAMGSetMaxIter(solver, 1);
    HYPRE_BoomerAMGSetTol(solver, reltol);
#if 1
    // Barry used these four settings, at least for level solves:
    HYPRE_BoomerAMGSetInterpType(solver, 6);
    HYPRE_BoomerAMGSetCoarsenType(solver, 8);
    HYPRE_BoomerAMGSetStrongThreshold(solver, 0.25);
    HYPRE_BoomerAMGSetTruncFactor(solver, 0.);
#endif
    //HYPRE_BoomerAMGSetPrintLevel(solver, 2);
    //HYPRE_BoomerAMGSetTruncFactor(solver, 0.5);
    //HYPRE_BoomerAMGSetCoarsenType(solver, 6);
    //HYPRE_BoomerAMGSetLogging(solver, 2);
    HYPRE_BoomerAMGSetup(solver, par_A, par_b, par_x);
#if 0
    // The default num_sweeps seems to work best.  Note that changing
    // the default can't be done with a temp array on the stack, as the
    // array will not be copied.  The following code will leak since
    // num_sweeps is never deleted, but at least it will work:
    int num_sweeps[4] = {2,2,2,1};
    HYPRE_BoomerAMGSetNumGridSweeps(solver, num_sweeps);
#endif
    //HYPRE_BoomerAMGSetIOutDat(solver, 2);
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
    ParmParse pp("hmabec");
#if (AMREX_SPACEDIM == 1)
    int struct_method_id = 0;
#else
    int struct_method_id = 1;
#endif
    pp.query("struct_method_id", struct_method_id);
    if (struct_method_id == 0) {
      HYPRE_StructSMGDestroy((HYPRE_StructSolver) solver);
    }
    else {
      HYPRE_StructPFMGDestroy((HYPRE_StructSolver) solver);
    }
  }
  else if (solver_choice == 109) {
    ParmParse pp("hmabec");
#if (AMREX_SPACEDIM == 1)
    int struct_method_id = 0;
#else
    int struct_method_id = 1;
#endif
    pp.query("struct_method_id", struct_method_id);
    HYPRE_StructGMRESDestroy((HYPRE_StructSolver) solver);
    if (struct_method_id == 0) {
      HYPRE_StructSMGDestroy((HYPRE_StructSolver) precond);
    }
    else {
      HYPRE_StructPFMGDestroy((HYPRE_StructSolver) precond);
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

    Real reltol_new = (bnorm > 0.0
                       ? abstol / bnorm * std::sqrt(volume)
                       : reltol);

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
        ParmParse pp("hmabec");
#if (AMREX_SPACEDIM == 1)
        int struct_method_id = 0;
#else
        int struct_method_id = 1;
#endif
        pp.query("struct_method_id", struct_method_id);
        if (struct_method_id == 0) {
          HYPRE_StructSMGSetTol((HYPRE_StructSolver) solver, reltol_new);
        }
        else {
          HYPRE_StructPFMGSetTol((HYPRE_StructSolver) solver, reltol_new);
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
    //std::cout << "MultiLevelEllipticSolver: starting solver..." << std::endl;
    HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);
    //std::cout << "                                  done." << std::endl;
    //HYPRE_SStructVectorPrint("Xamg", x, 0);
    //HYPRE_SStructVectorPrint("Bamg", b, 0);
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
    ParmParse pp("hmabec");
#if (AMREX_SPACEDIM == 1)
    int struct_method_id = 0;
#else
    int struct_method_id = 1;
#endif
    pp.query("struct_method_id", struct_method_id);
    if (struct_method_id == 0) {
      HYPRE_StructSMGSolve((HYPRE_StructSolver) solver, s_A, s_b, s_x);
    }
    else {
      HYPRE_StructPFMGSolve((HYPRE_StructSolver) solver, s_A, s_b, s_x);
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
    //std::cout << "MultiLevelEllipticSolver: starting solver..." << std::endl;
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
    //std::cout << "                                  done." << std::endl;
    //HYPRE_SStructVectorPrint("Xamg", x, 0);
    //HYPRE_SStructVectorPrint("Bamg", b, 0);
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
      ParmParse pp("hmabec");
#if (AMREX_SPACEDIM == 1)
      int struct_method_id = 0;
#else
      int struct_method_id = 1;
#endif
      pp.query("struct_method_id", struct_method_id);
      HYPRE_StructSolver& struct_solve_handle = *(HYPRE_StructSolver*)&solver;
      if (struct_method_id == 0) {
        HYPRE_StructSMGGetNumIterations(struct_solve_handle, &num_iterations);
        HYPRE_StructSMGGetFinalRelativeResidualNorm(struct_solve_handle, &res);
      }
      else {
        HYPRE_StructPFMGGetNumIterations(struct_solve_handle, &num_iterations);
        HYPRE_StructPFMGGetFinalRelativeResidualNorm(struct_solve_handle, &res);
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
    if (dest.nGrow() == 0) { // need a temporary if dest is the wrong size
      f = &dest[mfi];
      fcomp = icomp;
    }
    else {
      f = &scratch_fab;
      f->resize(reg);

      fcomp = 0;
    }
    Elixir f_hold = scratch_fab.elixir();

    readVectorBox(x, part, reg, subgrids[level][i], *f, fcomp);

    if (dest.nGrow() != 0) {
        dest[mfi].copy<RunOn::Device>(*f, 0, icomp, 1);
    }
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
    ParmParse pp("hmabec");
#if (AMREX_SPACEDIM == 1)
    int struct_method_id = 0;
#else
    int struct_method_id = 1;
#endif
    pp.query("struct_method_id", struct_method_id);
    HYPRE_StructSolver& struct_solve_handle = *(HYPRE_StructSolver*)&solver;
    if (struct_method_id == 0) {
      HYPRE_StructSMGGetFinalRelativeResidualNorm(struct_solve_handle, &res);
    }
    else {
      HYPRE_StructPFMGGetFinalRelativeResidualNorm(struct_solve_handle, &res);
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
                    // In normal code operation only the fluxes at internal
                    // Dirichlet boundaries are used.  Some diagnostics use the
                    // fluxes computed at domain boundaries but these do not
                    // influence the evolution of the interior solution.
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
