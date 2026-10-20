#include <AMReX_ParmParse.H>
#include <AMReX_LO_BCTYPES.H>

#include <HypreSolvers.H>
#include <elliptic_face_ops.H>
#include <radiation_helpers.H>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <_hypre_struct_mv.h>

using namespace amrex;

static int ispow2(int i)
{
  return (!((i == 1)) ? ((((i <= 0) || (i & 1)) ? 0 : ispow2(i / 2))) : (1));
}

Real AlphaBetaSolver::flux_scale = 1.0;

#if (AMREX_SPACEDIM == 1)
static int vl[2] = { 0, 0 };
static int vh[2] = { 0, 0 };
#endif

static int* loV(const Box& b) {
#if (AMREX_SPACEDIM == 1)
  vl[0] = b.smallEnd(0);
  return vl;
#else
  return (int*) b.loVect();
#endif
}

static int* hiV(const Box& b) {
#if (AMREX_SPACEDIM == 1)
  vh[0] = b.bigEnd(0);
  return vh;
#else
  return (int*) b.hiVect();
#endif
}

AlphaBetaSolver::AlphaBetaSolver(const BoxArray& grids,
                     const DistributionMapping& dmap,
                     const Geometry& _geom,
                     int _solver_choice)
  : geom(_geom), solver_choice(_solver_choice)
{
  ParmParse pp("habec");

  pfmg_smoother_type = 1; pp.query("pfmg_smoother_type", pfmg_smoother_type);
  verbose = 0; pp.query("v", verbose); pp.query("verbose", verbose);
  log_threshold = 0; pp.query("log_threshold", log_threshold);

  static int first = 1;
  if (verbose >= 1 && first && ParallelDescriptor::IOProcessor()) {
    first = 0;
    if (solver_choice == 1 || solver_choice == 3 || solver_choice == 5) {
      std::cout << "habec.pfmg_smoother_type           = " << pfmg_smoother_type << std::endl;
    }
    std::cout << "habec.verbose                   = " << verbose << std::endl;
    std::cout << "habec.log_threshold         = " << log_threshold << std::endl;
  }
  bho = 0;

  for (int i = 0; i < AMREX_SPACEDIM; i++) {
    dx[i] = geom.CellSize(i);
  }

#if (AMREX_SPACEDIM == 1)

  HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &hgrid);

  if (geom.isAnyPeriodic()) {
    BL_ASSERT(geom.isPeriodic(0));
    BL_ASSERT(geom.Domain().smallEnd(0) == 0);

    int is_periodic[2];
    is_periodic[0] = geom.period(0);
    is_periodic[1] = 0;
    BL_ASSERT(ispow2(is_periodic[0]));

    HYPRE_StructGridSetPeriodic(hgrid, is_periodic);
  }

#else

  HYPRE_StructGridCreate(MPI_COMM_WORLD, AMREX_SPACEDIM, &hgrid);

  if (geom.isAnyPeriodic()) {
    int is_periodic[AMREX_SPACEDIM];
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      is_periodic[i] = 0;
      if (geom.isPeriodic(i)) {
        is_periodic[i] = geom.period(i);
        BL_ASSERT(ispow2(is_periodic[i]));
        BL_ASSERT(geom.Domain().smallEnd(i) == 0);
      }
    }
    HYPRE_StructGridSetPeriodic(hgrid, is_periodic);
  }
#endif

  if (ParallelDescriptor::NProcs() != 1) {
    for (int i = 0; i < grids.size(); i++) {
      if (dmap[i] == ParallelDescriptor::MyProc()) {
        HYPRE_StructGridSetExtents(hgrid, loV(grids[i]), hiV(grids[i]));
      }
    }
  }
  else {
    for (int i = 0; i < grids.size(); i++) {
      HYPRE_StructGridSetExtents(hgrid, loV(grids[i]), hiV(grids[i]));
    }
  }

  HYPRE_StructGridAssemble(hgrid);

#if (AMREX_SPACEDIM == 1)
  int offsets[2][2] = {{-1,  0},
                       { 0,  0}};
#elif (AMREX_SPACEDIM == 2)
  int offsets[3][2] = {{-1,  0},
                       { 0, -1},
                       { 0,  0}};
#elif (AMREX_SPACEDIM == 3)
  int offsets[4][3] = {{-1,  0,  0},
                       { 0, -1,  0},
                       { 0,  0, -1},
                       { 0,  0,  0}};
#endif

#if   (AMREX_SPACEDIM == 1)
  int matrix_ghost_widths[6] = { 1, 1, 0, 0, 0, 0 };
#elif (AMREX_SPACEDIM == 2)
  int matrix_ghost_widths[6] = { 1, 1, 1, 1, 0, 0 };
#elif (AMREX_SPACEDIM == 3)
  int matrix_ghost_widths[6] = { 1, 1, 1, 1, 1, 1 };
#endif

  HYPRE_StructStencil stencil;

#if (AMREX_SPACEDIM == 1)
  HYPRE_StructStencilCreate(2, 2, &stencil);
#else
  HYPRE_StructStencilCreate(AMREX_SPACEDIM, AMREX_SPACEDIM + 1, &stencil);
#endif

  for (int i = 0; i < AMREX_SPACEDIM + 1; i++) {
    HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
  }

  HYPRE_StructMatrixCreate(MPI_COMM_WORLD, hgrid, stencil, &A);
  HYPRE_StructMatrixSetSymmetric(A, 1);
  HYPRE_StructMatrixSetNumGhost(A, matrix_ghost_widths);
  HYPRE_StructMatrixInitialize(A);

  HYPRE_StructMatrixCreate(MPI_COMM_WORLD, hgrid, stencil, &A0);
  HYPRE_StructMatrixSetSymmetric(A0, 1);
  HYPRE_StructMatrixSetNumGhost(A0, matrix_ghost_widths);
  HYPRE_StructMatrixInitialize(A0);

  HYPRE_StructVectorCreate(MPI_COMM_WORLD, hgrid, &b);
  HYPRE_StructVectorCreate(MPI_COMM_WORLD, hgrid, &x);

  HYPRE_StructStencilDestroy(stencil);

  HYPRE_StructVectorInitialize(b);
  HYPRE_StructVectorInitialize(x);

  Gpu::synchronize();

  int ncomp=1;
  int ngrow=0;
  acoefs.reset(new MultiFab(grids, dmap, ncomp, ngrow));
  acoefs->setVal(0.0);

  for (int i = 0; i < AMREX_SPACEDIM; i++) {
    BoxArray edge_boxes(grids);
    edge_boxes.surroundingNodes(i);
    bcoefs[i].reset(new MultiFab(edge_boxes, dmap, ncomp, ngrow));
  }
}

AlphaBetaSolver::~AlphaBetaSolver()
{
  HYPRE_StructVectorDestroy(b);
  HYPRE_StructVectorDestroy(x);

  HYPRE_StructMatrixDestroy(A);
  HYPRE_StructMatrixDestroy(A0);

  HYPRE_StructGridDestroy(hgrid);
}

void AlphaBetaSolver::configureSolver(Real _rel_tolerance, Real _abs_tolerance, int maxiter)
{
  BL_PROFILE("AlphaBetaSolver::configureSolver");

  const BoxArray& grids = acoefs->boxArray();

  const int size = AMREX_SPACEDIM + 1;
  int i, idim;

  int stencil_indices[size];

  for (i = 0; i < size; i++) {
    stencil_indices[i] = i;
  }

  BaseFab<GpuArray<Real, size>> matfab;
  for (MFIter ai(*acoefs); ai.isValid(); ++ai) {
    i = ai.index();
    const Box &reg = grids[i];

    matfab.resize(reg);
    Elixir matfab_hold = matfab.elixir();
    Real* mat = (Real*) matfab.dataPtr();

    fillMassMat(reg, (*acoefs)[ai].array(), alpha, matfab.array());

    for (idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        fillFaceMat(reg, (*bcoefs[idim])[ai].array(), beta, dx, idim, matfab.array());
    }

    const BaseBoundary& bd = getBoundaryData();
    const Box& domain = bd.getDomain();
    for (OrientationIter oitr; oitr; oitr++) {
      int cdir(oitr());
      idim = oitr().coordDir();
      const RadBoundaryCondition &bct = bd.bndryConds(oitr())[i];
      const Real      &bcl = bd.bndryLocs(oitr())[i];
      const Mask      &msk = bd.bndryMasks(oitr(),i);
      const Box &bbox = (*bcoefs[idim])[ai].box();
      const Box &msb  = msk.box();
      if (reg[oitr()] == domain[oitr()]) {
        Array4<int const> tfp{};
        int bctype = bct;
        if (bd.isMixedFace(oitr())) {
          const BaseFab<int> &tf = *(bd.boundaryTypes(oitr())[i]);
          tfp = tf.array();
          bctype = -1;
        }
        const Box &fsb = bd.bndryValues(oitr())[ai].box();
        Array4<Real const> opt_bc_alpha_arr{};
        if (SPa != 0) {
          opt_bc_alpha_arr = (*SPa)[ai].array();
        }

        fillBcMat3(reg, oitr().isLow(), idim,
               cdir, bctype,
               tfp, bcl, msk.array(),
               (*bcoefs[idim])[ai].array(),
               beta, geom.data(), flux_scale,
               opt_bc_alpha_arr, matfab.array());
      }
      else {
          fillBcMat(reg, cdir, bct, bcl, msk.array(), (*bcoefs[idim])[ai].array(), beta, dx, matfab.array());
      }
    }

    HYPRE_StructMatrixSetBoxValues(A, loV(reg), hiV(reg),
                                   size, stencil_indices, mat);
    Gpu::synchronize();
  }

  HYPRE_StructMatrixAssemble(A);

  HYPRE_StructVectorAssemble(b);
  HYPRE_StructVectorAssemble(x);

  reltol = _rel_tolerance;
  abstol = _abs_tolerance;

  switch (solver_choice) {
  case 0:
    HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_StructSMGSetMemoryUse(solver, 0);
    HYPRE_StructSMGSetMaxIter(solver, maxiter);
    HYPRE_StructSMGSetRelChange(solver, 0);
    HYPRE_StructSMGSetTol(solver, reltol);
    HYPRE_StructSMGSetNumPreRelax(solver, 1);
    HYPRE_StructSMGSetNumPostRelax(solver, 1);
    HYPRE_StructSMGSetLogging(solver, 1);
    HYPRE_StructSMGSetup(solver, A, b, x);
    break;

  case 1:
    HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_StructPFMGSetSkipRelax(solver, 0);
    HYPRE_StructPFMGSetMaxIter(solver, maxiter);
    HYPRE_StructPFMGSetRelChange(solver, 0);
    HYPRE_StructPFMGSetTol(solver, reltol);
    HYPRE_StructPFMGSetRelaxType(solver, pfmg_smoother_type);
    HYPRE_StructPFMGSetNumPreRelax(solver, 1);
    HYPRE_StructPFMGSetNumPostRelax(solver, 1);
    HYPRE_StructPFMGSetLogging(solver, 1);
    HYPRE_StructPFMGSetup(solver, A, b, x);
    break;

  case 2:
    HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver);
    HYPRE_StructJacobiSetMaxIter(solver, maxiter);
    HYPRE_StructJacobiSetup(solver, A, b, x);
    break;

  case 3:
  case 4:
    HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_StructPCGSetMaxIter(solver, maxiter);
    HYPRE_StructPCGSetRelChange(solver, 0);
    HYPRE_StructPCGSetTol(solver, reltol);

    if (solver_choice == 3) {
      HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
      HYPRE_StructPFMGSetMaxIter(precond, 1);
      HYPRE_StructPFMGSetTol(precond, 0.0);
      HYPRE_StructPFMGSetZeroGuess(precond);
      HYPRE_StructPFMGSetRelaxType(precond, pfmg_smoother_type);
      HYPRE_StructPFMGSetNumPreRelax(precond, 1);
      HYPRE_StructPFMGSetNumPostRelax(precond, 1);
      HYPRE_StructPFMGSetSkipRelax(precond, 0);
      HYPRE_StructPFMGSetLogging(precond, 0);
      HYPRE_StructPCGSetPrecond(solver,
                                HYPRE_StructPFMGSolve,
                                HYPRE_StructPFMGSetup,
                                precond);
    }
    else if (solver_choice == 4) {
      HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
      HYPRE_StructSMGSetMemoryUse(precond, 0);
      HYPRE_StructSMGSetMaxIter(precond, 1);
      HYPRE_StructSMGSetRelChange(precond, 0);
      HYPRE_StructSMGSetTol(precond, 0.0);
      HYPRE_StructSMGSetNumPreRelax(precond, 1);
      HYPRE_StructSMGSetNumPostRelax(precond, 1);
      HYPRE_StructSMGSetLogging(precond, 0);
      HYPRE_StructPCGSetPrecond(solver,
                                HYPRE_StructSMGSolve,
                                HYPRE_StructSMGSetup,
                                precond);
    }

    HYPRE_StructPCGSetLogging(solver, 1);
    HYPRE_StructPCGSetup(solver, A, b, x);
    break;

  case 5:
  case 6:
    HYPRE_StructHybridCreate(MPI_COMM_WORLD, &solver);
    HYPRE_StructHybridSetDSCGMaxIter(solver, maxiter);
    HYPRE_StructHybridSetPCGMaxIter(solver, maxiter);
    HYPRE_StructHybridSetTol(solver, reltol);

    HYPRE_StructHybridSetConvergenceTol(solver, 0.90);
    HYPRE_StructHybridSetRelChange(solver, 0);
    HYPRE_StructHybridSetLogging(solver, 1);
    HYPRE_StructHybridSetSolverType(solver, 1);

    if (solver_choice == 5) {
      HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &precond);
      HYPRE_StructPFMGSetMaxIter(precond, 1);
      HYPRE_StructPFMGSetTol(precond, 0.0);
      HYPRE_StructPFMGSetZeroGuess(precond);
      HYPRE_StructPFMGSetRelaxType(precond, pfmg_smoother_type);
      HYPRE_StructPFMGSetNumPreRelax(precond, 1);
      HYPRE_StructPFMGSetNumPostRelax(precond, 1);
      HYPRE_StructPFMGSetSkipRelax(precond, 0);
      HYPRE_StructPFMGSetLogging(precond, 0);
      HYPRE_StructHybridSetPrecond(solver,
                                   HYPRE_StructPFMGSolve,
                                   HYPRE_StructPFMGSetup,
                                   precond);
    }
    else if (solver_choice == 6) {
      HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
      HYPRE_StructSMGSetMemoryUse(precond, 0);
      HYPRE_StructSMGSetMaxIter(precond, 1);
      HYPRE_StructSMGSetRelChange(precond, 0);
      HYPRE_StructSMGSetTol(precond, 0.0);
      HYPRE_StructSMGSetNumPreRelax(precond, 1);
      HYPRE_StructSMGSetNumPostRelax(precond, 1);
      HYPRE_StructSMGSetLogging(precond, 0);
      HYPRE_StructHybridSetPrecond(solver,
                                   HYPRE_StructSMGSolve,
                                   HYPRE_StructSMGSetup,
                                   precond);
    }

    HYPRE_StructHybridSetup(solver, A, b, x);
    break;

  default:
      amrex::Error("AlphaBetaSolver: no such solver");
  }
  Gpu::synchronize();
}

void AlphaBetaSolver::releaseSolver()
{
  BL_PROFILE("AlphaBetaSolver::releaseSolver");

  if (solver_choice == 0) {
    HYPRE_StructSMGDestroy(solver);
  }
  else if (solver_choice == 1) {
    HYPRE_StructPFMGDestroy(solver);
  }
  else if(solver_choice == 2) {
    HYPRE_StructJacobiDestroy(solver);
  }
  else if(solver_choice == 3 || solver_choice == 4) {
    HYPRE_StructPCGDestroy(solver);
    if (solver_choice == 3)
    {
       HYPRE_StructPFMGDestroy(precond);
    }
    else if (solver_choice == 4)
    {
       HYPRE_StructSMGDestroy(precond);
    }
  }
  else if(solver_choice == 5 || solver_choice == 6) {
    HYPRE_StructHybridDestroy(solver);
    if(solver_choice == 5) {
       HYPRE_StructPFMGDestroy(precond);
    }
    if(solver_choice == 6) {
       HYPRE_StructSMGDestroy(precond);
    }
  }
}

void AlphaBetaSolver::setScalars(Real Alpha, Real Beta)
{
  alpha = Alpha;
  beta  = Beta;
}

void AlphaBetaSolver::setMassCoefficients(const MultiFab &a)
{
  BL_ASSERT( a.ok() );
  BL_ASSERT( a.boxArray() == acoefs->boxArray() );
  MultiFab::Copy(*acoefs, a, 0, 0, 1, 0);
}

void AlphaBetaSolver::setFaceCoefficients(const MultiFab &b, int dir)
{
  BL_ASSERT( b.ok() );
  BL_ASSERT( b.boxArray() == bcoefs[dir]->boxArray() );
  MultiFab::Copy(*bcoefs[dir], b, 0, 0, 1, 0);
}

void AlphaBetaSolver::setBcAlphaCoef(const MultiFab& a)
{
  BL_ASSERT( a.ok() );
  if (SPa == 0) {
    const BoxArray& grids = a.boxArray();
    const DistributionMapping& dmap = a.DistributionMap();
    SPa.reset(new MultiFab(grids,dmap,1,0));
  }
  MultiFab::Copy(*SPa, a, 0, 0, 1, 0);
}

void AlphaBetaSolver::solve(MultiFab& dest, int icomp, MultiFab& rhs, BoundaryMode bc_variant)
{
  BL_PROFILE("AlphaBetaSolver::solve");

  const BoxArray& grids = dest.boxArray();

  int i, idim;

  Real *vec;
  FArrayBox scratch_fab;
  for (MFIter di(dest); di.isValid(); ++di) {
    i = di.index();
    const Box &reg = grids[i];

    FArrayBox *f;
    int fcomp;

    Array4<Real> const d_arr = dest.array(di);
    Array4<Real> const rhs_view_arr = rhs.array(di);

    if (dest.nGrow() == 0) {
      f = &dest[di];
      fcomp = icomp;
    }
    else {
      f = &scratch_fab;
      f->resize(reg);

      Array4<Real> const face_flux_arr = f->array();

      fcomp = 0;

      AMREX_PARALLEL_FOR_3D(reg, i, j, k, { face_flux_arr(i,j,k,fcomp) = d_arr(i,j,k,icomp); });
      Gpu::streamSynchronize();
    }
    Elixir f_hold = scratch_fab.elixir();

    vec = f->dataPtr(fcomp);

    HYPRE_StructVectorSetBoxValues(x, loV(reg), hiV(reg), vec);

    Gpu::streamSynchronize();

    Array4<Real> const face_flux_arr = f->array();

    AMREX_PARALLEL_FOR_3D(reg, i, j, k, { face_flux_arr(i,j,k,fcomp) = rhs_view_arr(i,j,k,0); });
    Gpu::streamSynchronize();

    if (bc_variant) {
      const BaseBoundary& bd = getBoundaryData();
      const Box& domain = bd.getDomain();
      for (OrientationIter oitr; oitr; oitr++) {
        int cdir(oitr());
        idim = oitr().coordDir();
        const RadBoundaryCondition &bct = bd.bndryConds(oitr())[i];
        const Real      &bcl = bd.bndryLocs(oitr())[i];
        const FArrayBox       &fs  = bd.bndryValues(oitr())[di];
        const Mask      &msk = bd.bndryMasks(oitr(),i);
        const Box &bbox = (*bcoefs[idim])[di].box();

        if (!(reg[oitr()] == domain[oitr()])) {
            fillBcVec(reg, f->array(fcomp),
                  cdir, bct, bho, bcl,
                  fs.array(bc_component), msk.array(),
                  (*bcoefs[idim])[di].array(),
                  beta, dx);
        } else {
          Array4<const int> tfp{};
          int bctype = bct;
          if (bd.isMixedFace(oitr())) {
            const BaseFab<int> &tf = *(bd.boundaryTypes(oitr())[i]);
            tfp = tf.array();
            bctype = -1;
          }
          fillBcVec3(reg,
                 oitr().isLow(), idim,
                 f->array(fcomp),
                 cdir, bct,
                 tfp,
                 bho, bcl,
                 fs.array(bc_component),
                 msk.array(),
                 (*bcoefs[idim])[di].array(),
                 beta, geom.data());
        }
      }
    }

    Gpu::streamSynchronize();

    HYPRE_StructVectorSetBoxValues(b, loV(reg), hiV(reg), vec);
  }

  HYPRE_StructVectorAssemble(b);
  HYPRE_StructVectorAssemble(x);
  Gpu::synchronize();

  if (abstol > 0.0) {
    Real bnorm;
    bnorm = hypre_StructInnerProd((hypre_StructVector *) b,
                                  (hypre_StructVector *) b);
    bnorm = std::sqrt(bnorm);

    const BoxArray& grids = acoefs->boxArray();
    Real volume = 0.0;
    for (int i = 0; i < grids.size(); i++) {
      volume += grids[i].numPts();
    }

    Real reltol_new = (bnorm > 0.0
                       ? abstol / bnorm * std::sqrt(volume)
                       : reltol);

    if (reltol_new > reltol) {
      if (solver_choice == 0) {
        HYPRE_StructSMGSetTol(solver, reltol_new);
      }
      else if(solver_choice == 1) {
        HYPRE_StructPFMGSetTol(solver, reltol_new);
      }
      else if(solver_choice == 2) {
      }
      else if(solver_choice == 3 || solver_choice == 4) {
        HYPRE_StructPCGSetTol(solver, reltol_new);
      }
    }
  }

  if (solver_choice == 0) {
    HYPRE_StructSMGSolve(solver, A, b, x);
  }
  else if (solver_choice == 1) {
    HYPRE_StructPFMGSolve(solver, A, b, x);
  }
  else if (solver_choice == 2) {
    HYPRE_StructJacobiSolve(solver, A, b, x);
  }
  else if (solver_choice == 3 || solver_choice == 4) {
    HYPRE_StructPCGSolve(solver, A, b, x);
  }
  else if (solver_choice == 5 || solver_choice == 6) {
    HYPRE_StructHybridSolve(solver, A, b, x);
  }

  Gpu::synchronize();

  for (MFIter di(dest); di.isValid(); ++di) {
    i = di.index();
    const Box &reg = grids[i];

    FArrayBox *f;
    int fcomp;
    if (!(dest.nGrow() == 0)) {
      f = &scratch_fab;
      f->resize(reg);
      fcomp = 0;
    } else {
      f = &dest[di];
      fcomp = icomp;
    }
    Elixir f_hold = scratch_fab.elixir();

    vec = f->dataPtr(fcomp);
    HYPRE_StructVectorGetBoxValues(x, loV(reg), hiV(reg),
                                   vec);
    Gpu::synchronize();

    if (dest.nGrow() != 0) {
        Array4<Real> const face_flux_arr = f->array();
        Array4<Real> const d_arr = dest.array(di);
        AMREX_PARALLEL_FOR_3D(reg, i, j, k, { d_arr(i,j,k,icomp) = face_flux_arr(i,j,k); });
    }
  }

  if (verbose >= 2 && ParallelDescriptor::IOProcessor()) {
    int num_iterations;
    Real res;
    if (solver_choice == 0) {
      HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
      HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &res);
    }
    else if(solver_choice == 1) {
      HYPRE_StructPFMGGetNumIterations(solver, &num_iterations);
      HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, &res);
    }
    else if(solver_choice == 2) {
      HYPRE_StructJacobiGetNumIterations(solver, &num_iterations);
      HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &res);
    }
    else if(solver_choice == 3 || solver_choice == 4) {
      HYPRE_StructPCGGetNumIterations(solver, &num_iterations);
      HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &res);
    }
    else if(solver_choice == 5 || solver_choice == 6) {
      HYPRE_StructHybridGetNumIterations(solver, &num_iterations);
      HYPRE_StructHybridGetFinalRelativeResidualNorm(solver, &res);
    }
    if (num_iterations >= log_threshold) {
      int oldprec = std::cout.precision(20);
      std::cout << num_iterations
           << " Hypre Multigrid Iterations, Relative Residual "
           << res << std::endl;
      std::cout.precision(oldprec);
    }
  }
  Gpu::synchronize();
}

Real AlphaBetaSolver::currentResidualNorm()
{
  BL_PROFILE("AlphaBetaSolver::currentResidualNorm");

  Real bnorm;
  bnorm = hypre_StructInnerProd((hypre_StructVector *) b,
                                (hypre_StructVector *) b);
  bnorm = std::sqrt(bnorm);

  Real res;
  if (solver_choice == 0) {
    HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &res);
  }
  else if(solver_choice == 1) {
    HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, &res);
  }
  else if(solver_choice == 2) {
    HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &res);
  }
  else if(solver_choice == 3 || solver_choice == 4) {
    HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &res);
  }
  else if(solver_choice == 5 || solver_choice == 6) {
    HYPRE_StructHybridGetFinalRelativeResidualNorm(solver, &res);
  }

  Gpu::synchronize();

  const BoxArray& grids = acoefs->boxArray();
  Real volume = 0.0;
  for (int i = 0; i < grids.size(); i++) {
    volume += grids[i].numPts();
  }

  return bnorm * res / std::sqrt(volume);
}
