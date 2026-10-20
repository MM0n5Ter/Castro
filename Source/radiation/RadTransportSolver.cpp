#include <AMReX_ParmParse.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_LO_BCTYPES.H>

#include <RadTransportSolver.H>
#include <Radiation.H>
#include <radiation_helpers.H>
#include <user_rad_source.H>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace amrex;

RadTransportSolver::RadTransportSolver (Amr* Parent, int level, const BoxArray& grids, const DistributionMapping& dmap)
    : parent(Parent)
{
    read_params();

    if (!(radiation_solve::level_solver_kind < 100)) {
        if (radiation_solve::use_nonsym_solver_terms == 0) {
            hm = std::make_unique<MultiLevelEllipticSolver>(level, level, radiation_solve::level_solver_kind);
            hm->attachLevel(level, parent->Geom(level), grids, dmap,
                         IntVect::TheUnitVector());
            hm->makeMatrixLayout();
        }
        else {
            hem = std::make_unique<ExtendedMultiLevelSolver>(level, level, radiation_solve::level_solver_kind);
            cSaved  = hem->advectionScale();
            d1Saved = hem->gradientScale1();
            d2Saved = hem->gradientScale2();
            hem->attachLevel(level, parent->Geom(level), grids, dmap,
                          IntVect::TheUnitVector());
            hem->makeMatrixLayout();
        }
    } else {
        hd = std::make_unique<AlphaBetaSolver>(grids, dmap, parent->Geom(level), radiation_solve::level_solver_kind);
    }
}

void
RadTransportSolver::read_params ()
{
    ParmParse pp("radiation_solve");

#include <radiation_solve_queries.H>

    if (Radiation::SolverKind == Radiation::SingleGroupFLDSolver ||
        Radiation::SolverKind == Radiation::MultiGroupFLDSolver) {
        radiation_solve::abstol = 0.0;
    }

    if (AMREX_SPACEDIM == 1) {
        if (radiation_solve::level_solver_kind == 1) {
            amrex::Error("radiation_solve.level_solver_kind = 1 is not supported in 1D");
        }
    }

    if (Radiation::SolverKind == Radiation::SingleGroupFLDSolver
        && Radiation::lorentz_energy_term) {

        if (radiation_solve::level_solver_kind < 100) {
            amrex::Error("To do Lorentz term implicitly level_solver_kind must be >= 100.");
        }

        if (radiation_solve::use_nonsym_solver_terms == 0) {
            amrex::Error("To do Lorentz term implicitly use_nonsym_solver_terms must be 1.");
        }
    }

    if (Radiation::SolverKind == Radiation::MultiGroupFLDSolver &&
        Radiation::accelerate == 2 && Radiation::nGroups > 1) {

        if (radiation_solve::level_solver_kind < 100) {
            amrex::Error("When accelerate is 2, level_solver_kind must be >= 100.");
        }

        if (radiation_solve::use_nonsym_solver_terms == 0) {
            amrex::Error("When accelerate is 2, use_nonsym_solver_terms must be 1.");
        }
    }

}

void RadTransportSolver::levelSetup(int level)
{
  BL_PROFILE("RadTransportSolver::levelSetup");
}

void RadTransportSolver::levelBoundary(RadiativeBoundary& bd)
{
  BL_PROFILE("RadTransportSolver::levelBoundary");

  if (hd) {
    hd->setBndry(bd);
  }
  else if (hm) {
    hm->setBndry(hm->crseLevel(), bd);
  }
  else if (hem) {
    hem->setBndry(hem->crseLevel(), bd);
  }
}

void RadTransportSolver::levelBoundary(MultiGroupBoundary& multigroup_boundary, const int comp)
{
  BL_PROFILE("RadTransportSolver::levelBndryMG (updated)");

  if (hd) {
    hd->setBndry(multigroup_boundary, comp);
  }
  else if (hm) {
    hm->setBndry(hm->crseLevel(), multigroup_boundary, comp);
  }
  else if (hem) {
    hem->setBndry(hem->crseLevel(), multigroup_boundary, comp);
  }
}

void RadTransportSolver::setLevelCellCoeffs(int level, const MultiFab& cell_coefs, int dir)
{
    if (hem) {
      hem->setAdvectionCoefficients(level, cell_coefs, dir);
    }
}

void RadTransportSolver::setLevelFaceCoeffs(int level, const MultiFab& bcoefs, int dir)
{
    if (hd) {
        hd->setFaceCoefficients(bcoefs, dir);
    }
    else if (hm) {
        hm->setFaceCoefficients(level, bcoefs, dir);
    }
    else if (hem) {
        hem->setFaceCoefficients(level, bcoefs, dir);
    }
}

void RadTransportSolver::setLevelDiagCoeffs(int level, const MultiFab& acoefs)
{
    if (hd) {
        hd->setMassCoefficients(acoefs);
    }
    else if (hm) {
        hm->setMassCoefficients(level, acoefs);
    }
    else if (hem) {
        hem->setMassCoefficients(level, acoefs);
    }
}

void RadTransportSolver::setSolverScale(Real cScale, Real d1Scale, Real d2Scale)
{
  if (hem) {
    hem-> advectionScale() =  cScale;
    hem->gradientScale1() = d1Scale;
    hem->gradientScale2() = d2Scale;
  }
}

void RadTransportSolver::levelSolveSystem(int level,
                          MultiFab& Er, int rad_group_idx, MultiFab& rhs,
                          Real sync_residual_scale)
{
  BL_PROFILE("RadTransportSolver::levelSolveSystem");

  if (hd) {
    hd->setScalars(radiation_solve::alpha, radiation_solve::beta);
  }
  else if (hm) {
    hm->setScalars(radiation_solve::alpha, radiation_solve::beta);
  }
  else if (hem) {
    hem->setScalars(radiation_solve::alpha, radiation_solve::beta);
  }

  auto solve_on_hierarchy = [&] (auto& solver) {
    solver.assembleMatrix();
    solver.commitMatrix();
    solver.fillLevelVectors(level, Er, rad_group_idx, rhs, Nonzero_BC);
    solver.commitVectors();
    solver.configureSolver(radiation_solve::reltol, radiation_solve::abstol, radiation_solve::maxiter);
    solver.solve();
    solver.getSolution(level, Er, rad_group_idx);
    Real res = solver.currentResidualNorm();
    if (verbose >= 2 && ParallelDescriptor::IOProcessor()) {
      int oldprec = std::cout.precision(20);
      std::cout << "Absolute residual = " << res << std::endl;
      std::cout.precision(oldprec);
    }
    res *= sync_residual_scale;
    solver.releaseSolver();
  };

  if (hd) {
    hd->configureSolver(radiation_solve::reltol, radiation_solve::abstol, radiation_solve::maxiter);
    hd->solve(Er, rad_group_idx, rhs, Nonzero_BC);
    Real res = hd->currentResidualNorm();
    if (verbose >= 2 && ParallelDescriptor::IOProcessor()) {
      int oldprec = std::cout.precision(20);
      std::cout << "Absolute residual = " << res << std::endl;
      std::cout.precision(oldprec);
    }
    res *= sync_residual_scale;
    hd->releaseSolver();
  }
  else if (hm) {
    solve_on_hierarchy(*hm);
  }
  else if (hem) {
    solve_on_hierarchy(*hem);
  }
}

void RadTransportSolver::restoreSolverScale()
{
  if (hem) {
    hem-> advectionScale() =  cSaved;
    hem->gradientScale1() = d1Saved;
    hem->gradientScale2() = d2Saved;
  }
}
