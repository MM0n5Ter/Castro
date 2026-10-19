
#include <AMReX_ParmParse.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_LO_BCTYPES.H>

#include <RadTransportSolver.H>
#include <Radiation.H>  // for access to static physical constants only
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

    if (radiation_solve::level_solver_kind < 100) {
        hd.reset(new AlphaBetaSolver(grids, dmap, parent->Geom(level), radiation_solve::level_solver_kind));
    }
    else {
        if (radiation_solve::use_nonsym_solver_terms == 0) {
            hm.reset(new MultiLevelEllipticSolver(level, level, radiation_solve::level_solver_kind));
            hm->attachLevel(level, parent->Geom(level), grids, dmap,
                         IntVect::TheUnitVector());
            hm->makeMatrixLayout();
        }
        else {
            hem.reset(new ExtendedMultiLevelSolver(level, level, radiation_solve::level_solver_kind));
            cSaved  = hem->advectionScale();
            d1Saved = hem->gradientScale1();
            d2Saved = hem->gradientScale2();
            hem->attachLevel(level, parent->Geom(level), grids, dmap,
                          IntVect::TheUnitVector());
            hem->makeMatrixLayout();
        }
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

    // Check for unsupported options.

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

// update multigroup version
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

void RadTransportSolver::applyCellMetrics(int level, MultiFab& cc)
{
    BL_PROFILE("RadTransportSolver::applyCellMetrics");
    BL_ASSERT(cc.nGrow() == 0);

    auto geomdata = parent->Geom(level).data();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(cc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        auto cell_center_arr = cc[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Real r, s;
            cell_center_geometry_factors(i, j, k, geomdata, r, s);

            cell_center_arr(i,j,k) *= r * s;
        });
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

void RadTransportSolver::setLevelCellCoeffs(int level, const MultiFab& cell_coefs, int dir)
{
    if (hem) {
      hem->setAdvectionCoefficients(level, cell_coefs, dir);
    }
}

void RadTransportSolver::levelDiagCoeffs(int level,
                            MultiFab& fkp, MultiFab& eta, MultiFab& coupling_inverse,
                            Real c, Real delta_t, Real theta)
{
  BL_PROFILE("RadTransportSolver::levelDiagCoeffs");
  const BoxArray& grids = parent->boxArray(level);
  const DistributionMapping& dmap = parent->DistributionMap(level);

  auto geomdata = parent->Geom(level).data();

  // Allocate space for ABecLapacian acoeffs, fill with values

  int Ncomp  = 1;
  int Nghost = 0;

  MultiFab acoefs(grids, dmap, Ncomp, Nghost);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(fkp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.tilebox();

      auto a = acoefs[mfi].array();
      auto scaled_planck_arr = fkp[mfi].array();
      auto coupling_arr = eta[mfi].array();
      auto coupling_inverse_arr = coupling_inverse[mfi].array();

      const Real dtm = 1.e0_rt / delta_t;

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          Real r, s;
          cell_center_geometry_factors(i, j, k, geomdata, r, s);

          if (AMREX_SPACEDIM == 1) {
              s = 1.e0_rt;
          }

          a(i,j,k) = r * s *
                     (scaled_planck_arr(i,j,k) * coupling_inverse_arr(i,j,k) * c + dtm) /
                     (1.e0_rt - (1.e0_rt - theta) * coupling_arr(i,j,k));
      });
  }

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

void RadTransportSolver::levelBoundaryScale(int level, Array<MultiFab, AMREX_SPACEDIM>& lambda, int rad_group_idx,
                         int lo_bc[3], int hi_bc[3])
{
  const BoxArray& grids = parent->boxArray(level);
  const DistributionMapping& dmap = parent->DistributionMap(level);
  const Geometry& geom = parent->Geom(level);
  const Box& geom_domain = geom.Domain();

  MultiFab spa(grids, dmap, 1, 0);
#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(spa,true); mfi.isValid(); ++mfi) {
      const Box& reg  = mfi.tilebox();

      spa[mfi].setVal<RunOn::Host>(1.e210,reg,0);

      bool near_boundary=false;
      for (int idim=0; idim<AMREX_SPACEDIM; idim++) {
          if (lo_bc[idim] == AMREX_LO_SANCHEZ_POMRANING &&
              reg.smallEnd(idim) == geom_domain.smallEnd(idim)) {
              near_boundary=true;
              break;
          }
          if (hi_bc[idim] == AMREX_LO_SANCHEZ_POMRANING &&
              reg.bigEnd(idim) == geom_domain.bigEnd(idim)) {
              near_boundary=true;
              break;
          }
      }

      if (near_boundary) {
          auto boundary_scale_arr = spa[mfi].array();

          auto lmx = lambda[0][mfi].array();
#if AMREX_SPACEDIM >= 2
          auto lmy = lambda[1][mfi].array();
#endif
#if AMREX_SPACEDIM == 3
          auto lmz = lambda[2][mfi].array();
#endif

          amrex::ParallelFor(reg,
          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
          {
              Real lam;

              if (i == reg.loVect()[0] || i == reg.hiVect()[0] ||
                  j == reg.loVect()[1] || j == reg.hiVect()[1] ||
                  k == reg.hiVect()[2] || k == reg.hiVect()[2]) {
#if AMREX_SPACEDIM == 1
                  if (i == reg.loVect()[0]) {
                      lam = lmx(i,j,k,rad_group_idx);
                  }
                  else {
                      lam = lmx(i+1,j,k,rad_group_idx);
                  }
#elif AMREX_SPACEDIM == 2
                  lam = 0.25_rt * (lmx(i,j,k,rad_group_idx) + lmx(i+1,j  ,k,rad_group_idx) +
                                   lmy(i,j,k,rad_group_idx) + lmy(i  ,j+1,k,rad_group_idx));
#else
                  lam = (lmx(i,j,k,rad_group_idx) + lmx(i+1,j  ,k  ,rad_group_idx) +
                         lmy(i,j,k,rad_group_idx) + lmy(i  ,j+1,k  ,rad_group_idx) +
                         lmz(i,j,k,rad_group_idx) + lmz(i  ,j  ,k+1,rad_group_idx)) / 6.e0_rt;
#endif

                  boundary_scale_arr(i,j,k) = LimitedDiffusionAlpha(lam);
              }
          });
      }
  }

  if (hm) {
    hm->setBcAlphaCoef(level, spa);
  }
  else if (hem) {
    hem->setBcAlphaCoef(level, spa);
  }
  else if (hd) {
    hd->setBcAlphaCoef(spa);
  }
  else {
    amrex::Abort("Should not be in RadTransportSolver::levelBoundaryScale");
  }
}

void RadTransportSolver::levelFaceCoeffs(int level,
                            Array<MultiFab, AMREX_SPACEDIM>& lambda,
                            MultiFab& kappa_r, int kcomp,
                            Real c, int limiter_comp)
{
  BL_PROFILE("RadTransportSolver::levelFaceCoeffs");
  BL_ASSERT(kappa_r.nGrow() == 1);

  auto geomdata = parent->Geom(level).data();
  auto dx = parent->Geom(level).CellSizeArray();

  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

    MultiFab bcoefs(lambda[idim].boxArray(), lambda[idim].DistributionMap(), 1, 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(lambda[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto dir_coef_arr = bcoefs[mfi].array();
        auto limiter_arr = lambda[idim][mfi].array(limiter_comp);
        auto kappa_ross_arr = kappa_r[mfi].array(kcomp);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            if (idim == 0) {

                Real r, s;
                edge_center_geometry_factors(i, j, k, idim, geomdata, r, s);

                if (AMREX_SPACEDIM == 1) {
                    s = 1.e0_rt;
                }

                Real kap = face_opacity_average(kappa_ross_arr(i-1,j,k), kappa_ross_arr(i,j,k), dx[0], -1);
                dir_coef_arr(i,j,k) = r * s * c * limiter_arr(i,j,k) / kap;

            }
            else if (idim == 1) {

                Real r, s;
                edge_center_geometry_factors(i, j, k, idim, geomdata, r, s);

                if (AMREX_SPACEDIM == 1) {
                    s = 1.e0_rt;
                }

                Real kap = face_opacity_average(kappa_ross_arr(i,j-1,k), kappa_ross_arr(i,j,k), dx[1], -1);
                dir_coef_arr(i,j,k) = r * s * c * limiter_arr(i,j,k) / kap;

            }
            else {

                Real r, s;
                edge_center_geometry_factors(i, j, k, idim, geomdata, r, s);

                if (AMREX_SPACEDIM == 1) {
                    s = 1.e0_rt;
                }

                Real kap = face_opacity_average(kappa_ross_arr(i,j,k-1), kappa_ross_arr(i,j,k), dx[2], -1);
                dir_coef_arr(i,j,k) = r * s * c * limiter_arr(i,j,k) / kap;

            }
        });
    }

    if (hd) {
        hd->setFaceCoefficients(bcoefs, idim);
    }
    else if (hm) {
      hm->setFaceCoefficients(level, bcoefs, idim);
    }
    else if (hem) {
      hem->setFaceCoefficients(level, bcoefs, idim);
    }
  } // -->> over dimension
}

void RadTransportSolver::levelDiffCoeffs(int level, Array<MultiFab, AMREX_SPACEDIM>& lambda,
                            MultiFab& vel, MultiFab& dcf)
{
    BL_PROFILE("RadTransportSolver::levelDiffCoeffs");
    const HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
    const DistributionMapping& dm = hydrosim->DistributionMap();
    const Geometry& geom = parent->Geom(level);
    const auto dx = geom.CellSizeArray();
    const auto geomdata = geom.data();

    for (int idim=0; idim<AMREX_SPACEDIM; idim++) {

        MultiFab diff_coefs(hydrosim->getEdgeBoxArray(idim), dm, 1, 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(diff_coefs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

            const Box& bx = mfi.tilebox();

            auto diff_coefs_arr = diff_coefs[mfi].array();
            auto limiter_arr = lambda[idim][mfi].array();
            auto advect_vel_arr = vel[mfi].array();
            auto diff_factor_arr = dcf[mfi].array();

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                Real r, s;

                if (idim == 0) {

                    edge_center_geometry_factors(i, j, k, idim, geomdata, r, s);

                    if (advect_vel_arr(i-1,j,k,0) + advect_vel_arr(i,j,k,0) > 0.e0_rt) {
                        diff_coefs_arr(i,j,k) = diff_factor_arr(i-1,j,k) * advect_vel_arr(i-1,j,k,0) * limiter_arr(i,j,k);
                    }
                    else if (advect_vel_arr(i-1,j,k,0) + advect_vel_arr(i,j,k,0) < 0.e0_rt) {
                        diff_coefs_arr(i,j,k) = diff_factor_arr(i,j,k) * advect_vel_arr(i,j,k,0) * limiter_arr(i,j,k);
                    }
                    else {
                        diff_coefs_arr(i,j,k) = 0.e0_rt;
                    }

                    diff_coefs_arr(i,j,k) = diff_coefs_arr(i,j,k) * r;

                }
                else if (idim == 1) {

                    edge_center_geometry_factors(i, j, k, idim, geomdata, r, s);

                    if (advect_vel_arr(i,j-1,k,1) + advect_vel_arr(i,j,k,1) > 0.e0_rt) {
                        diff_coefs_arr(i,j,k) = diff_factor_arr(i,j-1,k) * advect_vel_arr(i,j-1,k,1) * limiter_arr(i,j,k);
                    }
                    else if (advect_vel_arr(i,j-1,k,1) + advect_vel_arr(i,j,k,1) < 0.e0_rt) {
                        diff_coefs_arr(i,j,k) = diff_factor_arr(i,j,k) * advect_vel_arr(i,j,k,1) * limiter_arr(i,j,k);
                    }
                    else {
                        diff_coefs_arr(i,j,k) = 0.e0_rt;
                    }

                    diff_coefs_arr(i,j,k) = diff_coefs_arr(i,j,k) * r;

                }
                else {

                    edge_center_geometry_factors(i, j, k, idim, geomdata, r, s);

                    if (advect_vel_arr(i,j,k-1,2) + advect_vel_arr(i,j,k,2) > 0.e0_rt) {
                        diff_coefs_arr(i,j,k) = diff_factor_arr(i,j,k-1) * advect_vel_arr(i,j,k-1,2) * limiter_arr(i,j,k);
                    }
                    else if (advect_vel_arr(i,j,k-1,2) + advect_vel_arr(i,j,k,2) < 0.e0_rt) {
                        diff_coefs_arr(i,j,k) = diff_factor_arr(i,j,k) * advect_vel_arr(i,j,k,2) * limiter_arr(i,j,k);
                    }
                    else {
                        diff_coefs_arr(i,j,k) = 0.e0_rt;
                    }

                    diff_coefs_arr(i,j,k) = diff_coefs_arr(i,j,k) * r;

                }

            });
        }

        hem->setGradientCoefficients2(level, diff_coefs, idim);
        hem->gradientScale2() = 1.0;
    }
}

void RadTransportSolver::levelAssembleRhs(int level, MultiFab& rhs,
                        MultiFab& temp,
                        MultiFab& fkp, MultiFab& eta, MultiFab& coupling_inverse,
                        MultiFab& matter_energy, MultiFab& state_energy,
                        MultiFab& flux_div_old, MultiFab& rad_energy_old, MultiFab& Edot,
                        Real delta_t, Real sigma, Real c, Real theta,
                        FluxRegister* fine_correction, Real scale,
                        int rad_group_idx, Real nu, Real dnu)
{
  BL_PROFILE("RadTransportSolver::levelAssembleRhs");
  BL_ASSERT(rhs.nGrow() == 0);

  const Geometry& geom = parent->Geom(level);
  auto geomdata = geom.data();

  rhs.setVal(0.0);
  if (fine_correction) {
    // This works trivially for a multilevel solve since the finer level is
    // present in the rhs to overwrite the junk produced under it.
    // In a single-level version we have to be sure that fine_correction
    // has been cleaned up using ClearInternalBorders.

    // Hack:  For the single group case rad_group_idx defaults to -1, which is
    // significant later in this routine.  So we have to construct the
    // correct component number here:

    int reflux_group = (rad_group_idx < 0) ? 0 : rad_group_idx;

    fine_correction->Reflux(rhs, scale, reflux_group, 0, 1, parent->Geom(level));
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      auto rhs_arr = rhs[mfi].array();
      auto temp_field_arr = temp[mfi].array();
      auto scaled_planck_arr = fkp[mfi].array();
      auto coupling_arr = eta[mfi].array();
      auto coupling_inverse_arr = coupling_inverse[mfi].array();
      auto matter_energy_arr = matter_energy[mfi].array();
      auto state_energy_arr = state_energy[mfi].array();
      auto flux_div_prev_arr = flux_div_old[mfi].array();
      auto rad_energy_old_arr = rad_energy_old[mfi].array(0);
      auto energy_rate_arr = Edot[mfi].array();

      const Real dtm = 1.0_rt / delta_t;

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          Real ek = scaled_planck_arr(i,j,k) * coupling_arr(i,j,k);
          Real bs = coupling_inverse_arr(i,j,k) * 4.e0_rt * sigma * scaled_planck_arr(i,j,k) * std::pow(temp_field_arr(i,j,k), 4);
          Real es = coupling_arr(i,j,k) * (matter_energy_arr(i,j,k) - state_energy_arr(i,j,k));
          Real ekt = (1.0_rt - theta) * coupling_arr(i,j,k);

          Real r, s;
          cell_center_geometry_factors(i, j, k, geomdata, r, s);

          if (AMREX_SPACEDIM == 1) {
              s = 1.0_rt;
          }

          rhs_arr(i,j,k) = (rhs_arr(i,j,k) + r * s *
                            (bs + dtm * (rad_energy_old_arr(i,j,k) + es) +
                             ek * c * energy_rate_arr(i,j,k) -
                             ekt * flux_div_prev_arr(i,j,k))) / (1.0_rt - ekt);
      });
  }
}


void RadTransportSolver::levelSolveSystem(int level,
                          MultiFab& Er, int rad_group_idx, MultiFab& rhs,
                          Real sync_residual_scale)
{
  BL_PROFILE("RadTransportSolver::levelSolveSystem");

  // Set coeffs, build solver, solve
  if (hd) {
    hd->setScalars(radiation_solve::alpha, radiation_solve::beta);
  }
  else if (hm) {
    hm->setScalars(radiation_solve::alpha, radiation_solve::beta);
  }
  else if (hem) {
    hem->setScalars(radiation_solve::alpha, radiation_solve::beta);
  }

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
    hm->assembleMatrix();
    hm->commitMatrix();
    hm->fillLevelVectors(level, Er, rad_group_idx, rhs, Nonzero_BC);
    hm->commitVectors();
    hm->configureSolver(radiation_solve::reltol, radiation_solve::abstol, radiation_solve::maxiter);
    hm->solve();
    hm->getSolution(level, Er, rad_group_idx);
    Real res = hm->currentResidualNorm();
    if (verbose >= 2 && ParallelDescriptor::IOProcessor()) {
      int oldprec = std::cout.precision(20);
      std::cout << "Absolute residual = " << res << std::endl;
      std::cout.precision(oldprec);
    }
    res *= sync_residual_scale;
    hm->releaseSolver();
  }
  else if (hem) {
    hem->assembleMatrix();
    hem->commitMatrix();
    hem->fillLevelVectors(level, Er, rad_group_idx, rhs, Nonzero_BC);
    hem->commitVectors();
    hem->configureSolver(radiation_solve::reltol, radiation_solve::abstol, radiation_solve::maxiter);
    hem->solve();
    hem->getSolution(level, Er, rad_group_idx);
    Real res = hem->currentResidualNorm();
    if (verbose >= 2 && ParallelDescriptor::IOProcessor()) {
      int oldprec = std::cout.precision(20);
      std::cout << "Absolute residual = " << res << std::endl;
      std::cout.precision(oldprec);
    }
    res *= sync_residual_scale;
    hem->releaseSolver();
  }
}

void RadTransportSolver::levelFaceToCellFlux(int level, const Array<MultiFab, AMREX_SPACEDIM>& Flux,
                                     MultiFab& flx, int flux_comp_base)
{
    int num_flux_comps = flx.nComp();

    const Geometry& geom = parent->Geom(level);
    auto geomdata = geom.data();

    for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
    {
#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(flx,true); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            auto t = flx[mfi].array();
            auto f = Flux[idim][mfi].array();

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                int it = idim * NGROUPS + flux_comp_base;

                Real radius_left, s_lo;
                edge_center_geometry_factors(i, j, k, idim, geomdata, radius_left, s_lo);

                Real radius_right, s_hi;
                edge_center_geometry_factors(i+1, j, k, idim, geomdata, radius_right, s_hi);

                if (idim == 0) {
                    t(i,j,k,it) = (f(i,j,k) / (radius_left + 1.e-50_rt) + f(i+1,j,k) / radius_right) * 0.5_rt;
                }
                else if (idim == 1) {
                    t(i,j,k,it) = (f(i,j,k) / radius_left + f(i,j+1,k) / radius_left) * 0.5_rt;
                }
                else {
                    t(i,j,k,it) = (f(i,j,k) + f(i,j,k+1)) * 0.5_rt;
                }
            });
        }
    }
}

void RadTransportSolver::levelFaceFlux(int level,
                         Array<MultiFab, AMREX_SPACEDIM>& Flux,
                         MultiFab& Er, int rad_group_idx)
{
  BL_PROFILE("RadTransportSolver::levelFaceFlux");
  const BoxArray& grids = parent->boxArray(level);
  const DistributionMapping& dmap = parent->DistributionMap(level);

  // grow a larger MultiFab to hold Er so we can difference across faces
  MultiFab RadEnergyGrown(grids, dmap, 1, 1);
  RadEnergyGrown.setVal(0.0);
  MultiFab::Copy(RadEnergyGrown, Er, rad_group_idx, 0, 1, 0);

  RadEnergyGrown.FillBoundary(parent->Geom(level).periodicity()); // zeroes left in off-level boundaries

  auto dx = parent->Geom(level).CellSizeArray();

  for (int n = 0; n < AMREX_SPACEDIM; n++) {

      const MultiFab *bp;

      if (hd) {
          bp = &hd->setFaceCoefficients(n);
      }
      else if (hm) {
          bp = &hm->setFaceCoefficients(level, n);
      }
      else if (hem) {
          bp = &hem->setFaceCoefficients(level, n);
      }

      MultiFab &bcoef = *(MultiFab*)bp;

#ifdef _OPENMP
#pragma omp parallel
#endif
      for (MFIter mfi(Flux[n], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
          const Box& bx = mfi.tilebox();

          auto RadEnergyBorder_arr = RadEnergyGrown[mfi].array();
          auto face_coef_arr = bcoef[mfi].array();
          auto diff_flux_arr = Flux[n][mfi].array();

          Real beta = radiation_solve::beta;

          amrex::ParallelFor(bx,
          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
          {
              if (n == 0) {

                  const Real fac = -beta / dx[0];

                  diff_flux_arr(i,j,k) = face_coef_arr(i,j,k) * (RadEnergyBorder_arr(i,j,k) - RadEnergyBorder_arr(i-1,j,k)) * fac;

              }
              else if (n == 1) {

                  const Real fac = -beta / dx[1];

                  diff_flux_arr(i,j,k) = face_coef_arr(i,j,k) * (RadEnergyBorder_arr(i,j,k) - RadEnergyBorder_arr(i,j-1,k)) * fac;

              }
              else {

                  const Real fac = -beta / dx[2];

                  diff_flux_arr(i,j,k) = face_coef_arr(i,j,k) * (RadEnergyBorder_arr(i,j,k) - RadEnergyBorder_arr(i,j,k-1)) * fac;

              }
          });
      }

  }

  // Correct fluxes at physical and coarse-fine boundaries.

  // Note: It would be good to move much of this section into the
  // linear solver drivers themselves (one layer down), to enable
  // an implementation that does not compute fluxes everywhere in
  // an EdgeVar.  We can't just move the nonsymmetric pieces easily
  // by themselves, though, because the current implementation
  // trashes the boundary fluxes before fixing them.

  if (hd) {
    hd->faceFluxAtBoundary(&Flux[0], Er, rad_group_idx, Nonzero_BC);
  }
  else if (hm) {
    hm->faceFluxAtBoundary(level, &Flux[0], Er, rad_group_idx, Nonzero_BC);
  }
}

void RadTransportSolver::levelCoarseFineFlux(int level,
                            FluxRegister* flux_from_fine, FluxRegister* flux_to_coarse,
                            const Array<MultiFab, AMREX_SPACEDIM>& Flux,
                            int rad_group_idx)
{
  BL_PROFILE("RadTransportSolver::levelCoarseFineFlux");

  const Real* dx = parent->Geom(level).CellSize();

  const Real volume = AMREX_D_TERM(dx[0], * dx[1], * dx[2]);

  if (flux_from_fine) {
    for (int n = 0; n < AMREX_SPACEDIM; n++) {
      const Real scale = volume / dx[n];
      flux_from_fine->CrseInit(Flux[n], n, 0, rad_group_idx, 1, scale);
    }
  }
  if (flux_to_coarse) {
    for (OrientationIter face; face; ++face) {
      Orientation ori = face();
      (*flux_to_coarse)[ori].setVal(0.0, rad_group_idx, 1);
    }
    for (int n = 0; n < AMREX_SPACEDIM; n++) {
      const Real scale = volume / dx[n];
      flux_to_coarse->FineAdd(Flux[n], n, 0, rad_group_idx, 1, scale);
    }
  }
}

void RadTransportSolver::levelDiffusionTerm(int level, MultiFab& diffusive_face_term, MultiFab& Er, int rad_group_idx)
{
  BL_PROFILE("RadTransportSolver::levelDiffusionTerm");
  const BoxArray& grids = parent->boxArray(level);
  const DistributionMapping& dmap = parent->DistributionMap(level);
  const Geometry& geom = parent->Geom(level);
  const GeometryData& geomdata = geom.data();
  auto dx = parent->Geom(level).CellSizeArray();
  const HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));

  Array<MultiFab, AMREX_SPACEDIM> diffusion_term_face;
  for (int idim=0; idim<AMREX_SPACEDIM; idim++) {
      diffusion_term_face[idim].define(hydrosim->getEdgeBoxArray(idim), dmap, 1, 0);
  }

  // grow a larger MultiFab to hold Er so we can difference across faces
  MultiFab RadEnergyGrown(grids, dmap, 1, 1);
  RadEnergyGrown.setVal(0.0);
  MultiFab::Copy(RadEnergyGrown, Er, rad_group_idx, 0, 1, 0);

  RadEnergyGrown.FillBoundary(parent->Geom(level).periodicity()); // zeroes left in off-level boundaries

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (int n = 0; n < AMREX_SPACEDIM; n++) {
      const MultiFab *dp;

      dp = &hem->setGradientCoefficients2(level, n);
      MultiFab &diff_coef = *(MultiFab*)dp;

      for (MFIter fi(diff_coef,true); fi.isValid(); ++fi) {
          const Box& bx = fi.tilebox();

          auto Er = RadEnergyGrown[fi].array();
          auto dc = diff_coef[fi].array();
          auto dtf = diffusion_term_face[n][fi].array();

          amrex::ParallelFor(bx,
          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
          {
              if (n == 0) {
                  dtf(i,j,k) = (Er(i,j,k) - Er(i-1,j,k)) / dx[0] * dc(i,j,k);
              }
              else if (n == 1) {
                  dtf(i,j,k) = (Er(i,j,k) - Er(i,j-1,k)) / dx[1] * dc(i,j,k);
              }
              else {
                  dtf(i,j,k) = (Er(i,j,k) - Er(i,j,k-1)) / dx[2] * dc(i,j,k);
              }
          });
      }
  }

  // Correct D terms at physical and coarse-fine boundaries.
  hem->computeBoundaryTerm(level, &diffusion_term_face[0], Er, rad_group_idx);

  // Correct for metric terms (only has an effect in non-Cartesian geometries).
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
#ifdef _OPENMP
#pragma omp parallel
#endif
      for (MFIter mfi(diffusion_term_face[dir], true); mfi.isValid(); ++mfi) {
          const Box& box = mfi.tilebox();
          Array4<Real> const d = diffusion_term_face[dir][mfi].array();

          amrex::ParallelFor(box,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              if (dir == 0 && (geomdata.Coord() == CoordSys::SPHERICAL || geomdata.Coord() == CoordSys::RZ)) {
                  Real r, s;
                  edge_center_geometry_factors(i, j, k, dir, geomdata, r, s);

                  d(i,j,k) = d(i,j,k) / (r + 1.0e-50_rt);
              }
              else if (dir == 1 && geomdata.Coord() == CoordSys::RZ) {
                  Real r, s;
                  cell_center_geometry_factors(i, j, k, geomdata, r, s);

                  d(i,j,k) = d(i,j,k) / r;
              }
          });
      }
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter fi(diffusive_face_term,true); fi.isValid(); ++fi) {
      const Box& bx = fi.tilebox();

      auto Dx = diffusion_term_face[0][fi].array();
#if AMREX_SPACEDIM >= 2
      auto Dy = diffusion_term_face[1][fi].array();
#endif
#if AMREX_SPACEDIM == 3
      auto Dz = diffusion_term_face[2][fi].array();
#endif

      auto D = diffusive_face_term[fi].array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
#if AMREX_SPACEDIM == 1
          D(i,j,k) = (Dx(i,j,k) + Dx(i+1,j,k)) * 0.5_rt;
#elif AMREX_SPACEDIM == 2
          D(i,j,k) = (Dx(i,j,k) + Dx(i+1,j,k) +
                      Dy(i,j,k) + Dy(i,j+1,k)) * 0.25_rt;
#else
          D(i,j,k) = (Dx(i,j,k) + Dx(i+1,j,k) +
                      Dy(i,j,k) + Dy(i,j+1,k) +
                      Dz(i,j,k) + Dz(i,j,k+1)) * (1.0_rt / 6.0_rt);
#endif
      });
  }
}

// <MultigroupDiffusion routines>
void RadTransportSolver::computeFaceCoeffs(MultiFab& bcoefs, int idim,
                              MultiFab& kappa_r, int kcomp,
                              MultiFab& lambda, int limiter_comp,
                              Real c, const Geometry& geom)
{
  BL_PROFILE("RadTransportSolver::computeFaceCoeffs (MultigroupDiffusion)");
  BL_ASSERT(kappa_r.nGrow() == 1);

  auto geomdata = geom.data();
  auto dx = geom.CellSizeArray();

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(lambda, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      auto dir_coef_arr = bcoefs[mfi].array();
      auto limiter_arr = lambda[mfi].array(limiter_comp);
      auto kappa_ross_arr = kappa_r[mfi].array(kcomp);

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          if (idim == 0) {

              Real r, s;
              edge_center_geometry_factors(i, j, k, idim, geomdata, r, s);

              if (AMREX_SPACEDIM == 1) {
                  s = 1.e0_rt;
              }

              Real kap = face_opacity_average(kappa_ross_arr(i-1,j,k), kappa_ross_arr(i,j,k), dx[0], -1);
              dir_coef_arr(i,j,k) = r * s * c * limiter_arr(i,j,k) / kap;

          }
          else if (idim == 1) {

              Real r, s;
              edge_center_geometry_factors(i, j, k, idim, geomdata, r, s);

              if (AMREX_SPACEDIM == 1) {
                  s = 1.e0_rt;
              }

              Real kap = face_opacity_average(kappa_ross_arr(i,j-1,k), kappa_ross_arr(i,j,k), dx[1], -1);
              dir_coef_arr(i,j,k) = r * s * c * limiter_arr(i,j,k) / kap;

          }
          else {

              Real r, s;
              edge_center_geometry_factors(i, j, k, idim, geomdata, r, s);

              if (AMREX_SPACEDIM == 1) {
                  s = 1.e0_rt;
              }

              Real kap = face_opacity_average(kappa_ross_arr(i,j,k-1), kappa_ross_arr(i,j,k), dx[2], -1);
              dir_coef_arr(i,j,k) = r * s * c * limiter_arr(i,j,k) / kap;

          }
      });
  }
}

void RadTransportSolver::levelDiagCoeffs(int level, MultiFab& kpp,
                            Real delta_t, Real c, int rad_group_idx, Real pseudo_transient_tau)
{
  BL_PROFILE("RadTransportSolver::levelDiagCoeffs (MultigroupDiffusion)");
  const BoxArray& grids = parent->boxArray(level);
  const DistributionMapping& dmap = parent->DistributionMap(level);
  const auto geomdata = parent->Geom(level).data();

  // allocate space for ABecLaplacian acoeffs, fill with values

  int Ncomp = 1;
  int Nghost = 0;
  MultiFab acoefs(grids, dmap, Ncomp, Nghost);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(kpp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

      const Box& bx = mfi.tilebox();

      Real pseudo_dt = delta_t / (1.0 + pseudo_transient_tau);

      auto diag_coef_arr = acoefs[mfi].array();
      auto kappa_group_arr = kpp[mfi].array(rad_group_idx);

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          Real r, s;
          cell_center_geometry_factors(i, j, k, geomdata, r, s);

          diag_coef_arr(i,j,k) = c * kappa_group_arr(i,j,k) + 1.e0_rt / pseudo_dt;
          diag_coef_arr(i,j,k) = r * s * diag_coef_arr(i,j,k);
      });
  }

  // set a coefficients
  if (hd) {
    hd->setMassCoefficients(acoefs);
  }
  else if (hm) {
    hm->setMassCoefficients(level,acoefs);
  }
  else if (hem) {
    hem->setMassCoefficients(level,acoefs);
  }
}


void RadTransportSolver::levelAssembleRhs(int level, MultiFab& rhs, const MultiFab& jg,
                        const MultiFab& group_weight_mf,
                        const MultiFab& couple_source,
                        const MultiFab& thermal_frac,
                        const MultiFab& rad_energy_step, const MultiFab& gas_energy_step,
                        const MultiFab& rad_energy_star, const MultiFab& gas_energy_star,
                        Real delta_t, int rad_group_idx, int it, Real pseudo_transient_tau)
{
  BL_PROFILE("RadTransportSolver::levelAssembleRhs (MultigroupDiffusion version)");
  HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
  Real time = hydrosim->get_state_data(Radiation_Type).curTime();
  const Real* dx = parent->Geom(level).CellSize();
  auto geomdata = parent->Geom(level).data();

  const Real dt1 = 1.0_rt / delta_t;

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter ri(rhs, TilingIfNotGPU()); ri.isValid(); ++ri) {

      const Box& bx = ri.tilebox();

      auto rhs_arr = rhs[ri].array();
      auto emission_arr = jg[ri].array();
      auto group_weight_arr = group_weight_mf[ri].array();
      auto couple_source_arr = couple_source[ri].array();
      auto thermal_frac_arr = thermal_frac[ri].array();
      auto rad_energy_step_arr = rad_energy_step[ri].array();
      auto step_energy_arr = gas_energy_step[ri].array();
      auto rad_energy_star_arr = rad_energy_star[ri].array();
      auto star_energy_arr = gas_energy_star[ri].array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          Real Hg = group_weight_arr(i,j,k,rad_group_idx) * thermal_frac_arr(i,j,k);

          rhs_arr(i,j,k) = C::c_light * (emission_arr(i,j,k,rad_group_idx) + Hg * couple_source_arr(i,j,k))
                           + dt1 * (rad_energy_step_arr(i,j,k,rad_group_idx) - Hg * (star_energy_arr(i,j,k) - step_energy_arr(i,j,k))
                                    + pseudo_transient_tau * rad_energy_star_arr(i,j,k,rad_group_idx));

          Real r, s;
          cell_center_geometry_factors(i, j, k, geomdata, r, s);

          rhs_arr(i,j,k) *= r;

          user_rad_source(i, j, k, rhs_arr, geomdata, time, delta_t, rad_group_idx);
      });
  }
}

// </ MultigroupDiffusion routines>

void RadTransportSolver::setSolverScale(Real cScale, Real d1Scale, Real d2Scale)
{
  if (hem) {
    hem-> advectionScale() =  cScale;
    hem->gradientScale1() = d1Scale;
    hem->gradientScale2() = d2Scale;
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

void RadTransportSolver::edgeGeomWeights(int idim, const Geometry& geom, const Box& edge_region,
                             Vector<Real>& r, Vector<Real>& s)
{
    const Box& reg = amrex::enclosedCells(edge_region);
    const int I = (AMREX_SPACEDIM >= 2) ? 1 : 0;
    if (geom.IsCartesian()) {
        r.resize(reg.length(0)+1, 1);
        s.resize(reg.length(I)+1, 1);
    }
    else if (geom.IsRZ()) {
        if (idim == 0) {
            geom.GetEdgeLoc(r, reg, 0);
        }
        else {
            geom.GetCellLoc(r, reg, 0);
        }
        s.resize(reg.length(I)+1, 1);
    }
    else {
      if (idim == 0) {
        geom.GetEdgeLoc(r, reg, 0);
        geom.GetCellLoc(s, reg, I);
      }
      else {
        geom.GetCellLoc(r, reg, 0);
        geom.GetEdgeLoc(s, reg, I);
      }
      const Real *dx = geom.CellSize();

      if (idim == 0) {
          for (int i = edge_region.loVect()[0]; i <= edge_region.hiVect()[0]; ++i) {
              r[i] *= r[i];
          }
#if AMREX_SPACEDIM >= 2
          Real h2 = 0.5e0_rt * dx[1];
          Real d2 = 1.e0_rt / dx[1];
          for (int j = edge_region.loVect()[1]; j <= edge_region.hiVect()[1]; ++j) {
              s[j] = d2 * (std::cos(s[j] - h2) - std::cos(s[j] + h2));
          }
#endif
      }
      else {
          Real h1 = 0.5e0_rt * dx[0];
          Real d1 = 1.e0_rt / (3.e0_rt * dx[0]);
          for (int i = edge_region.loVect()[0]; i <= edge_region.hiVect()[0]; ++i) {
              r[i] = d1 * (std::pow(r[i] + h1, 3) - std::pow(r[i] - h1, 3));
          }
#if AMREX_SPACEDIM >= 2
          for (int j = edge_region.loVect()[1]; j <= edge_region.hiVect()[1]; ++j) {
              s[j] = std::sin(s[j]);
          }
#endif
      }
    }
}

