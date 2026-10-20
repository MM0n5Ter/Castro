
#include <AMReX_LO_BCTYPES.H>

#include <Radiation.H>
#include <RadTransportSolver.H>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace amrex;

void Radiation::advance_single_group(int level, int iteration, int ncycle)
{
  BL_PROFILE("Radiation::advance_single_group");
  if (verbose && ParallelDescriptor::IOProcessor()) {
    std::cout << "Radiation implicit update, level " << level << "..." << std::endl;
  }

  bool has_diffusion_coefs = (Radiation::SolverKind == Radiation::SingleGroupFLDSolver &&
                     Radiation::lorentz_energy_term);

  int group = 0;
  int fine_level = parent->finestLevel();


  HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
  const BoxArray& grids = hydrosim->boxArray();
  const DistributionMapping& dmap = hydrosim->DistributionMap();
  Real delta_t          = parent->dtLevel(level);
  Real time = hydrosim->get_state_data(Radiation_Type).curTime();

  MultiFab& S_new = hydrosim->get_new_data(State_Type);
  MultiFab& rad_energy_new = hydrosim->get_new_data(Radiation_Type);

  MultiFab rad_energy_old(grids, dmap, rad_energy_new.nComp(), rad_energy_new.nGrow());
  MultiFab::Copy(rad_energy_old, rad_energy_new, 0, 0, rad_energy_new.nComp(), 0);

  Array<MultiFab, AMREX_SPACEDIM> new_face_flux;

  for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
      new_face_flux[idim].define(hydrosim->getEdgeBoxArray(idim), dmap, 1, 0);
  }

  MultiFab diffusive_face_term;
  if (has_diffusion_coefs) {
      diffusive_face_term.define(grids, dmap, AMREX_SPACEDIM, 0);
  }

  MultiFab matter_energy_density(grids,dmap,1,0);
  MultiFab state_energy_density(grids,dmap,1,0);

  MultiFab temp(grids,dmap,1,0);
  MultiFab fkp(grids,dmap,1,0);

  MultiFab& flux_div_old = *flux_div[level];
  MultiFab flux_div_new(grids,dmap,1,0);

  MultiFab limited_rad_energy;

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(matter_energy_density,true); mfi.isValid(); ++mfi) {
      const Box& reg = mfi.tilebox();

      auto density_matter_energy_arr = matter_energy_density[mfi].array();
      auto new_state_view = S_new[mfi].array();

      amrex::ParallelFor(reg,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          density_matter_energy_arr(i,j,k) = new_state_view(i,j,k,UEINT);
      });
  }

  MultiFab::Copy(state_energy_density, matter_energy_density, 0, 0, 1, 0);


  MultiFab kappa_r(grids,dmap,1,1);

  MultiFab fluid_velocity_mf;
  MultiFab diffusion_coef_mf;
  if (has_diffusion_coefs) {
    fluid_velocity_mf.define(grids, dmap, AMREX_SPACEDIM, 1);
    diffusion_coef_mf.define(grids, dmap, 1, 1);
    derive_rosseland_velocity_coef(kappa_r, fluid_velocity_mf, diffusion_coef_mf, delta_t, c, hydrosim);
  }
  else {
    derive_rosseland_opacity(kappa_r, hydrosim);
  }

  MultiFab eta(grids,dmap,1,0);
  MultiFab coupling_inverse(grids,dmap,1,0);

  Array<MultiFab, AMREX_SPACEDIM> lambda;

  for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
      lambda[idim].define(hydrosim->getEdgeBoxArray(idim), dmap, 1, 0);
  }

  if (limiter_lag_iter == 0) {
    evaluateGradientRatio(level, lambda, kappa_r, 0, rad_energy_old, 0);

    computeFluxLimiter(level, lambda);
  }
  else if (limiter_lag_iter < 0) {
    MultiFab& rad_energy_lagged = hydrosim->get_old_data(Radiation_Type);
    evaluateGradientRatio(level, lambda, kappa_r, 0, rad_energy_lagged, 0);
    computeFluxLimiter(level, lambda);
  }


  RadiativeBoundary bd(grids, dmap, hydrosim->Geom());

  fetchBndryData(bd, rad_energy_new, time, level);

  bool has_transport_bc = false;
  int lo_bc[3]={0}, hi_bc[3]={0};
  for (int idim=0; idim<AMREX_SPACEDIM; idim++) {
    lo_bc[idim] = rad_boundary.lo(idim);
    hi_bc[idim] = rad_boundary.hi(idim);
    if (lo_bc[idim] == AMREX_LO_SANCHEZ_POMRANING ||
        hi_bc[idim] == AMREX_LO_SANCHEZ_POMRANING) {
      has_transport_bc = true;
    }
  }

  FluxRegister* flux_from_fine  =
      (level < fine_level) ? flux_candidate_regs[level+1].get() : nullptr;
  FluxRegister* flux_to_coarse =
      (level > 0) ? flux_candidate_regs[level].get() : nullptr;

  RadTransportSolver* const solver = hydrosim->rad_operator.get();

  solver->levelBoundary(bd);

  Real relative, absolute;
  int it = 0;
  int use_conservative_update = (matter_update_mode == 0) ? 1 : 0;
  int converged;
  Real underrelax_scale = 1.0;

  auto print_spot_energy = [&] () {
    if (verbose > 2) {
       for (MFIter fi(state_energy_density); fi.isValid(); ++fi) {
          if (grids[fi.index()].contains(spot)) {
             std::cout << "state_energy_density: ";
             for (int m=0; m<1; ++m) std::cout << state_energy_density[0](spot,m) << " ";
             std::cout << std::endl;
             break;
          }
       }
    }
  };

  do {

    it++;

    MultiFab::Copy(temp, state_energy_density, 0, 0, state_energy_density.nComp(), 0);

    if (it == 1 || it <= planck_lag_iter + 1) {
      derive_planck_and_temp(fkp, temp, S_new);
    }
    else {
      derive_planck_and_temp(eta, temp, S_new);
    }


    compute_coupling_factors(eta, coupling_inverse, S_new, temp, fkp, rad_energy_new,
                delta_t, c, underrelax_scale,
                (it > planck_lag_iter) ? PlanckCoupling::Frozen : PlanckCoupling::Refresh);

    underrelax_scale *= underrelax_factor;

    if (it > 1 && it <= rosseland_lag_iter + 1) {
      refresh_rosseland_from_temp(kappa_r, temp, S_new, hydrosim->Geom());
    }


    solver->levelDiagCoeffs(level, fkp, eta, coupling_inverse, c, delta_t, 1.0);

    if (limiter_lag_iter > 0 && it <= limiter_lag_iter + 1) {
      evaluateGradientRatio(level, lambda, kappa_r, 0, rad_energy_new, 0);
      computeFluxLimiter(level, lambda);
    }

    solver->levelFaceCoeffs(level, lambda, kappa_r, 0, c);

    if (has_transport_bc) {
      solver->levelBoundaryScale(level, lambda, 0, lo_bc, hi_bc);
    }

    if (has_diffusion_coefs) {
      if (it>1) {
        refresh_diffusion_coef(diffusion_coef_mf, coupling_inverse, fkp, kappa_r, hydrosim->Geom());
      }
      solver->levelDiffCoeffs(level, lambda, fluid_velocity_mf, diffusion_coef_mf);
    }

    {
      MultiFab rhs(grids,dmap,1,0);

      flux_div_new.setVal(0.0);
      solver->levelAssembleRhs(level, rhs, temp,
                       fkp, eta, coupling_inverse, matter_energy_density, state_energy_density,
                       flux_div_old, rad_energy_old, flux_div_new,
                       delta_t, sigma, 0.0, 1.0,
                       NULL);

      apply_delayed_reflux(level, rhs, 0);

      solver->levelSolveSystem(level, rad_energy_new, 0, rhs, 0.01);

      flux_div_new.setVal(0.0);

      solver->levelFaceFlux(level, new_face_flux, rad_energy_new, 0);

      if (has_diffusion_coefs) {
        solver->levelDiffusionTerm(level, diffusive_face_term, rad_energy_new, 0);
      }


      solver->levelCoarseFineFlux(level, flux_from_fine, flux_to_coarse, new_face_flux, 0);
    }


    UpdateNorms update_norms;

    if (use_conservative_update) {
      MultiFab& exch = temp;

      evaluate_energy_transfer(exch, rad_energy_new, fkp);

      if (has_diffusion_coefs) {
        update_norms = apply_implicit_energy(state_energy_density, matter_energy_density, eta, coupling_inverse,
                               flux_div_old, flux_div_new,
                               exch, diffusive_face_term, delta_t);
      }
      else {
        update_norms = apply_implicit_energy(state_energy_density, matter_energy_density, eta, coupling_inverse,
                               flux_div_old, flux_div_new,
                               exch, delta_t);
      }
    }
    else {
      update_norms = apply_nonconservative_correction(state_energy_density, matter_energy_density, eta, coupling_inverse,
                                    rad_energy_new, flux_div_old, flux_div_new,
                                    temp, fkp, S_new, delta_t);
      if (has_diffusion_coefs) {
        amrex::Error("Radiation::advance_single_group: must do conservative energy update when has_diffusion_coefs is true");
      }
    }

    relative = update_norms.relative;
    absolute = update_norms.absolute;

    if (verbose > 1 && ParallelDescriptor::IOProcessor()) {
      int oldprec = std::cout.precision(20);
      if (use_conservative_update)
        std::cout << "Radiation Update:  Iteration " << it;
      else
        std::cout << "Nonconservative:   Iteration " << it;
      std::cout << ", Relative = " << relative << std::endl;
      std::cout << "                              "
           << "  Absolute = " << absolute << std::endl;
      std::cout.precision(oldprec);
    }

    print_spot_energy();

    converged = (relative <= reltol || absolute <= abstol || it >= maxiter);

    if (converged && !use_conservative_update) {
      use_conservative_update = 1;
      converged = 0;
    }


  } while (!converged);

  if (verbose == 1 && ParallelDescriptor::IOProcessor()) {
    int oldprec = std::cout.precision(20);
    std::cout << "Radiation Update:  Iteration " << it
        << ", Relative = " << relative << std::endl;
    std::cout << "                              "
        << "  Absolute = " << absolute << std::endl;
    std::cout.precision(oldprec);
  }

  if (it == maxiter &&
      (relative > reltol && absolute > abstol)) {
    if (verbose > 0 && ParallelDescriptor::IOProcessor()) {
      std::cout << "Implicit Update Failed to Converge" << std::endl;
      amrex::Abort("You Lose");
    }
  }

  if (flux_from_fine) {
    for (OrientationIter face; face; ++face) {
      Orientation ori = face();
      (*flux_accum_reg[level+1])[ori].linComb(1.0, -1.0,
                                         (*flux_from_fine)[ori], group, group, 1);
    }
  }

  if (flux_to_coarse) {
    for (OrientationIter face; face; ++face) {
      Orientation ori = face();
      (*flux_accum_reg[level])[ori].linComb(1.0, 1.0 / ncycle,
                                       (*flux_to_coarse)[ori], group, group, 1);
    }
  }

  if (level == fine_level && iteration == 1) {


    for (int flev = level; flev > 0; flev--) {
      dt_previous[flev-1] = 0.0;
    }


    for (int flev = level+1; flev <= parent->maxLevel(); flev++) {
        conservative_register_old[flev].reset();
    }
  }

  sync_state_from_energy(S_new, state_energy_density);

  print_spot_energy();

  MultiFab::Copy(flux_div_old, flux_div_new, 0, 0, 1, 0);

  if (radiation::plot_flux_limiter) {
      store_limiter_plotvar(level, lambda);
  }

  if (radiation::plot_planck_opacity) {
      MultiFab::Copy(*plot_fields[level], fkp, 0, icomp_kappa_planck, 1, 0);
  }

  if (radiation::plot_rosseland_opacity) {
      MultiFab::Copy(*plot_fields[level], kappa_r, 0, icomp_kappa_rosseland, 1, 0);
  }

  if (radiation::plot_lab_rad_energy || radiation::plot_lab_rad_flux || radiation::plot_moving_flux) {
      MultiFab flx(grids, dmap, AMREX_SPACEDIM, 0);
      solver->levelFaceToCellFlux(level, new_face_flux, flx, 0);

      if (radiation::plot_lab_rad_energy) {
          store_lab_energy_plotvar(level, S_new, rad_energy_new, flx, 0);
      }

      if (radiation::plot_lab_rad_flux) {
          if (radiation::moving_frame) {
              store_flux_plotvar(level, S_new, lambda, rad_energy_new, flx, 0);
          } else {
              MultiFab::Copy(*plot_fields[level], flx, 0, icomp_lab_flux, AMREX_SPACEDIM, 0);
          }
      }

      if (radiation::plot_moving_flux) {
          if (radiation::moving_frame) {
              MultiFab::Copy(*plot_fields[level], flx, 0, icomp_comoving_flux, AMREX_SPACEDIM, 0);
          } else {
              store_flux_plotvar(level, S_new, lambda, rad_energy_new, flx, 0, -1.0);
          }
      }
  }

  if (verbose && ParallelDescriptor::IOProcessor()) {
    std::cout << "                                     done" << std::endl;
  }
}
