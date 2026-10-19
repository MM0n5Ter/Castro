
#include <AMReX_LO_BCTYPES.H>

#include <Radiation.H>
#include <RadTransportSolver.H>

#include <iostream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace amrex;

void Radiation::MultigroupDiffusion_implicit_update(int level, int iteration, int ncycle)
{
  BL_PROFILE("Radiation::MultigroupDiffusion_implicit_update");
  if (verbose) {
      amrex::Print() << "Radiation MultigroupDiffusion implicit update, level " << level << "..." << std::endl;
  }

  BL_ASSERT(Radiation::nGroups > 0);

  int fine_level =  parent->finestLevel();

  HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
  const BoxArray& grids = hydrosim->boxArray();
  const DistributionMapping& dmap = hydrosim->DistributionMap();
  Real delta_t = parent->dtLevel(level);

  int ngrow = 1;

  Real time = hydrosim->get_state_data(Radiation_Type).curTime();
  Real oldtime = hydrosim->get_state_data(Radiation_Type).prevTime();

  MultiFab& S_new = hydrosim->get_new_data(State_Type);
  FillPatchIterator fill_patch_new(*hydrosim, S_new, ngrow, time, State_Type, 0, S_new.nComp());
  MultiFab& state_new_border = fill_patch_new.get_mf();

  Array<MultiFab, AMREX_SPACEDIM> lambda;
  if (radiation::limiter > 0) {
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        lambda[idim].define(hydrosim->getEdgeBoxArray(idim), dmap, nGroups, 0);
    }

    if (inner_limiter_update_iters == -1) {
      MultiFab& rad_energy_lagged = hydrosim->get_old_data(Radiation_Type);
      rad_energy_lagged.setBndry(-1.0);
      rad_energy_lagged.FillBoundary(parent->Geom(level).periodicity());

      MultiFab& S_old = hydrosim->get_old_data(State_Type);
      FillPatchIterator fill_patch_old(*hydrosim, S_old, ngrow, oldtime, State_Type, 0, S_old.nComp());
      MultiFab& state_lagged = fill_patch_old.get_mf();

      MultiFab kappa_ross_lagged(grids,dmap,nGroups,1);
      MultigroupDiffusion_compute_rosseland(kappa_ross_lagged, state_lagged);

      for (int rad_group_idx=0; rad_group_idx<nGroups; ++rad_group_idx) {
        evaluateGradientRatio(level, lambda, kappa_ross_lagged, rad_group_idx, rad_energy_lagged, rad_group_idx, 1, rad_group_idx);
        computeFluxLimiter(level, lambda, rad_group_idx);
      }
    }
  }
  else {
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        lambda[idim].define(hydrosim->getEdgeBoxArray(idim), dmap, 1, 0);
      lambda[idim].setVal(1./3.);
    }
  }

  MultiFab& rad_energy_new = hydrosim->get_new_data(Radiation_Type);
  {
    MultiFab rhs(grids,dmap,1,0);
    for (int rad_group_idx=0; rad_group_idx<nGroups; rad_group_idx++) {
      rhs.setVal(0.0);
      apply_delayed_reflux(level, rhs, rad_group_idx);
      rhs_to_rad_energy(rhs, delta_t, level);
      MultiFab::Add(rad_energy_new, rhs, 0, rad_group_idx, 1, 0);
    }
  }
  MultiFab rad_energy_old(grids, dmap , rad_energy_new.nComp(), 0);
  MultiFab::Copy(rad_energy_old, rad_energy_new, 0, 0, rad_energy_new.nComp(), 0);
  MultiFab rad_energy_iter(grids,dmap,nGroups,1);
  MultiFab rad_energy_star(grids, dmap, nGroups, 1);
  rad_energy_iter.setBndry(-1.0);
  rad_energy_star.setBndry(-1.0);

  MultiFab rhoe_new(grids,dmap,1,0);
  MultiFab rhoe_old(grids,dmap,1,0);
  MultiFab gas_energy_star(grids,dmap,1,0);

  MultiFab rho(grids,dmap,1,1);
  MultiFab temp_next(grids,dmap,1,1);
  MultiFab temp_iterate(grids,dmap,1,0);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(state_new_border, true); mfi.isValid(); ++mfi) {
      const Box &gbx = mfi.growntilebox(1);
      const Box &bx  = mfi.tilebox();

      rho[mfi].copy<RunOn::Device>(state_new_border[mfi], gbx, URHO, gbx, 0, 1);

      rhoe_new[mfi].copy<RunOn::Device>(state_new_border[mfi], bx, UEINT, bx, 0, 1);
      rhoe_old[mfi].copy<RunOn::Device>(rhoe_new[mfi], bx);

      temp_next[mfi].copy<RunOn::Device>(state_new_border[mfi], gbx, UTEMP, gbx, 0, 1);
  }

  MultiFab kappa_p(grids,dmap,nGroups,1);
  MultiFab kappa_r(grids,dmap,nGroups,1);

  MultiFab jg(grids,dmap,nGroups,1);
  MultiFab dEmitT(grids,dmap,nGroups,1);
  MultiFab dOpacT(grids,dmap,nGroups,1);
  MultiFab thermal_frac(grids,dmap,1,0);
  MultiFab thermal_frac_z(grids,dmap,1,0);
  MultiFab rest_frac(grids,dmap,1,0);

  MultiFab& group_weight_mf = dEmitT;

  MultiFab dedT(grids,dmap,1,0);

  MultiFab couple_source(grids,dmap,1,0);

  MultiGroupBoundary multigroup_boundary(grids,dmap, nGroups, hydrosim->Geom());
  fetchMultigridBndry(multigroup_boundary, rad_energy_new, time, level);

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

  RadTransportSolver* const solver = hydrosim->rad_operator.get();

  Real relative_inner_resid, abs_residual_in, rad_energy_error;
  Real rel_gas_energy, abs_gas_energy;
  Real rel_temp, abs_temp;
  Real rel_resid, abs_resid;

  FluxRegister* flux_from_fine = (level < fine_level) ? flux_candidate_regs[level+1].get() : nullptr;
  FluxRegister* flux_to_coarse = (level > 0) ? flux_candidate_regs[level].get() : nullptr;

  Array<MultiFab, AMREX_SPACEDIM> Flux;
  for (int n = 0; n < AMREX_SPACEDIM; n++) {
      Flux[n].define(hydrosim->getEdgeBoxArray(n), dmap, 1, 0);
  }

  std::unique_ptr<MultiFab> flux_saved;
  MultiFab* flux_cell_center;
  int icomp_flux_index = -1;
  if (radiation::plot_moving_flux) {
      flux_cell_center = plot_fields[level].get();
      icomp_flux_index = icomp_comoving_flux;
  } else if (radiation::plot_lab_rad_energy || radiation::plot_lab_rad_flux) {
      flux_saved.reset(new MultiFab(grids, dmap, nGroups*AMREX_SPACEDIM, 0));
      flux_cell_center = flux_saved.get();
      icomp_flux_index = 0;
  }

  MultiFab& rad_energy_step = rad_energy_old;
  MultiFab& gas_energy_step = rhoe_old;

  Real inner_reltol = relInnerTol;
  Real pseudo_transient_tau = 0.0;

  int it = 0;
  bool energy_conserving_mode = false;
  bool outer_tol_met = false;
  bool converged = false;
  bool inner_solve_done = false;
  do {
    it++;

    if (it == 1) {
      eos_fill_opacity_emission(state_new_border, temp_next,
                             temp_iterate,
                             kappa_p, kappa_r, jg,
                             dEmitT, dOpacT, dedT,
                             level, it, 1);
    }

    MultiFab::Copy(gas_energy_star, rhoe_new, 0, 0, 1, 0);
    MultiFab::Copy(temp_iterate, temp_next, 0, 0, 1, 0);
    MultiFab::Copy(rad_energy_star, rad_energy_new, 0, 0, nGroups, 0);

    if (radiation::limiter>0 && inner_limiter_update_iters==0) {
      rad_energy_star.FillBoundary(parent->Geom(level).periodicity());

      for (int rad_group_idx=0; rad_group_idx<nGroups; ++rad_group_idx) {
        evaluateGradientRatio(level, lambda, kappa_r, rad_group_idx, rad_energy_star, rad_group_idx, 1, rad_group_idx);
        computeFluxLimiter(level, lambda, rad_group_idx);
      }
    }

    compute_thermal_fraction(thermal_frac, thermal_frac_z,
                 rest_frac, dEmitT,
                 dOpacT, dedT,
                 rad_energy_star, rho,
                 delta_t, pseudo_transient_tau);


    int innerIterCount = 0;
    inner_solve_done = false;
    Real relative_inner_resid_prev = 1.e200, absolute_inner_resid_prev = 1.e200;
    bool acceleration_allowed = true;
    do {
      innerIterCount++;

      MultiFab::Copy(rad_energy_iter, rad_energy_new, 0, 0, nGroups, 0);

      if (radiation::limiter>0 && inner_limiter_update_iters>0) {
        if (innerIterCount <= inner_limiter_update_iters) {
          rad_energy_iter.FillBoundary(parent->Geom(level).periodicity());

          for (int rad_group_idx=0; rad_group_idx<nGroups; ++rad_group_idx) {
            evaluateGradientRatio(level, lambda, kappa_r, rad_group_idx, rad_energy_iter, rad_group_idx, 1, rad_group_idx);
            computeFluxLimiter(level, lambda, rad_group_idx);
          }
        }
      }

      build_coupling_term(couple_source, kappa_p, rad_energy_iter, jg);

      for (int rad_group_idx=0; rad_group_idx<nGroups; ++rad_group_idx) {

        select_active_group(rad_group_idx);


        solver->levelBoundary(multigroup_boundary, rad_group_idx);

        solver->levelDiagCoeffs(level, kappa_p, delta_t, c, rad_group_idx, pseudo_transient_tau);

        int limiter_comp = (radiation::limiter==0) ? 0 : rad_group_idx;
        solver->levelFaceCoeffs(level, lambda, kappa_r, rad_group_idx, c, limiter_comp);

        if (has_transport_bc) {
          solver->levelBoundaryScale(level, lambda, rad_group_idx, lo_bc, hi_bc);
        }

        {

          MultiFab rhs(grids,dmap,1,0);

          solver->levelAssembleRhs(level, rhs, jg, group_weight_mf,
                           couple_source, thermal_frac,
                           rad_energy_step, gas_energy_step, rad_energy_star, gas_energy_star,
                           delta_t, rad_group_idx, it, pseudo_transient_tau);

          solver->levelSolveSystem(level, rad_energy_new, rad_group_idx, rhs, 0.01);
        }

        solver->levelFaceFlux(level, Flux, rad_energy_new, rad_group_idx);
        solver->levelCoarseFineFlux(level, flux_from_fine, flux_to_coarse, Flux, rad_group_idx);

        if (icomp_flux_index >= 0)
            solver->levelFaceToCellFlux(level, Flux, *flux_cell_center, icomp_flux_index+rad_group_idx);

      }

      check_rad_convergence(relative_inner_resid, abs_residual_in, rad_energy_error, rad_energy_new, rad_energy_iter,
                           kappa_p, thermal_frac_z, temp_next, delta_t);

      if (verbose >= 2) {
        int oldprec = std::cout.precision(3);
        amrex::Print() << "Outer = " << it << ", Inner = " << innerIterCount
                       << ", inner err =  " << std::setw(8) << relative_inner_resid << " (rel),  "
                       << std::setw(8) << abs_residual_in << " (abs)" << std::endl;
        std::cout.precision(oldprec);
      }

      if (relative_inner_resid < 1.e-15) {
        inner_solve_done = true;
      }
      else if (innerIterCount < minInnerIter) {
        inner_solve_done = false;
      }
      else if ( (relative_inner_resid <= inner_reltol || abs_residual_in <= absInnerTolerance)
                && rad_energy_error <= reltol ) {
        inner_solve_done = true;
      }

      if (!inner_solve_done) {
        Real stall_factor=1.+1.e-6;
        if (skipAccelIfNoGain &&
            relative_inner_resid>stall_factor*relative_inner_resid_prev &&
            abs_residual_in>stall_factor*absolute_inner_resid_prev) {
          acceleration_allowed = false;
          if (relative_inner_resid>10.*relative_inner_resid_prev &&
              abs_residual_in>10.*absolute_inner_resid_prev) {
            MultiFab::Copy(rad_energy_new, rad_energy_star, 0, 0, nGroups, 0);
          }
        }
        relative_inner_resid_prev = relative_inner_resid;
        absolute_inner_resid_prev = abs_residual_in;

        if (acceleration_allowed) {
          if (accelerate == 1) {
            local_convergence_boost(rad_energy_new, rad_energy_iter, kappa_p, thermal_frac,
                        group_weight_mf, delta_t, pseudo_transient_tau);
          }
          else if (accelerate == 2) {
            gray_correction(rad_energy_new, rad_energy_iter, kappa_p, kappa_r,
                       thermal_frac, rest_frac, group_weight_mf,
                       lambda, solver, multigroup_boundary, grids, level, time, delta_t, pseudo_transient_tau);
          }
        }
      }

    } while(!inner_solve_done && innerIterCount < maxInnerIter);

    if (verbose == 1) {
      int oldprec = std::cout.precision(3);
      amrex::Print() << "Outer = " << it << ", Inner = " << innerIterCount
                     << ", inner tol =  " << std::setw(8) << relative_inner_resid << "  "
                     << std::setw(8) << abs_residual_in << std::endl;
      std::cout.precision(oldprec);
    }

    if (matter_update_mode == 0) {
      energy_conserving_mode = true;
    }
    else if (matter_update_mode == 1) {
      if (outer_tol_met || it >= maxiter) {
        energy_conserving_mode = true;
      }
      else {
        energy_conserving_mode = false;
      }
    }
    else if (matter_update_mode == 2) {
      if (it%2 == 0) {
        energy_conserving_mode = true;
      }
      else {
        energy_conserving_mode = false;
      }
    }
    else if (matter_update_mode == 3) {
      if (outer_tol_met || it >= maxiter) {
        energy_conserving_mode = true;
      }
      else {
        if (it%2 == 0) {
          energy_conserving_mode = true;
        }
        else {
          energy_conserving_mode = false;
        }
      }
    }
    else {
      energy_conserving_mode = true;
    }

    advance_matter(rhoe_new, temp_next, rad_energy_new, rad_energy_iter,
                  gas_energy_star, gas_energy_step,
                  thermal_frac, thermal_frac_z, rest_frac,
                  couple_source,
                  kappa_p, jg, group_weight_mf,
                  state_new_border, level, delta_t, pseudo_transient_tau, it, energy_conserving_mode);

    eos_fill_opacity_emission(state_new_border, temp_next,
                           temp_iterate,
                           kappa_p, kappa_r, jg,
                           dEmitT, dOpacT, dedT,
                           level, it+1, 0);

    check_matter_convergence(rhoe_new, gas_energy_star, gas_energy_step, rad_energy_new,
                           temp_next, temp_iterate,
                           rho, kappa_p, jg, dedT,
                           rel_gas_energy, abs_gas_energy, rel_resid, abs_resid, rel_temp, abs_temp,
                           delta_t);

    Real relative_outer_resid, absolute_outer_resid;

    switch (convergence_check_mode) {
    case 1:
      relative_outer_resid = rel_gas_energy;
      absolute_outer_resid = abs_gas_energy;
      break;
    case 2:
      relative_outer_resid = rel_resid;
      absolute_outer_resid = abs_resid;
      break;
    case 3:
      relative_outer_resid = rel_temp;
      absolute_outer_resid = abs_temp;
      break;
    default:
      relative_outer_resid = rel_temp;
      if (energy_conserving_mode) {
        relative_outer_resid = (relative_outer_resid > rel_resid  ) ? relative_outer_resid : rel_resid;
      }
      absolute_outer_resid = abs_temp;
    }

    if (verbose >= 2) {
      int oldprec = std::cout.precision(4);
      amrex::Print() << "Update Errors for      rhoe,        FT,         T"
                     << std::endl;
      amrex::Print() << "       Relative = " << std::setw(9) << rel_gas_energy << ", "
                     << std::setw(9) << rel_resid << ", " << std::setw(9) << rel_temp
                     << std::endl;
      amrex::Print() << "       Absolute = " << std::setw(9) << abs_gas_energy << ", "
                     << std::setw(9) << abs_resid << ", " << std::setw(9) << abs_temp
                     << std::endl;
      std::cout.precision(oldprec);
    }

    if (it < min_outer_iter) {
      converged = false;
    }
    else if (relative_outer_resid <= reltol || absolute_outer_resid <= abstol) {
      converged = true;
    }
    else {
      converged = false;
    }

    if (relative_outer_resid <= reltol) {
      outer_tol_met = true;
    }

    if (!converged && it > bisect_after) {
      bisect_temperature(rhoe_new, temp_next,
                    gas_energy_star, temp_iterate,
                    state_new_border, grids, level);

      eos_fill_opacity_emission(state_new_border, temp_next,
                             temp_iterate,
                             kappa_p, kappa_r, jg,
                             dEmitT, dOpacT, dedT,
                             level, it+1, 0);
    }

  } while ( ((!converged || !inner_solve_done) && it<maxiter)
            || !energy_conserving_mode);

  if (verbose == 1) {
    int oldprec = std::cout.precision(4);
    amrex::Print() << "Update Errors for      rhoe,        FT,         T"
                   << std::endl;
    amrex::Print() << "       Relative = " << std::setw(9) << rel_gas_energy << ", "
                   << std::setw(9) << rel_resid << ", " << std::setw(9) << rel_temp
                   << std::endl;
    amrex::Print() << "       Absolute = " << std::setw(9) << abs_gas_energy << ", "
                   << std::setw(9) << abs_resid << ", " << std::setw(9) << abs_temp
                   << std::endl;
    std::cout.precision(oldprec);
  }

  if (!converged) {
      amrex::Abort("Implicit Update Failed to Converge");
  }


  flux_from_fine = (level < fine_level) ? flux_candidate_regs[level+1].get() : nullptr;
  if (flux_from_fine) {
      for (OrientationIter face; face; ++face) {
          Orientation ori = face();
          (*flux_accum_reg[level+1])[ori].linComb(1.0, -1.0,
                                             (*flux_from_fine)[ori], 0, 0, nGroups);
      }
  }

  flux_to_coarse = (level > 0) ? flux_candidate_regs[level].get() : nullptr;
  if (flux_to_coarse) {
      for (OrientationIter face; face; ++face) {
          Orientation ori = face();
          (*flux_accum_reg[level])[ori].linComb(1.0, 1.0 / ncycle,
                                        (*flux_to_coarse)[ori], 0, 0, nGroups);
    }
  }

  if (level == fine_level && iteration == 1) {


    for (int flev = level; flev > 0; flev--) {
      dt_previous[flev-1] = 0.0;
    }
  }

  Real energy_rel_chg = 0.0;
  Real deltaTempRatio = 0.0;

  update_state_energy(S_new, rhoe_new, temp_next, grids, energy_rel_chg, deltaTempRatio, level);

  delta_e_rat_level[level] = energy_rel_chg;
  delta_T_rat_level[level] = deltaTempRatio;

  if (verbose >= 2) {
      amrex::Print() << "Delta T      Ratio = " << deltaTempRatio << std::endl;
  }

  if (radiation::plot_flux_limiter) {
      store_limiter_plotvar(level, lambda);
  }

  if (radiation::plot_planck_opacity) {
      MultiFab::Copy(*plot_fields[level], kappa_p, 0, icomp_kappa_planck, nGroups, 0);
  }

  if (radiation::plot_rosseland_opacity) {
      MultiFab::Copy(*plot_fields[level], kappa_r, 0, icomp_kappa_rosseland, nGroups, 0);
  }

  if (radiation::plot_lab_rad_energy) {
      store_lab_energy_plotvar(level, S_new, rad_energy_new, *flux_cell_center, icomp_flux_index);
  }


  if (radiation::plot_lab_rad_flux) {
      store_flux_plotvar(level, S_new, lambda, rad_energy_new, *flux_cell_center, icomp_flux_index);
  }

  if (verbose) {
      amrex::Print() << "                                     done" << std::endl;
  }
}
