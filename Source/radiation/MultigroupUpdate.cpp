#include <Radiation.H>
#include <radiation_helpers.H>
#include <filter.H>
#include <planck_integrals.H>
#include <opacity.H>
#include <user_emissivity.H>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <AMReX_LO_BCTYPES.H>

#include <RadTransportSolver.H>

#include <iomanip>

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

void Radiation::check_rad_convergence(Real& relative, Real& absolute, Real& rad_energy_err,
                                     const MultiFab& rad_energy_new, const MultiFab& rad_energy_iter,
                                     const MultiFab& kappa_p,
                                     const MultiFab& thermal_frac_z,
                                     const MultiFab& temp_next,
                                     Real delta_t)
{
  BL_PROFILE("Radiation::check_rad_convergence (MultigroupDiffusion)");

  ReduceOps<ReduceOpMax, ReduceOpMax, ReduceOpMax> reduce_op;
  ReduceData<Real, Real, Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(rad_energy_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      const auto Ern = rad_energy_new[mfi].array();
      const auto Erl = rad_energy_iter[mfi].array();
      const auto kap = kappa_p[mfi].array();
      const auto thermal_frac_z_view = thermal_frac_z[mfi].array();
      const auto temp = temp_next[mfi].array();

      reduce_op.eval(bx, reduce_data,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real chg = 0.e0_rt;
          Real tot = 0.e0_rt;
          Real kde = 0.e0_rt;

          for (int g = 0; g < NGROUPS; ++g) {
              Real der = Ern(i,j,k,g) - Erl(i,j,k,g);
              chg = chg + std::abs(der);
              tot = tot + std::abs(Ern(i,j,k,g));
              kde = kde + kap(i,j,k,g) * der;
          }

          Real abs_change = chg;
          Real frac_chg = chg / (tot + 1.e-50_rt);

          Real temp_err = thermal_frac_z_view(i,j,k) * kde;
          Real temp_err_rel = std::abs(temp_err / (temp(i,j,k) + 1.e-50_rt));

          return {frac_chg, abs_change, temp_err_rel};
      });
  }

  ReduceTuple hv = reduce_data.value();

  relative = amrex::get<0>(hv);
  absolute = amrex::get<1>(hv);
  rad_energy_err   = amrex::get<2>(hv);

  Real data[3] = {relative, absolute, rad_energy_err};

  ParallelDescriptor::ReduceRealMax(data, 3);

  relative = data[0];
  absolute = data[1];
  rad_energy_err   = data[2];
}

void Radiation::check_matter_convergence(const MultiFab& rhoe_new, const MultiFab& gas_energy_star,
                                       const MultiFab& gas_energy_step, const MultiFab& rad_energy_new,
                                       const MultiFab& temp_next, const MultiFab& temp_iterate,
                                       const MultiFab& rho,
                                       const MultiFab& kappa_p, const MultiFab& jg,
                                       const MultiFab& dedT,
                                       Real& rel_gas_energy, Real& abs_gas_energy,
                                       Real& rel_resid,   Real& abs_resid,
                                       Real& rel_temp,    Real& abs_temp,
                                       Real delta_t)
{
  ReduceOps<ReduceOpMax, ReduceOpMax, ReduceOpMax, ReduceOpMax, ReduceOpMax, ReduceOpMax> reduce_op;
  ReduceData<Real, Real, Real, Real, Real, Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(rhoe_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      auto ren = rhoe_new[mfi].array();
      auto res = gas_energy_star[mfi].array();
      auto re2 = gas_energy_step[mfi].array();
      auto Ern = rad_energy_new[mfi].array();
      auto Tmn = temp_next[mfi].array();
      auto Tms = temp_iterate[mfi].array();
      auto mass_density_arr = rho[mfi].array();
      auto kap = kappa_p[mfi].array();
      auto emission_arr = jg[mfi].array();
      auto deT = dedT[mfi].array();

      int ng = Radiation::nGroups;
      Real cdt = C::c_light * delta_t;

      reduce_op.eval(bx, reduce_data,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real gas_energy_abs_chg = std::abs(ren(i,j,k) - res(i,j,k));
          Real gas_energy_rel_chg = std::abs(gas_energy_abs_chg / (ren(i,j,k) + 1.e-50_rt));

          Real temp_abs_chg = std::abs(Tmn(i,j,k) - Tms(i,j,k));
          Real temp_rel_chg = std::abs(temp_abs_chg / (Tmn(i,j,k) + 1.e-50_rt));

          Real FT = ren(i,j,k) - re2(i,j,k);
          for (int g = 0; g < ng; ++g) {
              FT -= cdt * (kap(i,j,k,g) * Ern(i,j,k,g) - emission_arr(i,j,k,g));
          }
          FT = std::abs(FT);

          Real dTe = Tmn(i,j,k);
          Real resid_scale = mass_density_arr(i,j,k) * std::abs(deT(i,j,k) * dTe);

          Real resid_abs_chg = FT;
          Real resid_rel_chg = FT / (resid_scale + 1.e-50_rt);

          return {gas_energy_rel_chg, gas_energy_abs_chg, resid_rel_chg, resid_abs_chg, temp_rel_chg, temp_abs_chg};
      });
  }

  ReduceTuple hv = reduce_data.value();
  rel_gas_energy = amrex::get<0>(hv);
  abs_gas_energy = amrex::get<1>(hv);
  rel_resid   = amrex::get<2>(hv);
  abs_resid   = amrex::get<3>(hv);
  rel_temp    = amrex::get<4>(hv);
  abs_temp    = amrex::get<5>(hv);

  int ndata = 6;
  Real data[6] = {rel_gas_energy, abs_gas_energy, rel_resid, abs_resid, rel_temp, abs_temp};

  ParallelDescriptor::ReduceRealMax(data, ndata);

  rel_gas_energy = data[0];
  abs_gas_energy = data[1];
  rel_resid   = data[2];
  abs_resid   = data[3];
  rel_temp    = data[4];
  abs_temp    = data[5];
}

void Radiation::build_coupling_term(MultiFab& couple_source,
                                 const MultiFab& kpp, const MultiFab& Eg,
                                 const MultiFab& jg)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(kpp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto couple_source_arr = couple_source[mfi].array();
        auto kappa_group_arr = kpp[mfi].array();
        auto group_energy_arr = Eg[mfi].array();
        auto emission_arr = jg[mfi].array();

        int ng = Radiation::nGroups;

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            couple_source_arr(i,j,k) = 0.0_rt;

            for (int g = 0; g < ng; ++g) {
                couple_source_arr(i,j,k) = couple_source_arr(i,j,k) + (kappa_group_arr(i,j,k,g) * group_energy_arr(i,j,k,g) - emission_arr(i,j,k,g));
            }
        });
    }
}

void Radiation::compute_thermal_fraction(MultiFab& thermal_frac, MultiFab& thermal_frac_z,
                             MultiFab& rest_frac, MultiFab& dEmitT,
                             const MultiFab& dOpacT, const MultiFab& dedT,
                             const MultiFab& rad_energy_star, const MultiFab& rho,
                             Real delta_t, Real pseudo_transient_tau)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(rho, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto thermal_frac_arr = thermal_frac[mfi].array();
        auto thermal_frac_z_arr = thermal_frac_z[mfi].array();
        auto rest_frac_arr = rest_frac[mfi].array();
        auto emission_dtemp_arr = dEmitT[mfi].array();
        auto kappa_dtemp_arr = dOpacT[mfi].array();
        auto energy_dtemp_arr = dedT[mfi].array();
        auto rad_energy_star_arr = rad_energy_star[mfi].array();
        auto mass_density_arr = rho[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Real sigma = 1.e0_rt + pseudo_transient_tau;
            Real cdt = C::c_light * delta_t;

            Real dz_dtemp[NGROUPS];
            Real sum_dz_dtemp = 0.0_rt;
            for (int g = 0; g < NGROUPS; ++g) {
                dz_dtemp[g] = emission_dtemp_arr(i,j,k,g) - kappa_dtemp_arr(i,j,k,g) * rad_energy_star_arr(i,j,k,g);
                sum_dz_dtemp += dz_dtemp[g];
            }

            if (sum_dz_dtemp == 0.0_rt) {
                sum_dz_dtemp = 1.e-50_rt;
            }

            Real foo = cdt * sum_dz_dtemp;
            Real bar = sigma * mass_density_arr(i,j,k) * energy_dtemp_arr(i,j,k);
            thermal_frac_arr(i,j,k) = foo / (foo + bar);
            thermal_frac_z_arr(i,j,k) = thermal_frac_arr(i,j,k) / sum_dz_dtemp;
            rest_frac_arr(i,j,k) = bar / (foo + bar);
            for (int g = 0; g < NGROUPS; ++g) {
                emission_dtemp_arr(i,j,k,g) = dz_dtemp[g] / sum_dz_dtemp;
            }
        });
    }
}

void Radiation::eos_fill_opacity_emission(const MultiFab& S_new,
                                       const MultiFab& temp_next,
                                       const MultiFab& temp_iterate,
                                       MultiFab& kappa_p, MultiFab& kappa_r, MultiFab& jg,
                                       MultiFab& dEmitT, MultiFab& dOpacT, MultiFab& dedT,
                                       int level, int it, int ngrow)
{
  int star_valid_margin = 1 - ngrow;

  int lagged_opac;
  if (it == 1) {
    lagged_opac = 0;
  }
  else if (it <= opacity_lag_iter) {
    lagged_opac = 0;
  }
  else {
    lagged_opac = 1;
  }

  const Geometry& geom = parent->Geom(level);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& box = mfi.tilebox();

      auto energy_dtemp_arr = dedT[mfi].array();
      auto temp_field_arr = temp_next[mfi].array();
      auto new_state_view = S_new[mfi].array();

      amrex::ParallelFor(box,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          Real rhoInv = 1.e0_rt / new_state_view(i,j,k,URHO);

          eos_re_t eos_state;
          eos_state.rho = new_state_view(i,j,k,URHO);
          eos_state.T   = temp_field_arr(i,j,k);
          for (int n = 0; n < NumSpec; ++n) {
              eos_state.xn[n] = new_state_view(i,j,k,UFS+n) * rhoInv;
          }
#if NAUX_NET > 0
          for (int n = 0; n < NumAux; ++n) {
              eos_state.aux[n] = new_state_view(i,j,k,UFX+n) * rhoInv;
          }
#endif

          eos(eos_input_rt, eos_state);

          energy_dtemp_arr(i,j,k) = eos_state.cv;
      });
  }

  if (dedT_safety > 1.0) {
    dedT.mult(dedT_safety);
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.growntilebox(ngrow);

      auto new_state_view = S_new[mfi].array();
      auto temp_next_arr = temp_next[mfi].array();
      auto temp_iterate_arr = temp_iterate[mfi].array();
      auto kappa_planck_arr = kappa_p[mfi].array();
      auto kappa_ross_arr = kappa_r[mfi].array();
      auto kappa_dtemp_arr = dOpacT[mfi].array();
      auto emission_arr = jg[mfi].array();
      auto emission_dtemp_arr = dEmitT[mfi].array();

      bool local_use_kappa_dtemp = use_kappa_deriv;

      GpuArray<Real, NGROUPS> group_freq_loc;
      for (int g = 0; g < NGROUPS; ++g) {
          group_freq_loc[g] = freq_center[g];
      }
      GpuArray<Real, NGROUPS+1> freq_edges_loc;
      for (int g = 0; g < NGROUPS+1; ++g) {
          freq_edges_loc[g] = xnu[g];
      }

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          const Real fac = 0.5e0_rt;
          const Real min_temp_step_frac = 1.e-8_rt;

          if (lagged_opac) {
              kappa_dtemp_arr(i,j,k) = 0.0_rt;
              return;
          }

          Real rho = new_state_view(i,j,k,URHO);
          Real temp = temp_next_arr(i,j,k);

          Real Ye;
          if (NumAux > 0) {
              Real Ye = new_state_view(i,j,k,UFX);
          } else {
              Ye = 0.e0_rt;
          }

          Real dT;
          if (star_valid_margin > 0) {
              dT = fac * std::abs(temp_iterate_arr(i,j,k) - temp_next_arr(i,j,k));
              dT = amrex::max(dT, min_temp_step_frac * temp_next_arr(i,j,k));
          } else {
              dT = temp_next_arr(i,j,k) * 1.e-3_rt + 1.e-50_rt;
          }

          for (int g = 0; g < NGROUPS; ++g) {
              Real nu = group_freq_loc[g];

              bool need_planck_opac = true;
              bool need_ross_opac = true;

              Real kp, kr;

              opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

              kappa_planck_arr(i,j,k,g) = kp;
              kappa_ross_arr(i,j,k,g) = kr;

              if (local_use_kappa_dtemp == 0) {

                  kappa_dtemp_arr(i,j,k,g) = 0.e0_rt;

              } else {

                  bool need_planck_opac = true;
                  bool need_ross_opac = false;

                  Real kp1, kr1, kp2, kr2;

                  opacity(kp1, kr1, rho, temp-dT, Ye, nu, need_planck_opac, need_ross_opac);
                  opacity(kp2, kr2, rho, temp+dT, Ye, nu, need_planck_opac, need_ross_opac);

                  kappa_dtemp_arr(i,j,k,g) = (kp2 - kp1) / (2.e0_rt * dT);
              }
          }
      });

      const Box& reg = mfi.tilebox();

      amrex::ParallelFor(reg,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {

          Real temp_floored = amrex::max(temp_next_arr(i,j,k), 1.e-50_rt);

          Real B1, dplanck_curr;
          PlanckAntiderivSlope(temp_floored, 0.0_rt, B1, dplanck_curr);

          for (int g = 0; g < NGROUPS; ++g) {

              Real freq_upper = freq_edges_loc[g+1];

              if (g == NGROUPS - 1) {
                  freq_upper = amrex::max(freq_upper, 1.e25_rt);
              }

              Real B0 = B1;
              Real dplanck_prev = dplanck_curr;
              PlanckAntiderivSlope(temp_floored, freq_upper, B1, dplanck_curr);
              Real Bg = B1 - B0;
              Real dplanck_group_diff = dplanck_curr - dplanck_prev;

              emission_arr(i,j,k,g) = Bg * kappa_planck_arr(i,j,k,g);
              emission_dtemp_arr(i,j,k,g) = kappa_dtemp_arr(i,j,k,g) * Bg + dplanck_group_diff * kappa_planck_arr(i,j,k,g);

              user_emissivity(i, j, k, g,
                                 group_freq_loc, freq_edges_loc,
                                 temp_next_arr(i,j,k), kappa_planck_arr(i,j,k,g),
                                 kappa_dtemp_arr(i,j,k,g), emission_arr(i,j,k,g), emission_dtemp_arr(i,j,k,g));
          }
      });
  }

  if (ngrow == 0 && !lagged_opac) {
      kappa_r.FillBoundary(geom.periodicity());
  }
}

void Radiation::gray_correction(MultiFab& rad_energy_new, MultiFab& rad_energy_iter,
                           MultiFab& kappa_p, MultiFab& kappa_r,
                           MultiFab& thermal_frac, MultiFab& rest_frac,
                           MultiFab& group_weight_mf,
                           Array<MultiFab, AMREX_SPACEDIM>& lambda,
                           RadTransportSolver* solver, MultiGroupBoundary& multigroup_boundary,
                           const BoxArray& grids, int level, Real time,
                           Real delta_t, Real pseudo_transient_tau)
{
  const Geometry& geom = parent->Geom(level);
  auto dx = parent->Geom(level).CellSizeArray();
  const HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
  const DistributionMapping& dmap = hydrosim->DistributionMap();

  if (nGroups > 1) {
    solver->setSolverScale(1.0);
  }
  else {
    solver->setSolverScale(0.0);
  }
  multigroup_boundary.enterCorrection();

  MultiFab rad_energy_zeros(grids, dmap, 1, 0);
  rad_energy_zeros.setVal(0.0);
  fetchMultigridBndryZeroRad(multigroup_boundary, rad_energy_zeros, level);

  MultiFab spec(grids, dmap, nGroups, 1);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(spec, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.tilebox();

    auto kappa_planck_arr = kappa_p[mfi].array();
    auto group_weight_arr = group_weight_mf[mfi].array();
    auto spectral_weight_arr = spec[mfi].array();

    Real inv_light_dt = 1.e0_rt / (C::c_light * delta_t);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        Real epsilon[NGROUPS];
        Real sum_epsilon = 0.0;
        for (int g = 0; g < NGROUPS; ++g) {
            Real kappa_shift = kappa_planck_arr(i,j,k,g) + (1.e0_rt + pseudo_transient_tau) * inv_light_dt;
            epsilon[g] = group_weight_arr(i,j,k,g) / kappa_shift;
            sum_epsilon += epsilon[g];
        }

        if (sum_epsilon == 0.e0_rt) {
            for (int g = 0; g < NGROUPS; ++g) {
                spectral_weight_arr(i,j,k,g) = 0.e0_rt;
            }
        } else {
            for (int g = 0; g < NGROUPS; ++g) {
                spectral_weight_arr(i,j,k,g) = epsilon[g] / sum_epsilon;
            }
        }
    });
  }

  for (int indx = 0; indx < nGroups; indx++) {
    extendBorderValues(spec, indx);
  }
  spec.FillBoundary(parent->Geom(level).periodicity());

  solver->levelBoundary(multigroup_boundary,0);

  MultiFab acoefs(grids, dmap, 1, 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(acoefs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      auto rest_frac_arr = rest_frac[mfi].array();
      auto spectral_weight_arr = spec[mfi].array();
      auto kappa_planck_arr = kappa_p[mfi].array();
      auto diag_coef_arr = acoefs[mfi].array();

      const Real dt1 = (1.e0_rt + pseudo_transient_tau) / delta_t;

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          Real kappa_avg = 0.0;
          for (int g = 0; g < NGROUPS; ++g) {
              kappa_avg += spectral_weight_arr(i,j,k,g) * kappa_planck_arr(i,j,k,g);
          }

          Real H1 = rest_frac_arr(i,j,k);

          diag_coef_arr(i,j,k) = H1 * kappa_avg * C::c_light + dt1;
      });
  }

  solver->applyCellMetrics(level, acoefs);
  solver->setLevelDiagCoeffs(level, acoefs);

  const DistributionMapping& dm = hydrosim->DistributionMap();

  Array<MultiFab, AMREX_SPACEDIM> bcoefs, cell_coefs, edge_group_coef;
  for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
    const BoxArray& edge_boxes = hydrosim->getEdgeBoxArray(idim);

    bcoefs[idim].define(edge_boxes, dm, 1, 0);
    bcoefs[idim].setVal(0.0);

    edge_group_coef [idim].define(edge_boxes, dm, 1, 0);

    if (nGroups > 1) {
        cell_coefs[idim].define(edge_boxes, dm, 2, 0);
        cell_coefs[idim].setVal(0.0);
    }
  }

  for (int rad_group_idx = 0; rad_group_idx < nGroups; rad_group_idx++) {
    for (int idim=0; idim<AMREX_SPACEDIM; idim++) {
      solver->computeFaceCoeffs(edge_group_coef[idim], idim, kappa_r, rad_group_idx,
                            lambda[idim], rad_group_idx, c, geom);

#ifdef _OPENMP
#pragma omp parallel
#endif
      for (MFIter mfi(spec, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
          const Box&  bx  = mfi.nodaltilebox(idim);

          auto dir_coef_arr = bcoefs[idim][mfi].array();
          auto edge_group_coef_arr = edge_group_coef[idim][mfi].array();
          auto spectral_weight_arr = spec[mfi].array(rad_group_idx);
          auto grad_coef_arr = cell_coefs[idim][mfi].array();

          amrex::ParallelFor(bx,
          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
          {
              if (idim == 0) {
                  dir_coef_arr(i,j,k) += 0.5e0_rt * (spectral_weight_arr(i-1,j,k) + spectral_weight_arr(i,j,k)) * edge_group_coef_arr(i,j,k);
              }
              else if (idim == 1) {
                  dir_coef_arr(i,j,k) += 0.5e0_rt * (spectral_weight_arr(i,j-1,k) + spectral_weight_arr(i,j,k)) * edge_group_coef_arr(i,j,k);
              }
              else {
                  dir_coef_arr(i,j,k) += 0.5e0_rt * (spectral_weight_arr(i,j,k-1) + spectral_weight_arr(i,j,k)) * edge_group_coef_arr(i,j,k);
              }
          });

          if (nGroups > 1) {
              amrex::ParallelFor(bx,
              [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
              {
                  int ioff, joff, koff;
                  Real h1;

                  if (idim == 0) {
                      ioff = 1;
                      joff = 0;
                      koff = 0;
                      h1 = 1.e0_rt / dx[0];
                  }
                  else if (idim == 1) {
                      ioff = 0;
                      joff = 1;
                      koff = 0;
                      h1 = 1.e0_rt / dx[1];
                  }
                  else {
                      ioff = 0;
                      joff = 0;
                      koff = 1;
                      h1 = 1.e0_rt / dx[2];
                  }

                  Real spec_grad = (spectral_weight_arr(i,j,k) - spectral_weight_arr(i-ioff,j-joff,k-koff)) * h1;
                  Real foo = -0.5e0_rt * edge_group_coef_arr(i,j,k) * spec_grad;
                  grad_coef_arr(i,j,k,0) += foo;
                  grad_coef_arr(i,j,k,1) += foo;
              });
          }
      }
    }
  }

  for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
    solver->setLevelFaceCoeffs(level, bcoefs[idim], idim);

    if (nGroups > 1) {
      solver->setLevelCellCoeffs(level, cell_coefs[idim], idim);
    }
  }

  MultiFab rhs(grids,dmap,1,0);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      auto Ern = rad_energy_new[mfi].array();
      auto Erl = rad_energy_iter[mfi].array();
      auto kap = kappa_p[mfi].array();
      auto thermal_frac_arr = thermal_frac[mfi].array();
      auto rhs_arr = rhs[mfi].array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          Real rad_source_term = 0.0;
          for (int g = 0; g < NGROUPS; ++g) {
              rad_source_term += kap(i,j,k,g) * (Ern(i,j,k,g) - Erl(i,j,k,g));
          }

          Real H = thermal_frac_arr(i,j,k);

          rhs_arr(i,j,k) = C::c_light * H * rad_source_term;
      });
  }

  solver->applyCellMetrics(level, rhs);

  MultiFab accel(grids,dmap,1,0);
  accel.setVal(0.0);
  solver->levelSolveSystem(level, accel, 0, rhs, 0.01);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(spec, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& reg  = mfi.tilebox();

      auto rad_energy_new_arr = rad_energy_new[mfi].array();
      auto spectral_weight_arr = spec[mfi].array();
      auto correction_arr = accel[mfi].array();

      amrex::ParallelFor(reg,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          for (int n = 0; n < NGROUPS; ++n) {
              rad_energy_new_arr(i,j,k,n) = rad_energy_new_arr(i,j,k,n) + spectral_weight_arr(i,j,k,n) * correction_arr(i,j,k);
          }
      });
  }

  multigroup_boundary.leaveCorrection();
  fetchMultigridBndry(multigroup_boundary, rad_energy_new, time, level);

  solver->restoreSolverScale();
}

void Radiation::local_convergence_boost(MultiFab& rad_energy_new, const MultiFab& rad_energy_iter,
                            const MultiFab& kappa_p,
                            const MultiFab& thermal_frac,
                            const MultiFab& group_weight_mf,
                            Real delta_t, Real pseudo_transient_tau)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(rad_energy_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto Ern = rad_energy_new[mfi].array();
        auto Erl = rad_energy_iter[mfi].array();
        auto kap = kappa_p[mfi].array();
        auto thermal_frac_arr = thermal_frac[mfi].array();
        auto group_weight_arr = group_weight_mf[mfi].array();

        Real inv_light_dt = 1.0_rt / (C::c_light * delta_t);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Real Hg[NGROUPS], kappa_shift[NGROUPS];
            Real rad_source_term = 0.0_rt;
            Real p = 1.0_rt;

            for (int g = 0; g < NGROUPS; ++g) {
                Hg[g] = group_weight_arr(i,j,k,g) * thermal_frac_arr(i,j,k);
                kappa_shift[g] = kap(i,j,k,g) + (1.e0_rt + pseudo_transient_tau) * inv_light_dt;
                Real kk = kap(i,j,k,g) / kappa_shift[g];

                p -= Hg[g] * kk;
                rad_source_term += kap(i,j,k,g) * (Ern(i,j,k,g) - Erl(i,j,k,g));
            }

            for (int g = 0; g < NGROUPS; ++g) {
                Ern(i,j,k,g) = Ern(i,j,k,g) + (Hg[g] * rad_source_term) / (kappa_shift[g] * p + 1.e-50_rt);
            }
        });
    }
}

void Radiation::update_state_energy(MultiFab& state, const MultiFab& rhoe,
                                    const MultiFab& temp,
                                    const BoxArray& grids,
                                    Real& energy_rel_chg, Real& dT, int level)
{
  BL_PROFILE("Radiation::update_state_energy (MultigroupDiffusion)");

  const DistributionMapping& dmap = state.DistributionMap();

  MultiFab msk(grids,dmap,1,0);

  {
      BoxArray baf;

      if (level < parent->finestLevel()) {
          baf = parent->boxArray(level+1);
          baf.coarsen(parent->refRatio(level));
      }

#ifdef _OPENMP
#pragma omp parallel
#endif
      for (MFIter mfi(msk); mfi.isValid(); ++mfi)
      {
          msk[mfi].setVal<RunOn::Device>(1.0);

          if (level < parent->finestLevel()) {
              std::vector< std::pair<int,Box> > isects = baf.intersections(mfi.validbox());
              for (int ii = 0; ii < isects.size(); ii++) {
                  msk[mfi].setVal<RunOn::Device>(0.0, isects[ii].second, 0);
              }
          }
      }
  }

  ReduceOps<ReduceOpMax, ReduceOpMax> reduce_op;
  ReduceData<Real, Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(state,true); mfi.isValid(); ++mfi)
  {
      const Box& reg  = mfi.tilebox();

      auto state_arr = state[mfi].array();
      auto temp_field_arr = temp[mfi].array();
      auto gas_energy_arr = rhoe[mfi].array();
      auto valid_mask_arr = msk[mfi].array();

      reduce_op.eval(reg, reduce_data,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real ei = state_arr(i,j,k,UEINT);
          Real energy_rel_loc = std::abs((gas_energy_arr(i,j,k) - ei) *
                                    valid_mask_arr(i,j,k) / (ei + 1.e-50_rt));
          Real ek = state_arr(i,j,k,UEDEN) - state_arr(i,j,k,UEINT);
          state_arr(i,j,k,UEINT) = gas_energy_arr(i,j,k);
          state_arr(i,j,k,UEDEN) = gas_energy_arr(i,j,k) + ek;

          Real temp_old = state_arr(i,j,k,UTEMP);
          Real temp_rel_loc = std::abs((temp_field_arr(i,j,k) - temp_old) *
                                    valid_mask_arr(i,j,k) / (temp_old + 1.e-50_rt));
          state_arr(i,j,k,UTEMP) = temp_field_arr(i,j,k);

          return {energy_rel_loc, temp_rel_loc};
      });

  }

  ReduceTuple hv = reduce_data.value();
  energy_rel_chg = amrex::get<0>(hv);
  dT = amrex::get<1>(hv);

  ParallelDescriptor::ReduceRealMax(energy_rel_chg);
  ParallelDescriptor::ReduceRealMax(dT);
}

void Radiation::advance_matter(MultiFab& rhoe_new, MultiFab& temp_next,
                              const MultiFab& rad_energy_new, const MultiFab& rad_energy_iter,
                              const MultiFab& gas_energy_star,
                              const MultiFab& gas_energy_step,
                              const MultiFab& thermal_frac, const MultiFab& thermal_frac_z,
                              const MultiFab& rest_frac,
                              const MultiFab& couple_source,
                              const MultiFab& kappa_p, const MultiFab& jg,
                              const MultiFab& group_weight_mf,
                              const MultiFab& S_new,
                              int level, Real delta_t, Real pseudo_transient_tau,
                              int it, bool energy_conserving_mode)
{
    const Real cdt = C::c_light * delta_t;
    const Real inv_light_dt = 1.e0_rt / (C::c_light * delta_t);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(rhoe_new,true); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto gas_energy_new_view = rhoe_new[mfi].array();
        auto new_state_view = S_new[mfi].array();
        auto temp_next_view = temp_next[mfi].array();
        auto rad_energy_new_view = rad_energy_new[mfi].array();
        auto rad_energy_prev = rad_energy_iter[mfi].array();
        auto gas_energy_star_view = gas_energy_star[mfi].array();
        auto gas_energy_step_view = gas_energy_step[mfi].array();
        auto rest_frac_arr = rest_frac[mfi].array();
        auto cpT = couple_source[mfi].array();
        auto thermal_frac_z_view = thermal_frac_z[mfi].array();
        auto kpp  = kappa_p[mfi].array();
        auto emission_arr = jg[mfi].array();

        if (energy_conserving_mode) {

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                Real H1 = rest_frac_arr(i,j,k);

                Real kappa_weighted_rad_chg = 0.0;
                for (int g = 0; g < NGROUPS; ++g) {
                    kappa_weighted_rad_chg += kpp(i,j,k,g) * (rad_energy_new_view(i,j,k,g) - rad_energy_prev(i,j,k,g));
                }

                Real chg = cdt * kappa_weighted_rad_chg + H1 * ((gas_energy_step_view(i,j,k) - gas_energy_star_view(i,j,k)) + cdt * cpT(i,j,k));

                gas_energy_new_view(i,j,k) = gas_energy_star_view(i,j,k) + chg;

                gas_energy_new_view(i,j,k) = (gas_energy_new_view(i,j,k) + pseudo_transient_tau * gas_energy_star_view(i,j,k)) / (1.e0_rt + pseudo_transient_tau);
            });

            temp_next[mfi].copy<RunOn::Device>(rhoe_new[mfi],bx);
            Gpu::synchronize();

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {

                if (temp_next_view(i,j,k) <= 0.e0_rt)
                {
                    temp_next_view(i,j,k) = small_temp;
                }
                else
                {
                    Real rhoInv = 1.e0_rt / new_state_view(i,j,k,URHO);

                    eos_re_t eos_state;
                    eos_state.rho = new_state_view(i,j,k,URHO);
                    eos_state.T   = new_state_view(i,j,k,UTEMP);
                    eos_state.e   = temp_next_view(i,j,k) * rhoInv;
                    for (int n = 0; n < NumSpec; ++n) {
                        eos_state.xn[n] = new_state_view(i,j,k,UFS+n) * rhoInv;
                    }
#if NAUX_NET > 0
                    for (int n = 0; n < NumAux; ++n) {
                        eos_state.aux[n] = new_state_view(i,j,k,UFX+n) * rhoInv;
                    }
#endif

                    eos(eos_input_re, eos_state);

                    temp_next_view(i,j,k) = eos_state.T;
                }
            });
            Gpu::synchronize();
        }
        else {

            const Real fac = 0.01_rt;

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                Real cpT = 0.e0_rt;

                for (int g = 0; g < NGROUPS; ++g) {
                    cpT = cpT + kpp(i,j,k,g) * rad_energy_new_view(i,j,k,g) - emission_arr(i,j,k,g);
                }

                Real scratch_energy = cpT - (gas_energy_star_view(i,j,k) - gas_energy_step_view(i,j,k)) * inv_light_dt;

                Real temp_update = thermal_frac_z_view(i,j,k) * scratch_energy;

                if (std::abs(temp_update / (temp_next_view(i,j,k) + 1.e-50_rt)) > fac) {
                    temp_update = std::copysign(fac * temp_next_view(i,j,k), temp_update);
                }

                temp_next_view(i,j,k) = temp_next_view(i,j,k) + temp_update;

                Real rhoInv = 1.e0_rt / new_state_view(i,j,k,URHO);

                eos_re_t eos_state;
                eos_state.rho = new_state_view(i,j,k,URHO);
                eos_state.T   = temp_next_view(i,j,k);
                for (int n = 0; n < NumSpec; ++n) {
                    eos_state.xn[n] = new_state_view(i,j,k,UFS+n) * rhoInv;
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; ++n) {
                    eos_state.aux[n] = new_state_view(i,j,k,UFX+n) * rhoInv;
                }
#endif

                eos(eos_input_rt, eos_state);

                gas_energy_new_view(i,j,k) = eos_state.rho * eos_state.e;
            });
        }
    }
}

void Radiation::bisect_temperature(MultiFab& rhoe_new, MultiFab& temp_next,
                              const MultiFab& gas_energy_star, const MultiFab& temp_iterate,
                              const MultiFab& S_new, const BoxArray& grids, int level)
{
  temp_next.plus(temp_iterate, 0, 1, 0);
  temp_next.mult(0.5, 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(rhoe_new,true); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      auto rhoe = rhoe_new[mfi].array();
      auto temp = temp_next[mfi].array();
      auto state = S_new[mfi].array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          Real rhoInv = 1.e0_rt / state(i,j,k,URHO);

          eos_re_t eos_state;
          eos_state.rho = state(i,j,k,URHO);
          eos_state.T   =  temp(i,j,k);
          for (int n = 0; n < NumSpec; ++n) {
              eos_state.xn[n] = state(i,j,k,UFS+n) * rhoInv;
          }
#if NAUX_NET > 0
          for (int n = 0; n < NumAux; ++n) {
              eos_state.aux[n] = state(i,j,k,UFX+n) * rhoInv;
          }
#endif

          eos(eos_input_rt, eos_state);

          rhoe(i,j,k) = eos_state.rho * eos_state.e;
      });
  }
}

void Radiation::rhs_to_rad_energy(MultiFab& rhs, Real dt, int level)
{
    auto geomdata = parent->Geom(level).data();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter ri(rhs, TilingIfNotGPU()); ri.isValid(); ++ri)
    {
        const Box& bx = ri.tilebox();

        auto rhs_arr = rhs[ri].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Real r, s;
            cell_center_geometry_factors(i, j, k, geomdata, r, s);

            rhs_arr(i,j,k) *= dt / r;
        });
    }
}
