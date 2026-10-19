
#include <HydroSim.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef GRAVITY
#include <Gravity.H>
#endif

#include <cmath>
#include <climits>
#include <format>

#include <user_initialize_state_data.H>

using std::string;
using namespace amrex;

Real
HydroSim::advance (Real time,
                 Real dt,
                 int  amr_pass,
                 int  amr_cycles)

  // the main driver for a single level.  This will do either the SDC
  // algorithm or the Strang-split reactions algorithm.
  //
  // arguments:
  //    time          : the current simulation time
  //    dt            : the timestep to advance (e.g., go from time to
  //                    time + dt)
  //    amr_pass : where we are in the current AMR subcycle.  Each
  //                    level will take a number of steps to reach the
  //                    final time of the coarser level below it.  This
  //                    counter starts at 1
  //    amr_cycles    : the number of subcycles at this level

{
    if (parent->subcyclingMode() == "None" && level > 0) {
        amrex::Print() << "\n  Advance at this level has already been completed.\n\n";
        return dt;
    }

    BL_PROFILE("HydroSim::advance()");

    // Save the wall time when we started the step.

    wall_clock_start = ParallelDescriptor::second();

    MultiFab::RegionTag level_region_tag(std::format("AmrLevel_Level_{}", level));

    Real dt_new = dt;

    int top_level_to_advance = level;

    if (parent->subcyclingMode() == "None" && level == 0) {
        top_level_to_advance = parent->finestLevel();
    }

    for (int lev = level; lev <= top_level_to_advance; ++lev) {
        getLevel(lev).begin_advance(time, dt, amr_pass);
    }

    // Do the advance.

    if (time_scheme == CornerTrace || time_scheme == SimplifiedIterCorrection) {

        dt_new = std::min(dt_new, advance_subcycles_ctu(time, dt, amr_pass, amr_cycles));

#ifndef MHD
#ifndef AMREX_USE_GPU
#ifdef TRUE_SDC
    } else if (time_scheme == IterativeCorrection) {

      for (int iter = 0; iter < sdc_accuracy_order+sdc_extra_passes; ++iter) {
        correction_iter = iter;
        dt_new = run_advance_sdc(time, dt, amr_pass, amr_cycles);
      }

#endif // TRUE_SDC
#endif // AMREX_USE_GPU
#endif //MHD
    }

    // If the user requests, indicate that we want a regrid at the end of the step.

    if (regrid_each_step == 1) {
        post_step_regrid = 1;
    }

    for (int lev = level; lev <= top_level_to_advance; ++lev) {
#ifdef GRAVITY
        // Update the point mass.
        if (use_central_mass == 1) {
            getLevel(lev).centralmass_update(time, dt);
        }
#endif

#ifdef RADIATION
        MultiFab& S_new = getLevel(lev).get_new_data(State_Type);
        getLevel(lev).close_radiation_step(S_new, amr_pass, amr_cycles);
#endif

#ifdef AMREX_PARTICLES
        getLevel(lev).step_particles(amr_pass, time, dt);
#endif

        getLevel(lev).end_advance();
    }

    return dt_new;
}


step_result
HydroSim::begin_step_advance (Real time, Real dt)
{
    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::begin_step_advance()");

    step_result status {};

#ifdef RADIATION
    // make sure these are filled to avoid check/plot file errors:
    if (do_radiative_transfer) {
      get_old_data(Radiation_Type).setBndry(0.0);
      get_new_data(Radiation_Type).setBndry(0.0);
    }
    else {
      get_old_data(Radiation_Type).setVal(0.0);
      get_new_data(Radiation_Type).setVal(0.0);
    }
#endif

#ifdef GRAVITY
    if (track_center == 1) {
        define_tracked_center(get_old_data(State_Type), time);
    }
#endif

    // For the hydrodynamics update we need to have NUM_GROW ghost
    // zones available, but the state data does not carry ghost
    // zones. So we use a FillPatch using the state data to give us
    // Sborder, which does have ghost zones.

    if (time_scheme == CornerTrace || time_scheme == SimplifiedIterCorrection) {
#ifdef MHD
      MultiFab& bfield_x_old = get_old_data(Bfield_Type_x);
      MultiFab& bfield_y_old = get_old_data(Bfield_Type_y);
      MultiFab& bfield_z_old = get_old_data(Bfield_Type_z);

      Bx_old_grown.define(bfield_x_old.boxArray(), bfield_x_old.DistributionMap(), 1, NUM_GROW);
      By_old_grown.define(bfield_y_old.boxArray(), bfield_y_old.DistributionMap(), 1, NUM_GROW);
      Bz_old_grown.define(bfield_z_old.boxArray(), bfield_z_old.DistributionMap(), 1, NUM_GROW);

      FillPatch(*this, Bx_old_grown, NUM_GROW, time, Bfield_Type_x, 0, 1);
      FillPatch(*this, By_old_grown, NUM_GROW, time, Bfield_Type_y, 0, 1);
      FillPatch(*this, Bz_old_grown, NUM_GROW, time, Bfield_Type_z, 0, 1);
#endif
      // for the CTU unsplit method, we always start with the old
      // state note: although clean_state has already been done on
      // the old state in begin_advance, we still need to do
      // another here to ensure the ghost zones are thermodynamically
      // consistent
      Sborder.define(grids, dmap, NUM_STATE, NUM_GROW, MFInfo().SetTag("Sborder"));
      const Real prev_time = state[State_Type].prevTime();
      fill_ghost_field(Sborder, prev_time, NUM_GROW);
      clean_state(
#ifdef MHD
                  Bx_old_grown, By_old_grown, Bz_old_grown,
#endif
                  Sborder, prev_time, NUM_GROW);

    } else if (time_scheme == IterativeCorrection) {

      // we'll handle the filling inside of run_advance_sdc
      Sborder.define(grids, dmap, NUM_STATE, NUM_GROW, MFInfo().SetTag("Sborder"));

    } else {
      amrex::Abort("invalid time_scheme");
    }

#ifdef SHOCK_VAR
    // Zero out the shock data, and fill it during the advance.
    // For subcycling cases this will always give the shock
    // variable for the latest subcycle, rather than averaging.

    Sborder.setVal(0.0, UFLAG_SHOCK, 1, Sborder.nGrow());
#endif

    // Create any correctors to the source term data. This must be done
    // before the source term data is overwritten below. Note: we do
    // not create the corrector source if we're currently retrying the
    // step; we will already have done it, and aside from avoiding
    // duplicate work, we have already lost the data needed to do this
    // calculation since we overwrote the data from the previous step.

    if (hydrosim::time_scheme == CornerTrace ||
        hydrosim::time_scheme == SimplifiedIterCorrection) {
        if (!retry_active) {
            create_forcing_corrector();
        }
    }

    // Check for NaN's.

    const MultiFab& S_old = get_old_data(State_Type);

    scan_for_nan(S_old);

    // If we're doing a step later than the first on each level, the fluid
    // state might have evolved to the point where the AMR timestep could be
    // significantly too large, but we don't have freedom to adjust the AMR
    // timestep at that point. Trying to evolve with a dt that is too large
    // could result in catastrophic behavior such that we don't even get to
    // the point where we can bail out later in the advance, so let's just
    // go directly into a retry now if we're too far away from the needed dt.

    bool is_opening_step_on_level = true;

    for (int lev = level; lev >= 0; --lev) {
        if (getLevel(lev).iteration > 1) {
            is_opening_step_on_level = false;
            break;
        }
    }

    if (hydrosim::dt_validity_before_advance && !is_opening_step_on_level) {
        int use_new_state = 0;

        // We only display the estTimeStep output if the validity check fails
        std::string dtEstimate_output;

        Real dt_old_estimate;

        {
            StdoutCapture redirection;
            dt_old_estimate = estTimeStep(use_new_state);
            dtEstimate_output = redirection.getBufferedText();
        }

        if (hydrosim::change_max * dt_old_estimate < dt) {
            status.success = false;
            std::cout << dtEstimate_output;
            status.reason = "pre-advance timestep validity check failed";
        }
    }

    return status;
}



step_result
HydroSim::end_step_advance (Real time, Real dt)
{
    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::end_step_advance()");

    step_result status {};

    // Check if this timestep violated our stability criteria. Our idea is,
    // if the timestep created a velocity v and sound speed at the new time
    // such that (v+c) * dt / dx < CFL / change_max, where CFL is the user's
    // chosen timestep constraint and change_max is the factor that determines
    // how much the timestep can change during an advance, consider the advance
    // to have failed. This prevents the timestep from shrinking too much,
    // whereas in computeNewDt change_max prevents the timestep from growing
    // too much. The same reasoning applies for the other timestep limiters.

    if (hydrosim::time_scheme == CornerTrace ||
        hydrosim::time_scheme == SimplifiedIterCorrection) {
        if (hydrosim::dt_validity_after_advance) {

            // But don't do this check if we're using simplified SDC and we're not yet
            // on the final SDC iteration, since we're not yet at the final advance.

            bool do_step_validation = true;

            if (hydrosim::time_scheme == SimplifiedIterCorrection &&
                correction_iter < deferred_iters - 1) {
                do_step_validation = false;
            }

            if (do_step_validation) {
                int use_new_state = 1;

                // We only display the estTimeStep output if the validity check fails
                std::string dtEstimate_output;

                Real candidate_dt;

                {
                    StdoutCapture redirection;
                    candidate_dt = estTimeStep(use_new_state);
                    dtEstimate_output = redirection.getBufferedText();
                }

                if (hydrosim::change_max * candidate_dt < dt) {
                    status.success = false;
                    std::cout << dtEstimate_output;
                    status.reason = "post-advance timestep validity check failed";
                    return status;
                }
            }
        }
    }

#ifdef RADIATION
    if (!do_hydro && Radiation::rad_hydro_coupled) {
        MultiFab& rad_energy_old = get_old_data(Radiation_Type);
        MultiFab& rad_energy_new = get_new_data(Radiation_Type);
        MultiFab::Copy(rad_energy_new, rad_energy_old, 0, 0, rad_energy_old.nComp(), 0);
    }
#endif

    Sborder.clear();

    return status;
}



void
HydroSim::begin_advance(Real time, Real dt, int amr_pass)
{
    BL_PROFILE("HydroSim::begin_advance()");

    // Save the current iteration.

    iteration = amr_pass;

    sub_pass = 0;
    sub_cycles = 0;
    dt_cycle_step = 1.e200;
    dt_level_step = dt;

    retain_prev_field = false;

    // Reset the retry information.

    retry_active = 0;
    num_sub_taken = 1;

    if (regrid_each_step == 1 && level > 0) {

        if (getLevel(level-1).post_step_regrid == 1 && amr_pass == 1) {

            // If the level below this just triggered a special regrid,
            // the coarse contribution to this level's FluxRegister
            // is no longer valid because the grids have, in general, changed.
            // Zero it out, and add them back using the saved copy of the fluxes.

            getLevel(level-1).FluxRegCoarseInit();

        }

    }

    // The option of whether to do a multilevel initialization is
    // controlled within the radiation class.  This step belongs
    // before the swap.

#ifdef RADIATION
    if (do_radiative_transfer) {
        radiation->begin_timestep(level);
    }

    RadEnergyGrown.define(grids, dmap, Radiation::nGroups, NUM_GROW);
    rad_lambda_grown.define(grids, dmap, Radiation::nGroups, NUM_GROW);
#endif

#ifdef GRAVITY
    // If we're on level 0, update the maximum density used in the gravity solver
    // for setting the tolerances. This will be used in all level solves to follow.
    // This must be done before the swap because it relies on the new data.

    if (level == 0 && do_gravity_flag == 1 && gravity->get_solver_kind() == "PoissonGrav") {
        gravity->update_max_density();
    }
#endif

    // This array holds the source term corrector.

    forcing_corrector.define(grids, dmap, NUM_SOURCE, SRC_GHOST_WIDTH);
    forcing_corrector.setVal(0.0, SRC_GHOST_WIDTH);

    // Swap the new data from the last timestep into the old state data.

    swap_field_time_levels(dt);

#ifdef GRAVITY
    if (do_gravity_flag == 1) {
        gravity->swapTimeLevels(level);
    }
#endif

    MultiFab& S_old = get_old_data(State_Type);

    // if we are doing drive_initial_convection, check to see if we
    // need to reinitialize the thermodynamic data (while keeping the
    // velocity unchanged)

#ifndef MHD

    const Real cur_time = state[State_Type].curTime();
    const Real dt_level = parent->dtLevel(level);

    if (drive_initial_convection == 1 && cur_time <= convect_drive_end_time) {

        // Calculate the new dt by comparing to the dt needed to get
        // to the next multiple of convect_drive_reset_period

        const Real phaseOffset = std::fmod(cur_time, convect_drive_reset_period);

        Real convect_reset_dt;

        // Note that if we are just about exactly on a multiple of
        // convect_drive_reset_period, then we need to be
        // careful to avoid floating point issues.


        if (std::abs(phaseOffset - convect_drive_reset_period) <=
            std::numeric_limits<Real>::epsilon() * cur_time) {
                convect_reset_dt = convect_drive_reset_period +
                    (convect_drive_reset_period - phaseOffset);
        } else {
            convect_reset_dt = convect_drive_reset_period - phaseOffset;
        }

        if (convect_reset_dt < dt_level) {

          amrex::Print() << "<<<<< drive initial convection reset >>>>" << std::endl;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
            for (MFIter mfi(S_old, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& box = mfi.tilebox();

                auto s = S_old[mfi].array();
                auto geomdata = geom.data();

#ifdef RNG_STATE_INIT
                amrex::Error("drive initial convection not yet supported for random initialization");
#else
                amrex::ParallelFor(box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // redo the problem initialization.  We want to preserve
                    // the current velocity though, so save that and then
                    // restore it afterwards.

                    Real vx_saved = s(i,j,k,UMX) / s(i,j,k,URHO);
                    Real vy_saved = s(i,j,k,UMY) / s(i,j,k,URHO);
                    Real vz_saved = s(i,j,k,UMZ) / s(i,j,k,URHO);

                    user_initialize_state_data(i, j, k, s, geomdata);

                    s(i,j,k,UMX) = s(i,j,k,URHO) * vx_saved;
                    s(i,j,k,UMY) = s(i,j,k,URHO) * vy_saved;
                    s(i,j,k,UMZ) = s(i,j,k,URHO) * vz_saved;

                    s(i,j,k,UEDEN) = s(i,j,k,UEINT) + 0.5_rt * s(i,j,k,URHO) *
                        (vx_saved * vx_saved + vy_saved * vy_saved + vz_saved * vz_saved);

                });
#endif

            }

        }
    }
#endif


    // Ensure data is valid before beginning advance. This addresses
    // the fact that we may have new data on this level that was interpolated
    // from a coarser level, and the interpolation in general cannot be
    // trusted to respect the consistency between certain state variables
    // (e.g. UEINT and UEDEN) that we demand in every zone.

    clean_state(
#ifdef MHD
                 get_old_data(Bfield_Type_x),
                 get_old_data(Bfield_Type_y),
                 get_old_data(Bfield_Type_z),
#endif
                  S_old, time, S_old.nGrow());


    // Initialize the previous state data container now, so that we can
    // always ask if it has valid data.

    for (int k = 0; k < num_field_types; ++k) {
        prev_state[k] = std::make_unique<StateData>();
    }


    // Allocate space for the primitive variables.

#ifdef TRUE_SDC
    q.define(grids, dmap, NQ, NUM_GROW);
    q.setVal(0.0);
    qthermo.define(grids, dmap, NQEXTRAQ, NUM_GROW);


    if (sdc_accuracy_order == 4) {
      q_second.define(grids, dmap, NQ, NUM_GROW);
      qthermo_second.define(grids, dmap, NQEXTRAQ, NUM_GROW);
#ifdef DIFFUSION
      Temperature_cc.define(grids, dmap, 1, NUM_GROW);
#endif
    }

    if (time_scheme == IterativeCorrection) {

      node_new.resize(COLLOC_NODES);

      node_new[0] = std::make_unique<MultiFab>(S_old, amrex::make_alias, 0, NUM_STATE);
      for (int n = 1; n < COLLOC_NODES; ++n) {
        node_new[n] = std::make_unique<MultiFab>(grids, dmap, NUM_STATE, 0);
        node_new[n]->setVal(0.0);
      }

      Advect_old.resize(COLLOC_NODES);
      for (int n = 0; n < COLLOC_NODES; ++n) {
        Advect_old[n] = std::make_unique<MultiFab>(grids, dmap, NUM_STATE, 0);
        Advect_old[n]->setVal(0.0);
      }

      Advect_new.resize(COLLOC_NODES);
      Advect_new[0] = std::make_unique<MultiFab>(*Advect_old[0], amrex::make_alias, 0, NUM_STATE);
      for (int n = 1; n < COLLOC_NODES; ++n) {
        Advect_new[n] = std::make_unique<MultiFab>(grids, dmap, NUM_STATE, 0);
        Advect_new[n]->setVal(0.0);
      }

      // We use BurnData a few ways for the SDC integration.  First, we
      // use it to store the initial guess to the nonlinear solve.
      // Second, at the end of the SDC update, we copy the cell-center
      // reaction source into it, including one ghost cell, for later
      // filling of the plotfile.  Finally, we use it as a temporary
      // buffer for when we convert the state to centers while making the
      // source term
      BurnData.define(grids, dmap, NUM_STATE, 2);

#ifdef REACTIONS
      Reactions_old.resize(COLLOC_NODES);
      for (int n = 0; n < COLLOC_NODES; ++n) {
        Reactions_old[n] = std::make_unique<MultiFab>(grids, dmap, NUM_STATE, 0);
        Reactions_old[n]->setVal(0.0);
      }
#endif

    }
#endif

    // Zero out the current fluxes.

    for (int dir = 0; dir < 3; ++dir) {
        fluxes[dir]->setVal(0.0);
        mass_face_flux[dir]->setVal(0.0);
    }

#if (AMREX_SPACEDIM <= 2)
    if (!Geom().IsCartesian()) {
        Pressure_radial.setVal(0.0);
    }
#endif

#ifdef RADIATION
    if (Radiation::rad_hydro_coupled) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            rad_face_flux[dir]->setVal(0.0);
        }
    }
#endif

#ifdef REACTIONS
    if (keep_burn_weights == 1) {
        burn_split_data.setVal(0.0);
    }
#endif

}



void
HydroSim::end_advance()
{
    BL_PROFILE("HydroSim::end_advance()");

    if (do_reflux == 1 && parent->subcyclingMode() != "None") {
        FluxRegCoarseInit();
        FluxRegFineAccum();
    }

    // The mass_face_flux array currently holds only the fluxes
    // from the last subcycle. Override this with the sum of
    // the fluxes from the full timestep (this will be used
    // later during the reflux operation).

    if (do_reflux == 1 && refresh_sources_after_reflux == 1 && parent->subcyclingMode() != "None") {
        for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
            MultiFab::Copy(*mass_face_flux[idir], *fluxes[idir], URHO, 0, 1, 0);
        }
    }

#ifdef TRUE_SDC
    q.clear();
    qthermo.clear();

    if (sdc_accuracy_order == 4) {
      q_second.clear();
      qthermo_second.clear();
#ifdef DIFFUSION
      Temperature_cc.clear();
#endif
    }
#endif

#ifdef RADIATION
    RadEnergyGrown.clear();
    rad_lambda_grown.clear();
#endif

    forcing_corrector.clear();

    if (!retain_prev_field) {
        amrex::FillNull(prev_state);
    }

#ifdef TRUE_SDC
    if (time_scheme == IterativeCorrection) {
      node_new.clear();
      Advect_new.clear();
      Advect_old.clear();
#ifdef REACTIONS
      Reactions_old.clear();
      BurnData.clear();
#endif
    }
#endif

    // Each level is finalized separately, even when advanced together.
    // Count only this level's zones in the cumulative performance statistic.

    cells_advanced_total += static_cast<Real>(grids.numPts()) / static_cast<Real>(getLevel(0).grids.numPts());

    // For the per-advance timing report, count all levels advanced together.

    int top_level_to_advance = level;

    if (parent->subcyclingMode() == "None") {
        top_level_to_advance = parent->finestLevel();
    }

    long zone_count_advanced = 0;

    for (int lev = level; lev <= top_level_to_advance; ++lev) {
        zone_count_advanced += getLevel(lev).grids.numPts();
    }

    Real wall_time = ParallelDescriptor::second() - wall_clock_start;

    Real zone_throughput_advance = static_cast<Real>(zone_count_advanced) / wall_time / 1.e6;

    if (verbose >= 1) {
        if (top_level_to_advance > 0) {
            if (level == 0) {
                amrex::Print() << "  Zones advanced per microsecond from level " << level << " to level "
                               << top_level_to_advance << ": " << zone_throughput_advance << std::endl << std::endl;
            }
        }
        else {
            amrex::Print() << "  Zones advanced per microsecond at this level: "
                           << zone_throughput_advance << std::endl << std::endl;
        }
    }
}
