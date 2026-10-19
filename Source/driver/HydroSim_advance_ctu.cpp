
#include <HydroSim.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef GRAVITY
#include <Gravity.H>
#endif

using namespace amrex;

step_result
HydroSim::run_advance_ctu (Real time, Real dt)  // NOLINT(readability-convert-member-functions-to-static)
{
    // this routine will advance the old state data (called Sborder here)
    // to the new time, for a single level.  The new data is called
    // S_new here.  The update includes reactions (if we are not doing
    // SDC), hydro, and the source terms.

    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    BL_PROFILE("HydroSim::run_advance_ctu()");

    step_result status {};

#ifndef TRUE_SDC

    // Advance simultaneously on all levels that are not subcycling
    // relative to this level.

    int top_level_to_advance = level;

    if (parent->subcyclingMode() == "None" && level == 0) {
        top_level_to_advance = parent->finestLevel();
    }

    const Real prev_time = state[State_Type].prevTime();
    const Real  cur_time = state[State_Type].curTime();

    for (int lev = level; lev <= top_level_to_advance; ++lev) {
        // Perform initialization steps.

        status = getLevel(lev).begin_step_advance(time, dt);

        if (status.success == false) {
            return status;
        }

        // Perform all pre-advance operations and then initialize
        // the new-time state with the output of those operators.

        status = getLevel(lev).before_advance_hooks(prev_time, dt);

        if (status.success == false) {
            return status;
        }

        // Construct the old-time sources from Sborder. This will already
        // be applied to S_new (with full dt weighting), to be corrected
        // later. Note that this does not affect the prediction of the
        // interface state; an explicit source will be traced there as
        // needed.

        status = getLevel(lev).run_old_sources(prev_time, dt);

        if (status.success == false) {
            return status;
        }

        // Perform any operations that occur after the sources but before the hydro.

        status = getLevel(lev).before_hydro_hooks(prev_time, dt);

        if (status.success == false) {
            return status;
        }

        // Do the hydro update. We build directly off of Sborder, which
        // is the state that has already seen the burn.

#ifndef MHD
        status = getLevel(lev).assemble_ctu_hydro_source(prev_time, dt);
#else
        status = getLevel(lev).assemble_ctu_mhd_source(prev_time, dt);
#endif

        if (status.success == false) {
            return status;
        }
    }

    // We can perform the reflux immediately if there's no subcycling
    // above this level.

    if (do_reflux && level < top_level_to_advance) {
        reflux(level, top_level_to_advance, false);
    }

    for (int lev = level; lev <= top_level_to_advance; ++lev) {
        // Perform any operations that occur after the hydro but before
        // the corrector sources.

        status = getLevel(lev).after_hydro_hooks(cur_time, dt);

        if (status.success == false) {
            return status;
        }

        // Construct and apply new-time source terms.

        status = getLevel(lev).run_new_sources(cur_time, dt);

        if (status.success == false) {
            return status;
        }

        // Do the second half of the reactions for Strang, or the full burn for simplified SDC.

        status = getLevel(lev).after_advance_hooks(cur_time, dt);

        if (status.success == false) {
            return status;
        }

        // Perform finalization steps.

        status = getLevel(lev).end_step_advance(cur_time, dt);

        if (status.success == false) {
            return status;
        }
    }

#endif

    return status;

}





bool
HydroSim::rerun_advance_ctu(Real dt, const step_result& status)
{
    BL_PROFILE("HydroSim::rerun_advance_ctu()");

    bool retry_needed = false;

    if (!status.success) {
        retry_needed = true;
    }

    if (retry_needed) {

        int top_level_to_advance = level;

        if (parent->subcyclingMode() == "None" && level == 0) {
            top_level_to_advance = parent->finestLevel();
        }

        for (int lev = level; lev <= top_level_to_advance; ++lev) {
            if (status.proposed_dt > 0.0_rt && status.proposed_dt < dt) {
                getLevel(lev).dt_cycle_step = status.proposed_dt;
            } else {
                getLevel(lev).dt_cycle_step = std::min(dt, getLevel(lev).dt_cycle_step) * subcycle_backoff_factor;
            }
        }

        if (verbose && ParallelDescriptor::IOProcessor()) {
            std::cout << std::endl;
            std::cout << Font::Bold << FGColor::Red;
            std::cout << "  Timestep " << dt << " rejected at level " << level << "." << std::endl;
            std::cout << "  Performing a retry, with subcycled timesteps of maximum length dt = " << dt_cycle_step << std::endl;
            std::cout << ResetDisplay;
            std::cout << std::endl;
        }

        // If we are doing a retry and this is the first attempt
        // at the advance, make a copy of the state data. This will
        // be useful to us at the end of the timestep when we need
        // to restore the original old data.

        for (int lev = level; lev <= top_level_to_advance; ++lev) {
            getLevel(lev).snapshot_for_rerun();

            // Clear the contribution to the fluxes from this step.

            for (int dir = 0; dir < 3; ++dir) {
                getLevel(lev).fluxes[dir]->setVal(0.0);
            }

            for (int dir = 0; dir < 3; ++dir) {
                getLevel(lev).mass_face_flux[dir]->setVal(0.0);
            }

#if (AMREX_SPACEDIM <= 2)
            if (!Geom().IsCartesian()) {
                getLevel(lev).Pressure_radial.setVal(0.0);
            }
#endif

#ifdef RADIATION
            if (Radiation::rad_hydro_coupled) {
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                    getLevel(lev).rad_face_flux[dir]->setVal(0.0);
                }
            }
#endif

#ifdef REACTIONS
            if (hydrosim::keep_burn_weights) {
                getLevel(lev).burn_split_data.setVal(0.0);
            }
#endif

            // For simplified SDC, we'll have garbage data if we
            // attempt to use the lagged source terms (both reacting
            // and non-reacting) from the last timestep, since that
            // advance failed and we don't know if we can trust it.
            // So we zero out both source term correctors.

            if (time_scheme == SimplifiedIterCorrection) {
                getLevel(lev).forcing_corrector.setVal(0.0, getLevel(lev).forcing_corrector.nGrow());

#ifdef REACTIONS
                MultiFab& SDC_burn_new = getLevel(lev).get_new_data(Burn_Corr_Type);
                SDC_burn_new.setVal(0.0, SDC_burn_new.nGrow());
#endif
            }

        }

    }

    return retry_needed;
}



Real
HydroSim::advance_subcycles_ctu(const Real time, const Real dt, int amr_pass, int amr_cycles)
{
    BL_PROFILE("HydroSim::advance_subcycles_ctu()");

    // Start the subcycle time off with the main dt,
    // unless we already came in here with an estimate
    // that is different from the initial value we assigned,
    // for example from the post-step regrid algorithm.

    if (dt_cycle_step == 1.e200) {
        dt_cycle_step = dt;
    }

    Real subcycle_clock = time;

    Real dt_new = 1.e200;

    sub_pass = 0;

    int top_level_to_advance = level;

    if (parent->subcyclingMode() == "None" && level == 0) {
        top_level_to_advance = parent->finestLevel();
    }

    // Subcycle until we've reached the target time.
    // Compare against a slightly smaller number to
    // avoid roundoff concerns.

    Real eps = 1.0e-14;

    bool do_swap = false;

    Real final_dt_subcycle = 1.e200;

    while (subcycle_clock < (1.0 - eps) * (time + dt)) {

        // Save the dt_cycle_step before modifying it, we will use it later.

        final_dt_subcycle = dt_cycle_step;

        // Shorten the last timestep so that we don't overshoot
        // the ending time. Relatedly, we also don't want to
        // undershoot the ending time, which would cause us to
        // have to take a very short final timestep to get to
        // the desired time. We'll use a slightly larger tolerance
        // factor than we do in the outer while loop, to avoid
        // roundoff issues, under the logic that as long as the
        // tolerance is still small, there is no meaningful harm
        // from the timestep being slightly larger than our recommended
        // safe dt in the subcycling.

        AMREX_ASSERT(min_dt_time_ratio > eps);

        if (subcycle_clock + dt_cycle_step > (1.0 - min_dt_time_ratio) * (time + dt)) {
          dt_cycle_step = (time + dt) - subcycle_clock;
        }

        // Determine whether we're below the cutoff timestep.

        if (dt_cycle_step <= min_dt_time_ratio * time) {
            if (ParallelDescriptor::IOProcessor()) {
                std::cout << std::endl;
                std::cout << "  The subcycle mechanism requested subcycled timesteps of maximum length dt = " << dt_cycle_step << "," << std::endl
                          << "  but this timestep is shorter than the user-defined minimum, " << std::endl
                          << "  hydrosim.min_dt_time_ratio, multiplied by the current time (" << min_dt_time_ratio * time << "). Aborting." << std::endl;
            }
            amrex::Abort("Error: subcycled timesteps too short.");
        }

        // Check on whether we are going to take too many subcycles.

        int subcycles_left = int(round(((time + dt) - subcycle_clock) / dt_cycle_step));

        if (subcycles_left > subcycle_cap) {
            amrex::Print() << std::endl
                           << "  The subcycle mechanism requested " << subcycles_left
                           << " subcycled timesteps, which is larger than the maximum of "
                           << subcycle_cap << "." << std::endl
                           << "  If you would like to override this, increase the parameter hydrosim.subcycle_cap." << std::endl;
            amrex::Abort("Error: too many subcycles.");
        }

        // If we get to this point, we survived the sanity checks. Print out the current subcycle iteration.

        if (verbose && ParallelDescriptor::IOProcessor()) {
            std::cout << std::endl;
            std::cout << Font::Bold << FGColor::Green
                      << "  Beginning subcycle " << sub_pass + 1
                      << " starting at time " << subcycle_clock
                      << " with dt = " << dt_cycle_step << ResetDisplay << std::endl;
            std::cout << "  Estimated number of subcycles remaining: "
                      << subcycles_left << std::endl << std::endl;
        }

        // Swap the time levels. Only do this after the first iteration;
        // the first iteration used the swap done in begin_advance.
        // After that, the only exception will be if we do a retry.

        if (do_swap) {

            for (int lev = level; lev <= top_level_to_advance; ++lev) {
                getLevel(lev).swap_field_time_levels(0.0);

#ifdef GRAVITY
                if (do_gravity_flag) {
                    getLevel(lev).gravity->swapTimeLevels(lev);
                }
#endif
            }

        }  else {
            do_swap = true;
        }

        // Set the relevant time levels.

        for (int lev = level; lev <= top_level_to_advance; ++lev) {
            for (int k = 0; k < num_field_types; k++) {
                getLevel(lev).state[k].setTimeLevel(subcycle_clock + dt_cycle_step, dt_cycle_step, 0.0);
            }
        }

        // Do the advance and construct the relevant source terms. For CTU this
        // will include Strang-split reactions; for simplified SDC, we defer the
        // burn until after the advance.

        int sub_iter_count = 1;

        // Save the number of SDC iterations in case we are about to modify it.

        int sdc_passes_old = deferred_iters;

        if (time_scheme == SimplifiedIterCorrection) {
            // If we're in a retry we want to add one more iteration. This is
            // because we will have zeroed out the lagged corrector from the
            // last iteration/timestep and so having another iteration will
            // approximately compensate for that.

            if (retry_active) {
                deferred_iters += 1;
                amrex::Print() << "Adding an SDC iteration due to the retry." << std::endl << std::endl;
            }

            sub_iter_count = deferred_iters;
        }

        step_result status {};

        for (int n = 0; n < sub_iter_count; ++n) {

            if (time_scheme == SimplifiedIterCorrection) {
                for (int lev = level; lev <= top_level_to_advance; ++lev) {
                    getLevel(lev).correction_iter = n;
                }

                amrex::Print() << "Beginning SDC iteration " << n + 1 << " of " << sub_iter_count << "." << std::endl << std::endl;
            }

            // We do the hydro advance here, and record whether we completed it.

            status = run_advance_ctu(subcycle_clock, dt_cycle_step);

            for (int lev = level; lev <= top_level_to_advance; ++lev) {
                getLevel(lev).retry_active = false;
            }

            if (!status.success) {
                if (allow_step_retry) {
                    amrex::Print() << "Advance was unsuccessful with reason: " << status.reason << "; proceeding to a retry." << std::endl << std::endl;
                } else {
                    amrex::Abort("Advance was unsuccessful.");
                }
                break;
            }

            if (time_scheme == SimplifiedIterCorrection) {
                amrex::Print() << "Ending SDC iteration " << n + 1 << " of " << sub_iter_count << "." << std::endl << std::endl;
            }

        }

        if (verbose && ParallelDescriptor::IOProcessor()) {
            std::cout << Font::Bold << FGColor::Green << "  Subcycle completed" << ResetDisplay << std::endl << std::endl;
        }

        // Set deferred_iters to its original value, in case we modified it above.

        deferred_iters = sdc_passes_old;

        // If we're allowing for retries, check for that here.

        if (allow_step_retry) {

            // If we hit a retry, signal that we want to try again.
            // The retry function will handle resetting the state,
            // and updating dt_cycle_step.

            if (rerun_advance_ctu(dt_cycle_step, status)) {
                do_swap = false;
                for (int lev = level; lev <= top_level_to_advance; ++lev) {
                    getLevel(lev).retry_active = true;
                }

                continue;
            }
            else {
                for (int lev = level; lev <= top_level_to_advance; ++lev) {
                    getLevel(lev).retry_active = false;
                }
            }

        }

        subcycle_clock += dt_cycle_step;
        sub_pass += 1;

        // Continually record the last timestep we took on each level advanced
        // in case we need it later. We only record it if the subcycle
        // was completed successfully (i.e. we got to this point).
        // Note: this is different from final_dt_subcycle. This variable
        // records the actual timestep taken in this subcycle, while
        // the other one records the timestep as if it had not been
        // modified by the constraint of matching the final time.

        for (int lev = level; lev <= top_level_to_advance; ++lev) {
            getLevel(lev).lastStepDt = dt_cycle_step;
        }

    }

    if (verbose) {
        amrex::Print() << "  Subcycling complete" << std::endl << std::endl;
    }

    // Record the number of subcycles we took for diagnostic purposes.

    for (int lev = level; lev <= top_level_to_advance; ++lev) {
        getLevel(lev).num_sub_taken = sub_pass;
    }

    if (sub_pass > 1) {

        // Finally, copy the original data back to the old state
        // data so that externally it appears like we took only
        // a single timestep. We'll do this as a swap so that
        // we still have the last iteration's old data if we need
        // it later.

        for (int lev = level; lev <= top_level_to_advance; ++lev) {
            for (int k = 0; k < num_field_types; k++) {
                if (getLevel(lev).prev_state[k]->hasOldData()) {
                    getLevel(lev).state[k].replaceOldData(*getLevel(lev).prev_state[k]);
                }
                getLevel(lev).state[k].setTimeLevel(time + dt, dt, 0.0);
                getLevel(lev).prev_state[k]->setTimeLevel(time + dt, dt_cycle_step, 0.0);
            }
        }

        // If we took more than one step and are going to do a reflux,
        // keep the data past the end of the step.

        if (do_reflux && refresh_sources_after_reflux) {

            // Note that since we only want to do this if there's actually a
            // reflux immediately following this, skip this if we're on the
            // finest level and this is not the last iteration.

            for (int lev = level; lev <= top_level_to_advance; ++lev) {
                if (!(amr_pass < amr_cycles && lev == parent->finestLevel())) {
                    getLevel(lev).retain_prev_field = true;
                }
            }

        }

    }

    // We want to return the subcycled timestep as a suggestion.
    // Let's be sure to return the subcycled timestep that was
    // unmodified by anything we had to do in the last subcycle
    // to reach the target time.

    dt_cycle_step = final_dt_subcycle;

    dt_new = std::min(dt_new, dt_cycle_step);

    return dt_new;

}
