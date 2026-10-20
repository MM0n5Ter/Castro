#include <HydroSim.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef GRAVITY
#include <Gravity.H>
#endif
#include <array>
#include <cmath>

#include <climits>

#include <format>

#include <user_initialize_state_data.H>

using namespace amrex;

int
HydroSim::last_level_in_step () const
{
    if (parent->subcyclingMode() == "None" && level == 0) {
        return parent->finestLevel();
    }
    return level;
}

Real
HydroSim::advance (Real time,
                 Real dt,
                 int  amr_pass,
                 int  amr_cycles)


{
    if (parent->subcyclingMode() == "None" && level > 0) {
        amrex::Print() << "\n  Advance at this level has already been completed.\n\n";
        return dt;
    }

    BL_PROFILE("HydroSim::advance()");


    wall_clock_start = ParallelDescriptor::second();

    MultiFab::RegionTag level_region_tag(std::format("AmrLevel_Level_{}", level));

    Real dt_new = dt;

    const int top_level_to_advance = last_level_in_step();

    for (int lev = level; lev <= top_level_to_advance; ++lev) {
        getLevel(lev).begin_advance(time, dt, amr_pass);
    }


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

#endif
#endif
#endif
    }


    if (regrid_each_step == 1) {
        post_step_regrid = 1;
    }

    for (int lev = level; lev <= top_level_to_advance; ++lev) {
#ifdef GRAVITY
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

Real
HydroSim::advance_subcycles_ctu(const Real time, const Real dt, int amr_pass, int amr_cycles)
{
    BL_PROFILE("HydroSim::advance_subcycles_ctu()");


    if (dt_cycle_step == 1.e200) {
        dt_cycle_step = dt;
    }

    Real subcycle_clock = time;

    Real dt_new = 1.e200;

    sub_pass = 0;

    const int top_level_to_advance = last_level_in_step();


    Real eps = 1.0e-14;

    auto do_swap = false;

    Real final_dt_subcycle = 1.e200;

    while (subcycle_clock < (1.0 - eps) * (time + dt)) {


        final_dt_subcycle = dt_cycle_step;


        AMREX_ASSERT(min_dt_time_ratio > eps);

        if (subcycle_clock + dt_cycle_step > (1.0 - min_dt_time_ratio) * (time + dt)) {
          dt_cycle_step = (time + dt) - subcycle_clock;
        }


        if (dt_cycle_step <= min_dt_time_ratio * time) {
            if (ParallelDescriptor::IOProcessor()) {
                std::cout << std::endl;
                std::cout << "  The subcycle mechanism requested subcycled timesteps of maximum length dt = " << dt_cycle_step << "," << std::endl
                          << "  but this timestep is shorter than the user-defined minimum, " << std::endl
                          << "  hydrosim.min_dt_time_ratio, multiplied by the current time (" << min_dt_time_ratio * time << "). Aborting." << std::endl;
            }
            amrex::Abort("Error: subcycled timesteps too short.");
        }


        int subcycles_left = int(round(((time + dt) - subcycle_clock) / dt_cycle_step));

        if (subcycles_left > subcycle_cap) {
            amrex::Print() << std::endl
                           << "  The subcycle mechanism requested " << subcycles_left
                           << " subcycled timesteps, which is larger than the maximum of "
                           << subcycle_cap << "." << std::endl
                           << "  If you would like to override this, increase the parameter hydrosim.subcycle_cap." << std::endl;
            amrex::Abort("Error: too many subcycles.");
        }


        if (verbose && ParallelDescriptor::IOProcessor()) {
            std::cout << std::endl;
            std::cout << Font::Bold << FGColor::Green
                      << "  Beginning subcycle " << sub_pass + 1
                      << " starting at time " << subcycle_clock
                      << " with dt = " << dt_cycle_step << ResetDisplay << std::endl;
            std::cout << "  Estimated number of subcycles remaining: "
                      << subcycles_left << std::endl << std::endl;
        }


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


        for (int lev = level; lev <= top_level_to_advance; ++lev) {
            for (int k = 0; k < num_field_types; k++) {
                getLevel(lev).state[k].setTimeLevel(subcycle_clock + dt_cycle_step, dt_cycle_step, 0.0);
            }
        }


        int sub_iter_count = 1;


        int sdc_passes_old = deferred_iters;

        if (time_scheme == SimplifiedIterCorrection) {

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


            status = run_advance_ctu(subcycle_clock, dt_cycle_step);

            for (int lev = level; lev <= top_level_to_advance; ++lev) {
                getLevel(lev).retry_active = false;
            }

            if (!status.success) {
                if (!(allow_step_retry)) {
                    amrex::Abort("Advance was unsuccessful.");
                } else {
                    amrex::Print() << "Advance was unsuccessful with reason: " << status.reason << "; proceeding to a retry." << std::endl << std::endl;
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


        deferred_iters = sdc_passes_old;


        if (allow_step_retry) {


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


        for (int lev = level; lev <= top_level_to_advance; ++lev) {
            getLevel(lev).lastStepDt = dt_cycle_step;
        }

    }

    if (verbose) {
        amrex::Print() << "  Subcycling complete" << std::endl << std::endl;
    }


    for (int lev = level; lev <= top_level_to_advance; ++lev) {
        getLevel(lev).num_sub_taken = sub_pass;
    }

    if (sub_pass > 1) {


        for (int lev = level; lev <= top_level_to_advance; ++lev) {
            for (int k = 0; k < num_field_types; k++) {
                if (getLevel(lev).prev_state[k]->hasOldData()) {
                    getLevel(lev).state[k].replaceOldData(*getLevel(lev).prev_state[k]);
                }
                getLevel(lev).state[k].setTimeLevel(time + dt, dt, 0.0);
                getLevel(lev).prev_state[k]->setTimeLevel(time + dt, dt_cycle_step, 0.0);
            }
        }


        if (do_reflux && refresh_sources_after_reflux) {


            for (int lev = level; lev <= top_level_to_advance; ++lev) {
                if (!(amr_pass < amr_cycles && lev == parent->finestLevel())) {
                    getLevel(lev).retain_prev_field = true;
                }
            }

        }

    }


    dt_cycle_step = final_dt_subcycle;

    dt_new = std::min(dt_new, dt_cycle_step);

    return dt_new;

}

step_result
HydroSim::run_advance_ctu (Real time, Real dt)
{

    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    BL_PROFILE("HydroSim::run_advance_ctu()");

    step_result status {};

#ifndef TRUE_SDC


    const int top_level_to_advance = last_level_in_step();

    const Real prev_time = state[State_Type].prevTime();
    const Real  cur_time = state[State_Type].curTime();

    for (int lev = level; lev <= top_level_to_advance; ++lev) {

        status = getLevel(lev).begin_step_advance(time, dt);

        if (status.success == false) {
            return status;
        }


        status = getLevel(lev).before_advance_hooks(prev_time, dt);

        if (status.success == false) {
            return status;
        }


        status = getLevel(lev).run_old_sources(prev_time, dt);

        if (status.success == false) {
            return status;
        }


        status = getLevel(lev).before_hydro_hooks(prev_time, dt);

        if (status.success == false) {
            return status;
        }


#ifndef MHD
        status = getLevel(lev).assemble_ctu_hydro_source(prev_time, dt);
#else
        status = getLevel(lev).assemble_ctu_mhd_source(prev_time, dt);
#endif

        if (status.success == false) {
            return status;
        }
    }


    if (do_reflux && level < top_level_to_advance) {
        reflux(level, top_level_to_advance, false);
    }

    static constexpr std::array<step_result (HydroSim::*)(Real, Real), 4> finishing_steps = {
        &HydroSim::after_hydro_hooks,
        &HydroSim::run_new_sources,
        &HydroSim::after_advance_hooks,
        &HydroSim::end_step_advance
    };

    for (int lev = level; lev <= top_level_to_advance; ++lev) {

        for (auto finishing_step : finishing_steps) {
            status = (getLevel(lev).*finishing_step)(cur_time, dt);

            if (status.success == false) {
                return status;
            }
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

        const int top_level_to_advance = last_level_in_step();

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


        for (int lev = level; lev <= top_level_to_advance; ++lev) {
            getLevel(lev).snapshot_for_rerun();


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
