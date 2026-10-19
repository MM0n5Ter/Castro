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

step_result
HydroSim::begin_step_advance (Real time, Real dt)
{
    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::begin_step_advance()");

    step_result status {};

#ifdef RADIATION
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
      Sborder.define(grids, dmap, NUM_STATE, NUM_GROW, MFInfo().SetTag("Sborder"));
      const Real prev_time = state[State_Type].prevTime();
      fill_ghost_field(Sborder, prev_time, NUM_GROW);
      clean_state(
#ifdef MHD
                  Bx_old_grown, By_old_grown, Bz_old_grown,
#endif
                  Sborder, prev_time, NUM_GROW);

    } else if (time_scheme == IterativeCorrection) {

      Sborder.define(grids, dmap, NUM_STATE, NUM_GROW, MFInfo().SetTag("Sborder"));

    } else {
      amrex::Abort("invalid time_scheme");
    }

#ifdef SHOCK_VAR

    Sborder.setVal(0.0, UFLAG_SHOCK, 1, Sborder.nGrow());
#endif

    if (hydrosim::time_scheme == CornerTrace ||
        hydrosim::time_scheme == SimplifiedIterCorrection) {
        if (!retry_active) {
            create_forcing_corrector();
        }
    }

    const MultiFab& S_old = get_old_data(State_Type);

    scan_for_nan(S_old);

    bool is_opening_step_on_level = true;

    for (int lev = level; lev >= 0; --lev) {
        if (getLevel(lev).iteration > 1) {
            is_opening_step_on_level = false;
            break;
        }
    }

    if (hydrosim::dt_validity_before_advance && !is_opening_step_on_level) {
        int use_new_state = 0;

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

    if (hydrosim::time_scheme == CornerTrace ||
        hydrosim::time_scheme == SimplifiedIterCorrection) {
        if (hydrosim::dt_validity_after_advance) {

            bool do_step_validation = true;

            if (hydrosim::time_scheme == SimplifiedIterCorrection &&
                correction_iter < deferred_iters - 1) {
                do_step_validation = false;
            }

            if (do_step_validation) {
                int use_new_state = 1;

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

    iteration = amr_pass;

    sub_pass = 0;
    sub_cycles = 0;
    dt_cycle_step = 1.e200;
    dt_level_step = dt;

    retain_prev_field = false;

    retry_active = 0;
    num_sub_taken = 1;

    if (regrid_each_step == 1 && level > 0) {

        if (getLevel(level-1).post_step_regrid == 1 && amr_pass == 1) {

            getLevel(level-1).FluxRegCoarseInit();

        }

    }

#ifdef RADIATION
    if (do_radiative_transfer) {
        radiation->begin_timestep(level);
    }

    RadEnergyGrown.define(grids, dmap, Radiation::nGroups, NUM_GROW);
    rad_lambda_grown.define(grids, dmap, Radiation::nGroups, NUM_GROW);
#endif

#ifdef GRAVITY

    if (level == 0 && do_gravity_flag == 1 && gravity->get_solver_kind() == "PoissonGrav") {
        gravity->update_max_density();
    }
#endif

    forcing_corrector.define(grids, dmap, NUM_SOURCE, SRC_GHOST_WIDTH);
    forcing_corrector.setVal(0.0, SRC_GHOST_WIDTH);

    swap_field_time_levels(dt);

#ifdef GRAVITY
    if (do_gravity_flag == 1) {
        gravity->swapTimeLevels(level);
    }
#endif

    MultiFab& S_old = get_old_data(State_Type);

#ifndef MHD

    const Real cur_time = state[State_Type].curTime();
    const Real dt_level = parent->dtLevel(level);

    if (drive_initial_convection == 1 && cur_time <= convect_drive_end_time) {

        const Real phaseOffset = std::fmod(cur_time, convect_drive_reset_period);

        Real convect_reset_dt;

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

    clean_state(
#ifdef MHD
                 get_old_data(Bfield_Type_x),
                 get_old_data(Bfield_Type_y),
                 get_old_data(Bfield_Type_z),
#endif
                  S_old, time, S_old.nGrow());

    for (int k = 0; k < num_field_types; ++k) {
        prev_state[k] = std::make_unique<StateData>();
    }

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

    cells_advanced_total += static_cast<Real>(grids.numPts()) / static_cast<Real>(getLevel(0).grids.numPts());

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
