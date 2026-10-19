#ifndef WIN32
#include <unistd.h>
#endif

#include <iomanip>

#include <algorithm>
#include <cstdio>
#include <vector>
#include <iostream>
#include <string>
#include <ctime>
#include <memory>
#include <numbers>
#include <format>

#include <AMReX_Utility.H>
#include <AMReX_CONSTANTS.H>
#include <HydroSim.H>
#include <global.H>
#include <runtime_parameters.H>
#include <AMReX_VisMF.H>
#include <AMReX_TagBox.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_ParmParse.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef AMREX_PARTICLES
#include <AMReX_Particles.H>
#endif

#ifdef GRAVITY
#include <Gravity.H>
#endif

#ifdef DIFFUSION
#include <Diffusion.H>
#endif

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <extern_parameters.H>
#include <prob_parameters.H>

#include <ambient.H>
#include <hydrosim_limits.H>

#include <solver_constants.H>

using namespace amrex;

void
HydroSim::post_init (Real  )
{
    BL_PROFILE("HydroSim::post_init()");

    if (level > 0) {
        return;
    }

    int finest_level = parent->finestLevel();
    for (int k = finest_level-1; k>= 0; k--) {
      getLevel(k).avgDown();
    }

#ifdef GRAVITY

    if (do_gravity_flag) {

       Real cur_time = state[State_Type].curTime();

       if (gravity->get_solver_kind() == "PoissonGrav") {

          gravity->update_max_density();

          gravity->set_mass_shift(cur_time);

          gravity->multilevel_potential_solve(level,finest_level);
          if (gravity->get_self_check_flag() == 1) {
              gravity->check_composite_potential(level);
          }
       }

       for (int k = 0; k <= parent->finestLevel(); k++)
       {
          BoxArray ba = getLevel(k).boxArray();
          MultiFab& gravity_new = getLevel(k).get_new_data(Accel_Type);
          gravity->get_new_gravity_vector(k,gravity_new,cur_time);
       }
    }
#endif

#ifdef RADIATION
    if (do_radiative_transfer) {

      radiation->post_init(level);

      for (int k = finest_level-1; k>= 0; k--) {
        getLevel(k).avgDown(Radiation_Type);
      }

      print_energy_budget();

    }
#endif

#ifdef DO_USER_POST_INIT

    user_post_init();

#endif

#ifdef GRAVITY
#ifdef ROTATION
    if (do_equilibrium_start) {
        scf_solve();
    }
#endif
#endif

        int nstep = parent->levelSteps(0);
        Real level0_dt = parent->dtLevel(0);
        Real cumtime = parent->cumTime();
        if (cumtime != 0.0) {
            cumtime += level0_dt;
        }

        bool sum_step_hit = false;

        if (sum_step_period > 0) {

          if (nstep%sum_step_period == 0) {
            sum_step_hit = true;
          }
        }

        bool sum_time_hit = false;

        if (sum_time_period > 0.0) {

          const int num_per_old = static_cast<int>(std::floor((cumtime - level0_dt) / sum_time_period));
          const int num_per_new = static_cast<int>(std::floor((cumtime        ) / sum_time_period));

          if (num_per_old != num_per_new) {
            sum_time_hit = true;
          }

        }

        if (sum_step_hit || sum_time_hit) {
          tally_integrated_quantities();
        }

#ifdef GRAVITY
    if (level == 0 && track_center == 1) {
       write_tracked_center();
    }
#endif
}

void
HydroSim::post_restart ()
{
   BL_PROFILE("HydroSim::post_restart()");

#ifdef AMREX_PARTICLES
   ParticleAfterRestart(parent->theRestartFile());
#endif

#ifdef GRAVITY
    if (do_gravity_flag)
    {
        Real cur_time = state[State_Type].curTime();

        if (level == 0)
        {
            int grid_points_1d = count_points_1d ();

#if (AMREX_SPACEDIM == 1)
            grid_points_1d += 2 * NUM_GROW;
#endif

            gravity->set_grid_points_gravity(grid_points_1d);

            for (int lev = 0; lev <= parent->finestLevel(); lev++)
            {
                AmrLevel& level_handle  = getLevel(lev);
                HydroSim& sim_level = getLevel(lev);
                gravity->attach_level(lev,&level_handle,
                                       sim_level.Volume(),sim_level.Area());
            }

            if (track_center == 1)
            {
               const MultiFab& S_new = get_new_data(State_Type);
               define_tracked_center(S_new,cur_time);
            }

            gravity->set_mass_shift(cur_time);

            if ( gravity->get_solver_kind() == "PoissonGrav")
            {

                gravity->update_max_density();

                gravity->multilevel_potential_solve(0, parent->finestLevel());
                if (gravity->get_self_check_flag() == 1) {
                    gravity->check_composite_potential(level);
                }
            }

            if (restart_grow_factor > 1) {
                after_grow_restart();
            }
        }
    }
#endif

#ifdef DIFFUSION
      if (diffusion == nullptr) {
          diffusion = std::make_unique<Diffusion>(parent, &phys_bc);
      }

      if (level == 0) {
          for (int lev = 0; lev <= parent->finestLevel(); lev++) {
              AmrLevel& level_handle = getLevel(lev);
              HydroSim& sim_level = getLevel(lev);
              diffusion->attach_level(lev,&level_handle,
                                       sim_level.Volume(),sim_level.Area());
          }
      }
#endif

#ifdef DO_USER_POST_RESTART
    user_post_restart();
#endif

}

void
HydroSim::after_grow_restart ()
{

    BL_PROFILE("HydroSim::after_grow_restart()");

    if (level > 0) {
        return;
    }

#ifdef GRAVITY
    if (do_gravity_flag) {
        int finest_level = parent->finestLevel();
        Real cur_time = state[State_Type].curTime();

        if (gravity->get_solver_kind() == "PoissonGrav") {

          gravity->update_max_density();

          gravity->set_mass_shift(cur_time);

          gravity->multilevel_potential_solve(level,finest_level);
          if (gravity->get_self_check_flag() == 1) {
              gravity->check_composite_potential(level);
          }
       }

       for (int k = 0; k <= parent->finestLevel(); k++)
       {
          MultiFab& gravity_new = getLevel(k).get_new_data(Accel_Type);
          gravity->get_new_gravity_vector(k,gravity_new,cur_time);
       }
    }
#endif

#ifdef RADIATION
    if (do_radiative_transfer) {

      radiation->post_init(level);

      int finest_level = parent->finestLevel();

      for (int k = finest_level-1; k>= 0; k--) {
        getLevel(k).avgDown(Radiation_Type);
      }

      print_energy_budget();

    }
#endif
}

void
HydroSim::post_regrid (int lbase,
                     int new_finest)
{

    amrex::ignore_unused(lbase);
    amrex::ignore_unused(new_finest);

    BL_PROFILE("HydroSim::post_regrid()");

    fine_mask.clear();

#ifdef AMREX_PARTICLES
    if (TracerPC && level == lbase) {
        TracerPC->Redistribute(lbase);
    }
#endif

#ifdef GRAVITY
    if (do_gravity_flag)
    {

        if (regrid_each_step && getLevel(lbase).post_step_regrid && gravity->get_solver_kind() == "PoissonGrav") {

           if (level > lbase) {

               PotentialGradBCFunct gradient_phys_bc;

               Interpolater* gradient_interp = &face_linear_interp;

               Vector<MultiFab*> potential_gradient_coarse = amrex::GetVecOfPtrs(gravity->get_potential_gradient_prev(level-1));
               Vector<MultiFab*> potential_gradient_fine = amrex::GetVecOfPtrs(gravity->get_potential_gradient_curr(level));

               Real time = getLevel(lbase).get_state_data(Accel_Type).prevTime();

               const Vector<BCRec>& gradient_bcs = getLevel(level).get_desc_lst()[Accel_Type].getBCs();

               for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                   amrex::InterpFromCoarseLevel(*potential_gradient_fine[n], time, *potential_gradient_coarse[n],
                                                0, 0, 1,
                                                parent->Geom(level-1), parent->Geom(level),
                                                gradient_phys_bc, 0, gradient_phys_bc, 0, parent->refRatio(level-1),
                                                gradient_interp, gradient_bcs, 0);
               }

           }

       } else {

            const Real cur_time = state[State_Type].curTime();
            if ( (level == lbase) && cur_time > 0.)
            {
                if (gravity->get_solver_kind() == "PoissonGrav") {

                    if (level == 0) {
                      gravity->update_max_density();
                    }

                    gravity->multilevel_potential_solve(level, new_finest);

                }

            }

        }

    }
#endif

    if (state[State_Type].hasOldData()) {

        MultiFab& S_old = get_old_data(State_Type);

        clean_state(
#ifdef MHD
                    get_old_data(Bfield_Type_x),
                    get_old_data(Bfield_Type_y),
                    get_old_data(Bfield_Type_z),
#endif
                    S_old, state[State_Type].prevTime(), S_old.nGrow());

    }

    if (state[State_Type].hasNewData()) {

        MultiFab& S_new = get_new_data(State_Type);

        clean_state(
#ifdef MHD
                    get_new_data(Bfield_Type_x),
                    get_new_data(Bfield_Type_y),
                    get_new_data(Bfield_Type_z),
#endif
                    S_new, state[State_Type].curTime(), S_new.nGrow());

    }
}

void
HydroSim::post_timestep (int cycle_index)
{

    amrex::ignore_unused(cycle_index);

    BL_PROFILE("HydroSim::post_timestep()");

    int finest_level = parent->finestLevel();

#ifdef RADIATION
    if (do_radiative_transfer && (level < finest_level)) {
        radiation->prepare_delayed_reflux(level);

        if (do_reflux) {
            radiation->reflux(level);
        }
    }
#endif

    if (do_reflux && level < parent->finestLevel()) {
        if (parent->subcyclingMode() != "None") {
            reflux(level, level+1, true);
        }
    }

    if (level < finest_level) {
        avgDown();
    }

#ifdef MHD
    MultiFab& magfield_x_new = get_new_data(Bfield_Type_x);
    MultiFab& magfield_y_new = get_new_data(Bfield_Type_y);
    MultiFab& magfield_z_new = get_new_data(Bfield_Type_z);
#endif

    MultiFab& S_new = get_new_data(State_Type);
    clean_state(
#ifdef MHD
                magfield_x_new, magfield_y_new, magfield_z_new,
#endif
                S_new, state[State_Type].curTime(), S_new.nGrow());

#ifdef DO_USER_POST_TIMESTEP

    user_post_timestep();

#endif

    if (level == 0)
    {
        int nstep = parent->levelSteps(0);
        Real level0_dt = parent->dtLevel(0);
        Real cumtime = parent->cumTime() + level0_dt;

        bool sum_step_hit = false;

        if (sum_step_period > 0) {

          if (nstep%sum_step_period == 0) {
            sum_step_hit = true;
          }

        }

        bool sum_time_hit = false;

        if (sum_time_period > 0.0) {

          const int num_per_old = static_cast<int>(std::floor((cumtime - level0_dt) / sum_time_period));
          const int num_per_new = static_cast<int>(std::floor((cumtime        ) / sum_time_period));

          if (num_per_old != num_per_new) {
            sum_time_hit = true;
          }

        }

        if (sum_step_hit || sum_time_hit) {
          tally_integrated_quantities();
        }

#ifdef GRAVITY
        if (track_center) {
          write_tracked_center();
        }
#endif
    }

#ifdef RADIATION

    if (level == 0) {
      print_energy_budget();
    }
#endif

#ifdef AMREX_PARTICLES
    if (TracerPC)
    {
        const int ncycle = parent->nCycle(level);
        if (cycle_index < ncycle || level == 0)
        {
            int ngrow = (level == 0) ? 0 : cycle_index;

            TracerPC->Redistribute(level, parent->finestLevel(), ngrow);

            StampParticles(ngrow+1);
        }
    }
#endif

    if (!hydrosim::stop_field_name.empty() && level == 0) {
        Real peak_field_value = std::numeric_limits<Real>::min();

        for (int lev = 0; lev <= parent->finestLevel(); ++lev) {
            auto mf = getLevel(lev).derive(hydrosim::stop_field_name, state[State_Type].curTime(), 0);
            peak_field_value = std::max(peak_field_value, mf->max(0));
        }

        int to_stop = (peak_field_value >= hydrosim::stop_field_threshold);

        amrex::ParallelDescriptor::ReduceIntMax(to_stop);

        if (to_stop) {

            amrex::Print() << std::endl
                           << "Ending simulation because we are above the user-specified stopping criterion threshold."
                           << std::endl;

            stopRequested = true;

            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::ofstream stop_flag_file;
                stop_flag_file.open("dump_and_stop", std::ofstream::out);
                stop_flag_file.close();
            }

            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::ofstream runDoneFile;
                runDoneFile.open("jobIsDone", std::ofstream::out);
                runDoneFile.close();
            }

        }

    }
}

void
HydroSim::postCoarseTimeStep (Real cumtime)
{
    BL_ASSERT(level == 0);
    AmrLevel::postCoarseTimeStep(cumtime);
#ifdef GRAVITY
    if (do_gravity_flag) {
        gravity->set_mass_shift(cumtime, false);
    }
#endif

}

int
HydroSim::okToContinue ()
{
    if (level > 0) {
      return 1;
    }

    int test = 1;

    if (stopRequested) {
      test = 0;
      if (ParallelDescriptor::IOProcessor()) {
        std::cout << " Signalling a stop of the run due to stopRequested = true." << std::endl;
      }
    }
    else if (parent->dtLevel(level) < min_dt_time_ratio * parent->cumTime()) {
      test = 0;
      if (ParallelDescriptor::IOProcessor()) {
        std::cout << " Signalling a stop of the run because dt < min_dt_time_ratio * time." << std::endl;
      }
    }

    return test;
}
