#ifndef WIN32
#include <unistd.h>
#endif

#include <iomanip>
#include <iostream>
#include <format>
#include <string>
#include <ctime>

#include <AMReX_Utility.H>
#include <HydroSim.H>
#include <run_output.H>
#include <AMReX_ParmParse.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef GRAVITY
#include <Gravity.H>
#endif

#ifdef DIFFUSION
#include <diffusion_operator.H>
#endif

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <user_initialize_state_data.H>
#include <user_checkpoint.H>
#include <user_restart.H>
#include <AMReX_buildInfo.H>

using std::string;
using namespace amrex;

namespace
{
    int restart_format = -1;
    int current_version = 12;

    std::string checkpoint_entry(const std::string& dir, const char* leaf)
    {
        return dir + "/" + leaf;
    }
}

void
HydroSim::checkPoint(const std::string& dir,
                   std::ostream&  os,
                   VisMF::How     how,
                   bool  )
{

  const Real io_begin_clock = ParallelDescriptor::second();

  AmrLevel::checkPoint(dir, os, how, dump_old);

  const Real write_seconds = ParallelDescriptor::second() - io_begin_clock;

#ifdef RADIATION
  if (do_radiative_transfer) {
    radiation->checkPoint(level, dir, os, how);
  }
#endif

#ifdef AMREX_PARTICLES
  ParticleSaveRestart(dir);
#endif

  if (level == 0 && ParallelDescriptor::IOProcessor())
    {
        {
            std::ofstream HydroSimHeaderFile;
            HydroSimHeaderFile.open(checkpoint_entry(dir, "CastroHeader").c_str(), std::ios::out);

            HydroSimHeaderFile << "Checkpoint version: " << current_version << std::endl;
            HydroSimHeaderFile.close();

            writeRunRecord(dir, write_seconds);

            std::ofstream VarNameListFile;
            VarNameListFile.open(checkpoint_entry(dir, "state_names.txt").c_str(), std::ios::out);

            for (int n = 0; n < NUM_STATE; n++) {
              VarNameListFile << desc_lst[State_Type].name(n) << "\n";
            }
            VarNameListFile.close();
        }

        if (lastDtWriteLimited == 1) {

            std::ofstream timestepFile;
            timestepFile.open(checkpoint_entry(dir, "dtHeader").c_str(), std::ios::out);

            timestepFile << lastDtBeforeWriteLimit << std::endl;
            timestepFile.close();

        }

        {
            std::ofstream CpuTimeFile;
            CpuTimeFile.open(checkpoint_entry(dir, "CPUtime").c_str(), std::ios::out);

            CpuTimeFile << std::setprecision(17) << getCpuSeconds();
            CpuTimeFile.close();
        }

#ifdef GRAVITY
        if (use_central_mass) {

            std::ofstream PointMassFile;
            PointMassFile.open(checkpoint_entry(dir, "point_mass").c_str(), std::ios::out);

            PointMassFile << std::setprecision(17) << point_mass << std::endl;

            PointMassFile.close();

        }
#endif

#ifdef ROTATION
        if (do_frame_spin) {
            std::ofstream SpinPeriodFile;
            SpinPeriodFile.open(checkpoint_entry(dir, "Rotation").c_str(), std::ios::out);

            SpinPeriodFile << std::scientific;
            SpinPeriodFile.precision(19);

            SpinPeriodFile << std::setw(30) << hydrosim::spin_period << std::endl;

            SpinPeriodFile.close();
        }
#endif

        {
            user_checkpoint(dir);
        }
    }

}

void
HydroSim::set_state_in_checkpoint (Vector<int>& state_in_checkpoint)
{
    for (int i=0; i<num_field_types; ++i) {
        state_in_checkpoint[i] = 1;
    }
}

void
HydroSim::restart (Amr&     papa,
                 istream& is,
                 bool     bReadSpecial)
{
    if (restart_format == -1) {
        if (ParallelDescriptor::IOProcessor()) {
            std::ifstream HydroSimHeaderFile;
            HydroSimHeaderFile.open(checkpoint_entry(papa.theRestartFile(), "CastroHeader").c_str(), std::ios::in);
            if (HydroSimHeaderFile.good()) {
                char foo[256];
                HydroSimHeaderFile.getline(foo, 256, ':');
                HydroSimHeaderFile >> restart_format;
                HydroSimHeaderFile.close();
            } else {
                restart_format = 0;
            }
        }
        ParallelDescriptor::Bcast(&restart_format, 1, ParallelDescriptor::IOProcessorNumber());
    }

    if (restart_format != current_version) {
        amrex::Error("Checkpoint format incompatible with current code");
    }

    if (ParallelDescriptor::IOProcessor()) {

        std::ifstream timestepFile;
        timestepFile.open(checkpoint_entry(papa.theRestartFile(), "dtHeader").c_str(), std::ios::in);

        if (timestepFile.good()) {

            lastDtWriteLimited = 1;
            timestepFile >> lastDtBeforeWriteLimit;
            timestepFile.close();

        }

    }

    if (ParallelDescriptor::IOProcessor()) {

      std::ifstream VarNameListFile;
      VarNameListFile.open(checkpoint_entry(papa.theRestartFile(), "state_names.txt").c_str(), std::ios::in);

      if (VarNameListFile.good()) {
        std::string var;
        for (int n = 0; n < NUM_STATE; n++) {
          VarNameListFile >> var;
          if (var != desc_lst[State_Type].name(n)) {
            amrex::Error("state variables do not agree");
          }
        }
        VarNameListFile.close();
      }
    }

    ParallelDescriptor::Bcast(&lastDtWriteLimited, 1, ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&lastDtBeforeWriteLimit, 1, ParallelDescriptor::IOProcessorNumber());

    AmrLevel::restart(papa,is,bReadSpecial);

    buildMetrics();

    allocFabs();

    if (level == 0 && ParallelDescriptor::IOProcessor())
    {
      std::ifstream CpuTimeFile;
      CpuTimeFile.open(checkpoint_entry(parent->theRestartFile(), "CPUtime").c_str(), std::ios::in);

      if (CpuTimeFile.good()) {

          CpuTimeFile >> priorCpuSeconds;
          CpuTimeFile.close();

      }

      std::cout << "read CPU time: " << priorCpuSeconds << "\n";

    }

#ifdef GRAVITY
    if (use_central_mass && level == 0)
    {

        std::ifstream PointMassFile;
        PointMassFile.open(checkpoint_entry(parent->theRestartFile(), "point_mass").c_str(), std::ios::in);

        if (PointMassFile.good()) {
            PointMassFile >> point_mass;
            PointMassFile.close();
        }

    }
#endif

#ifdef ROTATION
    if (do_frame_spin && level == 0)
    {
        std::ifstream SpinPeriodFile;
        SpinPeriodFile.open(checkpoint_entry(parent->theRestartFile(), "Rotation").c_str(), std::ios::in);

        if (SpinPeriodFile.is_open()) {
            SpinPeriodFile >> hydrosim::spin_period;
            amrex::Print() << "  Based on the checkpoint, setting the rotational period to "
                           << std::setprecision(7) << std::fixed << hydrosim::spin_period << " s.\n";
            SpinPeriodFile.close();
        }
    }
#endif

    if (level == 0)
    {
        std::string dir = parent->theRestartFile();

        user_restart(dir);
    }

    if ( (restart_grow_factor > 1) && (parent->maxLevel() < 1) )
    {
       std::cout << "restart_grow_factor is " << restart_grow_factor << std::endl;
       std::cout << "max_level is " << parent->maxLevel() << std::endl;
       amrex::Error("Must have max_level > 0 if doing special restart with restart_grow_factor");
    }

    if (restart_grow_factor > 1 && level == 0)
    {
       if (verbose && ParallelDescriptor::IOProcessor()) {
         std::cout << "Doing special restart with restart_grow_factor " << restart_grow_factor << std::endl;
       }
       MultiFab& S_new = get_new_data(State_Type);

       Box pre_grow_domain;
       if (body_at_center == 0) {
          pre_grow_domain = amrex::coarsen(geom.Domain(),restart_grow_factor);
       } else if (body_at_center == 1) {

          Box domain(geom.Domain());
          int lo=0, hi=0;
          if (geom.IsRZ()) {
             if (restart_grow_factor != 2) {
                amrex::Abort("Must have restart_grow_factor = 2");
             }

             int d = 0;
             int dlen =  domain.size()[d];
             lo = 0;
             hi = dlen/2;
             pre_grow_domain.setSmall(d,lo);
             pre_grow_domain.setBig(d,hi);

#if AMREX_SPACEDIM >= 2
             d = 1;
             dlen =  domain.size()[d];
             lo =   dlen/4    ;
             hi = 3*dlen/4 - 1;
             pre_grow_domain.setSmall(d,lo);
             pre_grow_domain.setBig(d,hi);
#endif

          } else {
             for (int d = 0; d < AMREX_SPACEDIM; d++)
             {
                int dlen =  domain.size()[d];
                switch (restart_grow_factor) {
                case 2:
                   lo =   dlen/4    ;
                   hi = 3*dlen/4 - 1;
                   break;
                case 3:
                   lo =   (dlen)/3    ;
                   hi = 2*(dlen)/3 - 1;
                   break;
                default:
                   amrex::Abort("Must have restart_grow_factor = 2 or 3");
                }
                pre_grow_domain.setSmall(d,lo);
                pre_grow_domain.setBig(d,hi);
             }
          }
       } else {
          if (ParallelDescriptor::IOProcessor()) {
             std::cout << "... invalid value of body_at_center: " << body_at_center << std::endl;
          }
          amrex::Abort();
       }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
       {
           const Box& bx = mfi.tilebox();

           if (!pre_grow_domain.contains(bx)) {
               auto s = S_new[mfi].array();
               auto geomdata = geom.data();

#ifdef RNG_STATE_INIT
               amrex::Error("Error: random initialization not yet supported with grown factor");
#else
               amrex::ParallelFor(bx,
               [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
               {
                   user_initialize_state_data(i, j, k, s, geomdata);
               });
#endif
           }
       }
    }

    if (restart_grow_factor > 1 && level == 1) {
      getLevel(0).avgDown();
    }

#if AMREX_SPACEDIM == 2
    if (prev_theta_cells > 0 && level == 0 && geom.Coord() == 2) {
        if (verbose && ParallelDescriptor::IOProcessor()) {
            std::cout << "Doing special restart with extended theta domain "
                      << "where the old level 0 domain has " << prev_theta_cells
                      << " cells along theta-dir" << std::endl;
        }

        MultiFab& S_new = get_new_data(State_Type);

        Box pre_grow_domain = geom.Domain();
        pre_grow_domain.setBig(1, prev_theta_cells-1);

        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            if (!pre_grow_domain.contains(bx)) {
                auto s = S_new[mfi].array();
                auto geomdata = geom.data();
                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (j >= prev_theta_cells) {
                        user_initialize_state_data(i, j, k, s, geomdata);
                    }
                });
            }
        }
    }
#endif

#ifdef GRAVITY
    if (do_gravity_flag && level == 0) {
       BL_ASSERT(gravity == nullptr);
       gravity = std::make_unique<Gravity>(parent, parent->finestLevel(), &phys_bc, URHO);
    }
#endif

#ifdef DIFFUSION
    if (level == 0) {
       BL_ASSERT(diffusion == nullptr);
       diffusion = std::make_unique<Diffusion>(parent, &phys_bc);
    }
#endif

#ifdef RADIATION
    if (do_radiative_transfer) {
      if (radiation == nullptr) {
        int rad_skip_init = 1;
        radiation = std::make_unique<Radiation>(parent, this, rad_skip_init);
      }
      radiation->regrid(level, grids, dmap);
      radiation->restart(level, grids, dmap, parent->theRestartFile(), is);

      rad_operator.reset(new RadTransportSolver(parent, level, grids, dmap));
    }
#endif

    if (restart_time_override > -1.e199) {

        if (!parent->RegridOnRestart()) {
          amrex::Error("restart_time_override only makes sense when amr.regrid_on_restart=1");
        }

        const Real cur_time = get_state_data(State_Type).curTime();
        const Real prev_time = get_state_data(State_Type).prevTime();
        const Real dt = cur_time - prev_time;

        parent->setStartTime(restart_time_override);
        parent->setCumTime(restart_time_override);

        for (int n = 0; n < num_field_types; ++n) {
            StateData& present_state = get_state_data(n);
            present_state.setOldTimeLevel(restart_time_override-dt);
            present_state.setNewTimeLevel(restart_time_override   );
        }

    }

    if (restart_step_override > -1) {

        if (!parent->RegridOnRestart()) {
          amrex::Error("restart_step_override only makes sense when amr.regrid_on_restart=1");
        }

        parent->setLevelSteps(level, restart_step_override);
        parent->setLevelCount(level, restart_step_override);

    }
}
