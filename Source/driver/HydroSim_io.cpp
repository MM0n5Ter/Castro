
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
#include <HydroSim_io.H>
#include <AMReX_ParmParse.H>

#ifdef RADIATION
#include <Radiation.H>
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
}


void
HydroSim::restart (Amr&     papa,
                 istream& is,
                 bool     bReadSpecial)
{
    if (restart_format == -1) {
        if (ParallelDescriptor::IOProcessor()) {
            std::ifstream HydroSimHeaderFile;
            std::string FullPathHydroSimHeaderFile = papa.theRestartFile();
            FullPathHydroSimHeaderFile += "/CastroHeader";
            HydroSimHeaderFile.open(FullPathHydroSimHeaderFile.c_str(), std::ios::in);
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
        std::string FullPathTimestepFile = papa.theRestartFile();
        FullPathTimestepFile += "/dtHeader";
        timestepFile.open(FullPathTimestepFile.c_str(), std::ios::in);

        if (timestepFile.good()) {

            lastDtWriteLimited = 1;
            timestepFile >> lastDtBeforeWriteLimit;
            timestepFile.close();

        }

    }


    if (ParallelDescriptor::IOProcessor()) {

      std::ifstream VarNameListFile;
      std::string FullPathVarNameList = papa.theRestartFile();
      FullPathVarNameList += "/state_names.txt";
      VarNameListFile.open(FullPathVarNameList.c_str(), std::ios::in);

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
      std::string FullPathCpuTimeFile = parent->theRestartFile();
      FullPathCpuTimeFile += "/CPUtime";
      CpuTimeFile.open(FullPathCpuTimeFile.c_str(), std::ios::in);

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
        std::string FullPathPointMassFile = parent->theRestartFile();
        FullPathPointMassFile += "/point_mass";
        PointMassFile.open(FullPathPointMassFile.c_str(), std::ios::in);

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
        std::string FullPathSpinPeriodFile = parent->theRestartFile();
        FullPathSpinPeriodFile += "/Rotation";
        SpinPeriodFile.open(FullPathSpinPeriodFile.c_str(), std::ios::in);

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
                if (restart_grow_factor == 2) {
                   lo =   dlen/4    ;
                   hi = 3*dlen/4 - 1;
                } else if (restart_grow_factor == 3) {
                   lo =   (dlen)/3    ;
                   hi = 2*(dlen)/3 - 1;
                } else {
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

void
HydroSim::set_state_in_checkpoint (Vector<int>& state_in_checkpoint)
{
    for (int i=0; i<num_field_types; ++i) {
        state_in_checkpoint[i] = 1;
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
            std::string FullPathHydroSimHeaderFile = dir;
            FullPathHydroSimHeaderFile += "/CastroHeader";
            HydroSimHeaderFile.open(FullPathHydroSimHeaderFile.c_str(), std::ios::out);

            HydroSimHeaderFile << "Checkpoint version: " << current_version << std::endl;
            HydroSimHeaderFile.close();

            writeRunRecord(dir, write_seconds);

            std::ofstream VarNameListFile;
            std::string FullPathVarNameList = dir;
            FullPathVarNameList += "/state_names.txt";
            VarNameListFile.open(FullPathVarNameList.c_str(), std::ios::out);

            for (int n = 0; n < NUM_STATE; n++) {
              VarNameListFile << desc_lst[State_Type].name(n) << "\n";
            }
            VarNameListFile.close();
        }


        if (lastDtWriteLimited == 1) {

            std::ofstream timestepFile;
            std::string FullPathTimestepFile = dir;
            FullPathTimestepFile += "/dtHeader";
            timestepFile.open(FullPathTimestepFile.c_str(), std::ios::out);

            timestepFile << lastDtBeforeWriteLimit << std::endl;
            timestepFile.close();

        }

        {
            std::ofstream CpuTimeFile;
            std::string FullPathCpuTimeFile = dir;
            FullPathCpuTimeFile += "/CPUtime";
            CpuTimeFile.open(FullPathCpuTimeFile.c_str(), std::ios::out);

            CpuTimeFile << std::setprecision(17) << getCpuSeconds();
            CpuTimeFile.close();
        }

#ifdef GRAVITY
        if (use_central_mass) {

            std::ofstream PointMassFile;
            std::string FullPathPointMassFile = dir;
            FullPathPointMassFile += "/point_mass";
            PointMassFile.open(FullPathPointMassFile.c_str(), std::ios::out);

            PointMassFile << std::setprecision(17) << point_mass << std::endl;

            PointMassFile.close();

        }
#endif

#ifdef ROTATION
        if (do_frame_spin) {
            std::ofstream SpinPeriodFile;
            std::string FullPathSpinPeriodFile = dir;
            FullPathSpinPeriodFile += "/Rotation";
            SpinPeriodFile.open(FullPathSpinPeriodFile.c_str(), std::ios::out);

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

std::string
HydroSim::thePlotFileType () const
{
    static const std::string the_plot_file_type("HyperCLaw-V1.1");

    return the_plot_file_type;
}

void
HydroSim::setPlotVariables ()
{
  AmrLevel::setPlotVariables();


  for (int i = 0; i < desc_lst[Forcing_Type].nComp(); i++) {
    parent->deleteStatePlotVar(desc_lst[Forcing_Type].name(i));
  }

#ifdef REACTIONS
  if (time_scheme == SimplifiedIterCorrection) {
      for (int i = 0; i < desc_lst[Burn_Corr_Type].nComp(); i++) {
          parent->deleteStatePlotVar(desc_lst[Burn_Corr_Type].name(i));
      }
  }
#endif

}



void
HydroSim::writeRunRecord (const std::string& dir, const Real write_seconds)
{

  std::ofstream jobInfoFile;
  std::string FullPathJobInfoFile = dir;
  FullPathJobInfoFile += "/job_info";
  jobInfoFile.open(FullPathJobInfoFile.c_str(), std::ios::out);

  std::string PrettyLine = std::string(78, '=') + "\n";
  std::string OtherLine = std::string(78, '-') + "\n";
  std::string SkipSpace = std::string(8, ' ');

  jobInfoFile << PrettyLine;
  jobInfoFile << " HydroSim Job Information\n";
  jobInfoFile << PrettyLine;

  jobInfoFile << "job name: " << job_label << "\n\n";
  jobInfoFile << "inputs file: " << inputs_name << "\n\n";

  jobInfoFile << "number of MPI processes: " << ParallelDescriptor::NProcs() << "\n";
#ifdef AMREX_USE_OMP
  jobInfoFile << "number of threads:       " << omp_get_max_threads() << "\n";
#endif
  jobInfoFile << "\n";
  jobInfoFile << "hydro tile size:         " << hydro_tile_size << "\n";

  jobInfoFile << "\n";
#ifdef AMREX_USE_GPU
  jobInfoFile << "GPU time used since start of simulation (GPU-hours): " <<
    getCpuSeconds()/3600.0;
#else
  jobInfoFile << "CPU time used since start of simulation (CPU-hours): " <<
    getCpuSeconds()/3600.0;
#endif

  jobInfoFile << "\n\n";

  jobInfoFile << PrettyLine;
  jobInfoFile << " Plotfile Information\n";
  jobInfoFile << PrettyLine;

  const std::time_t now = time(nullptr);
  jobInfoFile << "output date / time: "
              << std::put_time(std::localtime(&now), "%c\n") << "\n";

  jobInfoFile << "output dir:         " << amrex::FileSystem::CurrentPath() << "\n";

  jobInfoFile << "I/O time (s):       " << write_seconds << "\n";

  jobInfoFile << "\n\n";

#ifdef AMREX_USE_GPU

  jobInfoFile << PrettyLine;
  jobInfoFile << "GPU Information:       " << "\n";
  jobInfoFile << PrettyLine;

  jobInfoFile << "GPU model name: " << Gpu::Device::deviceName() << "\n";
  jobInfoFile << "Number of GPUs used: " << Gpu::Device::numDevicesUsed() << "\n";

  jobInfoFile << "\n\n";
#endif

  jobInfoFile << PrettyLine;
  jobInfoFile << " Build Information\n";
  jobInfoFile << PrettyLine;

  jobInfoFile << "build date:    " << buildInfoGetBuildDate() << "\n";
  jobInfoFile << "build machine: " << buildInfoGetBuildMachine() << "\n";
  jobInfoFile << "build dir:     " << buildInfoGetBuildDir() << "\n";
  jobInfoFile << "AMReX dir:     " << buildInfoGetAMReXDir() << "\n";

  jobInfoFile << "\n";

  jobInfoFile << "make flags:    " << buildInfoGetMakeFlags() << "\n";

  jobInfoFile << "\n";

  jobInfoFile << "COMP:          " << buildInfoGetComp() << "\n";
  jobInfoFile << "COMP version:  " << buildInfoGetCompVersion() << "\n";

  jobInfoFile << "\n";

#ifdef AMREX_USE_CUDA
  jobInfoFile << "CUDA version:  " << buildInfoGetCUDAVersion() << "\n";
  jobInfoFile << "\n";
#endif

  jobInfoFile << "C++ compiler:  " << buildInfoGetCXXName() << "\n";
  jobInfoFile << "C++ flags:     " << buildInfoGetCXXFlags() << "\n";

  jobInfoFile << "\n";

  jobInfoFile << "Fortran comp:  " << buildInfoGetFName() << "\n";
  jobInfoFile << "Fortran flags: " << buildInfoGetFFlags() << "\n";

  jobInfoFile << "\n";

  jobInfoFile << "Link flags:    " << buildInfoGetLinkFlags() << "\n";
  jobInfoFile << "Libraries:     " << buildInfoGetLibraries() << "\n";

  jobInfoFile << "\n";

  for (int n = 1; n <= buildInfoGetNumModules(); n++) {
    jobInfoFile << buildInfoGetModuleName(n) << ": " << buildInfoGetModuleVal(n) << "\n";
  }

  jobInfoFile << "\n";

  const char* githash1 = buildInfoGetGitHash(1);
  const char* githash2 = buildInfoGetGitHash(2);
  const char* physics_lib_rev = buildInfoGetGitHash(3);
  if (strlen(githash1) > 0) {
    jobInfoFile << "HydroSim       git describe: " << githash1 << "\n";
  }
  if (strlen(githash2) > 0) {
    jobInfoFile << "AMReX        git describe: " << githash2 << "\n";
  }
  if (strlen(physics_lib_rev) > 0) {
    jobInfoFile << "Microphysics git describe: " << physics_lib_rev << "\n";
  }

  const char* build_rev_hash = buildInfoGetBuildGitHash();
  const char* build_rev_name = buildInfoGetBuildGitName();
  if (strlen(build_rev_hash) > 0){
    jobInfoFile << build_rev_name << " git describe: " << build_rev_hash << "\n";
  }

  jobInfoFile << "\n\n";


  jobInfoFile << PrettyLine;
  jobInfoFile << " Grid Information\n";
  jobInfoFile << PrettyLine;

  int f_lev = parent->finestLevel();

  for (int i = 0; i <= f_lev; i++)
    {
      jobInfoFile << " level: " << i << "\n";
      jobInfoFile << "   number of boxes = " << parent->numGrids(i) << "\n";
      jobInfoFile << "   maximum zones   = ";
      for (int n = 0; n < AMREX_SPACEDIM; n++)
        {
          jobInfoFile << parent->Geom(i).Domain().length(n) << " ";
        }
      jobInfoFile << "\n\n";
    }

  jobInfoFile << " Boundary conditions\n";
  Vector<int> lo_bc_flags(AMREX_SPACEDIM), hi_bc_flags(AMREX_SPACEDIM);
  ParmParse pp("hydrosim");
  pp.getarr("lo_bc",lo_bc_flags,0,AMREX_SPACEDIM);
  pp.getarr("hi_bc",hi_bc_flags,0,AMREX_SPACEDIM);


  constexpr std::array<std::string_view, 6> bc_labels{
      "interior", "inflow", "outflow",
      "symmetry", "slipwall", "noslipwall"};


  jobInfoFile << "   -x: " << bc_labels[lo_bc_flags[0]] << "\n";
  jobInfoFile << "   +x: " << bc_labels[hi_bc_flags[0]] << "\n";
  if (AMREX_SPACEDIM >= 2) {
    jobInfoFile << "   -y: " << bc_labels[lo_bc_flags[1]] << "\n";
    jobInfoFile << "   +y: " << bc_labels[hi_bc_flags[1]] << "\n";
  }
  if (AMREX_SPACEDIM == 3) {
    jobInfoFile << "   -z: " << bc_labels[lo_bc_flags[2]] << "\n";
    jobInfoFile << "   +z: " << bc_labels[hi_bc_flags[2]] << "\n";
  }

  jobInfoFile << "\n\n";

  jobInfoFile << " Domain geometry info\n";

  jobInfoFile << "     center: " << problem::center[0] << " , " << problem::center[1] << " , " << problem::center[2] << "\n";
  jobInfoFile << "\n";

  jobInfoFile << "     geometry.is_periodic: ";
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    jobInfoFile << geom.isPeriodic(idir) << " ";
  }
  jobInfoFile << "\n";

  jobInfoFile << "     geometry.coord_sys:   " << geom.Coord() << "\n";

  jobInfoFile << "     geometry.prob_lo:     ";
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    jobInfoFile << geom.ProbLo(idir) << " ";
  }
  jobInfoFile << "\n";

  jobInfoFile << "     geometry.prob_hi:     ";
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    jobInfoFile << geom.ProbHi(idir) << " ";
  }
  jobInfoFile << "\n";

  jobInfoFile << "     amr.n_cell:           ";
  const int*  domain_lo = geom.Domain().loVect();
  const int*  domain_hi = geom.Domain().hiVect();
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    jobInfoFile << domain_hi[idir] - domain_lo[idir] + 1 << " ";
  }
  jobInfoFile << "\n";

  int max_level = parent->maxLevel();
  jobInfoFile << "     amr.max_level:        " << max_level << "\n";

  jobInfoFile << "     amr.ref_ratio:        ";
  for (int lev = 1; lev <= max_level; lev++) {
    IntVect ref_ratio = parent->refRatio(lev-1);
    jobInfoFile << ref_ratio[0] << " ";
  }
  jobInfoFile << "\n";

  jobInfoFile << "     amr.n_error_buf:      ";
  for (int lev = 1; lev <= max_level; lev++) {
    int err_buffer_cells = parent->nErrorBuf(lev-1);
    jobInfoFile << err_buffer_cells << " ";
  }
  jobInfoFile << "\n";

  jobInfoFile << "     amr.regrid_int:       ";
  for (int lev = 1; lev <= max_level; lev++) {
    int regrid_interval = parent->regridInt(lev-1);
    jobInfoFile << regrid_interval << " ";
  }
  jobInfoFile << "\n";

  jobInfoFile << "     amr.blocking_factor:  ";
  for (int lev = 1; lev <= max_level; lev++) {
    IntVect bf = parent->blockingFactor(lev-1);
    jobInfoFile << bf[0] << " ";
  }
  jobInfoFile << "\n";

  jobInfoFile << "     amr.max_grid_size:    ";
  for (int lev = 1; lev <= max_level; lev++) {
    IntVect mgs = parent->maxGridSize(lev-1);
    jobInfoFile << mgs[0] << " ";
  }
  jobInfoFile << "\n\n";

  jobInfoFile << "     amr.subcycling_mode: " << parent->subcyclingMode();

  jobInfoFile << "\n\n";



  int mlen = 20;

  jobInfoFile << PrettyLine;
  jobInfoFile << " Species Information\n";
  jobInfoFile << PrettyLine;

  jobInfoFile <<
    std::setw(6) << "index" << SkipSpace <<
    std::setw(mlen+1) << "name" << SkipSpace <<
    std::setw(7) << "A" << SkipSpace <<
    std::setw(7) << "Z" << "\n";
  jobInfoFile << OtherLine;

  for (int i = 0; i < NumSpec; i++)
    {
      jobInfoFile <<
        std::setw(6) << i << SkipSpace <<
        std::setw(mlen+1) << std::setfill(' ') << short_spec_names_cxx[i] << SkipSpace <<
        std::setw(7) << aion[i] << SkipSpace <<
        std::setw(7) << zion[i] << "\n";
    }
  jobInfoFile << "\n\n";

#ifdef STARLIB
  jobInfoFile << PrettyLine;
  jobInfoFile << " Deviates for StarLib Rates \n";
  jobInfoFile << PrettyLine;

  jobInfoFile <<
  std::setw(6) << "index" << SkipSpace <<
  std::setw(mlen+11) << "name" << SkipSpace <<
  std::setw(7) << "deviate" <<  "\n";
  jobInfoFile << OtherLine;

  int idx = 1;
  for (int i = 1; i <= Rates::NumRates ; i++) {
    if (Rates::rate_names[i].ends_with("_starlib")) {
      jobInfoFile <<
      std::setw(6) << idx << SkipSpace <<
      std::setw(mlen+11) << Rates::rate_names[i] << SkipSpace <<
      std::setw(7) << starlib::prand(idx) << "\n";
      idx++;
    }
  }
  jobInfoFile << "\n\n";

  AMREX_ALWAYS_ASSERT(starlib::NumStarLibRates == idx-1);
#endif

  jobInfoFile << PrettyLine;
  jobInfoFile << " Inputs File Parameters\n";
  jobInfoFile << PrettyLine;

#include <hydrosim_job_info_tests.H>
#ifdef AMREX_PARTICLES
#include <particles_job_info_tests.H>
#endif

#ifdef GRAVITY
  gravity->report_run_settings(jobInfoFile);
#endif
#ifdef DIFFUSION
  diffusion->report_run_settings(jobInfoFile);
#endif

#include <prob_job_info_tests.H>

#include <extern_job_info_tests.H>

  jobInfoFile.close();

}


void
HydroSim::writeBuildRecord ()
{
  std::string PrettyLine = std::string(78, '=') + "\n";

  std::cout << PrettyLine;
  std::cout << " HydroSim Build Information\n";
  std::cout << PrettyLine;

  std::cout << "build date:    " << buildInfoGetBuildDate() << "\n";
  std::cout << "build machine: " << buildInfoGetBuildMachine() << "\n";
  std::cout << "build dir:     " << buildInfoGetBuildDir() << "\n";
  std::cout << "AMReX dir:     " << buildInfoGetAMReXDir() << "\n";

  std::cout << "\n";

  std::cout << "make flags:    " << buildInfoGetMakeFlags() << "\n";

  std::cout << "\n";

  std::cout << "COMP:          " << buildInfoGetComp() << "\n";
  std::cout << "COMP version:  " << buildInfoGetCompVersion() << "\n";

  std::cout << "\n";

#ifdef AMREX_USE_CUDA
  std::cout << "CUDA version:  " << buildInfoGetCUDAVersion() << "\n";
  std::cout << "\n";
#endif

  std::cout << "C++ compiler:  " << buildInfoGetCXXName() << "\n";
  std::cout << "C++ flags:     " << buildInfoGetCXXFlags() << "\n";

  std::cout << "\n";

  std::cout << "Fortran comp:  " << buildInfoGetFName() << "\n";
  std::cout << "Fortran flags: " << buildInfoGetFFlags() << "\n";

  std::cout << "\n";

  std::cout << "Link flags:    " << buildInfoGetLinkFlags() << "\n";
  std::cout << "Libraries:     " << buildInfoGetLibraries() << "\n";

  std::cout << "\n";

  for (int n = 1; n <= buildInfoGetNumModules(); n++) {
    std::cout << buildInfoGetModuleName(n) << ": " << buildInfoGetModuleVal(n) << "\n";
  }

  std::cout << "\n";

  const char* githash1 = buildInfoGetGitHash(1);
  const char* githash2 = buildInfoGetGitHash(2);
  const char* physics_lib_rev = buildInfoGetGitHash(3);
  if (strlen(githash1) > 0) {
    std::cout << "HydroSim       git describe: " << githash1 << "\n";
  }
  if (strlen(githash2) > 0) {
    std::cout << "AMReX        git describe: " << githash2 << "\n";
  }
  if (strlen(physics_lib_rev) > 0) {
    std::cout << "Microphysics git describe: " << physics_lib_rev << "\n";
  }

  const char* build_rev_hash = buildInfoGetBuildGitHash();
  const char* build_rev_name = buildInfoGetBuildGitName();
  if (strlen(build_rev_hash) > 0){
    std::cout << build_rev_name << " git describe: " << build_rev_hash << "\n";
  }

  std::cout << "\n\n";
}

void
HydroSim::writePlotFile(const std::string& dir,
                      ostream& os,
                      VisMF::How how)
{
  writeSnapshot(dir, os, how, 0);
}


void
HydroSim::writeSmallPlotFile (const std::string& dir,
                            ostream&       os,
                            VisMF::How     how)
{
  writeSnapshot(dir, os, how, 1);
}


void
HydroSim::writeSnapshot(const std::string& dir,
                       ostream& os,
                       VisMF::How how,
                       const int is_small)
{
#ifdef AMREX_PARTICLES
  ParticleOutputFile(dir);
#endif

    std::vector<std::pair<int,int> > plot_var_map;
    for (int typ = 0; typ < desc_lst.size(); typ++) {
      for (int comp = 0; comp < desc_lst[typ].nComp(); comp++) {
        if (((parent->isStatePlotVar(desc_lst[typ].name(comp)) && is_small == 0) ||
             (parent->isStateSmallPlotVar(desc_lst[typ].name(comp)) && is_small == 1)) &&
            desc_lst[typ].getType() == IndexType::TheCellType()) {
            plot_var_map.emplace_back(typ, comp);
        }
      }
    }

    int num_derive = 0;
    std::list<std::string> derive_names;
    const std::list<DeriveRec>& dlist = derive_lst.dlist();

    for (const auto & dd : dlist) {

        if ((parent->isDerivePlotVar(dd.name()) && is_small == 0) ||
            (parent->isDeriveSmallPlotVar(dd.name()) && is_small == 1))
        {
#ifdef AMREX_PARTICLES
            if (dd.name() == "particle_count" ||
                dd.name() == "total_particle_count")
            {
                if (HydroSim::theTracerPC())
                {
                    derive_names.push_back(dd.name());
                    num_derive = num_derive + dd.numDerive();
                }
            } else
#endif
            {
               derive_names.push_back(dd.name());
               num_derive = num_derive + dd.numDerive();
            }
        }
    }

    int n_data_items = static_cast<int>(plot_var_map.size()) + num_derive;

#ifdef RADIATION
    if (Radiation::num_plot_vars > 0) n_data_items += Radiation::num_plot_vars;
#endif

#ifdef REACTIONS
#ifndef TRUE_SDC
    if (keep_burn_weights) {
        n_data_items += static_cast<int>(HydroSim::burn_split_names.size());
    }
#endif
#endif

    Real cur_time = state[State_Type].curTime();

    if (level == 0 && ParallelDescriptor::IOProcessor())
    {
        os << thePlotFileType() << '\n';

        if (n_data_items == 0) {
          amrex::Error("Must specify at least one valid data item to plot");
        }

        os << n_data_items << '\n';

        for (const auto& [typ, comp] : plot_var_map)
        {
            os << desc_lst[typ].name(comp) << '\n';
        }

        for (const auto &name : derive_names)
        {
            const DeriveRec* rec = derive_lst.get(name);
            if (rec->numDerive() > 1) {
                for (int i = 0; i < rec->numDerive(); ++i) {
                    os << std::format("{}_{}\n", rec->variableName(0), i);
                }
            }
            else {
                os << rec->variableName(0) << '\n';
            }
        }

#ifdef RADIATION
        for (int i=0; i<Radiation::num_plot_vars; ++i) {
          os << Radiation::plot_var_names[i] << '\n';
        }
#endif

#ifdef REACTIONS
#ifndef TRUE_SDC
        if (keep_burn_weights) {
            for (const auto& name: HydroSim::burn_split_names) {
                os << name << '\n';
            }
        }
#endif
#endif

        os << AMREX_SPACEDIM << '\n';
        os << parent->cumTime() << '\n';
        int f_lev = parent->finestLevel();
        os << f_lev << '\n';
        for (int i = 0; i < AMREX_SPACEDIM; i++) {
            os << geom.ProbLo(i) << ' ';
        }
        os << '\n';
        for (int i = 0; i < AMREX_SPACEDIM; i++) {
            os << geom.ProbHi(i) << ' ';
        }
        os << '\n';
        for (int i = 0; i < f_lev; i++) {
          os << parent->refRatio(i)[0] << ' ';
        }
        os << '\n';
        for (int i = 0; i <= f_lev; i++) {
          os << parent->Geom(i).Domain() << ' ';
        }
        os << '\n';
        for (int i = 0; i <= f_lev; i++) {
          os << parent->levelSteps(i) << ' ';
        }
        os << '\n';
        for (int i = 0; i <= f_lev; i++)
        {
            for (int k = 0; k < AMREX_SPACEDIM; k++) {
              os << parent->Geom(i).CellSize()[k] << ' ';
            }
            os << '\n';
        }
        os << (int) geom.Coord() << '\n';
        os << "0\n";

#ifdef RADIATION
        if (do_radiative_transfer && Radiation::groups_enabled) {
          std::ofstream energy_group_file;
          std::string FullPathEnergyGroupFile = dir;
          FullPathEnergyGroupFile += "/RadiationGroups";
          energy_group_file.open(FullPathEnergyGroupFile.c_str(), std::ios::out);

          radiation->dump_group_table(energy_group_file);

          energy_group_file.close();
        }
#endif
    }
    static const std::string BaseName = "/Cell";
    std::string Level = std::format("Level_{}", level);
    std::string FullPath = dir;
    if (!FullPath.ends_with('/')) {
      FullPath += '/';
    }
    FullPath += Level;
    if (ParallelDescriptor::IOProcessor()) {
        if (!amrex::UtilCreateDirectory(FullPath, 0755)) {
            amrex::CreateDirectoryFailed(FullPath);
        }
    }
    ParallelDescriptor::Barrier();

    if (ParallelDescriptor::IOProcessor())
    {
        os << level << ' ' << grids.size() << ' ' << cur_time << '\n';
        os << parent->levelSteps(level) << '\n';

        for (int i = 0; i < grids.size(); ++i)
        {
            RealBox gridloc = RealBox(grids[i],geom.CellSize(),geom.ProbLo());
            for (int n = 0; n < AMREX_SPACEDIM; n++) {
              os << gridloc.lo(n) << ' ' << gridloc.hi(n) << '\n';
            }
        }
        if (n_data_items > 0)
        {
            std::string PathNameInHeader = Level;
            PathNameInHeader += BaseName;
            os << PathNameInHeader << '\n';
        }
    }
    int       cnt   = 0;
    const int nGrow = 0;
    MultiFab  plotMF(grids,dmap,n_data_items,nGrow);
    MultiFab* this_dat = nullptr;
    for (const auto& [typ, comp] : plot_var_map) {
        this_dat = &state[typ].newData();
        MultiFab::Copy(plotMF, *this_dat, comp, cnt, 1, nGrow);
        cnt++;
    }
    if (!dlist.empty())
    {
        for (const auto & dd : dlist) {

            if ((parent->isDerivePlotVar(dd.name()) && is_small == 0) ||
                (parent->isDeriveSmallPlotVar(dd.name()) && is_small == 1)) {

                auto derived_mf = derive(dd.variableName(0), cur_time, nGrow);
                MultiFab::Copy(plotMF, *derived_mf, 0, cnt, dd.numDerive(), nGrow);
                cnt = cnt + dd.numDerive();
            }
        }
    }

#ifdef RADIATION
    if (Radiation::num_plot_vars > 0) {
        MultiFab::Copy(plotMF,*(radiation->plot_fields[level]),0,cnt,Radiation::num_plot_vars,0);
        cnt += Radiation::num_plot_vars;
    }
#endif

#ifdef REACTIONS
#ifndef TRUE_SDC
    if (keep_burn_weights) {
        MultiFab::Copy(plotMF, getLevel(level).burn_split_data, 0, cnt, static_cast<int>(HydroSim::burn_split_names.size()), 0);
        cnt += static_cast<int>(HydroSim::burn_split_names.size());
    }
#endif
#endif

    std::string TheFullPath = FullPath;
    TheFullPath += BaseName;

    const Real io_begin_clock = ParallelDescriptor::second();

    if (amrex::AsyncOut::UseAsyncOut()) {
        VisMF::AsyncWrite(std::move(plotMF),TheFullPath);
    } else {
        VisMF::Write(plotMF,TheFullPath,how,true);
    }

    const Real write_seconds = ParallelDescriptor::second() - io_begin_clock;

    if (level == 0 && ParallelDescriptor::IOProcessor()) {
        writeRunRecord(dir, write_seconds);
    }

#ifdef GRAVITY
    if (use_central_mass && level == 0) {

        std::ofstream PointMassFile;
        std::string FullPathPointMassFile = dir;
        FullPathPointMassFile += "/point_mass";
        PointMassFile.open(FullPathPointMassFile.c_str(), std::ios::out);

        PointMassFile << std::setprecision(17) << point_mass << std::endl;

        PointMassFile.close();

    }
#endif

}
