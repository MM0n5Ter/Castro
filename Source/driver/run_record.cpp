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
