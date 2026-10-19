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

#include <user_initialize.H>
#include <user_initialize_state_data.H>
#ifdef MHD
#include <user_initialize_mhd_data.H>
#endif
#ifdef RADIATION
#include <user_initialize_rad_data.H>
#endif
#include <user_tagging.H>

#include <ambient.H>
#include <hydrosim_limits.H>

#include <solver_constants.H>

using namespace amrex;

bool         HydroSim::stopRequested = false;

#ifdef REACTIONS
std::vector<std::string> HydroSim::burn_split_names;
#endif

std::vector<std::string> HydroSim::tag_field_names;
std::vector<int> HydroSim::tag_field_ghosts;
int          HydroSim::num_tag_fields_default = 0;
int          HydroSim::radial_grow   = 1;
BCRec        HydroSim::phys_bc;

int          HydroSim::lastDtWriteLimited = 0;
Real         HydroSim::lastDtBeforeWriteLimit = 0.0;

params_t     HydroSim::params;

Real         HydroSim::cells_advanced_total = 0.0;

Vector<std::string> HydroSim::forcing_names;

Vector<AMRErrorTag> HydroSim::tag_criteria;

Vector<std::unique_ptr<std::fstream>> HydroSim::run_logs;
Vector<std::unique_ptr<std::fstream>> HydroSim::user_run_logs;

#ifdef TRUE_SDC
int          HydroSim::COLLOC_NODES;
Vector<Real> HydroSim::node_dt_frac;
Vector<Real> HydroSim::node_quad_weights;
#endif

#ifdef GRAVITY
std::unique_ptr<Gravity> HydroSim::gravity;
#endif

#ifdef DIFFUSION
std::unique_ptr<Diffusion> HydroSim::diffusion;
#endif

#ifdef RADIATION

std::unique_ptr<Radiation> HydroSim::radiation;
#endif


#if AMREX_SPACEDIM == 1
#ifndef AMREX_USE_GPU
IntVect      HydroSim::hydro_tile_size(1024);
#else
IntVect      HydroSim::hydro_tile_size(1048576);
#endif
IntVect      HydroSim::untiled_size(1024);
#elif AMREX_SPACEDIM == 2
#ifndef AMREX_USE_GPU
IntVect      HydroSim::hydro_tile_size(1024,16);
#else
IntVect      HydroSim::hydro_tile_size(1048576,1048576);
#endif
IntVect      HydroSim::untiled_size(1024,1024);
#else
#ifndef AMREX_USE_GPU
IntVect      HydroSim::hydro_tile_size(1024,16,16);
#else
IntVect      HydroSim::hydro_tile_size(1048576,1048576,1048576);
#endif
IntVect      HydroSim::untiled_size(1024,1024,1024);
#endif

int          HydroSim::sweep_tile_tuned = 0;
Long         HydroSim::largest_tuned_box_pts = 0;

Real         HydroSim::priorCpuSeconds = 0.0;

Real         HydroSim::startCpuSeconds = 0.0;

int          HydroSim::Forcing_Type_Index = -1;
int          HydroSim::num_field_types = 0;

int          HydroSim::user_init_pending = 0;


void
HydroSim::variableCleanUp ()
{
#ifdef GRAVITY
  if (gravity != nullptr) {
    if (verbose > 1 && ParallelDescriptor::IOProcessor()) {
      std::cout << "Deleting gravity in variableCleanUp..." << '\n';
    }
    gravity.reset();
  }
#endif

#ifdef DIFFUSION
  if (diffusion != nullptr) {
    if (verbose > 1 && ParallelDescriptor::IOProcessor()) {
      std::cout << "Deleting diffusion in variableCleanUp..." << '\n';
    }
    diffusion.reset();
  }
#endif

#ifdef RADIATION
  if (radiation != nullptr) {
      int report = (verbose || radiation->verbose);
      if (report && ParallelDescriptor::IOProcessor()) {
          std::cout << "Deleting radiation in variableCleanUp..." << '\n';
      }
      radiation.reset();
      if (report && ParallelDescriptor::IOProcessor()) {
          std::cout << "                                        done" << std::endl;
      }
  }
#endif

#ifdef AMREX_PARTICLES
  TracerPC.reset();
#endif

    desc_lst.clear();

    eos_finalize();

}

void
HydroSim::read_params ()
{
    static bool done = false;

    if (done) {
        return;
    }

    done = true;

    load_compiled_params();

    ParmParse pp("hydrosim");
    ParmParse ppa("amr");

    using namespace hydrosim;


    Vector<int> lo_bc(AMREX_SPACEDIM), hi_bc(AMREX_SPACEDIM);
    pp.getarr("lo_bc",lo_bc,0,AMREX_SPACEDIM);
    pp.getarr("hi_bc",hi_bc,0,AMREX_SPACEDIM);
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        phys_bc.setLo(i,lo_bc[i]);
        phys_bc.setHi(i,hi_bc[i]);
    }

    const Geometry& default_geom = DefaultGeometry();

    if (default_geom.isAnyPeriodic())
    {
        for (int dir = 0; dir<AMREX_SPACEDIM; dir++)
        {
            if (default_geom.isPeriodic(dir))
            {
                if (lo_bc[dir] != amrex::PhysBCType::interior)
                {
                    std::cerr << "HydroSim::read_params:periodic in direction "
                              << dir
                              << " but low BC is not Interior\n";
                    amrex::Error();
                }
                if (hi_bc[dir] != amrex::PhysBCType::interior)
                {
                    std::cerr << "HydroSim::read_params:periodic in direction "
                              << dir
                              << " but high BC is not Interior\n";
                    amrex::Error();
                }
            }
        }
    }
    else
    {
        for (int dir=0; dir<AMREX_SPACEDIM; dir++)
        {
            if (lo_bc[dir] == amrex::PhysBCType::interior)
            {
                std::cerr << "HydroSim::read_params:interior bc in direction "
                          << dir
                          << " but not periodic\n";
                amrex::Error();
            }
            if (hi_bc[dir] == amrex::PhysBCType::interior)
            {
                std::cerr << "HydroSim::read_params:interior bc in direction "
                          << dir
                          << " but not periodic\n";
                amrex::Error();
            }
        }
    }

    if ( default_geom.IsRZ() && (lo_bc[0] != amrex::PhysBCType::symmetry) ) {
        std::cerr << "ERROR:HydroSim::read_params: must set r=0 boundary condition to Symmetry for r-z\n";
        amrex::Error();
    }

#if (AMREX_SPACEDIM == 1)
    if ( default_geom.IsSPHERICAL() )
    {
      if ( (lo_bc[0] != amrex::PhysBCType::symmetry) && (default_geom.ProbLo(0) == 0.0) )
      {
        std::cerr << "ERROR:HydroSim::read_params: must set r=0 boundary condition to Symmetry for spherical\n";
        amrex::Error();
      }
    }
#elif (AMREX_SPACEDIM == 2)
    if ( default_geom.IsSPHERICAL() )
      {
        if ( (lo_bc[1] != amrex::PhysBCType::symmetry) && (default_geom.ProbLo(1) == 0.0) )
        {
          std::cerr << "ERROR:HydroSim::read_params: must set theta=0 boundary condition to Symmetry for spherical\n";
          amrex::Error();
        }

        if ( (hi_bc[1] != amrex::PhysBCType::symmetry) && (std::abs(default_geom.ProbHi(1) - std::numbers::pi) <= 1.e-4_rt) )
        {
          std::cerr << "ERROR:HydroSim::read_params: must set theta=pi boundary condition to Symmetry for spherical\n";
          amrex::Error();
        }

        if ( (default_geom.ProbLo(1) < 0.0_rt) && (default_geom.ProbHi(1) > std::numbers::pi) )
        {
          amrex::Abort("ERROR:HydroSim::read_params: Theta must be within [0, Pi] for spherical coordinate system in 2D");
        }

        Vector<int> n_cell(AMREX_SPACEDIM);
        ppa.queryarr("n_cell",n_cell,0,AMREX_SPACEDIM);
        Real ghost_extent = static_cast<Real>(NUM_GROW) * (default_geom.ProbHi(0) - default_geom.ProbLo(0)) / static_cast<Real>(n_cell[0]);
        if ( default_geom.ProbLo(0) < ghost_extent )
        {
          amrex::Abort("ERROR:HydroSim::read_params: R-min must be large enough so ghost cells doesn't extend to negative R");
        }
      }
#elif (AMREX_SPACEDIM == 3)
    if ( default_geom.IsRZ() )
      {
        amrex::Abort("We don't support cylindrical coordinate systems in 3D");
      }
    else if ( default_geom.IsSPHERICAL() )
      {
        amrex::Abort("We don't support spherical coordinate systems in 3D");
      }
#endif

#ifdef HYBRID_MOMENTUM

    if (riemann_method == 2) {
        amrex::Abort("CONTACT_FLUX Riemann solver unsupported when using hybrid momentum.");
    }
#endif


    if (restart_grow_factor < 1) {
      amrex::Error("restart_grow_factor must be integer >= 1");
    }

    if (cfl <= 0.0 || cfl > 1.0) {
      amrex::Error("Invalid CFL factor; must be between zero and one.");
    }

#ifdef AMREX_USE_GPU
    if (time_scheme == IterativeCorrection) {
        amrex::Error("SDC is currently not enabled on GPUs.");
    }
#endif


#ifndef SIMPLIFIED_SDC
    if (time_scheme == SimplifiedIterCorrection) {
        amrex::Error("Simplified SDC currently requires USE_SIMPLIFIED_SDC=TRUE when compiling.");
    }
#else
    if (time_scheme != SimplifiedIterCorrection) {
        amrex::Error("When building with USE_SIMPLIFIED_SDC=TRUE, only simplified SDC can be used.");
    }
#endif

#ifdef TRUE_SDC
    int level_cap_tmp;
    ppa.query("max_level", level_cap_tmp);
    if (level_cap_tmp > 0) {
        amrex::Error("True SDC does not work with AMR.");
    }
#endif

#ifndef TRUE_SDC
    if (time_scheme == IterativeCorrection) {
        amrex::Error("True SDC currently requires USE_TRUE_SDC=TRUE when compiling.");
    }
#else
    if (time_scheme != IterativeCorrection) {
        amrex::Error("When building with USE_TRUE_SDC=TRUE, only true SDC can be used.");
    }
#endif

#ifndef AMREX_USE_GPU

#ifdef RADIATION
    if (shock_flux_switch == 1) {
        amrex::Error("ERROR: hybrid Riemann not supported for radiation");
    }

    if (riemann_method > 0) {
        amrex::Error("ERROR: only the CGF Riemann solver is supported for radiation");
    }
#endif

#if AMREX_SPACEDIM == 1
    if (riemann_method > 1) {
        amrex::Error("ERROR: CONTACT_FLUX not implemented for 1-d");
    }
#endif

#ifndef SHOCK_VAR
   if (skip_burning_in_shocks != 0) {
       amrex::Error("ERROR: skip_burning_in_shocks requires compiling with USE_SHOCK_VAR=TRUE");
   }
#endif

    if (riemann_method == 1) {
        if (shock_star_max_iters > solver_constants::STAR_HISTORY_DEPTH) {
            amrex::Error("shock_star_max_iters > STAR_HISTORY_DEPTH");
        }

        if (riemann_search_mode == 2 && shock_star_max_iters < 5) {
            amrex::Error("Error: need shock_star_max_iters >= 5 to do a bisection search on secant iteration failure.");
        }
    }
#endif

    if (shock_flux_switch && AMREX_SPACEDIM == 1)
      {
        std::cerr << "shock_flux_switch only implemented in 2- and 3-d\n";
        amrex::Error();
      }

    if (shock_flux_switch && (default_geom.IsSPHERICAL() || default_geom.IsRZ() ))
      {
        std::cerr << "shock_flux_switch should only be used for Cartesian coordinates\n";
        amrex::Error();
      }

    if (!do_hydro) {
      do_reflux = false;
    }

    if (max_dt < constant_dt)
      {
        std::cerr << "cannot have max_dt < constant_dt\n";
        amrex::Error();
      }

    if (change_max <= 1.0)
    {
        std::cerr << "change_max must be greater than 1.0\n";
        amrex::Error();
    }

    if (linear_interp_limiter == 2) {
        if (default_geom.IsSPHERICAL()) {
            amrex::Error("linear_interp_limiter == 2 is not currently implemented for spherical geometries");
        }
        else if (default_geom.IsRZ()) {
            amrex::Error("lim_limit_state_interp == 2 is not currently implemented for cylindrical geometries");
        }
    }

    std::string subcycling_mode;
    ppa.query("subcycling_mode", subcycling_mode);
    if (regrid_each_step == 1 && subcycling_mode == "None") {
        amrex::Error("hydrosim.regrid_each_step == 1 is not consistent with amr.subcycling_mode = None.");
    }

#ifdef AMREX_PARTICLES
    load_particle_config();
#endif

#ifdef RADIATION

    if (do_radiative_transfer == 1) {
      Radiation::load_static_settings();
    }

    if (do_radiative_transfer && time_scheme != CornerTrace) {
        amrex::Error("Radiation is currently only supported for CTU time advancement.");
    }
#endif

#ifdef ROTATION
    if (do_frame_spin == 1) {
      if (spin_period <= 0.0) {
        std::cerr << "Error:HydroSim::Rotation enabled but rotation period less than zero\n";
        amrex::Error();
      }
    }

    if (default_geom.IsRZ() == 1) {
        spin_axis = 2;
    }
#if (AMREX_SPACEDIM == 1)
    if (do_frame_spin == 1) {
        std::cerr << "ERROR:HydroSim::Rotation not implemented in 1d\n";
        amrex::Error();
    }
#endif

    AMREX_ALWAYS_ASSERT(spin_axis >= 1 && spin_axis <= 3);

#endif

#ifdef SPONGE
    if (do_damping_zone == 1) {
        if (damping_timescale <= 0.0) {
            amrex::Error("If using the sponge, the damping_timescale must be positive.");
        }
        if (amrex::max(damping_upper_radius, damping_upper_density, damping_upper_pressure) < 0.0) {
            amrex::Error("If using the sponge, at least one of the upper radius, density, "
                         "or pressure must be non-negative.");
        }
        if (amrex::max(damping_lower_radius, damping_lower_density, damping_lower_pressure) < 0.0) {
            amrex::Error("If using the sponge, at least one of the lower radius, density, "
                         "or pressure must be non-negative.");
        }
    }
#endif


#if !defined(GRAVITY) || !defined(ROTATION)
   if (do_equilibrium_start) {
       amrex::Error("SCF initial model construction is only permitted if USE_GRAV=TRUE and USE_ROTATION=TRUE at compile time.");
   }
#endif

   StateDescriptor::setBndryFuncThreadSafety(boundary_callback_threadsafe);


   if (sum_step_period > 0 && ParallelDescriptor::IOProcessor()) {

       run_logs.resize(4);

       run_logs[0] = std::make_unique<std::fstream>();
       run_logs[0]->open("grid_diag.out", std::ios::out | std::ios::app);
       if (!run_logs[0]->good()) {
           amrex::FileOpenFailed("grid_diag.out");
       }

       run_logs[1] = std::make_unique<std::fstream>();
#ifdef GRAVITY
       run_logs[1]->open("gravity_diag.out", std::ios::out | std::ios::app);
       if (!run_logs[1]->good()) {
           amrex::FileOpenFailed("gravity_diag.out");
       }
#endif

       run_logs[2] = std::make_unique<std::fstream>();
       run_logs[2]->open("species_diag.out", std::ios::out | std::ios::app);
       if (!run_logs[2]->good()) {
           amrex::FileOpenFailed("species_diag.out");
       }

       run_logs[3] = std::make_unique<std::fstream>();
       run_logs[3]->open("amr_diag.out", std::ios::out | std::ios::app);
       if (!run_logs[3]->good()) {
           amrex::FileOpenFailed("amr_diag.out");
       }

   }

    Vector<int> tilesize(AMREX_SPACEDIM);
    if (pp.queryarr("hydro_tile_size", tilesize, 0, AMREX_SPACEDIM))
    {
        for (int i=0; i<AMREX_SPACEDIM; i++) {
          hydro_tile_size[i] = tilesize[i];
        }
    }


    Amr::setComputeNewDtOnRegrid(true);


    Vector<std::string> refine_criteria;
    ppa.queryarr("refine_criteria", refine_criteria, 0, ppa.countval("refine_criteria"));

    for (const auto & refine_criterion : refine_criteria) {
        std::string refine_prefix = "amr.refine." + refine_criterion;

        ParmParse ppr(refine_prefix);

        AMRErrorTagInfo info;

        if (ppr.countval("start_time") > 0) {
            Real min_time;
            ppr.get("start_time", min_time);
            info.SetMinTime(min_time);
        }
        if (ppr.countval("end_time") > 0) {
            Real max_time;
            ppr.get("end_time", max_time);
            info.SetMaxTime(max_time);
        }
        if (ppr.countval("max_level") > 0) {
            int max_level;
            ppr.get("max_level", max_level);
            BL_ASSERT(max_level < LEVEL_LIMIT);
            info.SetMaxLevel(max_level);
        } else {
            info.SetMaxLevel(LEVEL_LIMIT);
        }
        if (ppr.countval("volume_weighting") > 0) {
            int volume_weighting;
            ppr.get("volume_weighting", volume_weighting);
            info.SetVolumeWeighting(volume_weighting);
        }
        if (ppr.countval("derefine") > 0) {
            int derefine;
            ppr.get("derefine", derefine);
            info.SetDerefine(derefine);
        }

        if (ppr.countval("value_greater")) {
            Vector<Real> value;
            ppr.getarr("value_greater", value, 0, ppr.countval("value_greater"));
            std::string field;
            ppr.get("field_name", field);
            tag_criteria.emplace_back(value, AMRErrorTag::GREATER, field, info);
        }
        else if (ppr.countval("value_less")) {
            Vector<Real> value;
            ppr.getarr("value_less", value, 0, ppr.countval("value_less"));
            std::string field;
            ppr.get("field_name", field);
            tag_criteria.emplace_back(value, AMRErrorTag::LESS, field, info);
        }
        else if (ppr.countval("gradient")) {
            Vector<Real> value;
            ppr.getarr("gradient", value, 0, ppr.countval("gradient"));
            std::string field;
            ppr.get("field_name", field);
            tag_criteria.emplace_back(value, AMRErrorTag::GRAD, field, info);
        }
        else if (ppr.countval("relative_gradient")) {
            Vector<Real> value;
            ppr.getarr("relative_gradient", value, 0, ppr.countval("relative_gradient"));
            std::string field;
            ppr.get("field_name", field);
            tag_criteria.emplace_back(value, AMRErrorTag::RELGRAD, field, info);
        }
        else {
            amrex::Abort("Unrecognized refinement indicator for " + refine_criterion);
        }
    }

}

HydroSim::HydroSim ()
    :
    prev_state(num_field_types)
{
}

HydroSim::HydroSim (Amr&            papa,
                int             lev,
                const Geometry& level_geom,
                const BoxArray& bl,
                const DistributionMapping& dm,
                Real            time)
    :
    AmrLevel(papa,lev,level_geom,bl,dm,time),
    prev_state(num_field_types)
{
    BL_PROFILE("HydroSim::HydroSim()");

    MultiFab::RegionTag level_region_tag(std::format("AmrLevel_Level_{}", lev));

    buildMetrics();

    allocFabs();


    if (do_hydro == 1 && do_reflux == 1) {
        for (int ilev = 0; ilev <= parent->maxLevel(); ++ilev) {
            if (parent->nErrorBuf(ilev) == 0) {
                amrex::Error("n_error_buf = 0 is unsupported when using hydro.");
            }
        }
    }

    for (int k = 0; k < num_field_types; k++) {
      MultiFab& data = get_new_data(k);
      data.setVal(0.0, data.nGrow());
    }

#ifdef GRAVITY

    if (do_gravity_flag == 1) {
      if (gravity == nullptr) {
        gravity = std::make_unique<Gravity>(parent, parent->finestLevel(), &phys_bc, URHO);
      }

      if (!level_geom.isAllPeriodic() && gravity != nullptr)
      {
         int grid_points_1d = count_points_1d();


#if (AMREX_SPACEDIM == 1)
         grid_points_1d += 2 * NUM_GROW;
#endif

         gravity->set_grid_points_gravity(grid_points_1d);
      }

      gravity->attach_level(level,this,volume,area.data());

      if (verbose && level == 0 &&  ParallelDescriptor::IOProcessor()) {
        std::cout << "Setting the gravity type to " << gravity->get_solver_kind() << std::endl;
      }
   }

#endif


#ifdef DIFFUSION
      if (diffusion == nullptr) {
        diffusion = std::make_unique<Diffusion>(parent, &phys_bc);
      }

      diffusion->attach_level(level,this,volume,area.data());
#endif

#ifdef RADIATION
    if (do_radiative_transfer) {
      if (radiation == nullptr) {
        radiation = std::make_unique<Radiation>(parent, this);
        global::radiation_instance = radiation.get();
      }
      radiation->regrid(level, grids, dmap);

      rad_operator = std::make_unique<RadTransportSolver>(parent, level, grids, dmap);
    }
#endif


    check_runparams();

}

HydroSim::~HydroSim ()
{
#ifdef RADIATION
    if (radiation != nullptr) {
      radiation->close(level);
    }
#endif
}

void
HydroSim::buildMetrics ()
{
    BL_PROFILE("HydroSim::buildMetrics()");

    const int box_count = static_cast<int>(grids.size());

    radius.resize(box_count);

    const Real* dx = geom.CellSize();

    for (int i = 0; i < box_count; i++)
    {
        const Box& b = grids[i];
        int ilo      = b.smallEnd(0)-radial_grow;
        int ihi      = b.bigEnd(0)+radial_grow;
        int len      = ihi - ilo + 1;

        radius[i].resize(len);

        Real* rad = radius[i].dataPtr();

        if (Geom().IsCartesian())
        {
            for (int j = 0; j < len; j++)
            {
                rad[j] = 1.0;
            }
        }
        else
        {
            RealBox gridloc = RealBox(grids[i],geom.CellSize(),geom.ProbLo());

            const Real xlo = gridloc.lo(0) + (0.5 - radial_grow)*dx[0];

            for (int j = 0; j < len; j++)
            {
                rad[j] = xlo + j*dx[0];
            }
        }
    }

    volume.clear();
    volume.define(grids,dmap,1,NUM_GROW);
    geom.GetVolume(volume);

    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        area[dir].clear();
        area[dir].define(getEdgeBoxArray(dir),dmap,1,NUM_GROW);
        geom.GetFaceArea(area[dir],dir);
    }
    for (int dir = AMREX_SPACEDIM; dir < 3; dir++)
    {
        area[dir].clear();
        area[dir].define(grids, dmap, 1, 0);
        area[dir].setVal(0.0);
    }

#if (AMREX_SPACEDIM <= 2)
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        dLogFaceMeasure[dir].clear();
        geom.GetDLogA(dLogFaceMeasure[dir],grids,dmap,dir,NUM_GROW);
    }

#if (AMREX_SPACEDIM == 1)
    dLogFaceMeasure[1].clear();
    dLogFaceMeasure[1].define(grids, dmap, 1, 0);
    dLogFaceMeasure[1].setVal(0.0);
#endif

#endif

    wall_clock_start = 0.0;
}


void
HydroSim::allocFabs()
{
    BL_PROFILE("HydroSim::allocFabs()");

    fluxes.resize(3);

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      fluxes[dir] = std::make_unique<MultiFab>(MultiFab(getEdgeBoxArray(dir), dmap, NUM_STATE, 0));
    }

    for (int dir = AMREX_SPACEDIM; dir < 3; ++dir) {
      fluxes[dir] = std::make_unique<MultiFab>(MultiFab(get_new_data(State_Type).boxArray(), dmap, NUM_STATE, 0));
    }

    mass_face_flux.resize(3);

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      mass_face_flux[dir] = std::make_unique<MultiFab>(MultiFab(getEdgeBoxArray(dir), dmap, 1, 0));
    }

    for (int dir = AMREX_SPACEDIM; dir < 3; ++dir) {
      mass_face_flux[dir] = std::make_unique<MultiFab>(MultiFab(get_new_data(State_Type).boxArray(), dmap, 1, 0));
    }

#if (AMREX_SPACEDIM <= 2)
    if (!Geom().IsCartesian()) {
      Pressure_radial.define(getEdgeBoxArray(0), dmap, 1, 0);
    }
#endif

#ifdef RADIATION
    if (Radiation::rad_hydro_coupled) {
        rad_face_flux.resize(AMREX_SPACEDIM);
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            rad_face_flux[dir] = std::make_unique<MultiFab>(MultiFab(getEdgeBoxArray(dir), dmap, Radiation::nGroups, 0));
        }
    }
#endif

    if (do_reflux && level > 0) {

        flux_reg.define(grids, dmap, crse_ratio, level, NUM_STATE);
        flux_reg.setVal(0.0);

#if (AMREX_SPACEDIM < 3)
        if (!Geom().IsCartesian()) {
            pressure_flux_reg.define(grids, dmap, crse_ratio, level, 1);
            pressure_flux_reg.setVal(0.0);
        }
#endif

#ifdef RADIATION
        if (Radiation::rad_hydro_coupled) {
            rad_edge_reg.define(grids, dmap, crse_ratio, level, Radiation::nGroups);
            rad_edge_reg.setVal(0.0);
        }
#endif

#ifdef GRAVITY
        if (do_gravity_flag && gravity->get_solver_kind() == "PoissonGrav" && gravity->NoSync() == 0) {
            pot_flux_reg.define(grids, dmap, crse_ratio, level, 1);
            pot_flux_reg.setVal(0.0);
        }
#endif

    }


#ifdef REACTIONS
    if (keep_burn_weights) {
        if (hydrosim::time_scheme == CornerTrace) {
            burn_split_data.define(grids, dmap, 2, 0);
        }
        else if (hydrosim::time_scheme == SimplifiedIterCorrection) {
            burn_split_data.define(grids, dmap, deferred_iters+1, 0);
        }

        burn_split_data.setVal(0.0);
    }
#endif


    if (do_reflux) {

        flux_coarse_scale = -1.0;
        flux_refined_scale = 1.0;



#if (AMREX_SPACEDIM == 1)
        pressure_coarse_scale = -1.0;
        pressure_refined_scale = 1.0;
#elif (AMREX_SPACEDIM == 2)
        pressure_coarse_scale = -1.0;
        pressure_refined_scale = 1.0 / crse_ratio[1];
#endif

    }

    post_step_regrid = 0;

    lastStepDt = 1.e200;

    if (user_init_pending == 0) {


      user_init_pending = 1;


      user_initialize();

    }

}

void
HydroSim::setTimeLevel (Real time,
                      Real dt_old,
                      Real dt_new)
{
    AmrLevel::setTimeLevel(time,dt_old,dt_new);
}


void
HydroSim::initData ()
{
    BL_PROFILE("HydroSim::initData()");

#if AMREX_SPACEDIM > 1
    const Real* dx  = geom.CellSize();
#endif
    MultiFab& S_new = get_new_data(State_Type);
    Real cur_time   = state[State_Type].curTime();

    S_new.setVal(0.);

    if (! allow_anisotropic_cells) {

#if (AMREX_SPACEDIM == 2)
        const Real SMALL = 1.e-13;
        if (std::abs(dx[0] - dx[1]) > SMALL*dx[0])
          {
            amrex::Abort("We don't support dx != dy");
          }
#elif (AMREX_SPACEDIM == 3)
        const Real SMALL = 1.e-13;
        if ( (std::abs(dx[0] - dx[1]) > SMALL*dx[0]) || (std::abs(dx[0] - dx[2]) > SMALL*dx[0]) )
          {
            amrex::Abort("We don't support dx != dy != dz");
          }
#endif

    }

    if (verbose && ParallelDescriptor::IOProcessor()) {
      std::cout << "Initializing the data at level " << level << std::endl;
    }

#ifdef MHD
   MultiFab& magfield_x_new   = get_new_data(Bfield_Type_x);
   magfield_x_new.setVal(0.0);

   MultiFab& magfield_y_new   = get_new_data(Bfield_Type_y);
   magfield_y_new.setVal(0.0);

   MultiFab& magfield_z_new  =  get_new_data(Bfield_Type_z);
   magfield_z_new.setVal(0.0);

#endif


    Gpu::Device::profilerStop();

#ifdef RADIATION
    MultiFab &radiation_new = get_new_data(Radiation_Type);
    radiation_new.setVal(0.);
#endif

#ifdef REACTIONS
    MultiFab &reaction_new = get_new_data(Burning_Type);
    reaction_new.setVal(0.);
#endif

#ifdef REACTIONS
   if (time_scheme == SimplifiedIterCorrection) {
       MultiFab& reaction_src_new = get_new_data(Burn_Corr_Type);
       reaction_src_new.setVal(0.0, reaction_src_new.nGrow());
   }
#endif

#ifdef LOWMACH_INIT
    LOWMACH_init();
#else
    {

#ifdef MHD
       magfield_x_new.setVal(0.0);
       magfield_y_new.setVal(0.0);
       magfield_z_new.setVal(0.0);

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

          auto geomdata = geom.data();

          auto magfield_x_arr = magfield_x_new.array(mfi);
          auto magfield_y_arr = magfield_y_new.array(mfi);
          auto magfield_z_arr = magfield_z_new.array(mfi);

          const Box& face_box_x = mfi.nodaltilebox(0);

          amrex::ParallelFor(face_box_x,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              user_initialize_mhd_data(i, j, k, magfield_x_arr, 0, geomdata);
          });

          const Box& face_box_y = mfi.nodaltilebox(1);

          amrex::ParallelFor(face_box_y,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              user_initialize_mhd_data(i, j, k, magfield_y_arr, 1, geomdata);
          });

          const Box& face_box_z = mfi.nodaltilebox(2);

          amrex::ParallelFor(face_box_z,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              user_initialize_mhd_data(i, j, k, magfield_z_arr, 2, geomdata);
          });

       }

#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
       {
          const Box& box = mfi.tilebox();

          auto s = S_new[mfi].array();
          auto geomdata = geom.data();

#ifdef RNG_STATE_INIT
          amrex::ParallelForRNG(box,
          [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
          {
              user_initialize_state_data(i, j, k, s, geomdata, engine);
          });
#else
          amrex::ParallelFor(box,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              user_initialize_state_data(i, j, k, s, geomdata);
          });
#endif
       }


#ifdef MHD
      add_bfield_energy(magfield_x_new, magfield_y_new, magfield_z_new, S_new);

      check_div_bfield(magfield_x_new, magfield_y_new, magfield_z_new, S_new);

#endif

       computeTemp(
#ifdef MHD
                   magfield_x_new, magfield_y_new, magfield_z_new,
#endif
                   S_new, cur_time, 0);

       ReduceOps<ReduceOpSum, ReduceOpSum> reduce_op;
       ReduceData<int, int> reduce_data(reduce_op);
       using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
       {
           const Box& bx = mfi.tilebox();

           auto cons_state_arr = S_new.array(mfi);

           Real ltemp_floor = small_temp;
           Real ldensity_floor = small_dens;

           reduce_op.eval(bx, reduce_data,
           [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
           {
               int temp_failed = 0;
               if (cons_state_arr(i,j,k,UTEMP) < ltemp_floor * 1.001) {
                   temp_failed = 1;
               }

               int density_failed = 0;
               if (cons_state_arr(i,j,k,URHO) < ldensity_floor * 1.001) {
                   density_failed = 1;
               }

               return {temp_failed, density_failed};
           });

       }

       ReduceTuple hv = reduce_data.value();
       int init_temp_failed   = amrex::get<0>(hv);
       int init_density_failed = amrex::get<1>(hv);

       if (init_density_failed != 0) {
         amrex::Error("Error: initial data has rho <~ small_dens");
       }

       if (init_temp_failed != 0) {
         amrex::Error("Error: initial data has T <~ small_temp");
       }

#ifdef HYBRID_MOMENTUM

       linear_to_split_momentum(S_new, 0);
#endif


#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const Box& bx = mfi.tilebox();

         auto cons_state_arr = S_new.array(mfi);

         amrex::ParallelFor(bx,
         [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
         {
           Real species_sum = 0.0_rt;
           for (int n = 0; n < NumSpec; n++) {
             species_sum += cons_state_arr(i,j,k,UFS+n);
           }
           if (std::abs(cons_state_arr(i,j,k,URHO) - species_sum) > 1.e-8_rt * cons_state_arr(i,j,k,URHO)) {
#ifndef AMREX_USE_GPU
             std::cout << "Sum of (rho X)_i vs rho at (i,j,k): "
                       << i << " " << j << " " << k << " "
                       << species_sum << " " << cons_state_arr(i,j,k,URHO) << std::endl;
#endif
             amrex::Error("Error: failed check of initial species summing to 1");
           }
         });
       }

#ifdef TRUE_SDC
       if (init_cell_averaged == 0) {

         reconcile_energy(
#ifdef MHD
                            magfield_x_new, magfield_y_new,magfield_z_new,
#endif
                            S_new);


#ifndef AMREX_USE_GPU
         if (sdc_accuracy_order == 4) {
           Sborder.define(grids, dmap, NUM_STATE, NUM_GROW);
           AmrLevel::FillPatch(*this, Sborder, NUM_GROW, cur_time, State_Type, 0, NUM_STATE);

           auto domain_lo = geom.Domain().loVect3d();
           auto domain_hi = geom.Domain().hiVect3d();

           FArrayBox tmp;

           for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
             {
               const Box& box = mfi.validbox();

               tmp.resize(box, 1, The_Async_Arena());
               auto tmp_arr = tmp.array();

               build_fourth_in_place(box, Sborder.array(mfi), tmp_arr, domain_lo, domain_hi);
             }

           MultiFab::Copy(S_new, Sborder, 0, 0, NUM_STATE, 0);
           Sborder.clear();
         }
#endif
       } else {

         Sborder.define(grids, dmap, NUM_STATE, NUM_GROW);
         AmrLevel::FillPatch(*this, Sborder, NUM_GROW, cur_time, State_Type, 0, NUM_STATE);

         auto domain_lo = geom.Domain().loVect3d();
         auto domain_hi = geom.Domain().hiVect3d();

         FArrayBox tmp;

         for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
           {
             const Box& box = mfi.growntilebox(2);

             tmp.resize(box, 1, The_Async_Arena());
             auto tmp_arr = tmp.array();

             build_center_state_in_place(box, Sborder.array(mfi), tmp_arr, domain_lo, domain_hi);
           }

         for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
           {
             const Box& box = mfi.growntilebox(1);

             auto cons_state_arr = Sborder.array(mfi);

             amrex::ParallelFor(box,
             [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
             {

               Real rhoInv = 1.0_rt / cons_state_arr(i,j,k,URHO);
               Real u = cons_state_arr(i,j,k,UMX) * rhoInv;
               Real v = cons_state_arr(i,j,k,UMY) * rhoInv;
               Real w = cons_state_arr(i,j,k,UMZ) * rhoInv;

               eos_re_t eos_state;
               eos_state.rho = cons_state_arr(i,j,k,URHO);
               eos_state.T = cons_state_arr(i,j,k,UTEMP);
               eos_state.e = cons_state_arr(i,j,k,UEDEN) * rhoInv - 0.5_rt * (u*u + v*v + w*w);
               for (int n = 0; n < NumSpec; n++) {
                 eos_state.xn[n] = cons_state_arr(i,j,k,UFS+n) * rhoInv;
               }
#if NAUX_NET > 0
               for (int n = 0; n < NumAux; n++) {
                 eos_state.aux[n] = cons_state_arr(i,j,k,UFX+n) * rhoInv;
               }
#endif

               eos(eos_input_re, eos_state);

               cons_state_arr(i,j,k,UTEMP) = eos_state.T;

               cons_state_arr(i,j,k,UEINT) = eos_state.rho * eos_state.e;
             });
           }

         for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
           {
             const Box& box = mfi.validbox();

             tmp.resize(box, 1, The_Async_Arena());
             auto tmp_arr = tmp.array();

             build_fourth_in_place(box, Sborder.array(mfi), tmp_arr, domain_lo, domain_hi);
           }

         MultiFab::Copy(S_new, Sborder, UEINT, UEINT, 1, 0);
         MultiFab::Copy(S_new, Sborder, UTEMP, UTEMP, 1, 0);
         Sborder.clear();

       }
#else
       reconcile_energy(
#ifdef MHD
                            magfield_x_new, magfield_y_new,magfield_z_new,
#endif
                            S_new);
#endif


       int ng = S_new.nGrow();

       if (ng > 0) {
         AmrLevel::FillPatch(*this, S_new, ng, cur_time, State_Type, 0, S_new.nComp());
       }
    }

    clean_state(
#ifdef MHD
                    magfield_x_new, magfield_y_new, magfield_z_new,
#endif
                    S_new, cur_time, S_new.nGrow());


#ifdef RADIATION
    if (do_radiative_transfer) {
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
      for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
          const Box& box = mfi.tilebox();

          auto r = radiation_new[mfi].array();
          auto geomdata = geom.data();

          GpuArray<Real, NGROUPS+1> nu_edge_pass = {0.0};
          GpuArray<Real, NGROUPS> nu_center_pass = {0.0};
          GpuArray<Real, NGROUPS> nu_width_pass = {0.0};
#if NGROUPS > 1
          for (int g = 0; g <= NGROUPS; g++) {
              nu_edge_pass[g] = radiation->xnu[g];
          }
          for (int g = 0; g < NGROUPS; g++) {
              nu_center_pass[g] = radiation->freq_center[g];
              nu_width_pass[g] = radiation->freq_width[g];
          }
#endif

          amrex::ParallelFor(box,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {

              user_initialize_rad_data(i, j, k, r, nu_edge_pass, nu_center_pass, nu_width_pass, geomdata);
          });
      }
    }
#endif

#endif


#ifdef GRAVITY
    MultiFab& gravity_accel_new = get_new_data(Accel_Type);
    gravity_accel_new.setVal(0.);

    MultiFab& phi_new = get_new_data(Potential_Type);
    phi_new.setVal(0.);
#endif

    MultiFab& source_terms_new = get_new_data(Forcing_Type);
    source_terms_new.setVal(0., source_terms_new.nGrow());

#ifdef AMREX_PARTICLES
    if (level == 0) {
      init_particles();
    }
#endif

    Gpu::Device::profilerStart();

    if (verbose && ParallelDescriptor::IOProcessor()) {
      std::cout << "Done initializing the level " << level << " data " << std::endl;
    }
}

void
HydroSim::init (AmrLevel &old)
{
    BL_PROFILE("HydroSim::init(old)");

    auto* oldlev = dynamic_cast<HydroSim*>(&old);

    Real dt_new    = parent->dtLevel(level);
    Real cur_time  = oldlev->state[State_Type].curTime();
    Real prev_time = oldlev->state[State_Type].prevTime();
    Real dt_old    = cur_time - prev_time;
    setTimeLevel(cur_time,dt_old,dt_new);

    for (int s = 0; s < num_field_types; ++s) {
        MultiFab& state_field = get_new_data(s);
        FillPatch(old, state_field, state_field.nGrow(), cur_time, s, 0, state_field.nComp());
        if (oldlev->state[s].hasOldData()) {
            if (!state[s].hasOldData()) {
                state[s].allocOldData();
            }
            MultiFab& old_state_field = get_old_data(s);
            FillPatch(old, old_state_field, old_state_field.nGrow(), prev_time, s, 0, old_state_field.nComp());
        }
    }


    iteration = oldlev->iteration;
    sub_pass = oldlev->sub_pass;

    sub_cycles = oldlev->sub_cycles;
    dt_cycle_step = oldlev->dt_cycle_step;
    dt_level_step = oldlev->dt_level_step;

    retain_prev_field = oldlev->retain_prev_field;

    retry_active = oldlev->retry_active;

}

void
HydroSim::init ()
{
    BL_PROFILE("HydroSim::init()");

    Real dt        = parent->dtLevel(level);
    Real cur_time  = getLevel(level-1).state[State_Type].curTime();
    Real prev_time = getLevel(level-1).state[State_Type].prevTime();

    Real dt_old = (cur_time - prev_time) / static_cast<Real>(parent->MaxRefRatio(level-1));

    Real time = cur_time;


    if (getLevel(level-1).post_step_regrid) {
      time = prev_time;
    }

    setTimeLevel(time,dt_old,dt);

    for (int s = 0; s < num_field_types; ++s) {
        MultiFab& state_field = get_new_data(s);
        FillCoarsePatch(state_field, 0, time, s, 0, state_field.nComp(), state_field.nGrow());
    }
}

Real
HydroSim::initialTimeStep ()
{
    Real start_dt_local  = 0.0;

    if (first_step_dt > 0.0)
    {
       start_dt_local = first_step_dt;
    }
    else
    {
       start_dt_local = init_dt_shrink*estTimeStep();
    }

    return start_dt_local;
}

Real
HydroSim::estTimeStep (int use_new_state)
{
    BL_PROFILE("HydroSim::estTimeStep()");

    if (constant_dt > 0.0) {
      return constant_dt;
    }

    Real estdt = max_dt;

    const MultiFab& activeState = use_new_state ? get_new_data(State_Type) : get_old_data(State_Type);
#ifdef MHD
    const MultiFab& Bx = use_new_state ? get_new_data(Bfield_Type_x) : get_old_data(Bfield_Type_x);
    const MultiFab& By = use_new_state ? get_new_data(Bfield_Type_y) : get_old_data(Bfield_Type_y);
    const MultiFab& Bz = use_new_state ? get_new_data(Bfield_Type_z) : get_old_data(Bfield_Type_z);
#endif
#ifdef RADIATION
    const MultiFab& radiationField = use_new_state ? get_new_data(Radiation_Type) : get_old_data(Radiation_Type);
#endif
    const auto geomdata = geom.data();


    bool exclude_covered_zones = false;

    if (level < parent->finestLevel() && parent->subcyclingMode() == "None") {
        exclude_covered_zones = true;
    }

    MultiFab tmp_coverMask;
    const MultiFab& coverMask = exclude_covered_zones ? getLevel(level+1).make_fine_mask() : tmp_coverMask;

    std::string limiter = "hydrosim.max_dt";


    Real dtlimit_hydro = max_dt / cfl;

    if (do_hydro)
    {

#ifdef RADIATION
        if (Radiation::rad_hydro_coupled) {

            Real ldtlimit_hydro = timestep::dtlimit_rad(activeState, radiationField, geomdata);
            ParallelDescriptor::ReduceRealMin(ldtlimit_hydro);
            dtlimit_hydro = amrex::min(dtlimit_hydro, ldtlimit_hydro) * cfl;
            if (verbose) {
                amrex::Print() << "...estimated hydro-limited timestep at level " << level << ": " << dtlimit_hydro << std::endl;
            }
        }
        else
        {
#endif

#ifdef MHD
          auto cfl_timestep = timestep::estdt<timestep::mhd>(geomdata, coverMask, activeState, Bx, By, Bz);
#else
          auto cfl_timestep = timestep::estdt<timestep::hydro>(geomdata, coverMask, activeState);
#endif

          amrex::ParallelAllReduce::Min(cfl_timestep, MPI_COMM_WORLD);
          dtlimit_hydro = amrex::min(dtlimit_hydro, cfl_timestep.value) * cfl;
          if (verbose) {
              amrex::Print() << "...estimated hydro-limited timestep at level " << level << ": " << dtlimit_hydro << std::endl;
              std::string index_label = "(i";
#if AMREX_SPACEDIM >= 2
              index_label += ",j";
#endif
#if AMREX_SPACEDIM == 3
              index_label += ",k";
#endif
              index_label += ")";

              amrex::Print() << "...hydro-limited CFL timestep constrained at " << index_label << " = " << cfl_timestep.index << std::endl;
          }

#ifdef RADIATION
        }
#endif



        if (dtlimit_hydro < estdt) {
            limiter = "hydro";
            estdt = dtlimit_hydro;
        }

    }

#ifdef DIFFUSION

    Real dtlimit_diffusion = max_dt / cfl;

    if (conduction_enabled)
    {
        auto diffusion_dt = timestep::estdt<timestep::diffusion>(geomdata, coverMask, activeState);
        ParallelAllReduce::Min(diffusion_dt, MPI_COMM_WORLD);
        dtlimit_diffusion = amrex::min(dtlimit_diffusion, diffusion_dt.value) * cfl;

        if (verbose) {
            amrex::Print() << "...estimated diffusion-limited timestep at level " << level << ": " << dtlimit_diffusion << std::endl;
            std::string index_label = "(i";
#if AMREX_SPACEDIM >= 2
            index_label += ",j";
#endif
#if AMREX_SPACEDIM == 3
            index_label += ",k";
#endif
            index_label += ")";

            amrex::Print() << "...diffusion-limited timestep constrained at " << index_label << " = " << diffusion_dt.index << std::endl;

        }
    }


    if (dtlimit_diffusion < estdt) {
        limiter = "diffusion";
        estdt = dtlimit_diffusion;
    }
#endif

#ifdef REACTIONS
    Real dtlimit_burn = max_dt;

    if (do_react && (hydrosim::burn_dt_energy < 1.e199_rt || hydrosim::burn_dt_massfrac < 1.e199_rt)) {


        auto burn_timestep = timestep::estdt<timestep::burning>(geomdata, coverMask, activeState);

        ParallelAllReduce::Min(burn_timestep, MPI_COMM_WORLD);
        dtlimit_burn = amrex::min(dtlimit_burn, burn_timestep.value);

        if (verbose) {
            amrex::Print() << "...estimated burning-limited timestep at level " << level << ": " << dtlimit_burn << std::endl;
            std::string index_label = "(i";
#if AMREX_SPACEDIM >= 2
            index_label += ",j";
#endif
#if AMREX_SPACEDIM == 3
            index_label += ",k";
#endif
            index_label += ")";

            amrex::Print() << "...burning-limited timestep constrained at " << index_label << " = " << burn_timestep.index << std::endl;
        }


        if (dtlimit_burn < estdt) {
            limiter = "burning";
            estdt = dtlimit_burn;
        }
    }
#endif

    if (verbose) {
        amrex::Print() << "HydroSim::estTimeStep (" << limiter << "-limited) at level " << level << ":  estdt = " << estdt << '\n' << std::endl;
    }

    return estdt;
}

void
HydroSim::computeNewDt (int                    finest_level,
                      int                     ,
                      Vector<int>&           n_cycle,
                      const Vector<IntVect>&  ,
                      Vector<Real>&          dt_min,
                      Vector<Real>&          dt_level,
                      Real                   stop_time,
                      int                    post_regrid_flag)
{
    BL_PROFILE("HydroSim::computeNewDt()");

    if (level > 0) {
        return;
    }

    Real dt_0 = 1.0e+100;
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        HydroSim& adv_level = getLevel(i);
        dt_min[i] = adv_level.estTimeStep();
    }

    if (constant_dt <= 0.0)
    {
       if (post_regrid_flag == 1)
       {
          for (int i = 0; i <= finest_level; i++)
          {
              dt_min[i] = std::min(dt_min[i],dt_level[i]);
          }
       }
       else
       {
          if (lastDtWriteLimited) {

              dt_min[0] = std::min(dt_min[0], lastDtBeforeWriteLimit);

              lastDtWriteLimited = 0;
              lastDtBeforeWriteLimit = 0.0;

          }
          else {

              for (int i = 0; i <= finest_level; i++)
              {
                  if (verbose && ParallelDescriptor::IOProcessor()) {
                    if (dt_min[i] > change_max*dt_level[i])
                      {
                          std::cout << "HydroSim::compute_new_dt : limiting dt at level "
                                    << i << '\n';
                          std::cout << " ... new dt computed: " << dt_min[i]
                                    << '\n';
                          std::cout << " ... but limiting to: "
                                    << change_max * dt_level[i] << " = " << change_max
                                    << " * " << dt_level[i] << '\n';
                      }
                  }
                  dt_min[i] = std::min(dt_min[i],change_max*dt_level[i]);
              }

          }
       }
    }

    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_min[i]);
    }

    if (plot_interval_exact) {

        const Real plot_per = parent->plotPer();

        if (plot_per > 0.0) {

            const Real cur_time = state[State_Type].curTime();


            const Real phaseOffset = std::fmod(cur_time, plot_per);

            Real nextPlotDt;


            if (std::abs(phaseOffset - plot_per) <= std::numeric_limits<Real>::epsilon() * cur_time) {
                nextPlotDt = plot_per + (plot_per - phaseOffset);
            }
            else {
                nextPlotDt = plot_per - phaseOffset;
            }

            if (nextPlotDt < dt_0) {
                lastDtWriteLimited = 1;
                lastDtBeforeWriteLimit = dt_0;
                dt_0 = nextPlotDt;


                const Real dtFloor = 1.e-4 * lastDtBeforeWriteLimit;
                dt_0 = std::max(dt_0, dtFloor);

                if (verbose) {
                  amrex::Print() << " ... limiting dt to " << dt_0 << " to hit the next plot interval.\n";
                }
            }

        }

    }

    if (small_interval_exact) {

        const Real small_plot_per = parent->smallplotPer();

        if (small_plot_per > 0.0) {

            const Real cur_time = state[State_Type].curTime();


            const Real phaseOffset = std::fmod(cur_time, small_plot_per);

            Real nextSmallPlotDt;

            if (std::abs(phaseOffset - small_plot_per) <= std::numeric_limits<Real>::epsilon() * cur_time) {
                nextSmallPlotDt = small_plot_per + (small_plot_per - phaseOffset);
            }
            else {
                nextSmallPlotDt = small_plot_per - phaseOffset;
            }

            if (nextSmallPlotDt < dt_0) {
                lastDtWriteLimited = 1;
                lastDtBeforeWriteLimit = dt_0;
                dt_0 = nextSmallPlotDt;

                const Real dtFloor = 1.e-4 * lastDtBeforeWriteLimit;
                dt_0 = std::max(dt_0, dtFloor);

                if (verbose) {
                    amrex::Print() << " ... limiting dt to " << dt_0 << " to hit the next smallplot interval.\n";
                }
            }

        }

    }

    const Real eps = std::numeric_limits<Real>::epsilon();
    Real cur_time = state[State_Type].curTime();
    if (stop_time >= 0.0) {
        if ((cur_time + dt_0) >= (stop_time - eps)) {
            dt_0 = stop_time - cur_time;
            if (verbose) {
                amrex::Print() << " ... limiting dt to " << dt_0 << " to hit the stop_time.\n";
            }
        }
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void
HydroSim::computeInitialDt (int                   finest_level,
                          int                    ,
                          Vector<int>&           n_cycle,
                          const Vector<IntVect>&  ,
                          Vector<Real>&          dt_level,
                          Real                  stop_time)
{
    BL_PROFILE("HydroSim::computeInitialDt()");

    if (level > 0) {
      return;
    }

    int i;

    Real dt_0 = 1.0e+100;
    int n_factor = 1;
    for (i = 0; i <= finest_level; i++)
    {
        dt_level[i] = getLevel(i).initialTimeStep();
        n_factor   *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_level[i]);
    }

    const Real eps = 0.001*dt_0;
    Real cur_time  = state[State_Type].curTime();
    if (stop_time >= 0.0) {
        if ((cur_time + dt_0) > (stop_time - eps)) {
          dt_0 = stop_time - cur_time;
        }
    }

    n_factor = 1;
    for (i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
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

void
HydroSim::FluxRegCoarseInit() {
    BL_PROFILE("HydroSim::FluxRegCoarseInit");

    if (level == parent->finestLevel()) {
      return;
    }

    HydroSim& fine_level = getLevel(level+1);

    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      fine_level.flux_reg.CrseInit(*fluxes[i], i, 0, 0, NUM_STATE, flux_coarse_scale);
    }

#if (AMREX_SPACEDIM <= 2)
    if (!Geom().IsCartesian()) {
      fine_level.pressure_flux_reg.CrseInit(Pressure_radial, 0, 0, 0, 1, pressure_coarse_scale);
    }
#endif

#ifdef RADIATION
    if (Radiation::rad_hydro_coupled) {
      for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        fine_level.rad_edge_reg.CrseInit(*rad_face_flux[i], i, 0, 0, Radiation::nGroups, flux_coarse_scale);
      }
    }
#endif

}


void
HydroSim::FluxRegFineAccum() {

    BL_PROFILE("HydroSim::FluxRegFineAccum()");

    if (level == 0) {
      return;
    }

    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      flux_reg.FineAdd(*fluxes[i], i, 0, 0, NUM_STATE, flux_refined_scale);
    }

#if (AMREX_SPACEDIM <= 2)
    if (!Geom().IsCartesian()) {
      getLevel(level).pressure_flux_reg.FineAdd(Pressure_radial, 0, 0, 0, 1, pressure_refined_scale);
    }
#endif

#ifdef RADIATION
    if (Radiation::rad_hydro_coupled) {
      for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        getLevel(level).rad_edge_reg.FineAdd(*rad_face_flux[i], i, 0, 0, Radiation::nGroups, flux_refined_scale);
      }
    }
#endif

}


void
HydroSim::reflux (int crse_level, int fine_level, bool from_post_step)
{
    BL_PROFILE("HydroSim::reflux()");

    BL_ASSERT(fine_level > crse_level);

    const Real strt = ParallelDescriptor::second();

#ifdef GRAVITY
    int nlevs = fine_level - crse_level + 1;

    Vector<std::unique_ptr<MultiFab> > drho(nlevs);
    Vector<std::unique_ptr<MultiFab> > dphi(nlevs);

    if (do_gravity_flag && gravity->get_solver_kind() == "PoissonGrav" && gravity->NoSync() == 0 && from_post_step)  {

        for (int lev = crse_level; lev <= fine_level; ++lev) {

            const auto& amrlevel = getLevel(lev);
            const auto& ba = amrlevel.boxArray();
            const auto& dm = amrlevel.DistributionMap();

            drho[lev - crse_level] = std::make_unique<MultiFab>(ba, dm, 1, 0);
            dphi[lev - crse_level] = std::make_unique<MultiFab>(ba, dm, 1, 0);

            drho[lev - crse_level]->setVal(0.0);
            dphi[lev - crse_level]->setVal(0.0);

        }

    }
#endif

    FluxRegister* reg = nullptr;

    for (int lev = fine_level; lev > crse_level; --lev) {

        reg = &getLevel(lev).flux_reg;

        HydroSim& crse_lev = getLevel(lev-1);
#ifdef GRAVITY
        HydroSim& fine_lev = getLevel(lev);
#endif

        MultiFab& crse_state = crse_lev.get_new_data(State_Type);


        MultiFab ghosted_coarse_state(crse_state.boxArray(), crse_state.DistributionMap(), crse_state.nComp(), 1);

        crse_lev.fill_ghost_field(ghosted_coarse_state, crse_lev.state[State_Type].curTime(), 1);


        reg->ClearInternalBorders(crse_lev.geom);




        MultiFab scratch_fluxes[AMREX_SPACEDIM];

        for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

            scratch_fluxes[idir].define(crse_lev.fluxes[idir]->boxArray(),
                                     crse_lev.fluxes[idir]->DistributionMap(),
                                     crse_lev.fluxes[idir]->nComp(), crse_lev.fluxes[idir]->nGrow());

            scratch_fluxes[idir].setVal(0.0);


            for (OrientationIter fi; fi.isValid(); ++fi) {
                const FabSet& fs = (*reg)[fi()];
                if (fi().coordDir() == idir) {
                    fs.copyTo(scratch_fluxes[idir], 0, 0, 0, scratch_fluxes[idir].nComp());
                }
            }

        }


#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        for (MFIter mfi(crse_state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();

            auto U = ghosted_coarse_state[mfi].array();
            auto V = crse_lev.volume[mfi].array();


            for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
                const Box& nbx = amrex::surroundingNodes(bx, idir);
                auto F = scratch_fluxes[idir][mfi].array();
#ifndef MHD
                auto A = crse_lev.area[idir][mfi].array();
                Real dt = parent->dtLevel(crse_level);

                bool scale_by_area_dt = false;
                crse_lev.clip_fluxes_at_density_floor(nbx, idir, U, V, F, A, dt, scale_by_area_dt);
#endif
                amrex::ParallelFor(nbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    bool clear_fluxes = false;

                    Real rho = U(i,j,k,URHO);
                    Real densityDelta = F(i,j,k,URHO) / V(i,j,k);
                    Real invDensityNew = 1.0_rt / (rho + densityDelta);

                    for (int n = 0; n < NumSpec; ++n) {
                        Real rhoX = U(i,j,k,UFS+n);
                        Real speciesDensityDelta = F(i,j,k,UFS+n) / V(i,j,k);
                        Real MassfracNew = (rhoX + AMREX_SPACEDIM * speciesDensityDelta) * invDensityNew;

                        if (MassfracNew < -hydrosim::massfrac_check_tolerance ||
                            MassfracNew > 1.0_rt + hydrosim::massfrac_check_tolerance) {
                            clear_fluxes = true;
                            break;
                        }
                    }

                    if (clear_fluxes) {
                        for (int n = 0; n < NUM_STATE; ++n) {
                            F(i,j,k,n) = 0.0;
                        }
                    }
                });
            }
        }

        for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {


            for (OrientationIter fi; fi.isValid(); ++fi) {
                if (fi().coordDir() == idir) {
                    FabSet& fs = (*reg)[fi()];
                    fs.copyFrom(scratch_fluxes[idir], 0, 0, 0, scratch_fluxes[idir].nComp());
                }
            }


            if (refresh_sources_after_reflux || !from_post_step) {

                MultiFab::Add(*crse_lev.fluxes[idir], scratch_fluxes[idir], 0, 0, crse_lev.fluxes[idir]->nComp(), 0);


                MultiFab::Add(*crse_lev.mass_face_flux[idir], scratch_fluxes[idir], URHO, 0, 1, 0);
            }

        }


        reg->Reflux(crse_state, crse_lev.volume, 1.0, 0, 0, NUM_STATE, crse_lev.geom);


#ifdef GRAVITY
        int ilev = lev - crse_level - 1;

        if (do_gravity_flag && gravity->get_solver_kind() == "PoissonGrav" && gravity->NoSync() == 0 && from_post_step) {
            reg->Reflux(*drho[ilev], crse_lev.volume, 1.0, 0, URHO, 1, crse_lev.geom);
            amrex::average_down(*drho[ilev + 1], *drho[ilev], 0, 1, getLevel(lev).crse_ratio);
        }
#endif



        reg->setVal(0.0);

#if (AMREX_SPACEDIM <= 2)
        if (!Geom().IsCartesian()) {

            reg = &getLevel(lev).pressure_flux_reg;

            MultiFab dr(crse_lev.grids, crse_lev.dmap, 1, 0);
            dr.setVal(crse_lev.geom.CellSize(0));

            reg->ClearInternalBorders(crse_lev.geom);

            reg->Reflux(crse_state, dr, 1.0, 0, UMX, 1, crse_lev.geom);

            if (refresh_sources_after_reflux || !from_post_step) {

                MultiFab scratch_radial_fluxes(crse_lev.Pressure_radial.boxArray(),
                                    crse_lev.Pressure_radial.DistributionMap(),
                                    crse_lev.Pressure_radial.nComp(), crse_lev.Pressure_radial.nGrow());

                scratch_radial_fluxes.setVal(0.0);

                for (OrientationIter fi; fi.isValid(); ++fi)
                {
                    const FabSet& fs = (*reg)[fi()];
                    if (fi().coordDir() == 0) {
                        fs.copyTo(scratch_radial_fluxes, 0, 0, 0, scratch_radial_fluxes.nComp());
                    }
                }

                MultiFab::Add(crse_lev.Pressure_radial, scratch_radial_fluxes, 0, 0, crse_lev.Pressure_radial.nComp(), 0);

            }

            reg->setVal(0.0);

        }
#endif

#ifdef RADIATION


        if (Radiation::rad_hydro_coupled) {

            reg = &getLevel(lev).rad_edge_reg;

            reg->ClearInternalBorders(crse_lev.geom);

            reg->Reflux(crse_lev.get_new_data(Radiation_Type), crse_lev.volume, 1.0, 0, 0, Radiation::nGroups, crse_lev.geom);

            if (refresh_sources_after_reflux || !from_post_step) {

                for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

                    MultiFab scratch_fluxes(crse_lev.rad_face_flux[idir]->boxArray(),
                                         crse_lev.rad_face_flux[idir]->DistributionMap(),
                                         crse_lev.rad_face_flux[idir]->nComp(), crse_lev.rad_face_flux[idir]->nGrow());

                    scratch_fluxes.setVal(0.0);

                    for (OrientationIter fi; fi.isValid(); ++fi) {
                        const FabSet& fs = (*reg)[fi()];
                        if (fi().coordDir() == idir) {
                            fs.copyTo(scratch_fluxes, 0, 0, 0, scratch_fluxes.nComp());
                        }
                    }

                    MultiFab::Add(*crse_lev.rad_face_flux[idir], scratch_fluxes, 0, 0, crse_lev.rad_face_flux[idir]->nComp(), 0);

                }

            }

            reg->setVal(0.0);

        }

#endif

#ifdef GRAVITY
        if (do_gravity_flag && gravity->get_solver_kind() == "PoissonGrav" && gravity->NoSync() == 0 && from_post_step)  {

            reg = &getLevel(lev).pot_flux_reg;


            for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                reg->CrseInit(*(gravity->get_potential_gradient_curr(lev-1)[i]), crse_lev.area[i], i, 0, 0, 1, -1.0);
                reg->FineAdd(*(gravity->get_potential_gradient_curr(lev)[i]), fine_lev.area[i], i, 0, 0, 1, 1.0);
            }

            reg->Reflux(*dphi[ilev], crse_lev.volume, 1.0, 0, 0, 1, crse_lev.geom);

            amrex::average_down(*dphi[ilev + 1], *dphi[ilev], 0, 1, getLevel(lev).crse_ratio);

            reg->setVal(0.0);

        }
#endif

    }


#ifdef GRAVITY
    if (do_gravity_flag && gravity->get_solver_kind() == "PoissonGrav" && gravity->NoSync() == 0 && from_post_step) {
      gravity->synchronize_gravity_levels(crse_level, fine_level, amrex::GetVecOfPtrs(drho), amrex::GetVecOfPtrs(dphi));
    }
#endif


    if (refresh_sources_after_reflux && from_post_step &&
        (time_scheme == CornerTrace ||
         time_scheme == SimplifiedIterCorrection)) {

        for (int lev = fine_level; lev >= crse_level; --lev) {

            MultiFab& S_old = getLevel(lev).get_old_data(State_Type);

            MultiFab& S_new = getLevel(lev).get_new_data(State_Type);
#ifdef MHD
            MultiFab& magfield_x_new = getLevel(lev).get_new_data(Bfield_Type_x);
            MultiFab& magfield_y_new = getLevel(lev).get_new_data(Bfield_Type_y);
            MultiFab& magfield_z_new = getLevel(lev).get_new_data(Bfield_Type_z);
#endif
            MultiFab& source = getLevel(lev).get_new_data(Forcing_Type);
            Real time = getLevel(lev).state[State_Type].curTime();
            Real dt_level_advance = getLevel(lev).dt_level_step;
            Real dt_expected = parent->dtLevel(lev);

            if (getLevel(lev).apply_source_terms()) {

                getLevel(lev).apply_source_term_to_state(S_new, source, -dt_level_advance, 0);
                getLevel(lev).clean_state(
#ifdef MHD
                                          magfield_x_new, magfield_y_new, magfield_z_new,
#endif
                                          S_new, time, 0);

            }


            if (allow_step_retry && dt_level_advance < dt_expected && getLevel(lev).retain_prev_field) {

                for (int k = 0; k < num_field_types; k++) {

                    if (getLevel(lev).prev_state[k]->hasOldData()) {


                        MultiFab& old = getLevel(lev).get_old_data(k);
                        MultiFab::Copy(getLevel(lev).prev_state[k]->newData(), old, 0, 0, old.nComp(), old.nGrow());
                        MultiFab::Copy(old, getLevel(lev).prev_state[k]->oldData(), 0, 0, old.nComp(), old.nGrow());

                        getLevel(lev).state[k].setTimeLevel(time, dt_level_advance, 0.0);
                        getLevel(lev).prev_state[k]->setTimeLevel(time, dt_expected, 0.0);

                    }

                }

            }

            if (getLevel(lev).apply_source_terms()) {
                bool apply_source_flag = true;
                getLevel(lev).run_new_sources(
#ifdef MHD
                                magfield_x_new, magfield_y_new, magfield_z_new,
#endif
                                source, S_old, S_new, time, dt_level_advance, apply_source_flag);
            }

            if (allow_step_retry && dt_level_advance < dt_expected && getLevel(lev).retain_prev_field) {

                for (int k = 0; k < num_field_types; k++) {

                    if (getLevel(lev).prev_state[k]->hasOldData()) {


                        MultiFab& old = getLevel(lev).get_old_data(k);
                        MultiFab::Copy(old, getLevel(lev).prev_state[k]->newData(), 0, 0, old.nComp(), old.nGrow());

                        getLevel(lev).state[k].setTimeLevel(time, dt_expected, 0.0);
                        getLevel(lev).prev_state[k]->setTimeLevel(time, dt_level_advance, 0.0);

                    }

                }


                if (lev == 0 || lev > level) {
                    amrex::FillNull(getLevel(lev).prev_state);
                    getLevel(lev).retain_prev_field = false;
                }

            }

        }

    }

    if (verbose)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real end = ParallelDescriptor::second() - strt;
        amrex::Real level_real = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(end,IOProc);
        if (ParallelDescriptor::IOProcessor()) {
            std::cout << "HydroSim::reflux() at level " << level_real
                      << " : time = " << end << std::endl;
        }
#ifdef BL_LAZY
        });
#endif
    }
}

void
HydroSim::avgDown ()
{
  BL_PROFILE("HydroSim::avgDown()");

  if (level == parent->finestLevel()) {
      return;
  }

  for (int k = 0; k < num_field_types; k++) {
      avgDown(k);
  }

}

void
HydroSim::renorm_species (MultiFab& S_new, int ng)
{
    BL_PROFILE("HydroSim::renorm_species()");

    Real lmassfrac_floor = network_rp::small_x;

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<int> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(ng);

        auto u = S_new.array(mfi);


        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
        {

            auto cons_cell = u.cellData(i, j, k);
            Real rhoX_sum = 0.0_rt;
            Real rho = cons_cell[URHO];
            Real rhoInv = 1.0_rt / rho;

            int failed{};

            for (int n = 0; n < NumSpec; ++n) {
                Real X = cons_cell[UFS+n] * rhoInv;

                if (rho >= hydrosim::massfrac_check_min_density) {
                    if (X < -hydrosim::massfrac_check_tolerance ||
                        X > 1.0_rt + hydrosim::massfrac_check_tolerance) {
#ifndef AMREX_USE_GPU
                        std::cout << "(i, j, k) = " << i << " " << j << " " << k << " " << ", X[" << n << "] = " << X
                                  << "  (density here is: " << rho << ")" << std::endl;
#elif defined(ALLOW_GPU_PRINTF)
                        AMREX_DEVICE_PRINTF("(i, j, k) = %d %d %d, X[%d] = %g  (density here is: %g)\n",
                                            i, j, k, n, X, rho);
#endif
                        failed = 1;
                    }
                }

                cons_cell[UFS+n] = amrex::Clamp(cons_cell[UFS+n], lmassfrac_floor * rho, rho);
                rhoX_sum += cons_cell[UFS+n];
            }

            Real fac = rho / rhoX_sum;

            amrex::constexpr_for<0, NumSpec>([&] (auto n) {
                cons_cell[UFS+n] *= fac;
            });

            return {failed};
        });
    }

    ReduceTuple hv = reduce_data.value();
    int num_failed = amrex::get<0>(hv);

    if (num_failed > 0) {
        amrex::Error("Invalid mass fraction in HydroSim::renorm_species()");
    }
}

void
HydroSim::reconcile_energy (
#ifdef MHD
                              MultiFab& Bx,
                              MultiFab& By,
                              MultiFab& Bz,
#endif
                              MultiFab& S)
{

    BL_PROFILE("HydroSim::reconcile_energy()");

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& box = mfi.tilebox();

        auto cons_state_arr = S.array(mfi);

#ifdef MHD
        auto magfield_x_arr = Bx.array(mfi);
        auto magfield_y_arr = By.array(mfi);
        auto magfield_z_arr = Bz.array(mfi);
#endif

        ParallelFor(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
          Real rhoInv = 1.0_rt / cons_state_arr(i,j,k,URHO);
          Real u = cons_state_arr(i,j,k,UMX) * rhoInv;
          Real v = cons_state_arr(i,j,k,UMY) * rhoInv;
          Real w = cons_state_arr(i,j,k,UMZ) * rhoInv;

          cons_state_arr(i,j,k,UEDEN) = cons_state_arr(i,j,k,UEINT) +
            0.5_rt * cons_state_arr(i,j,k,URHO) * (u*u + v*v + w*w);

#ifdef MHD
          Real magfield_x_cell = 0.5_rt * (magfield_x_arr(i,j,k) + magfield_x_arr(i+1,j,k));
          Real magfield_y_cell = 0.5_rt * (magfield_y_arr(i,j,k) + magfield_y_arr(i,j+1,k));
          Real magfield_z_cell = 0.5_rt * (magfield_z_arr(i,j,k) + magfield_z_arr(i,j,k+1));

          cons_state_arr(i,j,k,UEDEN) += 0.5_rt * (magfield_x_cell * magfield_x_cell +
                                          magfield_y_cell * magfield_y_cell +
                                          magfield_z_cell * magfield_z_cell);
#endif

        });

    }
}

void
HydroSim::clamp_min_density (MultiFab& state_in, int ng)
{

    BL_PROFILE("HydroSim::clamp_min_density()");


    MultiFab reset_delta;

    if (log_source_changes)
    {


        reset_delta.define(state_in.boxArray(), state_in.DistributionMap(), state_in.nComp(), 0);

        MultiFab::Copy(reset_delta, state_in, 0, 0, state_in.nComp(), 0);

    }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox(ng);

        clamp_state_density(bx, state_in.array(mfi), verbose);
    }

    if (log_source_changes)
    {

        MultiFab::Subtract(reset_delta, state_in, 0, 0, state_in.nComp(), 0);

        measure_and_log_source_change(reset_delta, 1.0, "negative density resets");
    }

}

step_result
HydroSim::scan_for_negative_mass_density ()
{
    BL_PROFILE("scan_for_negative_mass_density()");

    MultiFab& S_old = get_old_data(State_Type);
    MultiFab& S_new = get_new_data(State_Type);

    ReduceOps<ReduceOpMax, ReduceOpMax> reduce_op;
    ReduceData<int, int> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        const auto state_old_arr = S_old.array(mfi);
        const auto new_state_view = S_new.array(mfi);

        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
        {
            int density_check_failed = 0;
            int massfrac_check_failed = 0;

            const Real rho = new_state_view(i,j,k,URHO);
            const Real rhoInv = 1.0_rt / rho;


            if (state_old_arr(i,j,k,URHO) >= retry_density_cutoff && rho < small_dens) {
#ifndef AMREX_USE_GPU
                std::cout << "Invalid density = " << rho << " at index " << i << ", " << j << ", " << k << "\n";
#endif
                density_check_failed = 1;
            }

            if (rho >= hydrosim::massfrac_check_min_density) {

                for (int n = 0; n < NumSpec; ++n) {
                    Real X = new_state_view(i,j,k,UFS+n) * rhoInv;

                    if (X < -hydrosim::massfrac_check_tolerance ||
                        X > 1.0_rt + hydrosim::massfrac_check_tolerance) {
#ifndef AMREX_USE_GPU
                        std::cout << "Invalid X[" << n << "] = " << X << " in zone "
                                  << i << ", " << j << ", " << k
                                  << " with density = " << rho << std::endl;
#elif defined(ALLOW_GPU_PRINTF)
                        AMREX_DEVICE_PRINTF("Invalid X[%d] = %g in zone (%d,%d,%d) with density = %g\n",
                                            n, X, i, j, k, rho);
#endif
                        massfrac_check_failed = 1;
                    }
                }

            }

            return {density_check_failed, massfrac_check_failed};
        });

#ifdef ALLOW_GPU_PRINTF
        std::fflush(nullptr);
#endif

    }

    ReduceTuple hv = reduce_data.value();
    int density_check_failed = amrex::get<0>(hv);
    int massfrac_check_failed = amrex::get<1>(hv);

    int fails[] = {density_check_failed, massfrac_check_failed};
    ParallelDescriptor::ReduceIntMax(fails, 2);

    step_result status {};

    if (fails[0] == 1) {
        status.success = false;
        status.reason = "invalid density";
    }

    if (fails[1] == 1) {
        status.success = false;
        status.reason = "invalid X";
    }

    return status;
}

void
HydroSim::clamp_speed_limit (MultiFab& state_in, int ng)
{
    BL_PROFILE("HydroSim::clamp_speed_limit()");


    if (hydrosim::velocity_cap <= 0.0_rt) {
        return;
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox(ng);

        auto u = state_in[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real rho = u(i,j,k,URHO);
            Real rhoInv = 1.0_rt / rho;

            Real vx = u(i,j,k,UMX) * rhoInv;
            Real vy = u(i,j,k,UMY) * rhoInv;
            Real vz = u(i,j,k,UMZ) * rhoInv;

            Real v = std::sqrt(vx * vx + vy * vy + vz * vz);

            if (v > hydrosim::velocity_cap) {
                Real velocity_scale = hydrosim::velocity_cap / v;

                u(i,j,k,UMX) *= velocity_scale;
                u(i,j,k,UMY) *= velocity_scale;
                u(i,j,k,UMZ) *= velocity_scale;

                u(i,j,k,UEDEN) -= 0.5_rt * rhoInv * (rho * vx * rho * vx - u(i,j,k,UMX) * u(i,j,k,UMX) +
                                                     rho * vy * rho * vy - u(i,j,k,UMY) * u(i,j,k,UMY) +
                                                     rho * vz * rho * vz - u(i,j,k,UMZ) * u(i,j,k,UMZ));
            }
        });
    }
}

void
HydroSim::avgDown (int state_indx)
{
    BL_PROFILE("HydroSim::avgDown(state_indx)");

    if (level == parent->finestLevel()) {
        return;
    }

    HydroSim& fine_lev = getLevel(level+1);

    const Geometry& fgeom = fine_lev.geom;
    const Geometry& cgeom =          geom;

    MultiFab&  S_crse   = get_new_data(state_indx);
    MultiFab&  S_fine   = fine_lev.get_new_data(state_indx);

    amrex::average_down(S_fine, S_crse,
                         fgeom, cgeom,
                         0, S_fine.nComp(), fine_ratio);
}

void
HydroSim::allocOldData ()
{
    BL_PROFILE("HydroSim::allocOldData");

    MultiFab::RegionTag level_region_tag(std::format("AmrLevel_Level_{}", level));
    MultiFab::RegionTag statedata_tag(std::format("StateData_Level_{}", level));
    for (int k = 0; k < num_field_types; k++) {
        state[k].allocOldData();
    }
}

void
HydroSim::removeOldData()
{
    AmrLevel::removeOldData();
}

void
HydroSim::errorEst (TagBoxArray& tags,
                  int           ,
                  int           ,
                  Real         time,
                  int           ,
                  int           )
{
    BL_PROFILE("HydroSim::errorEst()");

    Real tag_time = time;


    if (post_step_regrid) {
      tag_time = get_state_data(State_Type).curTime();
    }


    for (const auto & etag : tag_criteria) {
        std::unique_ptr<MultiFab> mf;
        if (! etag.Field().empty()) {
            mf = derive(etag.Field(), tag_time, etag.NGrow());
        }
        etag(tags, mf.get(), TagBox::CLEAR, TagBox::SET, tag_time, level, geom);
    }


    apply_user_tags(tags, tag_time);


    apply_tag_limits(tags, tag_time);

}



void
HydroSim::apply_user_tags (TagBoxArray& tags, Real time)
{

    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::apply_user_tags()");

    MultiFab& S_new = get_new_data(State_Type);

    int lev = level;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
        for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            TagBox& tagfab = tags[mfi];

            auto tag_view = tagfab.array();
            const auto state_arr = S_new[mfi].array();
            const GeometryData& geomdata = geom.data();

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                user_tagging(i, j, k, tag_view, state_arr, lev, geomdata);
            });
        }
    }

}



void
HydroSim::apply_tag_limits(TagBoxArray& tags, [[maybe_unused]] Real time)
{
    BL_PROFILE("HydroSim::apply_tag_limits()");


#ifdef GRAVITY
    if (gravity::gravity_model_name == "PoissonGrav") {

        int lev = level;

        int n_error_buf[3] = {0};
        int ref_ratio[3] = {0};
        int domlo[3] = {0};
        int domhi[3] = {0};
        int physbc_lo[3] = {-1};
        int physbc_hi[3] = {-1};
        int blocking_factor[3] = {0};
        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
            n_error_buf[dim] = parent->nErrorBuf(lev, dim);
            ref_ratio[dim] = parent->refRatio(lev)[dim];
            domlo[dim] = geom.Domain().loVect()[dim];
            domhi[dim] = geom.Domain().hiVect()[dim];
            physbc_lo[dim] = phys_bc.lo()[dim];
            physbc_hi[dim] = phys_bc.hi()[dim];
            blocking_factor[dim] = parent->blockingFactor(lev)[dim];
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();

            auto tag = tags[mfi].array();

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                bool domain_edge_flags[3] = {false};

                const int idx[3] = {i, j, k};

                for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {

                    int boundary_width = n_error_buf[dim] + blocking_factor[dim] / ref_ratio[dim];

                    if ((physbc_lo[dim] != amrex::PhysBCType::symmetry &&
                         physbc_lo[dim] != amrex::PhysBCType::interior) &&
                        (idx[dim] <= domlo[dim] + boundary_width)) {
                        domain_edge_flags[dim] = true;
                    }

                    if ((physbc_hi[dim] != amrex::PhysBCType::symmetry &&
                         physbc_hi[dim] != amrex::PhysBCType::interior) &&
                        (idx[dim] >= domhi[dim] - boundary_width)) {
                        domain_edge_flags[dim] = true;
                    }
                }

                if (domain_edge_flags[0] || domain_edge_flags[1] || domain_edge_flags[2]) {

                    tag(i,j,k) = TagBox::CLEAR;

                }
            });
        }

    }
#endif

    auto geomdata = geom.data();

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto tag = tags[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real* problo = geomdata.ProbLo();
            const Real* probhi = geomdata.ProbHi();
            const Real* dx = geomdata.CellSize();

            GpuArray<Real, 3> loc = {0.0};

            loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];
#if AMREX_SPACEDIM >= 2
            loc[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
#endif
#if AMREX_SPACEDIM == 3
            loc[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
#endif

            Real r = distance(geomdata, loc);

            Real extent_lo = 0.0;
            Real extent_hi = 0.0;

            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
                extent_lo = amrex::max(extent_lo, std::abs(problo[dim] - problem::center[dim]));
                extent_hi = amrex::max(extent_hi, std::abs(probhi[dim] - problem::center[dim]));
            }

            if (r > hydrosim::tagging_radius_cap * amrex::max(extent_lo, extent_hi)) {
                tag(i,j,k) = TagBox::CLEAR;
            }
        });
    }
}



std::unique_ptr<MultiFab>
HydroSim::derive (const std::string& name,
                Real           time,
                int            ngrow)
{

    BL_PROFILE("HydroSim::derive()");

#ifdef AMREX_PARTICLES
  return ParticleQuantity(name,time,ngrow);
#else
   return AmrLevel::derive(name,time,ngrow);
#endif
}

void
HydroSim::derive (const std::string& name,
                Real           time,
                MultiFab&      mf,
                int            dcomp)
{

    BL_PROFILE("HydroSim::derive()");

    AmrLevel::derive(name,time,mf,dcomp);
}

void
HydroSim::microphys_init ()
{

  if (ParallelDescriptor::IOProcessor()) {
    std::cout << "reading extern runtime parameters ..." << std::endl;
  }

  init_extern_parameters();

}

void
HydroSim::repair_internal_energy(const Box& bx,
#ifdef MHD
                              Array4<Real> const& Bx, Array4<Real> const& By, Array4<Real> const& Bz,
#endif
                              Array4<Real> const& u)
{
    BL_PROFILE("HydroSim::repair_internal_energy(Fab)");

    Real ltemp_floor = small_temp;
    Real ldual_energy_switch = dual_energy_switch;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

        auto cons_cell = u.cellData(i, j, k);

        const Real rhoInv = 1.0_rt / cons_cell[URHO];
        const Real Up = cons_cell[UMX] * rhoInv;
        const Real Vp = cons_cell[UMY] * rhoInv;
        const Real Wp = cons_cell[UMZ] * rhoInv;
        const Real ke = 0.5_rt * (Up * Up + Vp * Vp + Wp * Wp);

        eos_re_t eos_state;

        eos_state.rho = cons_cell[URHO];
        eos_state.T   = ltemp_floor;
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = cons_cell[UFS+n] * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; ++n) {
            eos_state.aux[n] = cons_cell[UFX+n] * rhoInv;
        }
#endif

        eos(eos_input_rt, eos_state);

        const Real min_eint = eos_state.e;

#ifdef MHD
        const Real magfield_x_cell = 0.5_rt * (Bx(i,j,k) + Bx(i+1,j,k));
        const Real magfield_y_cell = 0.5_rt * (By(i,j,k) + By(i,j+1,k));
        const Real magfield_z_cell = 0.5_rt * (Bz(i,j,k) + Bz(i,j,k+1));

        const Real magnetic_energy = 0.5_rt * (magfield_x_cell*magfield_x_cell +
                                      magfield_y_cell*magfield_y_cell +
                                      magfield_z_cell*magfield_z_cell);
#else
        const Real magnetic_energy = 0.0_rt;
#endif


        cons_cell[UEINT] = amrex::max(cons_cell[UEINT], cons_cell[URHO] * min_eint);
        cons_cell[UEDEN] = amrex::max(cons_cell[UEDEN], cons_cell[URHO] * (min_eint + ke) + magnetic_energy);


        const Real internal_energy_density = cons_cell[UEDEN] - cons_cell[URHO] * ke - magnetic_energy;

        if (internal_energy_density > ldual_energy_switch * cons_cell[UEDEN]) {
            cons_cell[UEINT] = internal_energy_density;
        }
    });
}

void
HydroSim::repair_internal_energy(
#ifdef MHD
                              MultiFab& Bx,
                              MultiFab& By,
                              MultiFab& Bz,
#endif
                              MultiFab& State, int ng)

{

    BL_PROFILE("HydroSim::repair_internal_energy(MultiFab)");

    MultiFab old_state;


    if (log_source_changes)
    {
        old_state.define(State.boxArray(), State.DistributionMap(), State.nComp(), 0);
        MultiFab::Copy(old_state, State, 0, 0, State.nComp(), 0);
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(State, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(ng);

        repair_internal_energy(bx,
#ifdef MHD
                              Bx.array(mfi), By.array(mfi), Bz.array(mfi),
#endif
                              State.array(mfi));
    }

    if (log_source_changes)
    {

        MultiFab reset_delta(State.boxArray(), State.DistributionMap(), State.nComp(), 0);

        MultiFab::Copy(reset_delta, State, 0, 0, State.nComp(), 0);

        MultiFab::Subtract(reset_delta, old_state, 0, 0, old_state.nComp(), 0);

        measure_and_log_source_change(reset_delta, 1.0, "negative energy resets");
    }
}


#ifdef MHD
void
HydroSim::add_bfield_energy( MultiFab& Bx,
                        MultiFab& By,
                        MultiFab& Bz,
                        MultiFab& State)
{

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
  for (MFIter mfi(State, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
      const Box& box = mfi.tilebox();
      auto cons_state_arr = State.array(mfi);
      auto magfield_x_arr = Bx.array(mfi);
      auto magfield_y_arr = By.array(mfi);
      auto magfield_z_arr = Bz.array(mfi);


      ParallelFor(box,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

          Real magfield_x_cell = 0.5_rt * (magfield_x_arr(i,j,k) + magfield_x_arr(i+1,j,k));
          Real magfield_y_cell = 0.5_rt * (magfield_y_arr(i,j,k) + magfield_y_arr(i,j+1,k));
          Real magfield_z_cell = 0.5_rt * (magfield_z_arr(i,j,k) + magfield_z_arr(i,j,k+1));

          cons_state_arr(i,j,k,UEDEN) += 0.5_rt * (magfield_x_cell * magfield_x_cell +
                                          magfield_y_cell * magfield_y_cell +
                                          magfield_z_cell * magfield_z_cell);

      });

  }


}

void
HydroSim::check_div_bfield( MultiFab& Bx,
                     MultiFab& By,
                     MultiFab& Bz,
                     MultiFab& State)
{



  ReduceOps<ReduceOpSum> reduce_op;
  ReduceData<int> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;


#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
  for (MFIter mfi(State, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
      const Box& box = mfi.tilebox();
      auto magfield_x_arr = Bx.array(mfi);
      auto magfield_y_arr = By.array(mfi);
      auto magfield_z_arr = Bz.array(mfi);

      const auto dx = geom.CellSizeArray();

      reduce_op.eval(box, reduce_data,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
      {

          Real magDivergence = (magfield_x_arr(i+1,j,k) - magfield_x_arr(i,j,k))/dx[0] +
                      (magfield_y_arr(i,j+1,k) - magfield_y_arr(i,j,k))/dx[1] +
                      (magfield_z_arr(i,j,k+1) - magfield_z_arr(i,j,k))/dx[2];

          Real magfield_x_cell = 0.5_rt * (magfield_x_arr(i,j,k) + magfield_x_arr(i+1,j,k));
          Real magfield_y_cell = 0.5_rt * (magfield_y_arr(i,j,k) + magfield_y_arr(i,j+1,k));
          Real magfield_z_cell = 0.5_rt * (magfield_z_arr(i,j,k) + magfield_z_arr(i,j,k+1));

          Real magNorm = std::sqrt(magfield_x_cell * magfield_x_cell +
                                magfield_y_cell * magfield_y_cell +
                                magfield_z_cell * magfield_z_cell);


          int mag_div_failed = 0;

          if (std::abs(magDivergence) > 1.0e-10*magNorm){
             mag_div_failed = 1;
          }


          return {mag_div_failed};
      });

  }

  ReduceTuple hv = reduce_data.value();
  int init_mag_div_failed = amrex::get<0>(hv);

  if (init_mag_div_failed != 0) {
     amrex::Error("Error: initial data has divergence of B not zero");
  }


}
#endif

void
HydroSim::computeTemp(
#ifdef MHD
                    MultiFab& Bx,
                    MultiFab& By,
                    MultiFab& Bz,
#endif

                    MultiFab& State, [[maybe_unused]] Real time, int ng)

{

  BL_PROFILE("HydroSim::computeTemp()");

  MultiFab StateCopy;

  MultiFab energy_laplacian;

#ifdef TRUE_SDC
  if (sdc_accuracy_order == 4) {


    StateCopy.define(State.boxArray(), State.DistributionMap(), NUM_STATE, 2);
    fill_ghost_field(StateCopy, time, StateCopy.nGrow());

    energy_laplacian.define(State.boxArray(), State.DistributionMap(), 1, 0);

    auto domain_lo = geom.Domain().loVect3d();
    auto domain_hi = geom.Domain().hiVect3d();

    FArrayBox tmp;

    for (MFIter mfi(StateCopy); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.growntilebox(1);
      const Box& bx0 = mfi.tilebox();

      laplacian_of_state(bx0, StateCopy.array(mfi), energy_laplacian.array(mfi), UEINT,
                       domain_lo, domain_hi);

      tmp.resize(bx, 1, The_Async_Arena());
      auto tmp_arr = tmp.array();

      build_center_state_in_place(bx, StateCopy.array(mfi), tmp_arr, domain_lo, domain_hi);

    }

  }
#endif

#ifdef TRUE_SDC
  if (sdc_accuracy_order == 4) {
    clamp_min_density(StateCopy, StateCopy.nGrow());
    repair_internal_energy(StateCopy, StateCopy.nGrow());
  } else {
#endif
    repair_internal_energy(
#ifdef MHD
                          Bx, By, Bz,
#endif
                          State, ng);
#ifdef TRUE_SDC
  }
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
  for (MFIter mfi(State, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
      int num_ghost = ng;
#ifdef TRUE_SDC
      if (sdc_accuracy_order == 4) {
          num_ghost = 1;
      }
#endif

      const Box& bx = mfi.growntilebox(num_ghost);

#ifdef TRUE_SDC
      FArrayBox& state_box_data = (sdc_accuracy_order == 4) ? StateCopy[mfi] : State[mfi];
#else
      FArrayBox& state_box_data = State[mfi];
#endif

      Array4<Real> const u = state_box_data.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

          Real rhoInv = 1.0_rt / u(i,j,k,URHO);

          eos_re_t eos_state;

          eos_state.rho = u(i,j,k,URHO);
          eos_state.T   = u(i,j,k,UTEMP);
          eos_state.e   = u(i,j,k,UEINT) * rhoInv;
          for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = u(i,j,k,UFS+n) * rhoInv;
          }
#if NAUX_NET > 0
          for (int n = 0; n < NumAux; ++n) {
            eos_state.aux[n] = u(i,j,k,UFX+n) * rhoInv;
          }
#endif

          eos(eos_input_re, eos_state);

          u(i,j,k,UTEMP) = eos_state.T;

      });

      if (clamp_ambient_temperature == 1) {
          amrex::ParallelFor(bx,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              Real rhoInv = 1.0_rt / u(i,j,k,URHO);

              if (u(i,j,k,URHO) <= hydrosim::ambient_threshold_factor * ambient::ambient_reference_state[URHO]) {
                  u(i,j,k,UTEMP) = ambient::ambient_reference_state[UTEMP];
                  u(i,j,k,UEINT) = ambient::ambient_reference_state[UEINT] * (u(i,j,k,URHO) * rhoInv);
                  u(i,j,k,UEDEN) = u(i,j,k,UEINT) + 0.5_rt * rhoInv * (u(i,j,k,UMX) * u(i,j,k,UMX) +
                                                                       u(i,j,k,UMY) * u(i,j,k,UMY) +
                                                                       u(i,j,k,UMZ) * u(i,j,k,UMZ));
              }
          });
      }
  }

#ifdef TRUE_SDC
  if (sdc_accuracy_order == 4) {


    auto domain_lo = geom.Domain().loVect3d();
    auto domain_hi = geom.Domain().hiVect3d();

    FArrayBox tmp;

    for (MFIter mfi(StateCopy); mfi.isValid(); ++mfi) {

      const Box& bx = mfi.tilebox();

      tmp.resize(bx, 1, The_Async_Arena());
      auto tmp_arr = tmp.array();

      build_fourth_in_place_n(bx, StateCopy.array(mfi), UTEMP, tmp_arr, domain_lo, domain_hi);
    }

    MultiFab::Add(StateCopy, energy_laplacian, 0, UEINT, 1, 0);

    MultiFab::Copy(State, StateCopy, UTEMP, UTEMP, 1, 0);
    MultiFab::Copy(State, StateCopy, UEINT, UEINT, 1, 0);

    if (ng > 0) {
      AmrLevel::FillPatch(*this, State, State.nGrow(), time, State_Type, 0, NUM_STATE);
    }

    StateCopy.clear();
  }
#endif

}



void
HydroSim::create_forcing_corrector()
{

    BL_PROFILE("HydroSim::create_forcing_corrector()");

    if (time_scheme == CornerTrace && source_half_step == 1) {




        const Real time = get_state_data(Forcing_Type).prevTime();

        AmrLevel::FillPatch(*this, forcing_corrector, SRC_GHOST_WIDTH, time, Forcing_Type, UMX, 3, UMX);

        forcing_corrector.mult(2.0 / lastStepDt, SRC_GHOST_WIDTH);

    }
    else if (time_scheme == SimplifiedIterCorrection && source_half_step == 1) {


        if (correction_iter == 0) {
            const Real time = get_state_data(Forcing_Type).prevTime();

            AmrLevel::FillPatch(*this, forcing_corrector, SRC_GHOST_WIDTH, time, Forcing_Type, 0, NUM_SOURCE);

        } else {
            const Real time = get_state_data(Forcing_Type).curTime();

            AmrLevel::FillPatch(*this, forcing_corrector, SRC_GHOST_WIDTH, time, Forcing_Type, 0, NUM_SOURCE);
        }

    }

}



void
HydroSim::swap_field_time_levels(const Real dt)
{

    BL_PROFILE("HydroSim::swap_field_time_levels()");

    MultiFab::RegionTag statedata_tag(std::format("StateData_Level_{}", level));
    MultiFab::RegionTag level_region_tag(std::format("AmrLevel_Level_{}", level));

    for (int k = 0; k < num_field_types; k++) {


#ifdef REACTIONS
        if (time_scheme == SimplifiedIterCorrection) {
            if (k == Burn_Corr_Type) {
                state[k].swapTimeLevels(0.0);
            }
        }
#endif

#ifdef TRUE_SDC
#ifdef REACTIONS
        if (time_scheme == IterativeCorrection &&
            sdc_accuracy_order == 4 && k == Forcing_Type_Index) {
          state[k].swapTimeLevels(0.0);
        }
#endif
#endif
        state[k].allocOldData();

        state[k].swapTimeLevels(dt);

    }

}

#ifdef GRAVITY
int
HydroSim::count_points_1d ()
{
     int grid_points_1d;

     Box bx(geom.Domain());
     long nx = bx.size()[0];

#if (AMREX_SPACEDIM == 1)
     grid_points_1d = static_cast<int>(nx);
#elif (AMREX_SPACEDIM == 2)
     long ny = bx.size()[1];
     Real diag_length_sq = Real(nx*nx + ny*ny);
     grid_points_1d = int(std::sqrt(diag_length_sq))+2*NUM_GROW;
#elif (AMREX_SPACEDIM == 3)
     long ny = bx.size()[1];
     long nz = bx.size()[2];
     Real diag_length_sq = Real(nx*nx + ny*ny + nz*nz);
     grid_points_1d = int(std::sqrt(diag_length_sq))+2*NUM_GROW;
#endif

     if (verbose && ParallelDescriptor::IOProcessor()) {
         std::cout << "HydroSim::grid_points_1d at level  " << level << " is " << grid_points_1d << std::endl;
     }

     return grid_points_1d;
}

void
HydroSim::define_tracked_center(const MultiFab& S, Real time)
{
    BL_PROFILE("HydroSim::define_tracked_center()");

#if AMREX_SPACEDIM >= 2
    const Real* dx = geom.CellSize();
#endif

    IntVect max_index = S.maxIndex(URHO,0);
    Box bx(max_index,max_index);
    bx.grow(1);
    BoxArray ba(bx);
    int owner = ParallelDescriptor::IOProcessorNumber();
    DistributionMapping dm { Vector<int>(1,owner) };
    MultiFab mf(ba, dm, 1, 0, MFInfo().SetArena(The_Managed_Arena()));

    FillPatch(*this,mf,0,time,State_Type,URHO,1);

#if AMREX_SPACEDIM >= 2
    int mi[3] = {0};
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      mi[i] = max_index[i];
    }
#endif

    AMREX_ALWAYS_ASSERT(mf.local_size() == 1);

    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {

#if AMREX_SPACEDIM >= 2

        const Box& box = mfi.validbox();
        auto data = mf.array(mfi);

        Real cen = data(mi[0], mi[1], mi[2]);

        amrex::ParallelFor(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            data(i,j,k) -= cen;
        });

        Real snapped_center[3];
        for (int d = 0; d < AMREX_SPACEDIM; d++) {
            snapped_center[d] = geom.ProbLo(d) +  (static_cast<Real>(mi[d]) + 0.5_rt) * dx[d];
        }


        Real a = 0.5_rt * (data(mi[0]+1, mi[1], mi[2]) + data(mi[0]-1, mi[1], mi[2]));
        Real b = 0.5_rt * (data(mi[0]+1, mi[1], mi[2]) - data(mi[0]-1, mi[1], mi[2]));
        Real x = -b / (2.0_rt * a);
        problem::center[0] = snapped_center[0] + x * dx[0];

        a = 0.5_rt * (data(mi[0], mi[1]+1, mi[2]) + data(mi[0], mi[1]-1, mi[2]));
        b = 0.5_rt * (data(mi[0], mi[1]+1, mi[2]) - data(mi[0], mi[1]-1, mi[2]));
        Real y = -b / (2.0_rt * a);
        problem::center[1] = snapped_center[1] + y * dx[1];

#if AMREX_SPACEDIM == 3
        a = 0.5_rt * (data(mi[0], mi[1], mi[2]+1) + data(mi[0], mi[1], mi[2]-1));
        b = 0.5_rt * (data(mi[0], mi[1], mi[2]+1) - data(mi[0], mi[1], mi[2]-1));
        Real z = -b / (2.0_rt * a);
        problem::center[2] = snapped_center[2] + z * dx[2];
#endif

#endif

    }
    ParallelDescriptor::Bcast(&problem::center[0], AMREX_SPACEDIM, owner);

    if ( Geom().IsRZ() ) {
      problem::center[0] = 0;
    } else if ( Geom().IsSPHERICAL() ) {
        problem::center[0] = 0;
        problem::center[1] = 0;
    }

}

void
HydroSim::write_tracked_center ()
{
    int num_log_files = parent->NumDataLogs();

    if ( (track_center==1) && (num_log_files > 0) && ParallelDescriptor::IOProcessor())
    {
       std::ostream& center_log = parent->DataLog(0);

       int nstep = parent->levelSteps(0);
       Real time = state[State_Type].curTime();

       if (time == 0.0) {
           center_log << std::setw( 8) <<  "   nstep";
           center_log << std::setw(14) <<  "         time  ";
           center_log << std::setw(14) <<  "         center" << std::endl;;
       }

           center_log << std::setw( 8) <<  nstep;
           center_log << std::setw(14) <<  std::setprecision(6) <<  time;
           center_log << std::setw(14) <<  std::setprecision(6) << problem::center[0];
#if (AMREX_SPACEDIM >= 2)
           center_log << std::setw(14) <<  std::setprecision(6) << problem::center[1];
#endif
#if (AMREX_SPACEDIM == 3)
           center_log << std::setw(14) <<  std::setprecision(6) << problem::center[2];
#endif
           center_log << std::endl;
    }
}
#endif

Real
HydroSim::getCpuSeconds()
{

  int workerCount = ParallelDescriptor::NProcs();
#ifdef AMREX_USE_OMP
  workerCount = workerCount*omp_get_max_threads();
#endif

  Real T = workerCount*(ParallelDescriptor::second() - startCpuSeconds) +
    priorCpuSeconds;

  return T;
}


MultiFab&
HydroSim::make_fine_mask()
{
    BL_PROFILE("HydroSim::make_fine_mask()");

    BL_ASSERT(level > 0);

    if (fine_mask.empty()) {
        fine_mask = makeFineMask(parent->boxArray(level-1),
                                 parent->DistributionMap(level-1),
                                 parent->boxArray(level), crse_ratio,
                                 1.0,
                                 0.0);
    }

    return fine_mask;
}

iMultiFab&
HydroSim::make_boundary_mask (int ng)
{
    BL_PROFILE("HydroSim::make_boundary_mask()");

    for (const auto & ibm : interior_masks)
    {
        if (ibm->nGrow() == ng) {
            return *ibm;
        }
    }

    interior_masks.push_back(std::make_unique<iMultiFab>(grids, dmap, 1, ng));

    iMultiFab& imf = *interior_masks.back();

    int ghost_cover_flag = 0;
    int uncovered_flag = 1;

    imf.BuildMask(geom.Domain(), geom.periodicity(),
                  ghost_cover_flag, uncovered_flag, uncovered_flag, uncovered_flag);

    return imf;
}


void
HydroSim::fill_ghost_field(MultiFab& S, Real time, int ng)
{
  BL_PROFILE("HydroSim::fill_ghost_field()");

  BL_ASSERT(S.nGrow() >= ng);

  AmrLevel::FillPatch(*this, S, ng, time, State_Type, 0, NUM_STATE);
}


void
HydroSim::scan_for_nan(const MultiFab& state_in, int include_ghosts)
{
  BL_PROFILE("HydroSim::scan_for_nan()");

  int ng = 0;
  if (include_ghosts == 1) {
    ng = state_in.nGrow();
  }

  if (state_in.contains_nan(URHO,state_in.nComp(),ng,true)) {
#if defined(AMREX_USE_GPU) && !defined(ALLOW_GPU_PRINTF)
      std::string abort_message = std::string("State has NaNs in scan_for_nan()");
      amrex::Abort(abort_message.c_str());
#else
      for (int i = 0; i < state_in.nComp(); i++) {
          if (state_in.contains_nan(URHO + i, 1, ng, true)) {
              std::string abort_message = std::string("State has NaNs in the ") + desc_lst[State_Type].name(i) + std::string(" component::scan_for_nan()");
              amrex::Abort(abort_message.c_str());
          }
      }
#endif
  }
}


void
HydroSim::clean_state(
#ifdef MHD
                    MultiFab& bx,
                    MultiFab& by,
                    MultiFab& bz,
#endif
                    MultiFab& state_in, Real time, int ng) {

    BL_PROFILE("HydroSim::clean_state()");


    clamp_min_density(state_in, ng);


    clamp_speed_limit(state_in, ng);


    renorm_species(state_in, ng);


#ifdef HYBRID_MOMENTUM
    if (split_momentum_hydro) {
        split_to_linear_momentum(state_in, ng);
    }
#endif


    computeTemp(
#ifdef MHD
                bx, by, bz,
#endif
                state_in, time, ng);

}

void
HydroSim::snapshot_for_rerun ()
{
    BL_PROFILE("HydroSim::snapshot_for_rerun()");

    for (int k = 0; k < num_field_types; k++) {


        if (!prev_state[k]->hasOldData()) {

#ifdef AMREX_USE_GPU
            Arena* prev_arena = state[k].getArena();
            state[k].setArena(The_Pinned_Arena());
#endif
            *prev_state[k] = state[k];
#ifdef AMREX_USE_GPU
            state[k].setArena(prev_arena);
#endif
        }

    }

}
