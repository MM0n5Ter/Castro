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
#include <runtime_param_init.H>
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
#include <diffusion_operator.H>
#endif

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <extern_parameters.H>
#include <prob_parameters.H>

#include <user_initialize.H>

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
#elif AMREX_SPACEDIM == 2

#ifndef AMREX_USE_GPU
IntVect      HydroSim::hydro_tile_size(1024,16);
#else
IntVect      HydroSim::hydro_tile_size(1048576,1048576);
#endif
#else

#ifndef AMREX_USE_GPU
IntVect      HydroSim::hydro_tile_size(1024,16,16);
#else
IntVect      HydroSim::hydro_tile_size(1048576,1048576,1048576);
#endif
#endif

#if AMREX_SPACEDIM == 1
IntVect      HydroSim::untiled_size(1024);
#elif AMREX_SPACEDIM == 2
IntVect      HydroSim::untiled_size(1024,1024);
#else
IntVect      HydroSim::untiled_size(1024,1024,1024);
#endif

int          HydroSim::sweep_tile_tuned = 0;
Long         HydroSim::largest_tuned_box_pts = 0;
Real         HydroSim::priorCpuSeconds = 0.0;
Real         HydroSim::startCpuSeconds = 0.0;
int          HydroSim::Forcing_Type_Index = -1;
int          HydroSim::num_field_types = 0;
int          HydroSim::user_init_pending = 0;

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
HydroSim::setTimeLevel (Real time,
                      Real dt_old,
                      Real dt_new)
{
    AmrLevel::setTimeLevel(time,dt_old,dt_new);
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

       auto open_diag_log = [] (int slot, const char* file) {
           run_logs[slot] = std::make_unique<std::fstream>();
           run_logs[slot]->open(file, std::ios::out | std::ios::app);
           if (!run_logs[slot]->good()) {
               amrex::FileOpenFailed(file);
           }
       };

       open_diag_log(0, "grid_diag.out");

#ifdef GRAVITY
       open_diag_log(1, "gravity_diag.out");
#else
       run_logs[1] = std::make_unique<std::fstream>();
#endif

       open_diag_log(2, "species_diag.out");

       open_diag_log(3, "amr_diag.out");

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

        static const std::pair<const char*, AMRErrorTag::TEST> tests[] = {
            {"value_greater", AMRErrorTag::GREATER},
            {"value_less", AMRErrorTag::LESS},
            {"gradient", AMRErrorTag::GRAD},
            {"relative_gradient", AMRErrorTag::RELGRAD}
        };

        bool matched = false;
        for (const auto& [key, test] : tests) {
            if (ppr.countval(key)) {
                Vector<Real> value;
                ppr.getarr(key, value, 0, ppr.countval(key));
                std::string field;
                ppr.get("field_name", field);
                tag_criteria.emplace_back(value, test, field, info);
                matched = true;
                break;
            }
        }
        if (!matched) {
            amrex::Abort("Unrecognized refinement indicator for " + refine_criterion);
        }
    }

}

void
HydroSim::microphys_init ()
{

  if (ParallelDescriptor::IOProcessor()) {
    std::cout << "reading extern runtime parameters ..." << std::endl;
  }

  init_extern_parameters();

}

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
