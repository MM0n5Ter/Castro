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
// the gravity object
std::unique_ptr<Gravity> HydroSim::gravity;
#endif

#ifdef DIFFUSION
// the diffusion object
std::unique_ptr<Diffusion> HydroSim::diffusion;
#endif

#ifdef RADIATION

// the radiation object
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

// this records whether we have done tuning on the hydro tile size
int          HydroSim::sweep_tile_tuned = 0;
Long         HydroSim::largest_tuned_box_pts = 0;

// this will be reset upon restart
Real         HydroSim::priorCpuSeconds = 0.0;

Real         HydroSim::startCpuSeconds = 0.0;

int          HydroSim::Forcing_Type_Index = -1;
int          HydroSim::num_field_types = 0;

int          HydroSim::user_init_pending = 0;


// HydroSim::variableSetUp is in HydroSim_setup.cpp
// variableCleanUp is called once at the end of a simulation
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

    // C++ cleaning
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

    // this gets all of the parameters defined in _cpp_params, regardless of
    // namespace
    load_compiled_params();

    ParmParse pp("hydrosim");
    ParmParse ppa("amr");

    using namespace hydrosim;


    // Get boundary conditions
    Vector<int> lo_bc(AMREX_SPACEDIM), hi_bc(AMREX_SPACEDIM);
    pp.getarr("lo_bc",lo_bc,0,AMREX_SPACEDIM);
    pp.getarr("hi_bc",hi_bc,0,AMREX_SPACEDIM);
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        phys_bc.setLo(i,lo_bc[i]);
        phys_bc.setHi(i,hi_bc[i]);
    }

    const Geometry& default_geom = DefaultGeometry();

    //
    // Check phys_bc against possible periodic geometry
    // if periodic, must have internal BC marked.
    //
    if (default_geom.isAnyPeriodic())
    {
        //
        // Do idiot check.  Periodic means interior in those directions.
        //
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
        //
        // Do idiot check.  If not periodic, should be no interior.
        //
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
    // We do not support hybrid advection when using the CONTACT_FLUX solver.

    if (riemann_method == 2) {
        amrex::Abort("CONTACT_FLUX Riemann solver unsupported when using hybrid momentum.");
    }
#endif

    // sanity checks

    if (restart_grow_factor < 1) {
      amrex::Error("restart_grow_factor must be integer >= 1");
    }

    if (cfl <= 0.0 || cfl > 1.0) {
      amrex::Error("Invalid CFL factor; must be between zero and one.");
    }

    // SDC does not support GPUs yet
#ifdef AMREX_USE_GPU
    if (time_scheme == IterativeCorrection) {
        amrex::Error("SDC is currently not enabled on GPUs.");
    }
#endif


    // Simplified SDC currently requires USE_SIMPLIFIED_SDC to be defined.
    // Also, if we have USE_SIMPLIFIED_SDC defined, we can't use the other
    // time integration_methods, because only the SDC burner
    // interface is available in Microphysics in this case.
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

    // Make sure not to call refluxing if we're not actually doing any hydro.
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

    // This interpolation is not currently implemented for some geometries
    if (linear_interp_limiter == 2) {
        if (default_geom.IsSPHERICAL()) {
            amrex::Error("linear_interp_limiter == 2 is not currently implemented for spherical geometries");
        }
        else if (default_geom.IsRZ()) {
            amrex::Error("lim_limit_state_interp == 2 is not currently implemented for cylindrical geometries");
        }
    }

    // Post-timestep regrids only make sense if we're subcycling.
    std::string subcycling_mode;
    ppa.query("subcycling_mode", subcycling_mode);
    if (regrid_each_step == 1 && subcycling_mode == "None") {
        amrex::Error("hydrosim.regrid_each_step == 1 is not consistent with amr.subcycling_mode = None.");
    }

#ifdef AMREX_PARTICLES
    load_particle_config();
#endif

#ifdef RADIATION
    // Some radiation parameters are initialized here because they
    // may be used in variableSetUp, well before the call to the
    // Radiation constructor,

    if (do_radiative_transfer == 1) {
      Radiation::load_static_settings();
    }

    // radiation is only supported with CTU
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

   // SCF initial model construction can only be done if both
   // rotation and gravity have been compiled in.

#if !defined(GRAVITY) || !defined(ROTATION)
   if (do_equilibrium_start) {
       amrex::Error("SCF initial model construction is only permitted if USE_GRAV=TRUE and USE_ROTATION=TRUE at compile time.");
   }
#endif

   StateDescriptor::setBndryFuncThreadSafety(boundary_callback_threadsafe);

   // Open up HydroSim data logs
   // Note that this functionality also exists in the Amr class
   // but we implement it on our own to have a little more control.
   // Some of these will only be filled for certain ifdefs, but
   // we should use consistent indexing regardless of ifdefs (so some
   // logs may be unused in a given run).

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

    // Override Amr defaults. Note: this function is called after Amr::Initialize()
    // in Amr::InitAmr(), right before the ParmParse checks, so if the user opts to
    // override our overriding, they can do so.

    Amr::setComputeNewDtOnRegrid(true);

    // Read in custom refinement scheme.

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
            // the default max_level of AMRErrorTagInfo is 1000, but make sure
            // that it is reasonable for HydroSim
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

    // Coterminous AMR boundaries are not supported in HydroSim if we're doing refluxing.

    if (do_hydro == 1 && do_reflux == 1) {
        for (int ilev = 0; ilev <= parent->maxLevel(); ++ilev) {
            if (parent->nErrorBuf(ilev) == 0) {
                amrex::Error("n_error_buf = 0 is unsupported when using hydro.");
            }
        }
    }

    // initialize all the new time level data to zero
    for (int k = 0; k < num_field_types; k++) {
      MultiFab& data = get_new_data(k);
      data.setVal(0.0, data.nGrow());
    }

#ifdef GRAVITY

    if (do_gravity_flag == 1) {
      // gravity is a static object, only alloc if not already there
      if (gravity == nullptr) {
        gravity = std::make_unique<Gravity>(parent, parent->finestLevel(), &phys_bc, URHO);
      }

      // Passing grid_points_1d at level 0
      if (!level_geom.isAllPeriodic() && gravity != nullptr)
      {
         int grid_points_1d = count_points_1d();

         // For 1D, we need to add ghost cells to the numpts
         // given to us by HydroSim.

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
      // diffusion is a static object, only alloc if not already there
      if (diffusion == nullptr) {
        diffusion = std::make_unique<Diffusion>(parent, &phys_bc);
      }

      diffusion->attach_level(level,this,volume,area.data());
#endif

#ifdef RADIATION
    if (do_radiative_transfer) {
      if (radiation == nullptr) {
        // radiation is a static object, only alloc if not already there
        radiation = std::make_unique<Radiation>(parent, this);
        global::radiation_instance = radiation.get();
      }
      radiation->regrid(level, grids, dmap);

      rad_operator = std::make_unique<RadTransportSolver>(parent, level, grids, dmap);
    }
#endif

    // now check the runtime parameters to warn / abort if the user set
    // anything that isn't known to HydroSim

    check_runparams();

}

HydroSim::~HydroSim ()  // NOLINT(modernize-use-equals-default)
{
#ifdef RADIATION
    if (radiation != nullptr) {
      //radiation->cleanup(level);
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

// Initialize the MultiFabs and flux registers that live as class members.

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
            // we have 2 components: first half and second half
            burn_split_data.define(grids, dmap, 2, 0);
        }
        else if (hydrosim::time_scheme == SimplifiedIterCorrection) {
            // we have a component for each sdc iteration + 1 extra for retries
            burn_split_data.define(grids, dmap, deferred_iters+1, 0);
        }

        burn_split_data.setVal(0.0);
    }
#endif

    // Set the flux register scalings.

    if (do_reflux) {

        flux_coarse_scale = -1.0;
        flux_refined_scale = 1.0;

        // The fine pressure scaling depends on dimensionality,
        // as the dimensionality determines the number of
        // adjacent zones. In 1D the face is a point so
        // there's only one fine neighbor for a given coarse
        // face; in 2D there's crse_ratio[1] faces adjacent
        // to a face perpendicular to the radial dimension;
        // and in 3D there would be crse_ratio**2, though
        // we do not separate the pressure out in 3D. Note
        // that the scaling by dt has already been handled
        // in the construction of the Pressure_radial array.

        // The coarse pressure scaling is the same as for the
        // fluxes, we want the total refluxing contribution
        // over the full set of fine timesteps to equal Pressure_radial.

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

      // sync up some C++ values of the runtime parameters

      user_init_pending = 1;

      // If we're doing C++ problem initialization, do it here. We have to make
      // sure it's done after the above call to init_prob_parameters() in case
      // any changes are made to the problem parameters.

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

    //
    // Loop over grids, call initialization functions
    //
#if AMREX_SPACEDIM > 1
    const Real* dx  = geom.CellSize();
#endif
    MultiFab& S_new = get_new_data(State_Type);
    Real cur_time   = state[State_Type].curTime();

    S_new.setVal(0.);

    if (! allow_anisotropic_cells) {

        // make sure dx = dy = dz -- that's all we guarantee to support
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

    // Don't profile for this code, since there will be a lot of host
    // activity and GPU page faults that we're uninterested in.

    Gpu::Device::profilerStop();

#ifdef RADIATION
    // rad quantities are in the state even if (do_radiative_transfer == 0)
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
              // C++ MHD problem initialization; has no effect if not
              // implemented by a problem setup (defaults to an empty
              // routine).
              user_initialize_mhd_data(i, j, k, magfield_x_arr, 0, geomdata);
          });

          const Box& face_box_y = mfi.nodaltilebox(1);

          amrex::ParallelFor(face_box_y,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              // C++ MHD problem initialization; has no effect if not
              // implemented by a problem setup (defaults to an empty
              // routine).
              user_initialize_mhd_data(i, j, k, magfield_y_arr, 1, geomdata);
          });

          const Box& face_box_z = mfi.nodaltilebox(2);

          amrex::ParallelFor(face_box_z,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              // C++ MHD problem initialization; has no effect if not
              // implemented by a problem setup (defaults to an empty
              // routine).
              user_initialize_mhd_data(i, j, k, magfield_z_arr, 2, geomdata);
          });

       }

#endif //MHD

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
              // problem initialization
              user_initialize_state_data(i, j, k, s, geomdata, engine);
          });
#else
          amrex::ParallelFor(box,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              // problem initialization
              user_initialize_state_data(i, j, k, s, geomdata);
          });
#endif
       }


#ifdef MHD
      //correct energy density with the magnetic field contribution
      add_bfield_energy(magfield_x_new, magfield_y_new, magfield_z_new, S_new);

      //check magDivergence
      check_div_bfield(magfield_x_new, magfield_y_new, magfield_z_new, S_new);

#endif

       // it is not a requirement that the problem setup defines the
       // temperature, so we do that here _and_ ensure that we are
       // within any small limits
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
               // if the problem tried to initialize a thermodynamic
               // state that is at or below small_temp, then we abort.
               // This is dangerous and we should recommend a smaller
               // small_temp
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
       // Generate the initial hybrid momenta based on this user data.

       linear_to_split_momentum(S_new, 0);
#endif

       // Verify that the sum of (rho X)_i = rho at every cell

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
         // we are assuming that the initialization was done to cell-centers

         // Enforce that the total and internal energies are consistent.
         reconcile_energy(
#ifdef MHD
                            magfield_x_new, magfield_y_new,magfield_z_new,
#endif
                            S_new);

         // For fourth-order, we need to convert to cell-averages now.
         // (to second-order, these are cell-averages, so we're done in that case).

#ifndef AMREX_USE_GPU
         if (sdc_accuracy_order == 4) {
           Sborder.define(grids, dmap, NUM_STATE, NUM_GROW);
           AmrLevel::FillPatch(*this, Sborder, NUM_GROW, cur_time, State_Type, 0, NUM_STATE);

           // note: this cannot be tiled
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

           // now copy back the averages
           MultiFab::Copy(S_new, Sborder, 0, 0, NUM_STATE, 0);
           Sborder.clear();
         }
#endif
       } else {

         Sborder.define(grids, dmap, NUM_STATE, NUM_GROW);
         AmrLevel::FillPatch(*this, Sborder, NUM_GROW, cur_time, State_Type, 0, NUM_STATE);

         // convert to centers -- not tile safe
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

         // reset the energy -- do this in one ghost cell so we can average in place below
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

         // convert back to averages -- not tile safe
         for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
           {
             const Box& box = mfi.validbox();

             tmp.resize(box, 1, The_Async_Arena());
             auto tmp_arr = tmp.array();

             build_fourth_in_place(box, Sborder.array(mfi), tmp_arr, domain_lo, domain_hi);
           }

         // now copy back the averages for UEINT and UTEMP only
         MultiFab::Copy(S_new, Sborder, UEINT, UEINT, 1, 0);
         MultiFab::Copy(S_new, Sborder, UTEMP, UTEMP, 1, 0);
         Sborder.clear();

       }
#else
       // Enforce that the total and internal energies are consistent.
       reconcile_energy(
#ifdef MHD
                            magfield_x_new, magfield_y_new,magfield_z_new,
#endif
                            S_new);
#endif

       // Do a FillPatch so that we can get the ghost zones filled.

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
              // C++ problem initialization; has no effect if not implemented
              // by a problem setup (defaults to an empty routine).

              user_initialize_rad_data(i, j, k, r, nu_edge_pass, nu_center_pass, nu_width_pass, geomdata);
          });
      }
    }
#endif // RADIATION

#endif // LOWMACH_INIT


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

    //
    // Create new grid data by fillpatching from old.
    //
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

    // Copy some other data we need from the old class.
    // One reason this is necessary is if we are doing
    // a post-timestep regrid -- then we're going to need
    // to save information about whether there was a retry
    // during the timestep.

    iteration = oldlev->iteration;
    sub_pass = oldlev->sub_pass;

    sub_cycles = oldlev->sub_cycles;
    dt_cycle_step = oldlev->dt_cycle_step;
    dt_level_step = oldlev->dt_level_step;

    retain_prev_field = oldlev->retain_prev_field;

    retry_active = oldlev->retry_active;

}

//
// This version inits the data on a new level that did not
// exist before regridding.
//
void
HydroSim::init ()
{
    BL_PROFILE("HydroSim::init()");

    Real dt        = parent->dtLevel(level);
    Real cur_time  = getLevel(level-1).state[State_Type].curTime();
    Real prev_time = getLevel(level-1).state[State_Type].prevTime();

    Real dt_old = (cur_time - prev_time) / static_cast<Real>(parent->MaxRefRatio(level-1));

    Real time = cur_time;

    // If we just triggered a regrid, we need to account for the fact that
    // the data on the coarse level has already been advanced.

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

    // If we're not subcycling, we only need to do timestep estimation on leaf cells.

    bool exclude_covered_zones = false;

    if (level < parent->finestLevel() && parent->subcyclingMode() == "None") {
        exclude_covered_zones = true;
    }

    MultiFab tmp_coverMask;
    const MultiFab& coverMask = exclude_covered_zones ? getLevel(level+1).make_fine_mask() : tmp_coverMask;

    std::string limiter = "hydrosim.max_dt";

    // Start the hydro with the max_dt value, but divide by CFL
    // to account for the fact that we multiply by it at the end.
    // This ensures that if max_dt is more restrictive than the hydro
    // criterion, we will get exactly max_dt for a timestep.

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


        // Determine if this is more restrictive than the maximum timestep limiting

        if (dtlimit_hydro < estdt) {
            limiter = "hydro";
            estdt = dtlimit_hydro;
        }

    }

#ifdef DIFFUSION
    // Diffusion-limited timestep
    // Note that the diffusion uses the same CFL safety factor
    // as the main hydrodynamics timestep limiter.

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

    // Determine if this is more restrictive than the hydro limiting

    if (dtlimit_diffusion < estdt) {
        limiter = "diffusion";
        estdt = dtlimit_diffusion;
    }
#endif  // diffusion

#ifdef REACTIONS
    // Dummy value to start with
    Real dtlimit_burn = max_dt;

    if (do_react && (hydrosim::burn_dt_energy < 1.e199_rt || hydrosim::burn_dt_massfrac < 1.e199_rt)) {

        // Compute burning-limited timestep.

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

        // Determine if this is more restrictive than the hydro limiting

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
                      int                    /*sub_cycle*/,
                      Vector<int>&           n_cycle,
                      const Vector<IntVect>& /*ref_ratio*/,
                      Vector<Real>&          dt_min,
                      Vector<Real>&          dt_level,
                      Real                   stop_time,
                      int                    post_regrid_flag)
{
    BL_PROFILE("HydroSim::computeNewDt()");

    //
    // We are at the start of a coarse grid timecycle.
    // Compute the timesteps for the next iteration.
    //
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
          //
          // Limit dt's by pre-regrid dt
          //
          for (int i = 0; i <= finest_level; i++)
          {
              dt_min[i] = std::min(dt_min[i],dt_level[i]);
          }
       }
       else
       {
          //
          // Limit dt's by change_max * old dt,
          // if we didn't limit the last timestep
          // to hit a plotfile interval.
          //
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

    //
    // Find the minimum over all levels
    //
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_min[i]);
    }

    //
    // Optionally, limit dt's by the value of
    // plot_per or small_plot_per.
    //
    if (plot_interval_exact) {

        const Real plot_per = parent->plotPer();

        if (plot_per > 0.0) {

            const Real cur_time = state[State_Type].curTime();

            // Calculate the new dt by comparing to the dt needed to get
            // to the next multiple of plot_per.

            const Real phaseOffset = std::fmod(cur_time, plot_per);

            Real nextPlotDt;

            // Note that if we are just about exactly on a multiple of plot_per,
            // then we need to be careful to avoid floating point issues.

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

                // Avoid taking timesteps that are so small that
                // they may cause problems in the hydrodynamics.

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

            // Same logic as for plot_interval_exact.

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

    //
    // Limit dt's by the value of stop_time.
    //
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
                          int                   /*subcycle*/,
                          Vector<int>&           n_cycle,
                          const Vector<IntVect>& /*ref_ratio*/,
                          Vector<Real>&          dt_level,
                          Real                  stop_time)
{
    BL_PROFILE("HydroSim::computeInitialDt()");

    //
    // Grids have been constructed, compute dt for all levels.
    //
    if (level > 0) {
      return;
    }

    int i;

    Real dt_0 = 1.0e+100;
    int n_factor = 1;
    // TODO/DEBUG: This will need to change for optimal subcycling.
    for (i = 0; i <= finest_level; i++)
    {
        dt_level[i] = getLevel(i).initialTimeStep();
        n_factor   *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_level[i]);
    }

    //
    // Limit dt's by the value of stop_time.
    //
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

    //
    // Integration cycle on fine level grids is complete
    // do post_timestep stuff here.
    //
    int finest_level = parent->finestLevel();

#ifdef RADIATION
    if (do_radiative_transfer && (level < finest_level)) {
        // computeTemp is not needed before or after this call because
        // setup for deferred sync does not touch state, only flux registers.
        radiation->prepare_delayed_reflux(level);

        if (do_reflux) {
            radiation->reflux(level);
            // Since radiation->reflux does not touch the fluid state,
            // we do need to recompute Temp here.
        }
    }
#endif

    // Now do the refluxing. If we're using gravity it
    // will also do the sync solve associated with the reflux.

    if (do_reflux && level < parent->finestLevel()) {
        if (parent->subcyclingMode() != "None") {
            reflux(level, level+1, true);
        }
    }

    // Ensure consistency with finer grids.

    if (level < finest_level) {
        avgDown();
    }

#ifdef MHD
    MultiFab& magfield_x_new = get_new_data(Bfield_Type_x);
    MultiFab& magfield_y_new = get_new_data(Bfield_Type_y);
    MultiFab& magfield_z_new = get_new_data(Bfield_Type_z);
#endif

    // Clean up any aberrant state data generated by the reflux and average-down,
    // and then update quantities like temperature to be consistent.
    MultiFab& S_new = get_new_data(State_Type);
    clean_state(
#ifdef MHD
                magfield_x_new, magfield_y_new, magfield_z_new,
#endif
                S_new, state[State_Type].curTime(), S_new.nGrow());


#ifdef DO_USER_POST_TIMESTEP

    // Provide a hook for the user to do things after all of
    // the normal updates have been applied. The user is
    // responsible for any actions after this point, like
    // doing a computeTemp call if they change the state data.

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
    // diagnostic stuff

    if (level == 0) {
      print_energy_budget();
    }
#endif

#ifdef AMREX_PARTICLES
    if (TracerPC)
    {
        const int ncycle = parent->nCycle(level);
        //
        // Don't redistribute/timestamp on the final subiteration except on the coarsest grid.
        //
        if (cycle_index < ncycle || level == 0)
        {
            int ngrow = (level == 0) ? 0 : cycle_index;

            TracerPC->Redistribute(level, parent->finestLevel(), ngrow);

            StampParticles(ngrow+1);
        }
    }
#endif

    // Check to see if the user-supplied stopping criterion has been met.

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

            // Write out a checkpoint.

            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::ofstream stop_flag_file;
                stop_flag_file.open("dump_and_stop", std::ofstream::out);
                stop_flag_file.close();
            }

            // Write out a file that indicates the job has completed.

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
            // Passing grid_points_1d at level 0
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
                // Update the maximum density, used in setting the solver tolerance.

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
      // diffusion is a static object, only alloc if not already there
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
    // postCoarseTimeStep() is only called by level 0.
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

               // In the case where we're coming here during a regrid that occurs
               // after a timestep, we only want to interpolate the gravitational
               // field from the old time. The state data will already have been
               // filled, so all we need to do is interpolate the phi_gradient data.

               // Instantiate a bare physical BC function for phi_gradient. It doesn't do anything
               // since the fine levels for Poisson gravity do not touch the physical boundary.

               PotentialGradBCFunct gradient_phys_bc;

               // We need to use a interpolater that works with data on faces.

               Interpolater* gradient_interp = &face_linear_interp;

               Vector<MultiFab*> potential_gradient_coarse = amrex::GetVecOfPtrs(gravity->get_potential_gradient_prev(level-1));
               Vector<MultiFab*> potential_gradient_fine = amrex::GetVecOfPtrs(gravity->get_potential_gradient_curr(level));

               Real time = getLevel(lbase).get_state_data(Accel_Type).prevTime();

               // For the BCs, we will use the Accel_Type BCs for convenience, but these will
               // not do anything because we do not fill on physical boundaries.

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
                    // Update the maximum density, used in setting the solver tolerance.

                    if (level == 0) {
                      gravity->update_max_density();
                    }

                    gravity->multilevel_potential_solve(level, new_finest);

                }

            }

        }

    }
#endif

    // Ensure regridded data is valid. This addresses the fact that data
    // on this level that was interpolated from a coarser level may not
    // respect the consistency between certain state variables
    // (e.g. UEINT and UEDEN) that we demand in every zone.

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
HydroSim::post_init (Real /*stop_time*/)
{
    BL_PROFILE("HydroSim::post_init()");

    if (level > 0) {
        return;
    }

    //
    // Average data down from finer levels
    // so that conserved data is consistent between levels.
    //
    int finest_level = parent->finestLevel();
    for (int k = finest_level-1; k>= 0; k--) {
      getLevel(k).avgDown();
    }

#ifdef GRAVITY

    if (do_gravity_flag) {

       Real cur_time = state[State_Type].curTime();

       if (gravity->get_solver_kind() == "PoissonGrav") {

          // Update the maximum density, used in setting the solver tolerance.

          gravity->update_max_density();

          // Calculate offset before first multilevel solve.
          gravity->set_mass_shift(cur_time);

          gravity->multilevel_potential_solve(level,finest_level);
          if (gravity->get_self_check_flag() == 1) {
              gravity->check_composite_potential(level);
          }
       }

       // Make this call just to fill the initial state data.
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
      // The option of whether to do a multilevel initialization is
      // controlled within the radiation class.

      radiation->post_init(level);

      for (int k = finest_level-1; k>= 0; k--) {
        getLevel(k).avgDown(Radiation_Type);
      }

      print_energy_budget();

      // re-estimate the initial timestep using the initialized
      // fine-level data and radiation field.
      //if (level == 0)
      //post_init_estDT();
    }
#endif

// Allow the user to define their own post_init functions.

#ifdef DO_USER_POST_INIT

    user_post_init();

#endif

    // If we're doing SCF initialization, do it here.

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

          // Update the maximum density, used in setting the solver tolerance.

          gravity->update_max_density();

          // Calculate offset before first multilevel solve.
          gravity->set_mass_shift(cur_time);

          gravity->multilevel_potential_solve(level,finest_level);
          if (gravity->get_self_check_flag() == 1) {
              gravity->check_composite_potential(level);
          }
       }

       // Make this call just to fill the initial state data.
       for (int k = 0; k <= parent->finestLevel(); k++)
       {
          MultiFab& gravity_new = getLevel(k).get_new_data(Accel_Type);
          gravity->get_new_gravity_vector(k,gravity_new,cur_time);
       }
    }
#endif

#ifdef RADIATION
    if (do_radiative_transfer) {
      // The option of whether to do a multilevel initialization is
      // controlled within the radiation class.

      radiation->post_init(level);

      int finest_level = parent->finestLevel();

      for (int k = finest_level-1; k>= 0; k--) {
        getLevel(k).avgDown(Radiation_Type);
      }

      print_energy_budget();

      // re-estimate the initial timestep using the initialized
      // fine-level data and radiation field.
      //if (level == 0)
      //post_init_estDT();
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

// reflux() synchronizes fluxes between levels and has two modes of operation.
//
// When from_post_step = true, we are performing the reflux in AmrLevel's
// post_timestep() routine. This takes place after this level has completed
// its advance as well as all fine levels above this level have completed
// their advance. The main activities are:
//
// (1) Limit the flux correction so that it won't cause unphysical densities.
// (2) Update any copies of the full flux arrays with the flux correction.
// (3) Perform the flux correction on StateData using FluxRegister's Reflux().
// (4) Do a sync solve for Poisson gravity now that the density has changed.
// (5) Recompute the new-time sources now that StateData has changed.
//
// When from_post_step = false, we are performing the reflux immediately
// after the hydro step has taken place on all levels between crse_level and
// fine_level. This happens when amr.subcycling_mode = None, or in general when
// the fine level does not subcycle relative to the coarse level. In this mode
// we can skip steps (4) and (5), as those steps are only necessary to fix the
// updates on these levels that took place without full information about the
// hydro solve being synchronized. The advance on crse_level and fine_level will
// then be followed by a normal computation of the new-time gravity and new-time
// sources, which will not need to be corrected later.

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

        // Get a version of this state with one ghost zone.

        MultiFab ghosted_coarse_state(crse_state.boxArray(), crse_state.DistributionMap(), crse_state.nComp(), 1);

        crse_lev.fill_ghost_field(ghosted_coarse_state, crse_lev.state[State_Type].curTime(), 1);

        // Clear out the data that's not on coarse-fine boundaries so that this register only
        // modifies the fluxes on coarse-fine interfaces.

        reg->ClearInternalBorders(crse_lev.geom);

        // The reflux operation can cause a small or negative density (for the same reason this
        // can happen during the level advance). We want to avoid this scenario because our
        // only recourse will be a density reset, which is disruptive. So before we apply the reflux,
        // we need to clean up the flux register to limit any fluxes that would do this.
        // This is nonconservative, since we do not go back and retroactively apply any
        // correction on the fine grid. (Of course, a density reset would also be nonconservative.)
        // We assume that the amount of fluid material lost this way is small since refluxes
        // causing a small density should only happen around ambient material.

        // The simplest way to do this is make a copy of the flux register to a MultiFab,
        // then loop through the data and calculate what the reflux operation would be,
        // limiting the flux if it would result in a negative density. Then we overwrite
        // the flux register with the updated data. This is more straightforward than operating
        // on the flux register data directly, and we will anyway need this copy of the flux
        // data in MultiFab form later.

        // We also apply a similar check to ensure that 0 < X < 1 after the reflux.

        MultiFab scratch_fluxes[AMREX_SPACEDIM];

        for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

            scratch_fluxes[idir].define(crse_lev.fluxes[idir]->boxArray(),
                                     crse_lev.fluxes[idir]->DistributionMap(),
                                     crse_lev.fluxes[idir]->nComp(), crse_lev.fluxes[idir]->nGrow());

            scratch_fluxes[idir].setVal(0.0);

            // Start with a MultiFab version of the flux register.

            for (OrientationIter fi; fi.isValid(); ++fi) {
                const FabSet& fs = (*reg)[fi()];
                if (fi().coordDir() == idir) {
                    fs.copyTo(scratch_fluxes[idir], 0, 0, 0, scratch_fluxes[idir].nComp());
                }
            }

        }

        // Now zero out any problematic flux corrections.

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        for (MFIter mfi(crse_state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();

            auto U = ghosted_coarse_state[mfi].array();
            auto V = crse_lev.volume[mfi].array();

            // Limit fluxes that would cause a small/negative density.
            // Also check to see whether the flux would cause invalid X. We use a
            // safety factor of AMREX_SPACEDIM since multiple fluxes touching the
            // same zone could be conspiring in the same direction. If we do detect
            // a case where X would be invalid, we set that flux to zero.

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

            // Update the flux register now that we may have modified some of the flux corrections.

            for (OrientationIter fi; fi.isValid(); ++fi) {
                if (fi().coordDir() == idir) {
                    FabSet& fs = (*reg)[fi()];
                    fs.copyFrom(scratch_fluxes[idir], 0, 0, 0, scratch_fluxes[idir].nComp());
                }
            }

            // Update the coarse fluxes MultiFabs using the reflux data. This should only make
            // a difference if we re-evaluate the source terms later.

            if (refresh_sources_after_reflux || !from_post_step) {

                MultiFab::Add(*crse_lev.fluxes[idir], scratch_fluxes[idir], 0, 0, crse_lev.fluxes[idir]->nComp(), 0);

                // The gravity and rotation source terms depend on the mass fluxes.

                MultiFab::Add(*crse_lev.mass_face_flux[idir], scratch_fluxes[idir], URHO, 0, 1, 0);
            }

        }

        // Trigger the actual reflux on the coarse level now.

        reg->Reflux(crse_state, crse_lev.volume, 1.0, 0, 0, NUM_STATE, crse_lev.geom);

        // Store the density change, for the gravity sync.

#ifdef GRAVITY
        int ilev = lev - crse_level - 1;

        if (do_gravity_flag && gravity->get_solver_kind() == "PoissonGrav" && gravity->NoSync() == 0 && from_post_step) {
            reg->Reflux(*drho[ilev], crse_lev.volume, 1.0, 0, URHO, 1, crse_lev.geom);
            amrex::average_down(*drho[ilev + 1], *drho[ilev], 0, 1, getLevel(lev).crse_ratio);
        }
#endif


        // We no longer need the flux register data, so clear it out.

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

        // This follows the same logic as the pure hydro fluxes; see above for details.

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

            // Note that the scaling by the area here is corrected for by dividing by the
            // cell volume in the reflux. In this way we get a discrete divergence that
            // is analogous to the divergence of the flux in the hydrodynamics.
            // The dimensions of dphi are therefore actually
            // phi / cm**2, which makes it correct for the RHS of the Poisson equation.

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

    // Do the sync solve across all levels.

#ifdef GRAVITY
    if (do_gravity_flag && gravity->get_solver_kind() == "PoissonGrav" && gravity->NoSync() == 0 && from_post_step) {
      gravity->synchronize_gravity_levels(crse_level, fine_level, amrex::GetVecOfPtrs(drho), amrex::GetVecOfPtrs(dphi));
    }
#endif

    // Now subtract the new-time updates to the state data,
    // recompute it, and add it back. This corrects for the fact
    // that the new-time data was calculated the first time around
    // using a state that hadn't yet been refluxed. Note that this
    // needs to come after the gravity sync solve because the gravity
    // source depends on an up-to-date value of phi. We'll do this
    // on the fine level in addition to the coarser levels, because
    // global sources like gravity or source terms that rely on
    // ghost zone fills like diffusion depend on the data in the
    // coarser levels.

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
            Real dt_level_advance = getLevel(lev).dt_level_step; // Note that this may be shorter than the full timestep due to subcycling.
            Real dt_expected = parent->dtLevel(lev); // The full timestep expected by the Amr class.

            if (getLevel(lev).apply_source_terms()) {

                getLevel(lev).apply_source_term_to_state(S_new, source, -dt_level_advance, 0);
                getLevel(lev).clean_state(
#ifdef MHD
                                          magfield_x_new, magfield_y_new, magfield_z_new,
#endif
                                          S_new, time, 0);

            }

            // Temporarily restore the last iteration's old data for the purposes of recalculating the corrector.
            // This is only necessary if we've done subcycles on that level.

            if (allow_step_retry && dt_level_advance < dt_expected && getLevel(lev).retain_prev_field) {

                for (int k = 0; k < num_field_types; k++) {

                    if (getLevel(lev).prev_state[k]->hasOldData()) {

                        // Use the new-time data as a temporary buffer. Ideally this would be done
                        // as a pointer swap, but we cannot assume that the distribution mapping
                        // is the same between the current state and the original state, due to
                        // possible regrids that have occurred in between.

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

                        // Now retrieve the original old time data.

                        MultiFab& old = getLevel(lev).get_old_data(k);
                        MultiFab::Copy(old, getLevel(lev).prev_state[k]->newData(), 0, 0, old.nComp(), old.nGrow());

                        getLevel(lev).state[k].setTimeLevel(time, dt_expected, 0.0);
                        getLevel(lev).prev_state[k]->setTimeLevel(time, dt_level_advance, 0.0);

                    }

                }

                // Now deallocate the old data, it is no longer needed.

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

        // Ensure the species mass fractions are between small_x and 1,
        // then normalize them so that they sum to 1.

        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
        {

            auto cons_cell = u.cellData(i, j, k);
            Real rhoX_sum = 0.0_rt;
            Real rho = cons_cell[URHO];
            Real rhoInv = 1.0_rt / rho;

            int failed{};

            for (int n = 0; n < NumSpec; ++n) {
                // Abort if X is unphysically large.
                Real X = cons_cell[UFS+n] * rhoInv;

                // Only do the abort check if the density is greater than a user-defined cutoff.
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

    // This routine sets the density in state_in to be larger than the
    // density floor.  Note that it will operate everywhere on state_in,
    // including ghost zones.

    MultiFab reset_delta;

    if (log_source_changes)
    {

        // Before we do anything, make a copy of the state.

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
        // Evaluate what the effective reset source was.

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

            // Optionally, the user can ignore this if the starting
            // density is lower than a certain threshold. This is useful
            // if the minimum density occurs in material that is not
            // dynamically important; in that case, a density reset suffices.

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

    // This routine sets the velocity in state_in to be no larger than the
    // speed limit, if one has been applied.

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
                  int          /*clearval*/,
                  int          /*tagval*/,
                  Real         time,
                  int          /*n_error_buf*/,
                  int          /*ngrow*/)
{
    BL_PROFILE("HydroSim::errorEst()");

    Real tag_time = time;

    // If we are forcing a post-timestep regrid,
    // note that we need to use the new time here,
    // not the old time.

    if (post_step_regrid) {
      tag_time = get_state_data(State_Type).curTime();
    }

    // Apply each of the tagging criteria defined in the inputs.

    for (const auto & etag : tag_criteria) {
        std::unique_ptr<MultiFab> mf;
        if (! etag.Field().empty()) {
            mf = derive(etag.Field(), tag_time, etag.NGrow());
        }
        etag(tags, mf.get(), TagBox::CLEAR, TagBox::SET, tag_time, level, geom);
    }

    // Now we'll tag any user-specified zones using the full state array.

    apply_user_tags(tags, tag_time);

    // Finally we'll apply any tagging restrictions which must be obeyed by any setup.

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

    // If we are using Poisson gravity, we must ensure that the outermost zones are untagged
    // due to the Poisson equation boundary conditions (we currently do not know how to fill
    // the boundary conditions for fine levels that touch the physical boundary.)
    // To do this properly we need to be aware of AMReX's strategy for tagging, which is not
    // cell-based, but rather chunk-based. The size of the chunk on the coarse grid is given
    // by blocking_factor / ref_ratio -- the idea here being that blocking_factor is the
    // smallest possible group of cells on a given level, so the smallest chunk of cells
    // possible on the coarse grid is given by that number divided by the refinement ratio.
    // So we cannot tag anything within that distance from the boundary. Additionally we
    // need to stay a further amount n_error_buf away, since n_error_buf zones are always
    // added as padding around tagged zones.

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

    // Allow the user to limit tagging outside of some distance from the problem center.
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
  // initialize the external runtime parameters

  if (ParallelDescriptor::IOProcessor()) {
    std::cout << "reading extern runtime parameters ..." << std::endl;
  }

  // read any runtime parameters directly from inputs (via ParmParse)
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

        // Ensure the internal energy is at least as large as this minimum
        // from the EOS; the same holds true for the total energy.

        cons_cell[UEINT] = amrex::max(cons_cell[UEINT], cons_cell[URHO] * min_eint);
        cons_cell[UEDEN] = amrex::max(cons_cell[UEDEN], cons_cell[URHO] * (min_eint + ke) + magnetic_energy);

        // Apply the dual energy criterion: get e from E if (E - K) > eta * E.

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

    // Make a copy of the state so we can evaluate how much changed.

    if (log_source_changes)
    {
        old_state.define(State.boxArray(), State.DistributionMap(), State.nComp(), 0);
        MultiFab::Copy(old_state, State, 0, 0, State.nComp(), 0);
    }

    // Ensure (rho e) isn't too small or negative
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
        // Evaluate what the effective reset source was.

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

  // for 4th order, the only variables that may change here are Temp
  // and Eint.  To ensure that we don't modify values unless there is
  // a reset for Eint, we want to store the Laplacian that we use for
  // the cell-average -> cell-center conversion so we can use the same
  // Laplacian to convert back, resulting only in roundoff changes if
  // Eint is not modified.  We have to store it here, since we
  // overwrite the grown state as we work.
  MultiFab energy_laplacian;

#ifdef TRUE_SDC
  if (sdc_accuracy_order == 4) {

    // we need to make the data live at cell-centers first

    // fill StateCopy with S_new.
    // we only need 2 ghost cells here, then the build_center_state
    // makes 1 ghost cell a valid center, we compute its temp, and
    // then the final average results only in interior temps valid
    StateCopy.define(State.boxArray(), State.DistributionMap(), NUM_STATE, 2);
    fill_ghost_field(StateCopy, time, StateCopy.nGrow());

    // store the Laplacian term for the internal energy
    energy_laplacian.define(State.boxArray(), State.DistributionMap(), 1, 0);

    // convert to cell centers -- this will result in StateCopy being
    // cell centered only on 1 ghost cells
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
    // we need to enforce minimum density here, since the conversion
    // from cell-average to centers could have made rho < 0 near steep
    // gradients
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
          // only one ghost cell is at cell-centers
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
          eos_state.T   = u(i,j,k,UTEMP); // Initial guess for the EOS
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

    // we need to copy back from StateCopy into S_new, making it
    // cell-average in the process.  For temperature, we will
    // construct the Laplacian from the new state and use for the
    // correction, since the temperature was just updated.  For the
    // internal energy, most zones will not have been reset, so we
    // don't want to modify them from what they were before, therefore
    // we use the stored Laplacian computed above to convert back to
    // cell-averages -- this is 4th-order and will be a no-op for
    // those zones where e wasn't changed.

    auto domain_lo = geom.Domain().loVect3d();
    auto domain_hi = geom.Domain().hiVect3d();

    FArrayBox tmp;

    for (MFIter mfi(StateCopy); mfi.isValid(); ++mfi) {

      const Box& bx = mfi.tilebox();

      tmp.resize(bx, 1, The_Async_Arena());
      auto tmp_arr = tmp.array();

      // only temperature
      build_fourth_in_place_n(bx, StateCopy.array(mfi), UTEMP, tmp_arr, domain_lo, domain_hi);
    }

    // correct UEINT
    MultiFab::Add(StateCopy, energy_laplacian, 0, UEINT, 1, 0);

    // copy back UTEMP and UEINT -- those are the only things that
    // should have changed.
    MultiFab::Copy(State, StateCopy, UTEMP, UTEMP, 1, 0);
    MultiFab::Copy(State, StateCopy, UEINT, UEINT, 1, 0);

    // now that we redid these, redo the ghost fill -- technically,
    // only need this for UTEMP and UEINT, and only if ng > 0
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

        // Optionally predict the source terms to t + dt/2,
        // which is the time-level n+1/2 value, To do this we use a
        // lagged predictor estimate: dS/dt_n = (S_n - S_{n-1}) / dt, so
        // S_{n+1/2} = S_n + (dt / 2) * dS/dt_n. We'll add the S_n
        // terms later; now we add the second term. We defer the
        // multiplication by dt / 2 until the actual advance, since
        // we may be subcycling and thus not know yet what the
        // advance timestep is.

        // Note that since for dS/dt we want (S^{n+1} - S^{n}) / dt,
        // we only need to take twice the new-time source term from the
        // last timestep, since in the predictor-corrector approach,
        // the new-time source term is 1/2 * S^{n+1} - 1/2 * S^{n}.
        // This is untrue in general for the non-momentum sources,
        // so for safety we'll only apply it to the momentum sources.

        // Even though we're calculating this predictor from the last
        // timestep, we've already done the swap, so the "new" data
        // from the last timestep is currently residing in the "old"
        // StateData. (As a corollary, this operation must be done
        // prior to updating any of the source StateData.) Since the
        // dt comes from the last timestep, which is no longer equal
        // to the difference between prevTime and curTime, we rely
        // on our recording of the last dt from the previous advance.

        const Real time = get_state_data(Forcing_Type).prevTime();

        AmrLevel::FillPatch(*this, forcing_corrector, SRC_GHOST_WIDTH, time, Forcing_Type, UMX, 3, UMX);

        forcing_corrector.mult(2.0 / lastStepDt, SRC_GHOST_WIDTH);

    }
    else if (time_scheme == SimplifiedIterCorrection && source_half_step == 1) {

        // If we're doing simplified SDC, our approach depends on
        // iteration.  For the first iteration, we use the corrector
        // from the previous timestep, which when combined with the
        // old source, will give us a prediction of the time-centered
        // source term.  For later iterations, we use the previous
        // iteration's corrector.  Note that we swap time-levels
        // before the start of the step, so for iteration 0 we access
        // the previous step's corrector as the old data.  But there
        // is no swap between iterations, so for later iterations, we
        // use the new timelevel.

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

        // The following is a hack to make sure that we only
        // ever have new data for certain state types that only
        // ever need new time data; by doing a swap now, we'll
        // guarantee that allocOldData() does nothing. We do
        // this because we never need the old data, so we
        // don't want to allocate memory for it.

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

    // Define a cube 3-on-a-side around the point with the maximum density
    FillPatch(*this,mf,0,time,State_Type,URHO,1);

#if AMREX_SPACEDIM >= 2
    int mi[3] = {0};
    for (int i = 0; i < AMREX_SPACEDIM; i++) {
      mi[i] = max_index[i];
    }
#endif

    // Find the position of the "center" by interpolating from data at
    // cell centers We only define a single 3x3 (or 3x3x3) box around
    // max_index above, so this MultiFab should have a single local
    // FAB.
    AMREX_ALWAYS_ASSERT(mf.local_size() == 1);

    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {

#if AMREX_SPACEDIM >= 2
        // it only makes sense for the center to move in 2- or 3-d

        const Box& box = mfi.validbox();
        auto data = mf.array(mfi);

        Real cen = data(mi[0], mi[1], mi[2]);

        amrex::ParallelFor(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            data(i,j,k) -= cen;
        });

        // This puts the "center" at the cell center
        Real snapped_center[3];
        for (int d = 0; d < AMREX_SPACEDIM; d++) {
            snapped_center[d] = geom.ProbLo(d) +  (static_cast<Real>(mi[d]) + 0.5_rt) * dx[d];
        }

        // Fit parabola y = a x^2  + b x + c through three points
        // a = 1/2 ( y_1 + y_-1)
        // b = 1/2 ( y_1 - y_-1)
        // x_vertex = -b / 2a

        // ... in x-direction
        Real a = 0.5_rt * (data(mi[0]+1, mi[1], mi[2]) + data(mi[0]-1, mi[1], mi[2]));
        Real b = 0.5_rt * (data(mi[0]+1, mi[1], mi[2]) - data(mi[0]-1, mi[1], mi[2]));
        Real x = -b / (2.0_rt * a);
        problem::center[0] = snapped_center[0] + x * dx[0];

        // ... in y-direction
        a = 0.5_rt * (data(mi[0], mi[1]+1, mi[2]) + data(mi[0], mi[1]-1, mi[2]));
        b = 0.5_rt * (data(mi[0], mi[1]+1, mi[2]) - data(mi[0], mi[1]-1, mi[2]));
        Real y = -b / (2.0_rt * a);
        problem::center[1] = snapped_center[1] + y * dx[1];

#if AMREX_SPACEDIM == 3
        // ... in z-direction
        a = 0.5_rt * (data(mi[0], mi[1], mi[2]+1) + data(mi[0], mi[1], mi[2]-1));
        b = 0.5_rt * (data(mi[0], mi[1], mi[2]+1) - data(mi[0], mi[1], mi[2]-1));
        Real z = -b / (2.0_rt * a);
        problem::center[2] = snapped_center[2] + z * dx[2];
#endif

#endif

    }
    // Now broadcast to everyone else.
    ParallelDescriptor::Bcast(&problem::center[0], AMREX_SPACEDIM, owner);

    // Make sure if R-Z and SPHERICAL that center stays exactly on axis
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

    BL_ASSERT(level > 0); // because we are building a mask for the coarser level

    if (fine_mask.empty()) {
        fine_mask = makeFineMask(parent->boxArray(level-1),
                                 parent->DistributionMap(level-1),
                                 parent->boxArray(level), crse_ratio,
                                 1.0,  // coarse
                                 0.0); // fine
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

    //  If we got here, we need to build a new one
    interior_masks.push_back(std::make_unique<iMultiFab>(grids, dmap, 1, ng));

    iMultiFab& imf = *interior_masks.back();

    int ghost_cover_flag = 0;
    int uncovered_flag = 1; // uncovered ghost, valid, and outside domain cells are set to 1

    imf.BuildMask(geom.Domain(), geom.periodicity(),
                  ghost_cover_flag, uncovered_flag, uncovered_flag, uncovered_flag);

    return imf;
}

// Fill a version of the state with ng ghost zones from the state data.

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

// Given State_Type state data, perform a number of cleaning steps to make
// sure the data is sensible.

void
HydroSim::clean_state(
#ifdef MHD
                    MultiFab& bx,
                    MultiFab& by,
                    MultiFab& bz,
#endif
                    MultiFab& state_in, Real time, int ng) {

    BL_PROFILE("HydroSim::clean_state()");

    // Enforce a minimum density.

    clamp_min_density(state_in, ng);

    // Enforce the speed limit.

    clamp_speed_limit(state_in, ng);

    // Ensure all species are normalized.

    renorm_species(state_in, ng);

    // Sync the linear and hybrid momenta.

#ifdef HYBRID_MOMENTUM
    if (split_momentum_hydro) {
        split_to_linear_momentum(state_in, ng);
    }
#endif

    // Compute the temperature (note that this will also reset
    // the internal energy for consistency with the total energy).

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

        // We want to store the previous state in pinned memory
        // if we're running on a GPU. This helps us alleviate
        // pressure on the GPU memory, at the slight cost of
        // lower bandwidth when we are saving/restoring the state.
        // Since we're using operator= to copy the StateData,
        // we'll use a trick where we temporarily change the
        // the arena used by the main state and then immediately
        // restore it.

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
