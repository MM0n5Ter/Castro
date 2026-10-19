#ifndef HYDROSIM_H
#define HYDROSIM_H

#include <format>

#include <state_indices.H>

// runtime parameters
#include <hydrosim_params.H>
#include <prob_parameters.H>
#include <HydroSim_io.H>
#include <HydroSim_helpers.H>
#include <timestep.H>

using namespace hydrosim;

#ifdef AMREX_PARTICLES
#include <particles_params.H>
#endif

#include <fundamental_constants.H>

#include <AMReX_BC_TYPES.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ErrorList.H>
#include <AMReX_FluxRegister.H>
#include <network.H>
#include <eos.H>
#ifndef TRUE_SDC
#ifdef REACTIONS
#include <burner.H>
#endif
#endif

#ifdef BL_LAZY
#include <AMReX_Lazy.H>
#endif

#ifdef AMREX_PARTICLES
#include <AMReX_AmrParticles.H>
#endif

#ifdef RADIATION
#include <RadTransportSolver.H>
#endif

#include <memory>
#include <iostream>

#include <params_type.H>

using std::istream;
using std::ostream;

enum StateType : std::uint8_t {
               State_Type = 0,
#ifdef RADIATION
               Radiation_Type,
#endif
#ifdef GRAVITY
               Potential_Type,
               Accel_Type,
#endif
               Forcing_Type,
#ifdef REACTIONS
               Burning_Type,
#endif
#ifdef MHD
               Bfield_Type_x,
               Bfield_Type_y,
               Bfield_Type_z,
#endif
#ifdef REACTIONS
               Burn_Corr_Type
#endif
};

// Create storage for all source terms.

enum sources : std::uint8_t {
               thermodynamic_src = 0,
#ifdef DIFFUSION
               conduction_src,
#endif
#ifdef HYBRID_MOMENTUM
               hybrid_mom_source,
#endif
#ifdef GRAVITY
               gravity_src,
#endif
#ifdef ROTATION
               rotation_src,
#endif
#ifdef SPONGE
               damping_src,
#endif
               external_source,
               geometry_source,
               num_forcing };


// time integration method

enum stepper_kind : std::uint8_t {
    CornerTrace = 0,
    ReservedIntegrator,
    IterativeCorrection,
    SimplifiedIterCorrection
};

// Struct that returns information about
// why an advance failed.

struct step_result {
    bool success{true};
    Real proposed_dt{};
    std::string reason{};
    step_result() = default;
};

///
/// @class HydroSim
///
/// @brief ``AmrLevel``-derived class for hyperbolic conservation equations for stellar media
///
class HydroSim
    :
    public amrex::AmrLevel
{
public:

///
/// Default constructor.  Builds invalid object.
///
    HydroSim ();

///
/// The basic constructor.
///
/// @param papa     parent ``amrex::Amr`` object
/// @param lev      which level are we on
/// @param level_geom level geometry
/// @param bl       ``BoxArray`` object
/// @param dm       the mapping
/// @param time     current time
///
    HydroSim (amrex::Amr&            papa,
            int             lev,
            const amrex::Geometry& level_geom,
            const amrex::BoxArray& bl,
            const amrex::DistributionMapping& dm,
            amrex::Real            time);

///
/// The destructor.
///
    ~HydroSim () override;

///
/// Remove copy/move constructors/assignment operators.
///
    HydroSim (const HydroSim&) = delete;
    HydroSim (HydroSim&&) = delete;
    HydroSim& operator= (const HydroSim&) = delete;
    HydroSim& operator= (HydroSim&&) = delete;

///
/// Restart from a checkpoint file.
///
/// @param papa             parent ``amrex::Amr`` object
/// @param is               input stream for checkpoint data
/// @param bReadSpecial     whether to read problem-specific restart data
///
    void restart (amrex::Amr&     papa,
                  istream& is,
                  bool bReadSpecial = false) override;

///
/// This is called only when we restart from an old checkpoint.
/// Sets the Vector `state_in_checkpoint` depending on the input
/// version and type.
///
/// @param state_in_checkpoint      Vector of integer flags
///                                 describing the state
///
    void set_state_in_checkpoint (amrex::Vector<int>& state_in_checkpoint) override;

///
/// Call ``amrex::AmrLevel::checkPoint`` and then add radiation info
///
/// @param dir          Directory to store checkpoint in
/// @param os           ``std::ostream`` object
/// @param how          `V`isMF::How`` object
/// @param dump_old     boolean
///
    void checkPoint(const std::string& dir,
                    std::ostream&      os,
                    amrex::VisMF::How         how,
                    bool               dump_old) override;

///
/// A string written as the first item in writePlotFile() at
/// level zero. It is so we can distinguish between different
/// types of plot files. For HydroSim it has the form: HydroSim-Vnnn
///
    std::string thePlotFileType () const override;

    void setPlotVariables () override;

///
/// Write a plotfile to specified directory.
///
/// @param dir  directory to write to
/// @param os   output stream
/// @param how  ``VisMF::How`` object
///
    void writePlotFile (const std::string& dir,
                        ostream&       os,
                        amrex::VisMF::How     how) override;

///
/// @param dir  directory to write to
/// @param os   output stream
/// @param how  ``VisMF::How`` object
///
    void writeSmallPlotFile (const std::string& dir,
                             ostream&       os,
                             amrex::VisMF::How     how) override;

///
/// @param dir      directory to write to
/// @param os       output stream
/// @param how      ``VisMF::How`` object
/// @param is_small integer
///
    void writeSnapshot(const std::string& dir,
                        ostream& os,
                        amrex::VisMF::How how,
                        const int is_small);


///
/// Write job info to file
///
/// @param dir       directory to write to
/// @param write_seconds   how long it took to generate the plotfile/checkpoint
///
    void writeRunRecord (const std::string& dir, const amrex::Real write_seconds);


///
/// Dump build info
///
    static void writeBuildRecord ();


///
/// Define data descriptors.
///
/// ::variableSetUp is called in the constructor of Amr.cpp, so
/// it should get called every time we start or restart a job
///
    static void variableSetUp ();

///
/// Define tagging functions.
///
    static void SetupErrorTags ();

///
/// Cleanup data descriptors at end of run.
///
/// ::variableSetUp is in ``HydroSim_setup.cpp``
/// ``variableCleanUp`` is called once at the end of a simulation
///
    static void variableCleanUp ();

///
/// Initialize grid data at problem start-up.
///
    void initData () override;

#ifdef AMREX_PARTICLES

///
/// Read particle-related inputs
///
    static void load_particle_config ();

///
/// Initialize particle locations and velocities (and strengths if relevant)
///
    void init_particles ();

///
/// Write particles in checkpoint directories
///
/// @param dir  directory containing checkpoint
///
    void ParticleSaveRestart (const std::string& dir);

///
/// Write particles in plotfile directories
///
/// @param dir  directory containing plotfile
///
    void ParticleOutputFile (const std::string& dir);

///
/// How to initialize at restart
///
/// @param restart_path     path to restart file
///
    void ParticleAfterRestart (const std::string& restart_path);

///
/// Derived quantities associated with particles
///
/// @param name         name of particle variable to derive
/// @param time         current time
/// @param ngrow        Number of ghost cells
///
    std::unique_ptr<amrex::MultiFab> ParticleQuantity (const std::string& name,
                                                     amrex::Real           time,
                                                     int            ngrow);

///
/// Timestamp particles
///
/// @param ngrow    number of ghost cells
///
    void StampParticles (int ngrow);

///
/// Advance the particles by dt
///
/// @param iteration    where we are in the current AMR subcycle
/// @param time         current time
/// @param dt           timestep
///
    void step_particles (int iteration, amrex::Real time, amrex::Real dt);

#endif

#ifdef LOWMACH_INIT
    void LOWMACH_init ();
#endif


///
///
///
    amrex::MultiFab* Area ();

///
/// Area of the multifab with surface normal along ``dir``
///
/// @param dir  direction of normal
///
    amrex::MultiFab& Area (int dir);

///
/// The volume of the multifab.
///
    amrex::MultiFab& Volume ();

///
/// Return the n'th fluxes MultiFab.
///
/// @param dir  direction in which to find the fluxes along
///
    amrex::MultiFab& Fluxes (int dir);

///
/// Set time levels of state data.
///
/// @param time         current time
/// @param dt_old       old timestep
/// @param dt_new       new timestep
///
    void setTimeLevel (amrex::Real time,
                       amrex::Real dt_old,
                       amrex::Real dt_new) override;

///
/// Initialize data on this level from another HydroSim (during regrid).
///
/// @param old  amrex::AmrLevel object to use to initialize data
///
    void init (amrex::AmrLevel& old) override;

///
/// Initialize data on this level after regridding if old level did not
/// previously exist
///
    void init () override;

///
/// Proceed with next timestep?
///
    int okToContinue () override;

///
/// Advance grids at this level in time.
///
/// The main driver for a single level.  This will do either the SDC
/// algorithm or the Strang-split reactions algorithm.
///
/// @param time     the current simulation time
/// @param dt       the timestep to advance (e.g., go from time to
///                    time + dt)
/// @param iteration    where we are in the current AMR subcycle.  Each
///                    level will take a number of steps to reach the
///                    final time of the coarser level below it.  This
///                    counter starts at 1
/// @param ncycle   the number of subcycles at this level
///
    amrex::Real advance (amrex::Real time,
                         amrex::Real dt,
                         int  iteration,
                         int  ncycle) override;

///
/// This routine will advance the old state data (called ``S_old`` here)
/// to the new time, for a single level.  The new data is called
/// ``S_new`` here.  The update includes reactions (if we are not doing
/// SDC), hydro, and the source terms. The advance is done using the CTU
/// (corner transport upwind) approach. We also use this interface for the
/// simplified SDC update.
///
/// @param time     the current simulation time
/// @param dt       the timestep to advance (e.g., go from time to
///                    time + dt)
///
    step_result run_advance_ctu (amrex::Real time, amrex::Real dt);


#ifndef AMREX_USE_GPU
    amrex::Real run_advance_sdc (amrex::Real time,
                                amrex::Real dt,
                                int  amr_pass,
                                int  amr_cycles);
#endif

///
/// Save a copy of the old state data for use by a retry.
///
    void snapshot_for_rerun();

///
/// Should we retry advancing the simulation? By default, we
/// don't do a retry unless the criteria are violated. This is
/// used only for the CTU advance.
///
/// @param dt       the timestep to advance (e.g., go from time to
///                    time + dt)
/// @param step_result  was the advance successful?
///
    bool rerun_advance_ctu(amrex::Real dt, const step_result& status);

///
/// Subcycles until we've reached the target time, ``time`` + ``dt``.
/// The last timestep will be shortened if needed so that
/// it does not overshoot the ending time. This is used only for
/// the CTU advance.
///
/// @param time     the current simulation time
/// @param dt       the timestep to advance (e.g., go from time to
///                    time + dt)
/// @param amr_pass    where we are in the current AMR subcycle.  Each
///                    level will take a number of steps to reach the
///                    final time of the coarser level below it.  This
///                    counter starts at 1
/// @param amr_cycles   the number of subcycles at this level
///
    amrex::Real advance_subcycles_ctu(amrex::Real time, amrex::Real dt, int amr_pass, int amr_cycles);

///
/// This is run at the start of an ::advance on a single level.
/// It sets up all the necessary arrays and data containers,
/// ensures the data is valid and makes copies of the MultiFabs
/// in the old and new state in case a retry is performed.
///
/// @param time     the current simulation time
/// @param dt       the timestep to advance (e.g., go from time to
///                    time + dt)
/// @param amr_pass    where we are in the current AMR subcycle.  Each
///                    level will take a number of steps to reach the
///                    final time of the coarser level below it.  This
///                    counter starts at 1
///
    void begin_advance(amrex::Real time, amrex::Real dt, int amr_pass);

///
/// If required, adds material lost in timestep to cumulative
/// losses, does refluxing, and clears arrays/data containers
/// initialized at start of the level's advance.
///
    void end_advance();

///
/// Performed at the start of ::do_advance, resets flags and
/// ensures required ghost zones are filled.
///
/// @param time     the current simulation time
/// @param dt       the timestep to advance
///
    step_result begin_step_advance (amrex::Real time, amrex::Real dt);

///
/// Performed at the end of ::do_advance, cleans up temporary data
/// created by ::begin_step_advance
///
/// @param time     the current simulation time
/// @param dt       the timestep to advance
///
    step_result end_step_advance (amrex::Real time, amrex::Real dt);

///
/// Check for NaNs in the given MultiFab
///
/// @param state        MultiFab to check
/// @param include_ghosts  do we check the ghost cells?
///
    static void scan_for_nan(const amrex::MultiFab& state, int include_ghosts=0);


#include <HydroSim_forcing.H>


#ifdef GRAVITY
#include <HydroSim_gravity.H>
#endif


#ifdef MHD
#include <HydroSim_mhd.H>
#endif

///
/// Estimate time step.
///
    amrex::Real estTimeStep (int use_new_state = 1);

///
/// Compute initial time step.
///
    amrex::Real initialTimeStep ();

///
/// Compute initial ``dt``.
///
/// @param finest_level     Index of finest level
/// @param subcycle         unused
/// @param n_cycle          the number of subcycles at this level
/// @param ref_ratio        unused
/// @param dt_level         Real Vector, will contain initial timestep at each level.
/// @param stop_time        End time of simulation
///
    void computeInitialDt (int                   finest_level,
                           int                   /*subcycle*/,
                           amrex::Vector<int>&           n_cycle,
                           const amrex::Vector<amrex::IntVect>& /*ref_ratio*/,
                           amrex::Vector<amrex::Real>&          dt_level,
                           amrex::Real                  stop_time) override;

///
/// Compute new ``dt``.
///
/// @param finest_level     Index of finest level
/// @param subcycle         unused
/// @param n_cycle          the number of subcycles at this level
/// @param ref_ratio        unused
/// @param dt_min          minimum timestep allowed at each level
/// @param dt_level        timestep at each level
/// @param stop_time        End time of simulation
/// @param post_regrid_flag Have we done regridding yet?
///
    void computeNewDt (int                   finest_level,
                       int                   /*subcycle*/,
                       amrex::Vector<int>&           n_cycle,
                       const amrex::Vector<amrex::IntVect>& /*ref_ratio*/,
                       amrex::Vector<amrex::Real>&          dt_min,
                       amrex::Vector<amrex::Real>&          dt_level,
                       amrex::Real                  stop_time,
                       int                   post_regrid_flag) override;

///
/// Allocate data at old time.
///
    void allocOldData () override;

///
/// Remove data at old time.
///
    void removeOldData () override;


///
/// Print information about energy budget.
///
    void print_energy_budget();

///
/// Do work after timestep().
///
/// @param iteration    where we are in the current AMR subcycle
///
    void post_timestep (int iteration) override;

///
/// Contains operations to be done only after a full coarse timestep.
///
/// @param cumtime    cumulative simulation time
///
    void postCoarseTimeStep (amrex::Real cumtime) override;

///
/// Do work after regrid().
///
/// @param lbase         the coarsest level affected by regridding
/// @param new_finest    the new finest AMR level
///
    void post_regrid (int lbase,
                      int new_finest) override;

///
/// Do work after a restart().
///
    void post_restart () override;

///
/// Do work after init().
///
/// @param stop_time    End time of simulation
///
    void post_init (amrex::Real stop_time) override;

#ifdef GRAVITY
#ifdef ROTATION
///
/// Initialize a model using the self-consistent field method.
///
    void scf_solve();

///
/// Perform the Hachisu SCF method.
///
    void run_equilibrium_solve();
#endif
#endif

///
/// Do work after restart with ``restart_grow_factor`` > 1
///
    void after_grow_restart ();

#ifdef DO_PROBLEM_POST_SIMULATION

///
/// Do work at the end of the simulation - before the last outputs
///
/// @param amr_level    AMR levels in the simulation hierarchy
///
    static void user_post_simulation (amrex::Vector<std::unique_ptr<AmrLevel> >& amr_level);
#endif


///
/// Error estimation for regridding.
///
/// @param tags         TagBoxArray of tags
/// @param clearval     integer
/// @param tagval       integer
/// @param time         current time
/// @param n_error_buf  integer
/// @param ngrow        Number of ghost cells
///
    void errorEst (amrex::TagBoxArray& tags,
                   int          clearval,
                   int          tagval,
                   amrex::Real  time,
                   int          n_error_buf = 0,
                   int          ngrow = 0) override;


///
/// Apply the ::user_tagging routine.
///
/// @param tags         TagBoxArray of tags
/// @param time         current time
///
    void apply_user_tags (amrex::TagBoxArray& tags, amrex::Real time);


///
/// Apply any tagging restrictions that must be satisfied by all problems.
///
/// @param tags         TagBoxArray of tags
/// @param time         current time
///
    void apply_tag_limits (amrex::TagBoxArray& tags, amrex::Real time);


///
/// Returns a MultiFab containing the derived data for this level.
/// If ngrow>0 the MultiFab is built on the appropriately grown BoxArray.
///
/// @param name     Name of derived data
/// @param time     Current time
/// @param ngrow    Number of ghost cells
///
    std::unique_ptr<amrex::MultiFab> derive (const std::string& name,
                                             amrex::Real        time,
                                             int                ngrow) override;

///
/// This version of derive() fills the dcomp'th component of mf with the derived quantity.
///
/// @param name         Name of quantity to derive
/// @param time         current time
/// @param mf           MultiFab to store derived quantity in
/// @param dcomp        index of component of `mf` to fill with derived quantity
///
    void derive (const std::string& name,
                 amrex::Real               time,
                 amrex::MultiFab&          mf,
                 int                dcomp) override;


#ifdef REACTIONS
#include <HydroSim_burn.H>
#endif

#ifdef ROTATION
#include <HydroSim_rotation.H>
#endif

#ifdef RADIATION

///
/// @param S_new        state at the new time
/// @param iteration    where we are in the current AMR subcycle
/// @param ncycle       the number of subcycles at this level
///
    void close_radiation_step (amrex::MultiFab& S_new,
                               int iteration,
                               int ncycle);
#endif

///
/// Reset the internal energy
///
/// @param Bx       magnetic field in x
/// @param By       magnetic field in y
/// @param Bz       magnetic field in z
/// @param State    Current state (MultiFab)
/// @param ng       number of ghost cells
///
    void repair_internal_energy (
#ifdef MHD
                                amrex::MultiFab& Bx,
                                amrex::MultiFab& By,
                                amrex::MultiFab& Bz,
#endif
                                amrex::MultiFab& State, int ng);


///
/// Reset the internal energy
///
/// @param bx       Box to update
/// @param Bx       magnetic field in x
/// @param By       magnetic field in y
/// @param Bz       magnetic field in z
/// @param u        Current state (Fab)
///
    static void repair_internal_energy (const amrex::Box& bx,
#ifdef MHD
                                amrex::Array4<amrex::Real> const& Bx,
                                amrex::Array4<amrex::Real> const& By,
                                amrex::Array4<amrex::Real> const& Bz,
#endif
                                amrex::Array4<amrex::Real> const& u);


#ifdef MHD
///
/// Add magnetic contribution to energy density
/// @param Bx       magnetic field in x
/// @param By       magnetic field in y
/// @param Bz       magnetic field in z
/// @param state    the state to operate on
///
    static void add_bfield_energy (
        amrex::MultiFab& Bx,
        amrex::MultiFab& By,
        amrex::MultiFab& Bz,
        amrex::MultiFab& state);

///
/// Check if divergence of B is zero at initialization
/// @param Bx       magnetic field in x
/// @param By       magnetic field in y
/// @param Bz       magnetic field in z
/// @param state    the state to operate on
///
    void check_div_bfield (
                      amrex::MultiFab& Bx,
                      amrex::MultiFab& By,
                      amrex::MultiFab& Bz,
                      amrex::MultiFab& state);

#endif



///
/// Compute the current temperature
///
/// @param Bx       magnetic field in x
/// @param By       magnetic field in y
/// @param Bz       magnetic field in z
/// @param state    the state to operate on
/// @param time     current time
/// @param ng       number of ghost cells
///
    void computeTemp (
#ifdef MHD
                      amrex::MultiFab& Bx,
                      amrex::MultiFab& By,
                      amrex::MultiFab& Bz,
#endif
                      amrex::MultiFab& state, amrex::Real time, int ng);


///
/// Add any terms needed to correct the source terms.
/// Currently this is a lagged predictor for CTU and the
/// time-centering correction for simplified SDC.
///
    void create_forcing_corrector();


///
/// This is a hack to make sure that we only
/// ever have new data for certain state types that only
/// ever need new time data; by doing a swap now, we'll
/// guarantee that allocOldData() does nothing. We do
/// this because we never need the old data, so we
/// don't want to allocate memory for it.
///
/// @param dt   timestep
///
    void swap_field_time_levels (const amrex::Real dt);

#ifdef DIFFUSION
#include <HydroSim_conduction.H>
#endif


///
/// Volume weighted sum of given quantity
///
/// @param mf           MultiFab containing quantity
/// @param comp         integer index of quantity in MultiFab
/// @param local        boolean, is sum local (over each patch) or over entire MultiFab?
/// @param exclude_fine     boolean, should we build a mask to exclude finer levels?
///
    amrex::Real volWeightedTotal (const MultiFab& mf, int comp, bool local=false, bool exclude_fine=true);

///
/// Volume weighted sum of given quantity
///
/// @param name         Name of quantity
/// @param time         current time
/// @param local        boolean, is sum local (over each patch) or over entire MultiFab?
/// @param exclude_fine     boolean, should we build a mask to exclude finer levels?
///
    amrex::Real volWeightedTotal (const std::string& name, amrex::Real time, bool local=false, bool exclude_fine=true);

///
/// Sum weighted by volume multiplied by distance from center in given direction
///
/// @param mf       MultiFab containing quantity
/// @param comp     integer index of quantity in MultiFab
/// @param idir     Axis along which to compute distance from center
/// @param local    boolean, is sum local (over each patch) or over entire MultiFab?
///
    amrex::Real locWeightedSum (const MultiFab& mf, int comp, int idir, bool local=false);

///
/// Sum weighted by volume multiplied by distance from center in given direction
///
/// @param name     Name of quantity
/// @param time     current time
/// @param idir     Axis along which to compute distance from center
/// @param local    boolean, is sum local (over each patch) or over entire MultiFab?
///
    amrex::Real locWeightedSum (const std::string& name, amrex::Real time, int idir, bool local=false);

///
/// Volume weighted sum of the product of two quantities
///
/// @param mf1      MultiFab containing first quantity
/// @param mf2      MultiFab containing second quantity
/// @param comp1    integer index of quantity in first MultiFab
/// @param comp2    integer index of quantity in second MultiFab
/// @param local    boolean, is sum local (over each patch) or over entire MultiFab?
///
    amrex::Real volPairTotal (const MultiFab& mf1, const MultiFab& mf2, int comp1, int comp2, bool local=false);

///
/// Volume weighted sum of the product of two quantities
///
/// @param name1    Name of first quantity
/// @param name2    Name of second quantity
/// @param time     current time
/// @param local    boolean, is sum local (over each patch) or over entire MultiFab?
///
    amrex::Real volPairTotal (const std::string& name1, const std::string& name2, amrex::Real time, bool local=false);


///
/// Location weighted sum of (quantity) squared
///
/// @param name     Name of quantity
/// @param time     current time
/// @param idir     Direction along which to weight
/// @param local    boolean, is sum local (over each patch) or over entire MultiFab?
///
    amrex::Real locDistSquaredTotal (const std::string& name, amrex::Real time, int idir, bool local=false);

#ifdef GRAVITY
///
/// Calculate the gravitational wave signal
///
/// @param time       current time
/// @param strain_plus_1   gravitational wave signal in x-direction
/// @param strain_cross_1  gravitational wave signal in x-direction
/// @param strain_plus_2   gravitational wave signal in y-direction
/// @param strain_cross_2  gravitational wave signal in y-direction
/// @param strain_plus_3   gravitational wave signal in z-direction
/// @param strain_cross_3  gravitational wave signal in z-direction
/// @param local      is sum local

    void wave_strain (Real time,
                   Real& strain_plus_1, Real& strain_cross_1,
                   Real& strain_plus_2, Real& strain_cross_2,
                   Real& strain_plus_3, Real& strain_cross_3,
                   bool local);
#endif

#ifdef GRAVITY
///
/// Are we using point mass gravity?
///
    static int using_central_mass ();

///
/// Return the value of the point mass.
///
    static amrex::Real get_central_mass ();
#endif


///
/// Fill a version of the state with ``ng`` ghost zones from the state data.
/// After we do the fill, optionally clean the data using the iclean parameter.
///
/// @param S        MultiFab to be filled
/// @param time     current time
/// @param ng       number of ghost cells
///
    void fill_ghost_field(amrex::MultiFab& S, amrex::Real time, int ng);



// Hydrodynamics
#include <HydroSim_hydro.H>

// SDC
#ifdef TRUE_SDC
#include <HydroSim_deferred.H>
#endif

    static void microphys_init ();

    static void read_params ();


///
/// @param lev    AMR level index
///
    HydroSim& getLevel (int lev);

    void FluxRegCoarseInit();
    void FluxRegFineAccum();


///
/// Do refluxing
///
/// @param crse_level         coarse AMR level index
/// @param fine_level         fine AMR level index
/// @param from_post_step   true if called from post_timestep()
///
    void reflux (int crse_level, int fine_level, bool from_post_step);


///
/// Normalize species fractions so they sum to 1
///
/// @param S_new    State
/// @param ng       number of ghost cells
///
    static void renorm_species (amrex::MultiFab& S_new, int ng);


///
/// Enforces
/// \f[
/// \rho E = \rho e + \frac{1}{2} \rho (u^2 + v^2 + w^2)
/// \f]
///
/// @param S  State
///
    static void reconcile_energy (
#ifdef MHD
                               amrex::MultiFab& Bx, amrex::MultiFab& By, amrex::MultiFab& Bz,
#endif
                               amrex::MultiFab& S);

///
/// This routine sets the density in ``state`` to be larger than the density floor.
/// Note that it will operate everywhere on ``state``, including ghost zones.
///
/// @param state        state
/// @param ng           number of ghost cells
///
    void clamp_min_density (amrex::MultiFab& state, int ng);

///
/// After a hydro advance, check to see whether any density went too small.
///
    step_result scan_for_negative_mass_density ();

///
/// Ensure the magnitude of the velocity is not larger than ``hydrosim.velocity_cap``
/// in any zone.
///
/// @param state        state
/// @param ng           number of ghost cells
///
    static void clamp_speed_limit (amrex::MultiFab& state, int ng);

///
/// Given ``State_Type`` state data, perform a number of cleaning steps to make
/// sure the data is sensible.
///
/// @param state    State data
/// @param time     current time
/// @param ng       number of ghost cells
///
    void clean_state (
#ifdef MHD
                      amrex::MultiFab& Bx, amrex::MultiFab& By, amrex::MultiFab& Bz,
#endif
                      amrex::MultiFab& state, amrex::Real time, int ng);

///
/// Average new state from ``level+1`` down to ``level``
///
    void avgDown ();


///
/// Average given component of new state from ``level+1`` down to ``level``
///
/// @param state_indx   Index of state to average down
///
    void avgDown (int state_indx);

    void buildMetrics ();


///
/// Initialize the MultiFabs and flux registers that live as class members.
///
    void allocFabs ();

///
/// Calculates volume weight sums of variables and prints to screen
///
    void tally_integrated_quantities ();

///
/// Problem-specific diagnostics (called by tally_integrated_quantities)
///
    void user_diagnostics ();

    void write_info ();

///
/// Return BCs
///
    static amrex::BCRec physbc () { return phys_bc; }

#ifdef GRAVITY

///
/// Find the position of the "center" by interpolating from data at cell centers
///
/// @param S        Current state
/// @param time     current time
///
    void define_tracked_center (const amrex::MultiFab& S, amrex::Real time);
    void write_tracked_center ();

///
/// @param time     current time
/// @param dt       timestep
///
    void centralmass_update(amrex::Real time, amrex::Real dt);
#endif

    static void haltRun ();

///
/// Returns true if we're on the GPU and the total
/// memory on this level oversubscribes GPU memory.
///
    bool oversubscribing() {    // NOLINT(readability-convert-member-functions-to-static)
#ifdef AMREX_USE_GPU
        return (amrex::MultiFab::queryMemUsage(std::format("AmrLevel_Level_{}", level)) >=
                static_cast<Long>(amrex::Gpu::Device::totalGlobalMem()));
#else
        return false;
#endif
    }

#ifdef AMREX_PARTICLES
    static amrex::AmrTracerParticleContainer* theTracerPC () { return TracerPC.get(); }
#endif

    static amrex::Vector<std::string> forcing_names;

///
/// Vector storing list of error tags.
///
    static amrex::Vector<amrex::AMRErrorTag> tag_criteria;

///
/// This MultiFab is on the coarser level.  This is useful for the coarser level
///     to mask out the finer level.  We only build this when it is needed.
///     This coarse MultiFab has to live on the fine level because it must be updated
///     even when only the fine level changes.
///
    amrex::MultiFab fine_mask;
    amrex::MultiFab& make_fine_mask();


///
/// A record of how many cells we have advanced throughout the simulation.
///     This is saved as a real because we will be storing the number of zones
///     advanced as a ratio with the number of zones on the coarse grid (to
///     help prevent the number from getting too large), and that may be a
///     non-integer number.
///
    static amrex::Real cells_advanced_total;

///
/// diagnostics
///
    static Vector<std::unique_ptr<std::fstream> > run_logs;
    static Vector<std::unique_ptr<std::fstream> > user_run_logs;

///
/// runtime parameters
//
    static params_t params;


protected:


///
/// Build a mask where ghost cells overlapping interior cells in the same
/// multifab are set to 0, and all others are set to 1.
///
    amrex::Vector<std::unique_ptr<amrex::iMultiFab> > interior_masks;

///
/// @param ng number of ghost cells
///
    amrex::iMultiFab& make_boundary_mask (int ng);

#ifdef GRAVITY
    int count_points_1d();


///
/// Difference between composite and level solve for phi.
///
    amrex::MultiFab composite_minus_level_phi;
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > composite_minus_level_grad_phi;
#endif


#ifdef REACTIONS
///
/// Storage for the burn_split_data
///
    amrex::MultiFab burn_split_data;
    static std::vector<std::string> burn_split_names;
#endif


///
/// A state array with ghost zones.
///
    amrex::MultiFab Sborder;

#ifdef MHD
   amrex::MultiFab Bx_old_grown;
   amrex::MultiFab By_old_grown;
   amrex::MultiFab Bz_old_grown;
#endif

#ifdef RADIATION
    amrex::MultiFab RadEnergyGrown;
    amrex::MultiFab rad_lambda_grown;

    std::unique_ptr<RadTransportSolver> rad_operator;
#endif



#ifdef TRUE_SDC
///
/// A state array for reaction-related data used by SDC
///
    amrex::MultiFab BurnData;


///
/// The primitive variable state array.
///
    amrex::MultiFab q;

///
/// A second primitive-state array used by the fourth-order method
///
    amrex::MultiFab q_second;

///
/// Cell-centered temperature needed for fourth-order diffusion
//
#ifdef DIFFUSION
    amrex::MultiFab Temperature_cc;
#endif

///
/// The auxiliary primitive variable state array.
///
    amrex::MultiFab qthermo;
    amrex::MultiFab qthermo_second;
#endif


///
/// Source term corrector.
///
    amrex::MultiFab forcing_corrector;


///
/// Hydrodynamic (and radiation) fluxes.
///
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > fluxes;
#if (AMREX_SPACEDIM <= 2)
    amrex::MultiFab         Pressure_radial;
#endif
#ifdef RADIATION
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > rad_face_flux;
#endif

    amrex::Vector<std::unique_ptr<amrex::MultiFab> > mass_face_flux;

    amrex::FluxRegister flux_reg;
#if (AMREX_SPACEDIM <= 2)
    amrex::FluxRegister pressure_flux_reg;
#endif
#ifdef RADIATION
    amrex::FluxRegister rad_edge_reg;
#endif
#ifdef GRAVITY
    amrex::FluxRegister pot_flux_reg;
#endif

///
/// Scalings for the flux registers.
///
    amrex::Real flux_coarse_scale;
    amrex::Real flux_refined_scale;

#if (AMREX_SPACEDIM <= 2)
    amrex::Real pressure_coarse_scale;
    amrex::Real pressure_refined_scale;
#endif



///
/// State data to hold if we want to do a retry.
///
    amrex::Vector<std::unique_ptr<amrex::StateData> > prev_state;



///
/// Flag for indicating that we want to save prev_state until the reflux.
///
    bool retain_prev_field{};


#ifdef TRUE_SDC
    //
    // Storage for the SDC time integration

    // this is the new iterations solution at the time nodes
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > node_new;

    // this is the old value of the advective update at the
    // nodes of the time integration
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > Advect_old;

    // this is the new value of the advective update at the
    // nodes of the time integration
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > Advect_new;

    // this is the old value of the reaction source at the
    // nodes of the time integration
#ifdef REACTIONS
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > Reactions_old;
#endif

    static int COLLOC_NODES;
    static amrex::Vector<amrex::Real> node_dt_frac;
    static amrex::Vector<amrex::Real> node_quad_weights;
#endif

///
/// Wall time that we started the timestep
///
    amrex::Real wall_clock_start;


///
/// The data.
///
    amrex::MultiFab             volume;
    amrex::Array<amrex::MultiFab,3> area;
    amrex::Vector< amrex::Vector<amrex::Real> > radius;
#if AMREX_SPACEDIM <= 2
    amrex::Array<amrex::MultiFab,2> dLogFaceMeasure;
#endif

///
/// For initialization of the C++ values of the runtime parameters
  static int user_init_pending;


///
/// Keep track of which AMR iteration we're on.
///
    int iteration;

    //
    // Static data members.
    //

    static bool      stopRequested;
    static int       radial_grow;
    static std::vector<std::string> tag_field_names;
    static std::vector<int> tag_field_ghosts;
    static int              num_tag_fields_default;
    static amrex::BCRec     phys_bc;

    static int         lastDtWriteLimited;
    static amrex::Real lastDtBeforeWriteLimit;

    int retry_active;
    int num_sub_taken;

    amrex::Real lastStepDt;


///
/// for keeping track of the amount of CPU or GPU time used -- this will persist
/// after restarts
///
    static amrex::Real      priorCpuSeconds;
    static amrex::Real      startCpuSeconds;

///
/// Get total CPU time
///
    static amrex::Real getCpuSeconds();

    bool             OldStateFilled;


///
/// There can be only one Gravity object, it covers all levels:
///
    static std::unique_ptr<class Gravity> gravity;


///
/// There can be only one Diffusion object, it covers all levels:
///
    static std::unique_ptr<class Diffusion> diffusion;

#ifdef RADIATION


///
/// There can be only one Radiation object, it covers all levels:
///
    static std::unique_ptr<class Radiation> radiation;

    friend class Radiation;
#endif

#ifdef AMREX_PARTICLES
    static std::unique_ptr<amrex::AmrTracerParticleContainer> TracerPC;
#endif

    static amrex::IntVect hydro_tile_size;
    static amrex::IntVect untiled_size;

    static int sweep_tile_tuned;
    static Long largest_tuned_box_pts;

    static int Forcing_Type_Index;
    static int num_field_types;


    // counters for various retries in HydroSim

///
/// subcycle capability
///
    int sub_pass;
    int sub_cycles;
    amrex::Real dt_cycle_step;
    amrex::Real dt_level_step;


///
/// sdc
///
    int correction_iter;

#ifdef TRUE_SDC
    int current_colloc_node;
#endif


/* problem-specific includes */
public:
#include <Problem.H>

};

//
// Inlines.
//

inline
amrex::MultiFab*
HydroSim::Area ()
{
    return area.data();
}


///
/// @param dir    coordinate direction
///
inline
amrex::MultiFab&
HydroSim::Area (int dir)
{
    return area[dir];
}

inline
amrex::MultiFab&
HydroSim::Volume ()
{
    return volume;
}


///
/// @param lev    AMR level index
///
inline
HydroSim&
HydroSim::getLevel (int lev)
{
    return *dynamic_cast<HydroSim *>(&parent->getLevel(lev));
}

inline
void
HydroSim::haltRun()
{
  stopRequested = true;


}

inline
int
HydroSim::get_write_on_finish()
{
    return write_on_finish;
}

#ifdef GRAVITY
inline
amrex::Real
HydroSim::get_central_mass ()
{
    return point_mass;
}

inline
int
HydroSim::using_central_mass ()
{
    return use_central_mass;
}
#endif

#endif /*_HydroSim_H_*/
