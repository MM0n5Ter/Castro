#ifndef HYDROSIM_H
#define HYDROSIM_H

#include <format>

#include <state_indices.H>

// runtime parameters
#include <hydrosim_params.H>
#include <prob_parameters.H>
#include <HydroSim_io.H>
#include <HydroSim_util.H>
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
#include <RadSolve.H>
#endif

#include <memory>
#include <iostream>

#include <params_type.H>

using std::istream;
using std::ostream;

enum StateType : std::uint8_t {
               State_Type = 0,
#ifdef RADIATION
               Rad_Type,
#endif
#ifdef GRAVITY
               Potential_Type,
               Gravity_Type,
#endif
               Source_Type,
#ifdef REACTIONS
               Reactions_Type,
#endif
#ifdef MHD
               Mag_Type_x,
               Mag_Type_y,
               Mag_Type_z,
#endif
#ifdef REACTIONS
               Simplified_SDC_React_Type
#endif
};

// Create storage for all source terms.

enum sources : std::uint8_t {
               thermo_src = 0,
#ifdef DIFFUSION
               diff_src,
#endif
#ifdef HYBRID_MOMENTUM
               hybrid_src,
#endif
#ifdef GRAVITY
               grav_src,
#endif
#ifdef ROTATION
               rot_src,
#endif
#ifdef SPONGE
               sponge_src,
#endif
               ext_src,
               geom_src,
               num_src };


// time integration method

enum int_method : std::uint8_t {
    CornerTransportUpwind = 0,
    UnusedTimeIntegration,
    SpectralDeferredCorrections,
    SimplifiedSpectralDeferredCorrections
};

// Scope of a level sum: summed over all ranks, or only over
// the patches owned by this rank.

enum class SumScope : std::uint8_t {
    AllRanks = 0,
    ThisRank
};

// Whether a level sum leaves out the zones covered by the
// next finer level.

enum class CoveredZones : std::uint8_t {
    Exclude = 0,
    Include
};

// Which components of the interface state a Riemann solve stores:
// the reduced set (NIFC) needed by the rest of the update, or the
// full primitive state (NQ).

enum class FaceComponents : std::uint8_t {
    Reduced = 0,
    Full
};

// Which density resets are reported: none, only those in zones
// whose density is above the retry cutoff, or all of them.

enum class DensityResetReport : std::uint8_t {
    Quiet = 0,
    AboveRetryCutoff,
    All
};

// The density resets a verbosity level reports.

inline
DensityResetReport
density_reset_report (int verbosity)
{
    if (verbosity > 1) {
        return DensityResetReport::All;
    }
    if (verbosity > 0) {
        return DensityResetReport::AboveRetryCutoff;
    }
    return DensityResetReport::Quiet;
}

// Whether a source evaluation adds the sources it builds to the
// state, or only fills the source MultiFab.

enum class SourceUpdate : std::uint8_t {
    FillOnly = 0,
    ApplyToState
};

// Whether a check of a MultiFab covers only its valid zones or
// its ghost zones as well.

enum class GhostZones : std::uint8_t {
    Exclude = 0,
    Include
};

// Struct that returns information about
// why an advance failed.

struct advance_status {
    // What a failed advance passes on to the retry: a timestep to
    // retry with (used if positive and below the failed one) and
    // why the advance failed.
    struct retry_request {
        Real suggested_dt{};
        std::string reason{};
    };

    bool success{true};
    retry_request retry{};
    advance_status() = default;
};

// Checks and fixes of State_Type data that need nothing from the
// level the data belongs to.

///
/// Check for NaNs in the given MultiFab
///
/// @param state        MultiFab to check
/// @param ghost_zones  do we check the ghost cells?
///
void check_for_nan (const amrex::MultiFab& state, GhostZones ghost_zones = GhostZones::Exclude);

///
/// Normalize species fractions so they sum to 1
///
/// @param S_new    State
/// @param ng       number of ghost cells
///
void normalize_species (amrex::MultiFab& S_new, int ng);

///
/// Ensure the magnitude of the velocity is not larger than ``hydrosim.speed_limit``
/// in any zone.
///
/// @param state        state
/// @param ng           number of ghost cells
///
void enforce_speed_limit (amrex::MultiFab& state, int ng);

#ifdef TRUE_SDC
// Kernels of an SDC stage, for the Gauss-Lobatto and Radau node
// sets: the advective update from one time node to the next, the
// source term C of the reacting update, the initial guess for the
// reaction solve and the stored reaction state.

namespace sdc_stage {

void update_advection_o2_lobatto(const amrex::Box& bx,
                                 amrex::Real dt_m, amrex::Real dt,
                                 amrex::Array4<const amrex::Real> const& k_m,
                                 amrex::Array4<amrex::Real> const& k_n,
                                 amrex::Array4<const amrex::Real> const& A_m,
                                 amrex::Array4<const amrex::Real> const& A_0_old,
                                 amrex::Array4<const amrex::Real> const& A_1_old,
                                 int m_start);

void update_advection_o2_radau(const amrex::Box& bx,
                               amrex::Real dt_m, amrex::Real dt,
                               amrex::Array4<const amrex::Real> const& k_m,
                               amrex::Array4<amrex::Real> const& k_n,
                               amrex::Array4<const amrex::Real> const& A_m,
                               amrex::Array4<const amrex::Real> const& A_0_old,
                               amrex::Array4<const amrex::Real> const& A_1_old,
                               amrex::Array4<const amrex::Real> const& A_2_old,
                               int m_start);

void update_advection_o4_lobatto(const amrex::Box& bx,
                                 amrex::Real dt_m, amrex::Real dt,
                                 amrex::Array4<const amrex::Real> const& k_m,
                                 amrex::Array4<amrex::Real> const& k_n,
                                 amrex::Array4<const amrex::Real> const& A_m,
                                 amrex::Array4<const amrex::Real> const& A_0_old,
                                 amrex::Array4<const amrex::Real> const& A_1_old,
                                 amrex::Array4<const amrex::Real> const& A_2_old,
                                 int m_start);

void update_advection_o4_radau(const amrex::Box& bx,
                               amrex::Real dt_m, amrex::Real dt,
                               amrex::Array4<const amrex::Real> const& k_m,
                               amrex::Array4<amrex::Real> const& k_n,
                               amrex::Array4<const amrex::Real> const& A_m,
                               amrex::Array4<const amrex::Real> const& A_0_old,
                               amrex::Array4<const amrex::Real> const& A_1_old,
                               amrex::Array4<const amrex::Real> const& A_2_old,
                               amrex::Array4<const amrex::Real> const& A_3_old,
                               int m_start);

#ifdef REACTIONS
void compute_C2_lobatto(const amrex::Box& bx,
                        amrex::Real dt_m, amrex::Real dt,
                        amrex::Array4<const amrex::Real> const& A_m,
                        amrex::Array4<const amrex::Real> const& A_0_old,
                        amrex::Array4<const amrex::Real> const& A_1_old,
                        amrex::Array4<const amrex::Real> const& R_0_old,
                        amrex::Array4<const amrex::Real> const& R_1_old,
                        amrex::Array4<amrex::Real> const& C,
                        int m_start);

void compute_C2_radau(const amrex::Box& bx,
                      amrex::Real dt_m, amrex::Real dt,
                      amrex::Array4<const amrex::Real> const& A_m,
                      amrex::Array4<const amrex::Real> const& A_0_old,
                      amrex::Array4<const amrex::Real> const& A_1_old,
                      amrex::Array4<const amrex::Real> const& A_2_old,
                      amrex::Array4<const amrex::Real> const& R_0_old,
                      amrex::Array4<const amrex::Real> const& R_1_old,
                      amrex::Array4<const amrex::Real> const& R_2_old,
                      amrex::Array4<amrex::Real> const& C,
                      int m_start);

void compute_C4_lobatto(const amrex::Box& bx,
                        amrex::Real dt_m, amrex::Real dt,
                        amrex::Array4<const amrex::Real> const& A_m,
                        amrex::Array4<const amrex::Real> const& A_0_old,
                        amrex::Array4<const amrex::Real> const& A_1_old,
                        amrex::Array4<const amrex::Real> const& A_2_old,
                        amrex::Array4<const amrex::Real> const& R_0_old,
                        amrex::Array4<const amrex::Real> const& R_1_old,
                        amrex::Array4<const amrex::Real> const& R_2_old,
                        amrex::Array4<amrex::Real> const& C,
                        int m_start);

void compute_C4_radau(const amrex::Box& bx,
                      amrex::Real dt_m, amrex::Real dt,
                      amrex::Array4<const amrex::Real> const& A_m,
                      amrex::Array4<const amrex::Real> const& A_0_old,
                      amrex::Array4<const amrex::Real> const& A_1_old,
                      amrex::Array4<const amrex::Real> const& A_2_old,
                      amrex::Array4<const amrex::Real> const& A_3_old,
                      amrex::Array4<const amrex::Real> const& R_0_old,
                      amrex::Array4<const amrex::Real> const& R_1_old,
                      amrex::Array4<const amrex::Real> const& R_2_old,
                      amrex::Array4<const amrex::Real> const& R_3_old,
                      amrex::Array4<amrex::Real> const& C,
                      int m_start);

void conservative_update(const amrex::Box& bx, amrex::Real const dt_m,
                         amrex::Array4<const amrex::Real> const& U_old,
                         amrex::Array4<amrex::Real> const& U_new,
                         amrex::Array4<const amrex::Real> const& C,
                         amrex::Array4<const amrex::Real> const& R_new);
#endif

void compute_initial_guess(const amrex::Box& bx,
                           amrex::Array4<const amrex::Real> const& U_old,
                           amrex::Array4<const amrex::Real> const& U_new,
                           amrex::Array4<const amrex::Real> const& Adv_old,
                           amrex::Array4<const amrex::Real> const& React_old,
                           amrex::Array4<amrex::Real> const& U_guess,
                           amrex::Real const dt_m, int const sdc_iter);

#ifdef REACTIONS
void store_reaction_state(const amrex::Box& bx,
                          amrex::Array4<const amrex::Real> const& React_old,
                          amrex::Array4<amrex::Real> const& R_store);
#endif

}
#endif

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
/// Which variable lists a plotfile writes: the full plotfile
/// variables or the small plotfile variables.
///
    enum class PlotfileKind : std::uint8_t {
        Full = 0,
        Small
    };

///
/// @param dir      directory to write to
/// @param os       output stream
/// @param how      ``VisMF::How`` object
/// @param kind     full or small plotfile
///
    void plotFileOutput(const std::string& dir,
                        ostream& os,
                        amrex::VisMF::How how,
                        const PlotfileKind kind);


///
/// Write job info to file
///
/// @param dir       directory to write to
/// @param io_time   how long it took to generate the plotfile/checkpoint
///
    void writeJobInfo (const std::string& dir, const amrex::Real io_time);


///
/// Dump build info
///
    static void writeBuildInfo ();


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
    static void ErrorSetUp ();

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
    static void read_particle_params ();

///
/// Initialize particle locations and velocities (and strengths if relevant)
///
    void init_particles ();

///
/// Write particles in checkpoint directories
///
/// @param dir  directory containing checkpoint
///
    void ParticleCheckPoint (const std::string& dir);

///
/// Write particles in plotfile directories
///
/// @param dir  directory containing plotfile
///
    void ParticlePlotFile (const std::string& dir);

///
/// How to initialize at restart
///
/// @param restart_file     path to restart file
///
    void ParticlePostRestart (const std::string& restart_file);

///
/// Derived quantities associated with particles
///
/// @param name         name of particle variable to derive
/// @param time         current time
/// @param ngrow        Number of ghost cells
///
    std::unique_ptr<amrex::MultiFab> ParticleDerive (const std::string& name,
                                                     amrex::Real           time,
                                                     int            ngrow);

///
/// Timestamp particles
///
/// @param ngrow    number of ghost cells
///
    void TimestampParticles (int ngrow);

///
/// Advance the particles by dt
///
/// @param iteration    where we are in the current AMR subcycle
/// @param time         current time
/// @param dt           timestep
///
    void advance_particles (int iteration, amrex::Real time, amrex::Real dt);

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
/// The step an ::advance takes: the current simulation time, the
/// timestep, where we are in the current AMR subcycle (counting
/// from 1) and the number of subcycles at this level.
///
    struct LevelStep {
        amrex::Real time;
        amrex::Real dt;
        int amr_iteration;
        int amr_ncycle;
    };

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
    advance_status do_advance_ctu (amrex::Real time, amrex::Real dt);


#ifndef AMREX_USE_GPU
    amrex::Real do_advance_sdc (const LevelStep& step);
#endif

///
/// Save a copy of the old state data for use by a retry.
///
    void save_data_for_retry();

///
/// Zero the flux, mass flux, radial pressure and radiation flux
/// accumulators that are summed over the subcycles of an advance.
///
    void zero_flux_accumulators ();

///
/// Should we retry advancing the simulation? By default, we
/// don't do a retry unless the criteria are violated. This is
/// used only for the CTU advance.
///
/// @param dt       the timestep to advance (e.g., go from time to
///                    time + dt)
/// @param advance_status  was the advance successful?
///
    bool redo_advance_ctu(amrex::Real dt, const advance_status& status);

///
/// Subcycles until we've reached the target time, ``time`` + ``dt``.
/// The last timestep will be shortened if needed so that
/// it does not overshoot the ending time. This is used only for
/// the CTU advance.
///
/// @param step     the time, the timestep to advance (e.g., go from
///                 time to time + dt) and the AMR subcycle position
///                 of this advance
///
    amrex::Real subcycle_advance_ctu(const LevelStep& step);

///
/// This is run at the start of an ::advance on a single level.
/// It sets up all the necessary arrays and data containers,
/// ensures the data is valid and makes copies of the MultiFabs
/// in the old and new state in case a retry is performed.
///
/// @param step     the time, the timestep to advance and the
///                 AMR subcycle position of this advance
///
    void initialize_advance(const LevelStep& step);

///
/// If required, adds material lost in timestep to cumulative
/// losses, does refluxing, and clears arrays/data containers
/// initialized at start of the level's advance.
///
    void finalize_advance();

///
/// The finest level advanced together with this one. Without subcycling
/// the coarse level advances all levels at once; otherwise each level
/// advances on its own.
///
    int finest_level_advanced () const;

///
/// Performed at the start of ::do_advance, resets flags and
/// ensures required ghost zones are filled.
///
/// @param time     the current simulation time
/// @param dt       the timestep to advance
///
    advance_status initialize_do_advance (amrex::Real time, amrex::Real dt);

///
/// Performed at the end of ::do_advance, cleans up temporary data
/// created by ::initialize_do_advance
///
/// @param time     the current simulation time
/// @param dt       the timestep to advance
///
    advance_status finalize_do_advance (amrex::Real time, amrex::Real dt);

#ifdef MHD
///
/// The face-centered magnetic field of a level: one MultiFab per
/// field component, each on the faces normal to that direction.
///
    struct MagFaceFields {
        amrex::MultiFab& x;
        amrex::MultiFab& y;
        amrex::MultiFab& z;
    };
#endif

#include <HydroSim_sources.H>


#ifdef GRAVITY
#include <HydroSim_gravity.H>
#endif


#ifdef MHD
#include <HydroSim_mhd.H>
#endif

///
/// Which state an estimate is taken from: the old-time or the
/// new-time data of this level.
///
    enum class StateTime : std::uint8_t {
        Old = 0,
        New
    };

///
/// Estimate time step.
///
/// @param state_time   estimate from the old or the new state
///
    amrex::Real estTimeStep (StateTime state_time = StateTime::New);

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
    void do_energy_diagnostics();

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
    void scf_relaxation();

///
/// Perform the Hachisu SCF method.
///
    void do_hscf_solve();
#endif
#endif

///
/// Do work after restart with ``grown_factor`` > 1
///
    void post_grown_restart ();

#ifdef DO_PROBLEM_POST_SIMULATION

///
/// Do work at the end of the simulation - before the last outputs
///
/// @param amr_level    AMR levels in the simulation hierarchy
///
    static void problem_post_simulation (amrex::Vector<std::unique_ptr<AmrLevel> >& amr_level);
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
/// Apply the ::problem_tagging routine.
///
/// @param tags         TagBoxArray of tags
/// @param time         current time
///
    void apply_problem_tags (amrex::TagBoxArray& tags, amrex::Real time);


///
/// Apply any tagging restrictions that must be satisfied by all problems.
///
/// @param tags         TagBoxArray of tags
/// @param time         current time
///
    void apply_tagging_restrictions (amrex::TagBoxArray& tags, amrex::Real time);


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
#include <HydroSim_react.H>
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
    void final_radiation_call (amrex::MultiFab& S_new,
                               int iteration,
                               int ncycle);
#endif

///
/// Reset the internal energy
///
/// @param B        face-centered magnetic field
/// @param State    Current state (MultiFab)
/// @param ng       number of ghost cells
///
    void reset_internal_energy (
#ifdef MHD
                                const MagFaceFields& B,
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
    static void reset_internal_energy (const amrex::Box& bx,
#ifdef MHD
                                amrex::Array4<amrex::Real> const& Bx,
                                amrex::Array4<amrex::Real> const& By,
                                amrex::Array4<amrex::Real> const& Bz,
#endif
                                amrex::Array4<amrex::Real> const& u);


#ifdef MHD
///
/// Add magnetic contribution to energy density
/// @param B        face-centered magnetic field
/// @param state    the state to operate on
///
    static void add_magnetic_e (
        const MagFaceFields& B,
        amrex::MultiFab& state);

///
/// Check if divergence of B is zero at initialization
/// @param B        face-centered magnetic field
/// @param state    the state to operate on
///
    void check_div_B (
                      const MagFaceFields& B,
                      amrex::MultiFab& state);

#endif



///
/// Compute the current temperature
///
/// @param B        face-centered magnetic field
/// @param state    the state to operate on
/// @param time     current time
/// @param ng       number of ghost cells
///
    void computeTemp (
#ifdef MHD
                      const MagFaceFields& B,
#endif
                      amrex::MultiFab& state, amrex::Real time, int ng);


///
/// Add any terms needed to correct the source terms.
/// Currently this is a lagged predictor for CTU and the
/// time-centering correction for simplified SDC.
///
    void create_source_corrector();


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
    void swap_state_time_levels (const amrex::Real dt);

#ifdef DIFFUSION
#include <HydroSim_diffusion.H>
#endif


///
/// Volume weighted sum of given quantity
///
/// @param mf           MultiFab containing quantity
/// @param comp         integer index of quantity in MultiFab
/// @param scope        sum over all ranks, or only over this rank's patches?
/// @param covered      leave out the zones covered by the next finer level?
///
    amrex::Real volWgtSum (const MultiFab& mf, int comp,
                           SumScope scope=SumScope::AllRanks, CoveredZones covered=CoveredZones::Exclude);

///
/// Volume weighted sum of given quantity
///
/// @param name         Name of quantity
/// @param time         current time
/// @param scope        sum over all ranks, or only over this rank's patches?
/// @param covered      leave out the zones covered by the next finer level?
///
    amrex::Real volWgtSum (const std::string& name, amrex::Real time,
                           SumScope scope=SumScope::AllRanks, CoveredZones covered=CoveredZones::Exclude);

///
/// Sum weighted by volume multiplied by distance from center in given direction
///
/// @param mf       MultiFab containing quantity
/// @param comp     integer index of quantity in MultiFab
/// @param idir     Axis along which to compute distance from center
/// @param scope    sum over all ranks, or only over this rank's patches?
///
    amrex::Real locWgtSum (const MultiFab& mf, int comp, int idir, SumScope scope=SumScope::AllRanks);

///
/// Volume weighted sum of the product of two quantities
///
/// @param mf1      MultiFab containing first quantity
/// @param mf2      MultiFab containing second quantity
/// @param comp1    integer index of quantity in first MultiFab
/// @param comp2    integer index of quantity in second MultiFab
/// @param scope    sum over all ranks, or only over this rank's patches?
///
    amrex::Real volProductSum (const MultiFab& mf1, const MultiFab& mf2, int comp1, int comp2,
                               SumScope scope=SumScope::AllRanks);


///
/// Location weighted sum of (quantity) squared
///
/// @param name     Name of quantity
/// @param time     current time
/// @param idir     Direction along which to weight
/// @param scope    sum over all ranks, or only over this rank's patches?
///
    amrex::Real locSquaredSum (const std::string& name, amrex::Real time, int idir,
                               SumScope scope=SumScope::AllRanks);

#ifdef GRAVITY
///
/// Gravitational wave strain seen by an observer on each coordinate axis:
/// plus[n] and cross[n] are the two polarizations for the observer on axis n
///
    struct WaveStrain {
        amrex::Array<amrex::Real, 3> plus{};
        amrex::Array<amrex::Real, 3> cross{};
    };

///
/// Calculate the gravitational wave signal
///
/// @param time       current time
/// @param scope      sum over all ranks, or only over this rank's patches?
///
/// @return the contribution of this level to the strain
///
    WaveStrain gwstrain (Real time, SumScope scope);
#endif

#ifdef GRAVITY
///
/// Are we using point mass gravity?
///
    static int using_point_mass ();

///
/// Return the value of the point mass.
///
    static amrex::Real get_point_mass ();
#endif


///
/// Fill a version of the state with ``ng`` ghost zones from the state data.
/// After we do the fill, optionally clean the data using the iclean parameter.
///
/// @param S        MultiFab to be filled
/// @param time     current time
/// @param ng       number of ghost cells
///
    void fill_grown_state(amrex::MultiFab& S, amrex::Real time, int ng);



// Hydrodynamics
#include <HydroSim_hydro.H>

// SDC
#ifdef TRUE_SDC
#include <HydroSim_sdc.H>
#endif

    static void extern_init ();

    static void read_params ();


///
/// @param lev    AMR level index
///
    HydroSim& getLevel (int lev);

    void FluxRegCrseInit();
    void FluxRegFineAdd();


///
/// Where a reflux is done: right after the advance of the levels it
/// synchronizes, or in post_timestep() once the fine levels are done.
///
    enum class RefluxStage : std::uint8_t {
        AfterAdvance = 0,
        PostTimestep
    };

///
/// Do refluxing
///
/// @param crse_level         coarse AMR level index
/// @param fine_level         fine AMR level index
/// @param stage              where the reflux is done (see RefluxStage)
///
    void reflux (int crse_level, int fine_level, RefluxStage stage);


///
/// Enforces
/// \f[
/// \rho E = \rho e + \frac{1}{2} \rho (u^2 + v^2 + w^2)
/// \f]
///
/// @param S  State
///
    static void enforce_consistent_e (
#ifdef MHD
                               const MagFaceFields& B,
#endif
                               amrex::MultiFab& S);

///
/// This routine sets the density in ``state`` to be larger than the density floor.
/// Note that it will operate everywhere on ``state``, including ghost zones.
///
/// @param state        state
/// @param ng           number of ghost cells
///
    void enforce_min_density (amrex::MultiFab& state, int ng);

///
/// After a hydro advance, check to see whether any density went too small.
///
    advance_status check_for_negative_density ();

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
                      const MagFaceFields& B,
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
    void initMFs ();

///
/// Calculates volume weight sums of variables and prints to screen
///
    void sum_integrated_quantities ();

///
/// Whether the integrated quantities are due at coarse step ``nstep``
/// ending at ``cumtime``, either by step count (sum_interval) or because
/// the step crossed a multiple of sum_per.
///
/// @param nstep    coarse level step number
/// @param cumtime  simulation time at the end of the step
/// @param dtlev    coarse level timestep
///
    bool integral_sums_due (int nstep, amrex::Real cumtime, amrex::Real dtlev) const;

///
/// Problem-specific diagnostics (called by sum_integrated_quantities)
///
    void problem_diagnostics ();

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
    void define_new_center (const amrex::MultiFab& S, amrex::Real time);
    void write_center ();

///
/// @param time     current time
/// @param dt       timestep
///
    void pointmass_update(amrex::Real time, amrex::Real dt);
#endif

    static void stopJob ();

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

    static amrex::Vector<std::string> source_names;

///
/// Vector storing list of error tags.
///
    static amrex::Vector<amrex::AMRErrorTag> error_tags;

///
/// This MultiFab is on the coarser level.  This is useful for the coarser level
///     to mask out the finer level.  We only build this when it is needed.
///     This coarse MultiFab has to live on the fine level because it must be updated
///     even when only the fine level changes.
///
    amrex::MultiFab fine_mask;
    amrex::MultiFab& build_fine_mask();

///
/// Without subcycling, zones covered by the next finer level are
/// advanced there, so work on this level can skip them.
///
    bool skip_covered_zones () const;


///
/// A record of how many cells we have advanced throughout the simulation.
///     This is saved as a real because we will be storing the number of zones
///     advanced as a ratio with the number of zones on the coarse grid (to
///     help prevent the number from getting too large), and that may be a
///     non-integer number.
///
    static amrex::Real num_zones_advanced;

///
/// diagnostics
///
    static Vector<std::unique_ptr<std::fstream> > data_logs;
    static Vector<std::unique_ptr<std::fstream> > problem_data_logs;

///
/// runtime parameters
//
    static params_t params;


protected:

///
/// Owning list of MultiFabs, e.g. one per direction or per time node.
///
    using MultiFabVector = amrex::Vector<std::unique_ptr<amrex::MultiFab> >;


#ifdef GRAVITY
    int get_numpts();


///
/// Difference between composite and level solve for phi.
///
    amrex::MultiFab composite_level_phi_diff;
    MultiFabVector composite_level_grad_phi_diff;
#endif


#ifdef REACTIONS
///
/// Storage for the burn_weights
///
    amrex::MultiFab burn_weights;
    static std::vector<std::string> burn_weight_names;
#endif


///
/// A state array with ghost zones.
///
    amrex::MultiFab Sborder;

#ifdef MHD
   amrex::MultiFab Bx_old_tmp;
   amrex::MultiFab By_old_tmp;
   amrex::MultiFab Bz_old_tmp;
#endif

#ifdef RADIATION
    amrex::MultiFab Erborder;
    amrex::MultiFab lamborder;

    std::unique_ptr<RadSolve> rad_solver;
#endif



#ifdef TRUE_SDC
///
/// A state array for reaction-related data used by SDC
///
    amrex::MultiFab Sreact;


///
/// The primitive variable state array.
///
    amrex::MultiFab q;

///
/// A second primitive-state array used by the fourth-order method
///
    amrex::MultiFab q_bar;

///
/// Cell-centered temperature needed for fourth-order diffusion
//
#ifdef DIFFUSION
    amrex::MultiFab T_cc;
#endif

///
/// The auxiliary primitive variable state array.
///
    amrex::MultiFab qaux;
    amrex::MultiFab qaux_bar;
#endif


///
/// Source term corrector.
///
    amrex::MultiFab source_corrector;


///
/// Hydrodynamic (and radiation) fluxes.
///
    MultiFabVector fluxes;
#if (AMREX_SPACEDIM <= 2)
    amrex::MultiFab         P_radial;
#endif
#ifdef RADIATION
    MultiFabVector rad_fluxes;
#endif

    MultiFabVector mass_fluxes;

    amrex::FluxRegister flux_reg;
#if (AMREX_SPACEDIM <= 2)
    amrex::FluxRegister pres_reg;
#endif
#ifdef RADIATION
    amrex::FluxRegister rad_flux_reg;
#endif
#ifdef GRAVITY
    amrex::FluxRegister phi_reg;
#endif

///
/// Scalings for the flux registers.
///
    amrex::Real flux_crse_scale;
    amrex::Real flux_fine_scale;

#if (AMREX_SPACEDIM <= 2)
    amrex::Real pres_crse_scale;
    amrex::Real pres_fine_scale;
#endif



///
/// State data to hold if we want to do a retry.
///
    amrex::Vector<std::unique_ptr<amrex::StateData> > prev_state;



///
/// Flag for indicating that we want to save prev_state until the reflux.
///
    bool keep_prev_state{};


#ifdef TRUE_SDC
    //
    // Storage for the SDC time integration

    // this is the new iterations solution at the time nodes
    MultiFabVector k_new;

    // this is the old value of the advective update at the
    // nodes of the time integration
    MultiFabVector A_old;

    // this is the new value of the advective update at the
    // nodes of the time integration
    MultiFabVector A_new;

    // this is the old value of the reaction source at the
    // nodes of the time integration
#ifdef REACTIONS
    MultiFabVector R_old;
#endif

    static int SDC_NODES;
    static amrex::Vector<amrex::Real> dt_sdc;
    static amrex::Vector<amrex::Real> node_weights;
#endif

///
/// Wall time that we started the timestep
///
    amrex::Real wall_time_start;


///
/// The data.
///
    amrex::MultiFab             volume;
    amrex::Array<amrex::MultiFab,3> area;
    amrex::Vector< amrex::Vector<amrex::Real> > radius;
#if AMREX_SPACEDIM <= 2
    amrex::Array<amrex::MultiFab,2> dLogArea;
#endif

///
/// For initialization of the C++ values of the runtime parameters
  static int do_cxx_prob_initialize;


///
/// Keep track of which AMR iteration we're on.
///
    int iteration;

    //
    // Static data members.
    //

    static bool      signalStopJob;
    static int       radius_grow;
    static std::vector<std::string> err_list_names;
    static std::vector<int> err_list_ng;
    static int              num_err_list_default;
    static amrex::BCRec     phys_bc;

    static int         lastDtPlotLimited;
    static amrex::Real lastDtBeforePlotLimiting;

    int in_retry;
    int num_subcycles_taken;

    amrex::Real lastDt;


///
/// for keeping track of the amount of CPU or GPU time used -- this will persist
/// after restarts
///
    static amrex::Real      previousCPUTimeUsed;
    static amrex::Real      startCPUTime;

///
/// Get total CPU time
///
    static amrex::Real getCPUTime();

    bool             FillPatchedOldState_ok;


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
    static amrex::IntVect no_tile_size;

    static int hydro_tile_size_has_been_tuned;
    static Long largest_box_from_hydro_tile_size_tuning;

    static int SDC_Source_Type;
    static int num_state_type;


    // counters for various retries in HydroSim

///
/// subcycle capability
///
    int sub_iteration;
    int sub_ncycle;
    amrex::Real dt_subcycle;
    amrex::Real dt_advance;


///
/// sdc
///
    int sdc_iteration;

#ifdef TRUE_SDC
    int current_sdc_node;
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
HydroSim::stopJob()
{
  signalStopJob = true;


}

inline
int
HydroSim::get_output_at_completion()
{
    return output_at_completion;
}

#ifdef GRAVITY
inline
amrex::Real
HydroSim::get_point_mass ()
{
    return point_mass;
}

inline
int
HydroSim::using_point_mass ()
{
    return use_point_mass;
}
#endif

#endif /*_HydroSim_H_*/
