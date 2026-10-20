#ifndef HYDROSIM_H
#define HYDROSIM_H

#include <array>
#include <format>

#include <state_indices.H>

#include <hydrosim_params.H>
#include <prob_parameters.H>
#include <run_output.H>
#include <HydroSim_helpers.H>
#include <dt_limits.H>

using namespace hydrosim;


#include <fundamental_constants.H>

#include <AMReX_BC_TYPES.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ErrorList.H>
#include <AMReX_FluxRegister.H>
#include <network.H>
#include <eos.H>

#ifdef BL_LAZY
#include <AMReX_Lazy.H>
#endif



#include <memory>
#include <iostream>

#include <params_type.H>

using std::istream;
using std::ostream;

#include <level_defs.H>

enum class nan_scan_region : std::uint8_t { valid_only, with_ghosts };

enum class plotfile_kind : std::uint8_t { regular, reduced };

class HydroSim
    :
    public amrex::AmrLevel
{
public:

    HydroSim ();

    HydroSim (amrex::Amr&            papa,
            int             lev,
            const amrex::Geometry& level_geom,
            const amrex::BoxArray& bl,
            const amrex::DistributionMapping& dm,
            amrex::Real            time);

    ~HydroSim () override;

    HydroSim (const HydroSim&) = delete;
    HydroSim (HydroSim&&) = delete;
    HydroSim& operator= (const HydroSim&) = delete;
    HydroSim& operator= (HydroSim&&) = delete;

    void restart (amrex::Amr&     papa,
                  istream& is,
                  bool bReadSpecial = false) override;

    void set_state_in_checkpoint (amrex::Vector<int>& state_in_checkpoint) override;

    void checkPoint(const std::string& dir,
                    std::ostream&      os,
                    amrex::VisMF::How         how,
                    bool               dump_old) override;

    std::string thePlotFileType () const override;

    void setPlotVariables () override;

    void writePlotFile (const std::string& dir,
                        ostream&       os,
                        amrex::VisMF::How     how) override;

    void writeSmallPlotFile (const std::string& dir,
                             ostream&       os,
                             amrex::VisMF::How     how) override;

    void writeSnapshot(const std::string& dir,
                        ostream& os,
                        amrex::VisMF::How how,
                        plotfile_kind kind);


    void writeRunRecord (const std::string& dir, const amrex::Real write_seconds);


    static void writeBuildRecord ();


    static void variableSetUp ();

    static void SetupErrorTags ();

    static void variableCleanUp ();

    void initData () override;


#ifdef LOWMACH_INIT
    void LOWMACH_init ();
#endif


    amrex::MultiFab* Area () { return area.data(); }

    amrex::MultiFab& Area (int dir) { return area[dir]; }

    amrex::MultiFab& Volume () { return volume; }

    amrex::MultiFab& Fluxes (int dir);

    void setTimeLevel (amrex::Real time,
                       amrex::Real dt_old,
                       amrex::Real dt_new) override;

    void init (amrex::AmrLevel& old) override;

    void init () override;

    int okToContinue () override;

    amrex::Real advance (amrex::Real time,
                         amrex::Real dt,
                         int  iteration,
                         int  ncycle) override;

    step_result run_advance_ctu (amrex::Real time, amrex::Real dt);


#ifndef AMREX_USE_GPU
    amrex::Real run_advance_sdc (amrex::Real time,
                                amrex::Real dt,
                                int  amr_pass,
                                int  amr_cycles);
#endif

    void snapshot_for_rerun();

    int last_level_in_step () const;

    bool rerun_advance_ctu(amrex::Real dt, const step_result& status);

    amrex::Real advance_subcycles_ctu(amrex::Real time, amrex::Real dt, int amr_pass, int amr_cycles);

    void begin_advance(amrex::Real time, amrex::Real dt, int amr_pass);

    void end_advance();

    step_result begin_step_advance (amrex::Real time, amrex::Real dt);

    step_result end_step_advance (amrex::Real time, amrex::Real dt);

    static void scan_for_nan(const amrex::MultiFab& state, nan_scan_region region=nan_scan_region::valid_only);


#include <HydroSim_forcing.H>


#ifdef GRAVITY
#include <HydroSim_gravity.H>
#endif



    amrex::Real estTimeStep (state_epoch epoch = state_epoch::new_time);

    amrex::Real initialTimeStep ();

    void computeInitialDt (int                   finest_level,
                           int                    ,
                           amrex::Vector<int>&           n_cycle,
                           const amrex::Vector<amrex::IntVect>&  ,
                           amrex::Vector<amrex::Real>&          dt_level,
                           amrex::Real                  stop_time) override;

    void computeNewDt (int                   finest_level,
                       int                    ,
                       amrex::Vector<int>&           n_cycle,
                       const amrex::Vector<amrex::IntVect>&  ,
                       amrex::Vector<amrex::Real>&          dt_min,
                       amrex::Vector<amrex::Real>&          dt_level,
                       amrex::Real                  stop_time,
                       int                   post_regrid_flag) override;

    void allocOldData () override;

    void removeOldData () override;


    void print_energy_budget();

    void post_timestep (int iteration) override;

    void postCoarseTimeStep (amrex::Real cumtime) override;

    void post_regrid (int lbase,
                      int new_finest) override;

    void post_restart () override;

    void post_init (amrex::Real stop_time) override;

#ifdef GRAVITY
#endif

    void after_grow_restart ();

    void rebuild_gravity_and_radiation ();

#ifdef DO_PROBLEM_POST_SIMULATION

    static void user_post_simulation (amrex::Vector<std::unique_ptr<AmrLevel> >& amr_level);
#endif


    void errorEst (amrex::TagBoxArray& tags,
                   int          clearval,
                   int          tagval,
                   amrex::Real  time,
                   int          n_error_buf = 0,
                   int          ngrow = 0) override;


    void apply_user_tags (amrex::TagBoxArray& tags, amrex::Real time);


    void apply_tag_limits (amrex::TagBoxArray& tags, amrex::Real time);


    std::unique_ptr<amrex::MultiFab> derive (const std::string& name,
                                             amrex::Real        time,
                                             int                ngrow) override;

    void derive (const std::string& name,
                 amrex::Real               time,
                 amrex::MultiFab&          mf,
                 int                dcomp) override;





    void repair_internal_energy (
                                amrex::MultiFab& State, int ng);


    static void repair_internal_energy (const amrex::Box& bx,
                                amrex::Array4<amrex::Real> const& u);




    void computeTemp (
                      amrex::MultiFab& state, amrex::Real time, int ng);


    void create_forcing_corrector();


    void swap_field_time_levels (const amrex::Real dt);



    amrex::Real volWeightedTotal (const MultiFab& mf, int comp, bool local=false, bool exclude_fine=true);

    amrex::Real volWeightedTotal (const std::string& name, amrex::Real time, bool local=false, bool exclude_fine=true);

    amrex::Real locWeightedSum (const MultiFab& mf, int comp, int idir, bool local=false);

    amrex::Real locWeightedSum (const std::string& name, amrex::Real time, int idir, bool local=false);

    amrex::Real volPairTotal (const MultiFab& mf1, const MultiFab& mf2, int comp1, int comp2, bool local=false);

    amrex::Real volPairTotal (const std::string& name1, const std::string& name2, amrex::Real time, bool local=false);


    amrex::Real locDistSquaredTotal (const std::string& name, amrex::Real time, int idir, bool local=false);

#ifdef GRAVITY

    struct strain_components {
        std::array<Real, 3> plus{}, cross{};
    };

    strain_components wave_strain (Real time, bool local);
#endif

#ifdef GRAVITY
    static int using_central_mass ();

    static amrex::Real get_central_mass ();
#endif


    void fill_ghost_field(amrex::MultiFab& S, amrex::Real time, int ng);


#include <HydroSim_hydro.H>


    static void microphys_init ();

    static void read_params ();


    HydroSim& getLevel (int lev) { return *dynamic_cast<HydroSim *>(&parent->getLevel(lev)); }

    void FluxRegCoarseInit();
    void FluxRegFineAccum();


    void reflux (int crse_level, int fine_level, reflux_origin origin);


    static void renorm_species (amrex::MultiFab& S_new, int ng);


    static void reconcile_energy (
                               amrex::MultiFab& S);

    void clamp_min_density (amrex::MultiFab& state, int ng);

    step_result scan_for_negative_mass_density ();

    static void clamp_speed_limit (amrex::MultiFab& state, int ng);

    void clean_state (
                      amrex::MultiFab& state, amrex::Real time, int ng);

    void avgDown ();


    void avgDown (int state_indx);

    void buildMetrics ();


    void allocFabs ();

    void tally_integrated_quantities ();

    void user_diagnostics ();

    void write_info ();

    static amrex::BCRec physbc () { return phys_bc; }

#ifdef GRAVITY

    void define_tracked_center (const amrex::MultiFab& S, amrex::Real time);
    void write_tracked_center ();

    void centralmass_update(amrex::Real time, amrex::Real dt);
#endif

    static void haltRun ();

    bool oversubscribing() {
#ifdef AMREX_USE_GPU
        return (amrex::MultiFab::queryMemUsage(std::format("AmrLevel_Level_{}", level)) >=
                static_cast<Long>(amrex::Gpu::Device::totalGlobalMem()));
#else
        return false;
#endif
    }


    static amrex::Vector<std::string> forcing_names;

    static amrex::Vector<amrex::AMRErrorTag> tag_criteria;

    amrex::MultiFab fine_mask;
    amrex::MultiFab& make_fine_mask();


    static amrex::Real cells_advanced_total;

    static Vector<std::unique_ptr<std::fstream> > run_logs;
    static Vector<std::unique_ptr<std::fstream> > user_run_logs;

    static params_t params;


protected:


    amrex::Vector<std::unique_ptr<amrex::iMultiFab> > interior_masks;

    amrex::iMultiFab& make_boundary_mask (int ng);

#ifdef GRAVITY
    int count_points_1d();


    amrex::MultiFab composite_minus_level_phi;
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > composite_minus_level_grad_phi;
#endif




    amrex::MultiFab Sborder;






    amrex::MultiFab forcing_corrector;


    amrex::Vector<std::unique_ptr<amrex::MultiFab> > fluxes;
#if (AMREX_SPACEDIM <= 2)
    amrex::MultiFab         Pressure_radial;
#endif

    amrex::Vector<std::unique_ptr<amrex::MultiFab> > mass_face_flux;

    amrex::FluxRegister flux_reg;
#if (AMREX_SPACEDIM <= 2)
    amrex::FluxRegister pressure_flux_reg;
#endif
#ifdef GRAVITY
    amrex::FluxRegister pot_flux_reg;
#endif

    amrex::Real flux_coarse_scale;
    amrex::Real flux_refined_scale;

#if (AMREX_SPACEDIM <= 2)
    amrex::Real pressure_coarse_scale;
    amrex::Real pressure_refined_scale;
#endif


    amrex::Vector<std::unique_ptr<amrex::StateData> > prev_state;


    bool retain_prev_field{};



    amrex::Real wall_clock_start;


    amrex::MultiFab             volume;
    amrex::Array<amrex::MultiFab,3> area;
    amrex::Vector< amrex::Vector<amrex::Real> > radius;
#if AMREX_SPACEDIM <= 2
    amrex::Array<amrex::MultiFab,2> dLogFaceMeasure;
#endif

  static int user_init_pending;


    int iteration;


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


    static amrex::Real      priorCpuSeconds;
    static amrex::Real      startCpuSeconds;

    static amrex::Real getCpuSeconds();

    bool             OldStateFilled;


    static std::unique_ptr<class Gravity> gravity;


    static std::unique_ptr<class Diffusion> diffusion;



    static amrex::IntVect hydro_tile_size;
    static amrex::IntVect untiled_size;

    static int sweep_tile_tuned;
    static Long largest_tuned_box_pts;

    static int Forcing_Type_Index;
    static int num_field_types;


    int sub_pass;
    int sub_cycles;
    amrex::Real dt_cycle_step;
    amrex::Real dt_level_step;


    int correction_iter;



public:
#include <Problem.H>

};


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

#endif
