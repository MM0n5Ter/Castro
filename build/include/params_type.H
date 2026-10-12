
// This file is automatically created by parse_castro_params.py at build time.
// To update or add runtime parameters, please edit _cpp_parameters and rebuild.

#ifndef PARAMS_TYPE_H
#define PARAMS_TYPE_H

#include <castro_limits.H>

struct castro_t {
#ifdef GRAVITY
    bool use_point_mass{false};
    bool point_mass_fix_solution{false};
    bool point_mass_offset_is_true{false};
    amrex::Real point_mass{0.0_rt};
    amrex::Real gw_dist{0.0_rt};
    amrex::Real point_mass_location_offset{0.0_rt};
#endif
#ifdef SPONGE
    amrex::Real sponge_lower_radius{-1.0_rt};
    amrex::Real sponge_upper_radius{-1.0_rt};
    amrex::Real sponge_lower_density{-1.0_rt};
    amrex::Real sponge_upper_density{-1.0_rt};
    amrex::Real sponge_lower_pressure{-1.0_rt};
    amrex::Real sponge_upper_pressure{-1.0_rt};
    amrex::Real sponge_lower_factor{0.0_rt};
    amrex::Real sponge_upper_factor{1.0_rt};
    amrex::Real sponge_target_x_velocity{0.0_rt};
    amrex::Real sponge_target_y_velocity{0.0_rt};
    amrex::Real sponge_target_z_velocity{0.0_rt};
    amrex::Real sponge_timescale{-1.0_rt};
#endif
    bool do_reflux{true};
    bool update_sources_after_reflux{true};
    bool allow_non_unit_aspect_zones{false};
    bool do_hydro{true};
    bool limit_fourth_order{true};
    bool initialization_is_cell_average{false};
    bool use_reconstructed_gamma1{false};
    bool add_ext_src{false};
    bool hybrid_hydro{false};
    bool ppm_do_limiting{true};
    bool mhd_limit_characteristic{true};
    bool hybrid_riemann{false};
    bool use_flattening{true};
    bool transverse_use_eos{false};
    bool transverse_reset_density{true};
    bool transverse_reset_rhoe{false};
    bool use_pslope{false};
    bool ppm_well_balanced{false};
    bool limit_fluxes_on_small_dens{false};
    bool do_sponge{false};
    bool sponge_implicit{true};
    bool ext_src_implicit{false};
    bool source_term_predictor{false};
    bool first_order_hydro{false};
    bool hse_zero_vels{false};
    bool hse_interp_temp{false};
    bool hse_reflect_vels{false};
    bool fill_ambient_bc{false};
    bool clamp_ambient_temp{false};
    bool use_geom_source{true};
    bool add_sdc_react_source_to_advection{true};
    bool check_dt_before_advance{true};
    bool check_dt_after_advance{true};
    bool plot_per_is_exact{false};
    bool small_plot_per_is_exact{false};
    bool use_retry{true};
    bool use_post_step_regrid{false};
    bool do_react{true};
    bool disable_shock_burning{false};
    bool shock_detection_include_sources{true};
    bool drive_initial_convection{false};
    bool do_grav{true};
    bool moving_center{false};
    bool do_rotation{true};
    bool bndry_func_thread_safe{true};
    bool star_at_center{true};
    bool do_scf_initial_model{false};
    bool do_special_tagging{false};
    bool dump_old{false};
    bool domain_is_plane_parallel{false};
#ifdef AMREX_DEBUG
    bool print_update_diagnostics{true};
#else
    bool print_update_diagnostics{true};
#endif
    bool output_at_completion{true};
    bool store_omegadot{false};
    bool store_burn_weights{false};
    bool abort_on_invalid_params{false};
    bool do_radiation{true};
    int state_interp_order{1};
    int lin_limit_state_interp{0};
    int time_integration_method{0};
    int ppm_type{1};
    int ppm_temp_fix{0};
    int plm_iorder{2};
    int plm_limiter{2};
    int riemann_solver{0};
    int riemann_shock_maxiter{12};
    int riemann_cg_blend{2};
    int xl_ext_bc_type{-1};
    int xr_ext_bc_type{-1};
    int yl_ext_bc_type{-1};
    int yr_ext_bc_type{-1};
    int zl_ext_bc_type{-1};
    int zr_ext_bc_type{-1};
    int ambient_fill_dir{-1};
    int ambient_outflow_vel{0};
    int sdc_order{2};
    int sdc_quadrature{0};
    int sdc_extra{0};
    int sdc_solver{1};
    int max_subcycles{10};
    int sdc_iters{2};
    int grav_source_type{4};
    int grown_factor{1};
    int old_theta_ncell{-1};
    int scf_max_iterations{30};
    int verbose{0};
    int sum_interval{-1};
    int reset_checkpoint_step{-1};
    amrex::Real difmag{0.1_rt};
    amrex::Real small_dens{-1.e200_rt};
    amrex::Real small_temp{-1.e200_rt};
    amrex::Real small_pres{-1.e200_rt};
    amrex::Real small_ener{-1.e200_rt};
    amrex::Real riemann_pstar_tol{1.0e-5_rt};
    amrex::Real dual_energy_eta1{1.0e0_rt};
    amrex::Real dual_energy_eta2{1.0e-4_rt};
    amrex::Real pslope_cutoff_density{-1.e20_rt};
    amrex::Real speed_limit{0.0_rt};
    amrex::Real hse_fixed_temp{-1.e200_rt};
    amrex::Real ambient_safety_factor{1.1e0_rt};
    amrex::Real ambient_density{-1.e200_rt};
    amrex::Real ambient_temp{-1.e200_rt};
    amrex::Real ambient_energy{-1.e200_rt};
    amrex::Real hydro_memory_footprint_ratio{-1.0_rt};
    amrex::Real fixed_dt{-1.0_rt};
    amrex::Real initial_dt{-1.0_rt};
    amrex::Real dt_cutoff{1.e-12_rt};
    amrex::Real max_dt{1.e200_rt};
    amrex::Real cfl{0.8_rt};
    amrex::Real init_shrink{1.0_rt};
    amrex::Real change_max{1.1_rt};
    amrex::Real retry_subcycle_factor{0.5_rt};
    amrex::Real retry_small_density_cutoff{-1.e200_rt};
    amrex::Real abundance_failure_tolerance{1.e-2_rt};
    amrex::Real abundance_failure_rho_cutoff{-1.e200_rt};
    amrex::Real stopping_criterion_value{1.e200_rt};
    amrex::Real dtnuc_e{1.e200_rt};
    amrex::Real dtnuc_X{1.e200_rt};
    amrex::Real dtnuc_X_threshold{1.e-3_rt};
    amrex::Real react_T_min{0.0_rt};
    amrex::Real react_T_max{1.e200_rt};
    amrex::Real react_rho_min{0.0_rt};
    amrex::Real react_rho_max{1.e200_rt};
    amrex::Real shock_detection_threshold{0.6666666666666666666666_rt};
    amrex::Real T_guess{1.e8_rt};
    amrex::Real drive_initial_convection_tmax{1.e200_rt};
    amrex::Real drive_initial_convection_reinit_period{1.e200_rt};
    amrex::Real scf_maximum_density{-1.e6_rt};
    amrex::Real scf_equatorial_radius{-1.e9_rt};
    amrex::Real scf_polar_radius{-1.e9_rt};
    amrex::Real scf_relax_tol{1.e-3_rt};
    amrex::Real max_tagging_radius{10.0e0_rt};
    amrex::Real sum_per{-1.0e0_rt};
    amrex::Real reset_checkpoint_time{-1.e200_rt};
    std::string stopping_criterion_field{};
    std::string job_name{"Castro"};
};

struct diffusion_t {
    int verbose{0};
    int mlmg_maxorder{4};
};

struct gravity_t {
    bool direct_sum_bcs{false};
    bool no_sync{false};
    bool do_composite_phi_correction{true};
    bool get_g_from_phi{false};
    bool mlmg_agglomeration{true};
    bool mlmg_consolidation{true};
    bool mlmg_nsolve{false};
    int drdxfac{1};
    int lnum{0};
    int verbose{0};
    int max_solve_level{MAX_LEV-1};
    int mlmg_max_fmg_iter{0};
    amrex::Real const_grav{0.0_rt};
    std::string gravity_type{"fillme"};
};

struct particles_t {
    bool restart_from_nonparticle_chkfile{false};
    bool timestamp_density{true};
    bool timestamp_temperature{false};
    int particle_verbose{0};
    std::string particle_init_file{};
    std::string particle_restart_file{};
    std::string particle_output_file{};
    std::string timestamp_dir{};
};

struct radiation_t {
    bool comoving{true};
    bool plot_lambda{false};
    bool plot_kappa_p{false};
    bool plot_kappa_r{false};
    bool plot_lab_Er{false};
    bool plot_lab_flux{false};
    bool plot_com_flux{false};
    int closure{3};
    int limiter{2};
    int fspace_advection_type{2};
    amrex::Real prop_temp_floor{0.0_rt};
    amrex::Real flatten_pp_threshold{-1.0_rt};
};

struct radsolve_t {
    bool use_hypre_nonsymmetric_terms{false};
    int level_solver_flag{1};
    int maxiter{40};
    int verbose{0};
    amrex::Real reltol{1.e-10_rt};
    amrex::Real abstol{1.e-10_rt};
    amrex::Real alpha{1.0_rt};
    amrex::Real beta{1.0_rt};
};

struct params_t {
    castro_t castro;
    diffusion_t diffusion;
    gravity_t gravity;
    particles_t particles;
    radiation_t radiation;
    radsolve_t radsolve;
};

#endif
