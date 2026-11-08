
// This file is automatically created by parse_hydrosim_params.py at build time.
// To update or add runtime parameters, please edit _cpp_parameters and rebuild.

#ifndef PARAMS_TYPE_H
#define PARAMS_TYPE_H

#include <hydrosim_limits.H>

struct diffusion_t {
    int verbose{0};
    int solver_max_order{4};
};

struct gravity_t {
    bool boundary_direct_sum{false};
    bool no_sync{false};
    bool composite_phi_fixup{true};
    bool accel_from_phi_grad{false};
    bool mlmg_grid_merge{true};
    bool mlmg_proc_gather{true};
    bool mlmg_coarse_nsolve{false};
    int radial_bin_ratio{1};
    int lnum{0};
    int verbose{0};
    int solve_level_cap{LEVEL_LIMIT-1};
    int mlmg_fmg_cycles{0};
    amrex::Real uniform_accel{0.0_rt};
    std::string gravity_model_name{"unset_gravity_type"};
};

struct hydrosim_t {
#ifdef SPONGE
    amrex::Real damping_lower_radius{-1.0_rt};
    amrex::Real damping_upper_radius{-1.0_rt};
    amrex::Real damping_lower_density{-1.0_rt};
    amrex::Real damping_upper_density{-1.0_rt};
    amrex::Real damping_lower_pressure{-1.0_rt};
    amrex::Real damping_upper_pressure{-1.0_rt};
    amrex::Real damping_lower_factor{0.0_rt};
    amrex::Real damping_upper_factor{1.0_rt};
    amrex::Real damping_target_x_velocity{0.0_rt};
    amrex::Real damping_target_y_velocity{0.0_rt};
    amrex::Real damping_target_z_velocity{0.0_rt};
    amrex::Real damping_timescale{-1.0_rt};
#endif
    bool do_reflux{true};
    bool refresh_sources_after_reflux{true};
    bool allow_anisotropic_cells{false};
    bool do_hydro{true};
    bool clamp_fourth_order{true};
    bool init_cell_averaged{false};
    bool use_recon_gamma1{false};
    bool enable_extra_source{false};
    bool split_momentum_hydro{false};
    bool ppm_parabola_clip{true};
    bool mhd_char_limit{true};
    bool shock_flux_switch{false};
    bool use_shock_flatten{true};
    bool transverse_eos_pressure{false};
    bool transverse_revert_density{true};
    bool transverse_rebuild_rhoe{false};
    bool use_pressure_slope{false};
    bool ppm_equil_recon{false};
    bool clip_small_rho_flux{false};
    bool do_damping_zone{false};
    bool damping_implicit{true};
    bool extra_source_implicit{false};
    bool source_half_step{false};
    bool first_order_scheme{false};
    bool hse_rest_vel{false};
    bool hse_temp_extrap{false};
    bool hse_mirror_vel{false};
    bool apply_ambient_bc{false};
    bool clamp_ambient_temperature{false};
    bool use_curvilinear_src{true};
    bool include_react_in_sdc_advect{true};
    bool dt_validity_before_advance{true};
    bool dt_validity_after_advance{true};
    bool plot_interval_exact{false};
    bool small_interval_exact{false};
    bool allow_step_retry{true};
    bool regrid_each_step{false};
    bool do_react{true};
    bool skip_burning_in_shocks{false};
    bool shock_sensor_subtract_src{true};
    bool drive_initial_convection{false};
    bool do_gravity_flag{true};
    bool track_center{false};
    bool do_frame_spin{true};
    bool boundary_callback_threadsafe{true};
    bool body_at_center{true};
    bool do_equilibrium_start{false};
    bool enable_special_refine{false};
    bool dump_old{false};
    bool plane_parallel_domain{false};
#ifdef AMREX_DEBUG
    bool log_source_changes{true};
#else
    bool log_source_changes{true};
#endif
    bool write_on_finish{true};
    bool keep_omegadot{false};
    bool keep_burn_weights{false};
    bool stop_on_bad_params{false};
    bool do_radiative_transfer{true};
    int state_interp_degree{1};
    int linear_interp_limiter{0};
    int time_scheme{0};
    int ppm_recon_kind{1};
    int ppm_temp_mode{0};
    int plm_iorder{2};
    int plm_slope_limiter{2};
    int riemann_method{0};
    int shock_star_max_iters{12};
    int riemann_search_mode{2};
    int xl_bc_kind{-1};
    int xr_bc_kind{-1};
    int yl_bc_kind{-1};
    int yr_bc_kind{-1};
    int zl_bc_kind{-1};
    int zr_bc_kind{-1};
    int ambient_bc_axis{-1};
    int ambient_outflow_mode{0};
    int sdc_accuracy_order{2};
    int sdc_quadrature{0};
    int sdc_extra_passes{0};
    int sdc_nonlin_method{1};
    int subcycle_cap{10};
    int deferred_iters{2};
    int gravity_source_mode{4};
    int restart_grow_factor{1};
    int prev_theta_cells{-1};
    int scf_iter_cap{30};
    int verbose{0};
    int sum_step_period{-1};
    int restart_step_override{-1};
    amrex::Real viscosity_coeff{0.1_rt};
    amrex::Real small_dens{-1.e200_rt};
    amrex::Real small_temp{-1.e200_rt};
    amrex::Real pressure_cutoff{-1.e200_rt};
    amrex::Real eint_floor{-1.e200_rt};
    amrex::Real riemann_star_tol{1.0e-5_rt};
    amrex::Real dual_energy_switch_hi{1.0e0_rt};
    amrex::Real dual_energy_switch{1.0e-4_rt};
    amrex::Real slope_rho_cutoff{-1.e20_rt};
    amrex::Real velocity_cap{0.0_rt};
    amrex::Real hse_bound_temp{-1.e200_rt};
    amrex::Real ambient_threshold_factor{1.1e0_rt};
    amrex::Real background_density{-1.e200_rt};
    amrex::Real background_temp{-1.e200_rt};
    amrex::Real background_energy{-1.e200_rt};
    amrex::Real hydro_tile_mem_ratio{-1.0_rt};
    amrex::Real constant_dt{-1.0_rt};
    amrex::Real first_step_dt{-1.0_rt};
    amrex::Real min_dt_time_ratio{1.e-12_rt};
    amrex::Real max_dt{1.e200_rt};
    amrex::Real cfl{0.8_rt};
    amrex::Real init_dt_shrink{1.0_rt};
    amrex::Real change_max{1.1_rt};
    amrex::Real subcycle_backoff_factor{0.5_rt};
    amrex::Real retry_density_cutoff{-1.e200_rt};
    amrex::Real massfrac_check_tolerance{1.e-2_rt};
    amrex::Real massfrac_check_min_density{-1.e200_rt};
    amrex::Real stop_field_threshold{1.e200_rt};
    amrex::Real burn_dt_energy{1.e200_rt};
    amrex::Real burn_dt_massfrac{1.e200_rt};
    amrex::Real nuc_dt_abund_floor{1.e-3_rt};
    amrex::Real burn_temp_min{0.0_rt};
    amrex::Real burn_temp_max{1.e200_rt};
    amrex::Real burn_rho_min{0.0_rt};
    amrex::Real burn_rho_max{1.e200_rt};
    amrex::Real shock_sensor_threshold{0.6666666666666666666666_rt};
    amrex::Real eos_temp_guess{1.e8_rt};
    amrex::Real convect_drive_end_time{1.e200_rt};
    amrex::Real convect_drive_reset_period{1.e200_rt};
    amrex::Real scf_peak_density{-1.e6_rt};
    amrex::Real scf_equator_radius{-1.e9_rt};
    amrex::Real scf_pole_radius{-1.e9_rt};
    amrex::Real scf_residual_tol{1.e-3_rt};
    amrex::Real tagging_radius_cap{10.0e0_rt};
    amrex::Real sum_time_period{-1.0e0_rt};
    amrex::Real restart_time_override{-1.e200_rt};
    std::string stop_field_name{};
    std::string job_label{"HydroSim"};
};

struct particles_t {
    bool restart_chk_no_particles{false};
    bool snapshot_density{true};
    bool snapshot_temperature{false};
    int particle_log_level{0};
    std::string particle_seed_file{};
    std::string particle_resume_file{};
    std::string particle_dump_file{};
    std::string snapshot_dir{};
};

struct radiation_t {
    bool moving_frame{true};
    bool plot_flux_limiter{false};
    bool plot_planck_opacity{false};
    bool plot_rosseland_opacity{false};
    bool plot_lab_rad_energy{false};
    bool plot_lab_rad_flux{false};
    bool plot_moving_flux{false};
    int closure{3};
    int limiter{2};
    int frequency_advection_mode{2};
    amrex::Real prop_temp_min{0.0_rt};
    amrex::Real radiation_flatten_cutoff{-1.0_rt};
};

struct radiation_solve_t {
    bool use_nonsym_solver_terms{false};
    int level_solver_kind{1};
    int maxiter{40};
    int verbose{0};
    amrex::Real reltol{1.e-10_rt};
    amrex::Real abstol{1.e-10_rt};
    amrex::Real alpha{1.0_rt};
    amrex::Real beta{1.0_rt};
};

struct params_t {
    diffusion_t diffusion;
    gravity_t gravity;
    hydrosim_t hydrosim;
    particles_t particles;
    radiation_t radiation;
    radiation_solve_t radiation_solve;
};

#endif
