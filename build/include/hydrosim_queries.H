
// This file is automatically created by parse_hydrosim_params.py at build time.
// To update or add runtime parameters, please edit _cpp_parameters and rebuild.



#ifdef GRAVITY
hydrosim::use_central_mass = false;
pp.query("use_central_mass", hydrosim::use_central_mass);
pp.query("use_central_mass", HydroSim::params.hydrosim.use_central_mass);

hydrosim::point_mass = 0.0_rt;
pp.query("point_mass", hydrosim::point_mass);
pp.query("point_mass", HydroSim::params.hydrosim.point_mass);

hydrosim::point_mass_correct_solution = false;
pp.query("point_mass_correct_solution", hydrosim::point_mass_correct_solution);
pp.query("point_mass_correct_solution", HydroSim::params.hydrosim.point_mass_correct_solution);

hydrosim::wave_obs_distance = 0.0_rt;
pp.query("wave_obs_distance", hydrosim::wave_obs_distance);
pp.query("wave_obs_distance", HydroSim::params.hydrosim.wave_obs_distance);

hydrosim::point_mass_is_shifted = false;
pp.query("point_mass_is_shifted", hydrosim::point_mass_is_shifted);
pp.query("point_mass_is_shifted", HydroSim::params.hydrosim.point_mass_is_shifted);

hydrosim::point_mass_shift = 0.0_rt;
pp.query("point_mass_shift", hydrosim::point_mass_shift);
pp.query("point_mass_shift", HydroSim::params.hydrosim.point_mass_shift);

#endif


#ifdef SPONGE
hydrosim::damping_lower_radius = -1.0_rt;
pp.query("damping_lower_radius", hydrosim::damping_lower_radius);
pp.query("damping_lower_radius", HydroSim::params.hydrosim.damping_lower_radius);

hydrosim::damping_upper_radius = -1.0_rt;
pp.query("damping_upper_radius", hydrosim::damping_upper_radius);
pp.query("damping_upper_radius", HydroSim::params.hydrosim.damping_upper_radius);

hydrosim::damping_lower_density = -1.0_rt;
pp.query("damping_lower_density", hydrosim::damping_lower_density);
pp.query("damping_lower_density", HydroSim::params.hydrosim.damping_lower_density);

hydrosim::damping_upper_density = -1.0_rt;
pp.query("damping_upper_density", hydrosim::damping_upper_density);
pp.query("damping_upper_density", HydroSim::params.hydrosim.damping_upper_density);

hydrosim::damping_lower_pressure = -1.0_rt;
pp.query("damping_lower_pressure", hydrosim::damping_lower_pressure);
pp.query("damping_lower_pressure", HydroSim::params.hydrosim.damping_lower_pressure);

hydrosim::damping_upper_pressure = -1.0_rt;
pp.query("damping_upper_pressure", hydrosim::damping_upper_pressure);
pp.query("damping_upper_pressure", HydroSim::params.hydrosim.damping_upper_pressure);

hydrosim::damping_lower_factor = 0.0_rt;
pp.query("damping_lower_factor", hydrosim::damping_lower_factor);
pp.query("damping_lower_factor", HydroSim::params.hydrosim.damping_lower_factor);

hydrosim::damping_upper_factor = 1.0_rt;
pp.query("damping_upper_factor", hydrosim::damping_upper_factor);
pp.query("damping_upper_factor", HydroSim::params.hydrosim.damping_upper_factor);

hydrosim::damping_target_x_velocity = 0.0_rt;
pp.query("damping_target_x_velocity", hydrosim::damping_target_x_velocity);
pp.query("damping_target_x_velocity", HydroSim::params.hydrosim.damping_target_x_velocity);

hydrosim::damping_target_y_velocity = 0.0_rt;
pp.query("damping_target_y_velocity", hydrosim::damping_target_y_velocity);
pp.query("damping_target_y_velocity", HydroSim::params.hydrosim.damping_target_y_velocity);

hydrosim::damping_target_z_velocity = 0.0_rt;
pp.query("damping_target_z_velocity", hydrosim::damping_target_z_velocity);
pp.query("damping_target_z_velocity", HydroSim::params.hydrosim.damping_target_z_velocity);

hydrosim::damping_timescale = -1.0_rt;
pp.query("damping_timescale", hydrosim::damping_timescale);
pp.query("damping_timescale", HydroSim::params.hydrosim.damping_timescale);

#endif

hydrosim::state_interp_degree = 1;
pp.query("state_interp_degree", hydrosim::state_interp_degree);
pp.query("state_interp_degree", HydroSim::params.hydrosim.state_interp_degree);

hydrosim::linear_interp_limiter = 0;
pp.query("linear_interp_limiter", hydrosim::linear_interp_limiter);
pp.query("linear_interp_limiter", HydroSim::params.hydrosim.linear_interp_limiter);

hydrosim::do_reflux = true;
pp.query("do_reflux", hydrosim::do_reflux);
pp.query("do_reflux", HydroSim::params.hydrosim.do_reflux);

hydrosim::refresh_sources_after_reflux = true;
pp.query("refresh_sources_after_reflux", hydrosim::refresh_sources_after_reflux);
pp.query("refresh_sources_after_reflux", HydroSim::params.hydrosim.refresh_sources_after_reflux);

hydrosim::allow_anisotropic_cells = false;
pp.query("allow_anisotropic_cells", hydrosim::allow_anisotropic_cells);
pp.query("allow_anisotropic_cells", HydroSim::params.hydrosim.allow_anisotropic_cells);

hydrosim::viscosity_coeff = 0.1_rt;
pp.query("viscosity_coeff", hydrosim::viscosity_coeff);
pp.query("viscosity_coeff", HydroSim::params.hydrosim.viscosity_coeff);

hydrosim::small_dens = -1.e200_rt;
pp.query("small_dens", hydrosim::small_dens);
pp.query("small_dens", HydroSim::params.hydrosim.small_dens);

hydrosim::small_temp = -1.e200_rt;
pp.query("small_temp", hydrosim::small_temp);
pp.query("small_temp", HydroSim::params.hydrosim.small_temp);

hydrosim::pressure_cutoff = -1.e200_rt;
pp.query("pressure_cutoff", hydrosim::pressure_cutoff);
pp.query("pressure_cutoff", HydroSim::params.hydrosim.pressure_cutoff);

hydrosim::eint_floor = -1.e200_rt;
pp.query("eint_floor", hydrosim::eint_floor);
pp.query("eint_floor", HydroSim::params.hydrosim.eint_floor);

hydrosim::do_hydro = true;
pp.query("do_hydro", hydrosim::do_hydro);
pp.query("do_hydro", HydroSim::params.hydrosim.do_hydro);

hydrosim::time_scheme = 0;
pp.query("time_scheme", hydrosim::time_scheme);
pp.query("time_scheme", HydroSim::params.hydrosim.time_scheme);

hydrosim::clamp_fourth_order = true;
pp.query("clamp_fourth_order", hydrosim::clamp_fourth_order);
pp.query("clamp_fourth_order", HydroSim::params.hydrosim.clamp_fourth_order);

hydrosim::init_cell_averaged = false;
pp.query("init_cell_averaged", hydrosim::init_cell_averaged);
pp.query("init_cell_averaged", HydroSim::params.hydrosim.init_cell_averaged);

hydrosim::use_recon_gamma1 = false;
pp.query("use_recon_gamma1", hydrosim::use_recon_gamma1);
pp.query("use_recon_gamma1", HydroSim::params.hydrosim.use_recon_gamma1);

hydrosim::enable_extra_source = false;
pp.query("enable_extra_source", hydrosim::enable_extra_source);
pp.query("enable_extra_source", HydroSim::params.hydrosim.enable_extra_source);

hydrosim::split_momentum_hydro = false;
pp.query("split_momentum_hydro", hydrosim::split_momentum_hydro);
pp.query("split_momentum_hydro", HydroSim::params.hydrosim.split_momentum_hydro);

hydrosim::ppm_recon_kind = 1;
pp.query("ppm_recon_kind", hydrosim::ppm_recon_kind);
pp.query("ppm_recon_kind", HydroSim::params.hydrosim.ppm_recon_kind);

hydrosim::ppm_parabola_clip = true;
pp.query("ppm_parabola_clip", hydrosim::ppm_parabola_clip);
pp.query("ppm_parabola_clip", HydroSim::params.hydrosim.ppm_parabola_clip);

hydrosim::mhd_char_limit = true;
pp.query("mhd_char_limit", hydrosim::mhd_char_limit);
pp.query("mhd_char_limit", HydroSim::params.hydrosim.mhd_char_limit);

hydrosim::ppm_temp_mode = 0;
pp.query("ppm_temp_mode", hydrosim::ppm_temp_mode);
pp.query("ppm_temp_mode", HydroSim::params.hydrosim.ppm_temp_mode);

hydrosim::plm_iorder = 2;
pp.query("plm_iorder", hydrosim::plm_iorder);
pp.query("plm_iorder", HydroSim::params.hydrosim.plm_iorder);

hydrosim::plm_slope_limiter = 2;
pp.query("plm_slope_limiter", hydrosim::plm_slope_limiter);
pp.query("plm_slope_limiter", HydroSim::params.hydrosim.plm_slope_limiter);

hydrosim::shock_flux_switch = false;
pp.query("shock_flux_switch", hydrosim::shock_flux_switch);
pp.query("shock_flux_switch", HydroSim::params.hydrosim.shock_flux_switch);

hydrosim::riemann_method = 0;
pp.query("riemann_method", hydrosim::riemann_method);
pp.query("riemann_method", HydroSim::params.hydrosim.riemann_method);

hydrosim::shock_star_max_iters = 12;
pp.query("shock_star_max_iters", hydrosim::shock_star_max_iters);
pp.query("shock_star_max_iters", HydroSim::params.hydrosim.shock_star_max_iters);

hydrosim::riemann_star_tol = 1.0e-5_rt;
pp.query("riemann_star_tol", hydrosim::riemann_star_tol);
pp.query("riemann_star_tol", HydroSim::params.hydrosim.riemann_star_tol);

hydrosim::riemann_search_mode = 2;
pp.query("riemann_search_mode", hydrosim::riemann_search_mode);
pp.query("riemann_search_mode", HydroSim::params.hydrosim.riemann_search_mode);

hydrosim::use_shock_flatten = true;
pp.query("use_shock_flatten", hydrosim::use_shock_flatten);
pp.query("use_shock_flatten", HydroSim::params.hydrosim.use_shock_flatten);

hydrosim::transverse_eos_pressure = false;
pp.query("transverse_eos_pressure", hydrosim::transverse_eos_pressure);
pp.query("transverse_eos_pressure", HydroSim::params.hydrosim.transverse_eos_pressure);

hydrosim::transverse_revert_density = true;
pp.query("transverse_revert_density", hydrosim::transverse_revert_density);
pp.query("transverse_revert_density", HydroSim::params.hydrosim.transverse_revert_density);

hydrosim::transverse_rebuild_rhoe = false;
pp.query("transverse_rebuild_rhoe", hydrosim::transverse_rebuild_rhoe);
pp.query("transverse_rebuild_rhoe", HydroSim::params.hydrosim.transverse_rebuild_rhoe);

hydrosim::dual_energy_switch_hi = 1.0e0_rt;
pp.query("dual_energy_switch_hi", hydrosim::dual_energy_switch_hi);
pp.query("dual_energy_switch_hi", HydroSim::params.hydrosim.dual_energy_switch_hi);

hydrosim::dual_energy_switch = 1.0e-4_rt;
pp.query("dual_energy_switch", hydrosim::dual_energy_switch);
pp.query("dual_energy_switch", HydroSim::params.hydrosim.dual_energy_switch);

hydrosim::use_pressure_slope = false;
pp.query("use_pressure_slope", hydrosim::use_pressure_slope);
pp.query("use_pressure_slope", HydroSim::params.hydrosim.use_pressure_slope);

hydrosim::ppm_equil_recon = false;
pp.query("ppm_equil_recon", hydrosim::ppm_equil_recon);
pp.query("ppm_equil_recon", HydroSim::params.hydrosim.ppm_equil_recon);

hydrosim::slope_rho_cutoff = -1.e20_rt;
pp.query("slope_rho_cutoff", hydrosim::slope_rho_cutoff);
pp.query("slope_rho_cutoff", HydroSim::params.hydrosim.slope_rho_cutoff);

hydrosim::clip_small_rho_flux = false;
pp.query("clip_small_rho_flux", hydrosim::clip_small_rho_flux);
pp.query("clip_small_rho_flux", HydroSim::params.hydrosim.clip_small_rho_flux);

hydrosim::velocity_cap = 0.0_rt;
pp.query("velocity_cap", hydrosim::velocity_cap);
pp.query("velocity_cap", HydroSim::params.hydrosim.velocity_cap);

hydrosim::do_damping_zone = false;
pp.query("do_damping_zone", hydrosim::do_damping_zone);
pp.query("do_damping_zone", HydroSim::params.hydrosim.do_damping_zone);

hydrosim::damping_implicit = true;
pp.query("damping_implicit", hydrosim::damping_implicit);
pp.query("damping_implicit", HydroSim::params.hydrosim.damping_implicit);

hydrosim::extra_source_implicit = false;
pp.query("extra_source_implicit", hydrosim::extra_source_implicit);
pp.query("extra_source_implicit", HydroSim::params.hydrosim.extra_source_implicit);

hydrosim::source_half_step = false;
pp.query("source_half_step", hydrosim::source_half_step);
pp.query("source_half_step", HydroSim::params.hydrosim.source_half_step);

hydrosim::first_order_scheme = false;
pp.query("first_order_scheme", hydrosim::first_order_scheme);
pp.query("first_order_scheme", HydroSim::params.hydrosim.first_order_scheme);

hydrosim::xl_bc_kind = -1;
pp.query("xl_bc_kind", hydrosim::xl_bc_kind);
pp.query("xl_bc_kind", HydroSim::params.hydrosim.xl_bc_kind);

hydrosim::xr_bc_kind = -1;
pp.query("xr_bc_kind", hydrosim::xr_bc_kind);
pp.query("xr_bc_kind", HydroSim::params.hydrosim.xr_bc_kind);

hydrosim::yl_bc_kind = -1;
pp.query("yl_bc_kind", hydrosim::yl_bc_kind);
pp.query("yl_bc_kind", HydroSim::params.hydrosim.yl_bc_kind);

hydrosim::yr_bc_kind = -1;
pp.query("yr_bc_kind", hydrosim::yr_bc_kind);
pp.query("yr_bc_kind", HydroSim::params.hydrosim.yr_bc_kind);

hydrosim::zl_bc_kind = -1;
pp.query("zl_bc_kind", hydrosim::zl_bc_kind);
pp.query("zl_bc_kind", HydroSim::params.hydrosim.zl_bc_kind);

hydrosim::zr_bc_kind = -1;
pp.query("zr_bc_kind", hydrosim::zr_bc_kind);
pp.query("zr_bc_kind", HydroSim::params.hydrosim.zr_bc_kind);

hydrosim::hse_rest_vel = false;
pp.query("hse_rest_vel", hydrosim::hse_rest_vel);
pp.query("hse_rest_vel", HydroSim::params.hydrosim.hse_rest_vel);

hydrosim::hse_temp_extrap = false;
pp.query("hse_temp_extrap", hydrosim::hse_temp_extrap);
pp.query("hse_temp_extrap", HydroSim::params.hydrosim.hse_temp_extrap);

hydrosim::hse_bound_temp = -1.e200_rt;
pp.query("hse_bound_temp", hydrosim::hse_bound_temp);
pp.query("hse_bound_temp", HydroSim::params.hydrosim.hse_bound_temp);

hydrosim::hse_mirror_vel = false;
pp.query("hse_mirror_vel", hydrosim::hse_mirror_vel);
pp.query("hse_mirror_vel", HydroSim::params.hydrosim.hse_mirror_vel);

hydrosim::apply_ambient_bc = false;
pp.query("apply_ambient_bc", hydrosim::apply_ambient_bc);
pp.query("apply_ambient_bc", HydroSim::params.hydrosim.apply_ambient_bc);

hydrosim::ambient_bc_axis = -1;
pp.query("ambient_bc_axis", hydrosim::ambient_bc_axis);
pp.query("ambient_bc_axis", HydroSim::params.hydrosim.ambient_bc_axis);

hydrosim::ambient_outflow_mode = 0;
pp.query("ambient_outflow_mode", hydrosim::ambient_outflow_mode);
pp.query("ambient_outflow_mode", HydroSim::params.hydrosim.ambient_outflow_mode);

hydrosim::clamp_ambient_temperature = false;
pp.query("clamp_ambient_temperature", hydrosim::clamp_ambient_temperature);
pp.query("clamp_ambient_temperature", HydroSim::params.hydrosim.clamp_ambient_temperature);

hydrosim::ambient_threshold_factor = 1.1e0_rt;
pp.query("ambient_threshold_factor", hydrosim::ambient_threshold_factor);
pp.query("ambient_threshold_factor", HydroSim::params.hydrosim.ambient_threshold_factor);

hydrosim::background_density = -1.e200_rt;
pp.query("background_density", hydrosim::background_density);
pp.query("background_density", HydroSim::params.hydrosim.background_density);

hydrosim::background_temp = -1.e200_rt;
pp.query("background_temp", hydrosim::background_temp);
pp.query("background_temp", HydroSim::params.hydrosim.background_temp);

hydrosim::background_energy = -1.e200_rt;
pp.query("background_energy", hydrosim::background_energy);
pp.query("background_energy", HydroSim::params.hydrosim.background_energy);

hydrosim::sdc_accuracy_order = 2;
pp.query("sdc_accuracy_order", hydrosim::sdc_accuracy_order);
pp.query("sdc_accuracy_order", HydroSim::params.hydrosim.sdc_accuracy_order);

hydrosim::sdc_quadrature = 0;
pp.query("sdc_quadrature", hydrosim::sdc_quadrature);
pp.query("sdc_quadrature", HydroSim::params.hydrosim.sdc_quadrature);

hydrosim::sdc_extra_passes = 0;
pp.query("sdc_extra_passes", hydrosim::sdc_extra_passes);
pp.query("sdc_extra_passes", HydroSim::params.hydrosim.sdc_extra_passes);

hydrosim::sdc_nonlin_method = 1;
pp.query("sdc_nonlin_method", hydrosim::sdc_nonlin_method);
pp.query("sdc_nonlin_method", HydroSim::params.hydrosim.sdc_nonlin_method);

hydrosim::use_curvilinear_src = true;
pp.query("use_curvilinear_src", hydrosim::use_curvilinear_src);
pp.query("use_curvilinear_src", HydroSim::params.hydrosim.use_curvilinear_src);

hydrosim::include_react_in_sdc_advect = true;
pp.query("include_react_in_sdc_advect", hydrosim::include_react_in_sdc_advect);
pp.query("include_react_in_sdc_advect", HydroSim::params.hydrosim.include_react_in_sdc_advect);

hydrosim::hydro_tile_mem_ratio = -1.0_rt;
pp.query("hydro_tile_mem_ratio", hydrosim::hydro_tile_mem_ratio);
pp.query("hydro_tile_mem_ratio", HydroSim::params.hydrosim.hydro_tile_mem_ratio);

hydrosim::constant_dt = -1.0_rt;
pp.query("constant_dt", hydrosim::constant_dt);
pp.query("constant_dt", HydroSim::params.hydrosim.constant_dt);

hydrosim::first_step_dt = -1.0_rt;
pp.query("first_step_dt", hydrosim::first_step_dt);
pp.query("first_step_dt", HydroSim::params.hydrosim.first_step_dt);

hydrosim::min_dt_time_ratio = 1.e-12_rt;
pp.query("min_dt_time_ratio", hydrosim::min_dt_time_ratio);
pp.query("min_dt_time_ratio", HydroSim::params.hydrosim.min_dt_time_ratio);

hydrosim::max_dt = 1.e200_rt;
pp.query("max_dt", hydrosim::max_dt);
pp.query("max_dt", HydroSim::params.hydrosim.max_dt);

hydrosim::cfl = 0.8_rt;
pp.query("cfl", hydrosim::cfl);
pp.query("cfl", HydroSim::params.hydrosim.cfl);

hydrosim::init_dt_shrink = 1.0_rt;
pp.query("init_dt_shrink", hydrosim::init_dt_shrink);
pp.query("init_dt_shrink", HydroSim::params.hydrosim.init_dt_shrink);

hydrosim::change_max = 1.1_rt;
pp.query("change_max", hydrosim::change_max);
pp.query("change_max", HydroSim::params.hydrosim.change_max);

hydrosim::dt_validity_before_advance = true;
pp.query("dt_validity_before_advance", hydrosim::dt_validity_before_advance);
pp.query("dt_validity_before_advance", HydroSim::params.hydrosim.dt_validity_before_advance);

hydrosim::dt_validity_after_advance = true;
pp.query("dt_validity_after_advance", hydrosim::dt_validity_after_advance);
pp.query("dt_validity_after_advance", HydroSim::params.hydrosim.dt_validity_after_advance);

hydrosim::plot_interval_exact = false;
pp.query("plot_interval_exact", hydrosim::plot_interval_exact);
pp.query("plot_interval_exact", HydroSim::params.hydrosim.plot_interval_exact);

hydrosim::small_interval_exact = false;
pp.query("small_interval_exact", hydrosim::small_interval_exact);
pp.query("small_interval_exact", HydroSim::params.hydrosim.small_interval_exact);

hydrosim::allow_step_retry = true;
pp.query("allow_step_retry", hydrosim::allow_step_retry);
pp.query("allow_step_retry", HydroSim::params.hydrosim.allow_step_retry);

hydrosim::subcycle_backoff_factor = 0.5_rt;
pp.query("subcycle_backoff_factor", hydrosim::subcycle_backoff_factor);
pp.query("subcycle_backoff_factor", HydroSim::params.hydrosim.subcycle_backoff_factor);

hydrosim::retry_density_cutoff = -1.e200_rt;
pp.query("retry_density_cutoff", hydrosim::retry_density_cutoff);
pp.query("retry_density_cutoff", HydroSim::params.hydrosim.retry_density_cutoff);

hydrosim::massfrac_check_tolerance = 1.e-2_rt;
pp.query("massfrac_check_tolerance", hydrosim::massfrac_check_tolerance);
pp.query("massfrac_check_tolerance", HydroSim::params.hydrosim.massfrac_check_tolerance);

hydrosim::massfrac_check_min_density = -1.e200_rt;
pp.query("massfrac_check_min_density", hydrosim::massfrac_check_min_density);
pp.query("massfrac_check_min_density", HydroSim::params.hydrosim.massfrac_check_min_density);

hydrosim::regrid_each_step = false;
pp.query("regrid_each_step", hydrosim::regrid_each_step);
pp.query("regrid_each_step", HydroSim::params.hydrosim.regrid_each_step);

hydrosim::subcycle_cap = 10;
pp.query("subcycle_cap", hydrosim::subcycle_cap);
pp.query("subcycle_cap", HydroSim::params.hydrosim.subcycle_cap);

hydrosim::deferred_iters = 2;
pp.query("deferred_iters", hydrosim::deferred_iters);
pp.query("deferred_iters", HydroSim::params.hydrosim.deferred_iters);

hydrosim::stop_field_name = "";
pp.query("stop_field_name", hydrosim::stop_field_name);
pp.query("stop_field_name", HydroSim::params.hydrosim.stop_field_name);

hydrosim::stop_field_threshold = 1.e200_rt;
pp.query("stop_field_threshold", hydrosim::stop_field_threshold);
pp.query("stop_field_threshold", HydroSim::params.hydrosim.stop_field_threshold);

hydrosim::burn_dt_energy = 1.e200_rt;
pp.query("burn_dt_energy", hydrosim::burn_dt_energy);
pp.query("burn_dt_energy", HydroSim::params.hydrosim.burn_dt_energy);

hydrosim::burn_dt_massfrac = 1.e200_rt;
pp.query("burn_dt_massfrac", hydrosim::burn_dt_massfrac);
pp.query("burn_dt_massfrac", HydroSim::params.hydrosim.burn_dt_massfrac);

hydrosim::nuc_dt_abund_floor = 1.e-3_rt;
pp.query("nuc_dt_abund_floor", hydrosim::nuc_dt_abund_floor);
pp.query("nuc_dt_abund_floor", HydroSim::params.hydrosim.nuc_dt_abund_floor);

hydrosim::do_react = true;
pp.query("do_react", hydrosim::do_react);
pp.query("do_react", HydroSim::params.hydrosim.do_react);

hydrosim::burn_temp_min = 0.0_rt;
pp.query("burn_temp_min", hydrosim::burn_temp_min);
pp.query("burn_temp_min", HydroSim::params.hydrosim.burn_temp_min);

hydrosim::burn_temp_max = 1.e200_rt;
pp.query("burn_temp_max", hydrosim::burn_temp_max);
pp.query("burn_temp_max", HydroSim::params.hydrosim.burn_temp_max);

hydrosim::burn_rho_min = 0.0_rt;
pp.query("burn_rho_min", hydrosim::burn_rho_min);
pp.query("burn_rho_min", HydroSim::params.hydrosim.burn_rho_min);

hydrosim::burn_rho_max = 1.e200_rt;
pp.query("burn_rho_max", hydrosim::burn_rho_max);
pp.query("burn_rho_max", HydroSim::params.hydrosim.burn_rho_max);

hydrosim::skip_burning_in_shocks = false;
pp.query("skip_burning_in_shocks", hydrosim::skip_burning_in_shocks);
pp.query("skip_burning_in_shocks", HydroSim::params.hydrosim.skip_burning_in_shocks);

hydrosim::shock_sensor_threshold = 0.6666666666666666666666_rt;
pp.query("shock_sensor_threshold", hydrosim::shock_sensor_threshold);
pp.query("shock_sensor_threshold", HydroSim::params.hydrosim.shock_sensor_threshold);

hydrosim::shock_sensor_subtract_src = true;
pp.query("shock_sensor_subtract_src", hydrosim::shock_sensor_subtract_src);
pp.query("shock_sensor_subtract_src", HydroSim::params.hydrosim.shock_sensor_subtract_src);

hydrosim::eos_temp_guess = 1.e8_rt;
pp.query("eos_temp_guess", hydrosim::eos_temp_guess);
pp.query("eos_temp_guess", HydroSim::params.hydrosim.eos_temp_guess);

hydrosim::drive_initial_convection = false;
pp.query("drive_initial_convection", hydrosim::drive_initial_convection);
pp.query("drive_initial_convection", HydroSim::params.hydrosim.drive_initial_convection);

hydrosim::convect_drive_end_time = 1.e200_rt;
pp.query("convect_drive_end_time", hydrosim::convect_drive_end_time);
pp.query("convect_drive_end_time", HydroSim::params.hydrosim.convect_drive_end_time);

hydrosim::convect_drive_reset_period = 1.e200_rt;
pp.query("convect_drive_reset_period", hydrosim::convect_drive_reset_period);
pp.query("convect_drive_reset_period", HydroSim::params.hydrosim.convect_drive_reset_period);

hydrosim::do_gravity_flag = true;
pp.query("do_gravity_flag", hydrosim::do_gravity_flag);
pp.query("do_gravity_flag", HydroSim::params.hydrosim.do_gravity_flag);

hydrosim::track_center = false;
pp.query("track_center", hydrosim::track_center);
pp.query("track_center", HydroSim::params.hydrosim.track_center);

hydrosim::gravity_source_mode = 4;
pp.query("gravity_source_mode", hydrosim::gravity_source_mode);
pp.query("gravity_source_mode", HydroSim::params.hydrosim.gravity_source_mode);

hydrosim::do_frame_spin = true;
pp.query("do_frame_spin", hydrosim::do_frame_spin);
pp.query("do_frame_spin", HydroSim::params.hydrosim.do_frame_spin);

hydrosim::boundary_callback_threadsafe = true;
pp.query("boundary_callback_threadsafe", hydrosim::boundary_callback_threadsafe);
pp.query("boundary_callback_threadsafe", HydroSim::params.hydrosim.boundary_callback_threadsafe);

hydrosim::restart_grow_factor = 1;
pp.query("restart_grow_factor", hydrosim::restart_grow_factor);
pp.query("restart_grow_factor", HydroSim::params.hydrosim.restart_grow_factor);

hydrosim::body_at_center = true;
pp.query("body_at_center", hydrosim::body_at_center);
pp.query("body_at_center", HydroSim::params.hydrosim.body_at_center);

hydrosim::prev_theta_cells = -1;
pp.query("prev_theta_cells", hydrosim::prev_theta_cells);
pp.query("prev_theta_cells", HydroSim::params.hydrosim.prev_theta_cells);

hydrosim::do_equilibrium_start = false;
pp.query("do_equilibrium_start", hydrosim::do_equilibrium_start);
pp.query("do_equilibrium_start", HydroSim::params.hydrosim.do_equilibrium_start);

hydrosim::scf_peak_density = -1.e6_rt;
pp.query("scf_peak_density", hydrosim::scf_peak_density);
pp.query("scf_peak_density", HydroSim::params.hydrosim.scf_peak_density);

hydrosim::scf_equator_radius = -1.e9_rt;
pp.query("scf_equator_radius", hydrosim::scf_equator_radius);
pp.query("scf_equator_radius", HydroSim::params.hydrosim.scf_equator_radius);

hydrosim::scf_pole_radius = -1.e9_rt;
pp.query("scf_pole_radius", hydrosim::scf_pole_radius);
pp.query("scf_pole_radius", HydroSim::params.hydrosim.scf_pole_radius);

hydrosim::scf_residual_tol = 1.e-3_rt;
pp.query("scf_residual_tol", hydrosim::scf_residual_tol);
pp.query("scf_residual_tol", HydroSim::params.hydrosim.scf_residual_tol);

hydrosim::scf_iter_cap = 30;
pp.query("scf_iter_cap", hydrosim::scf_iter_cap);
pp.query("scf_iter_cap", HydroSim::params.hydrosim.scf_iter_cap);

hydrosim::enable_special_refine = false;
pp.query("enable_special_refine", hydrosim::enable_special_refine);
pp.query("enable_special_refine", HydroSim::params.hydrosim.enable_special_refine);

hydrosim::tagging_radius_cap = 10.0e0_rt;
pp.query("tagging_radius_cap", hydrosim::tagging_radius_cap);
pp.query("tagging_radius_cap", HydroSim::params.hydrosim.tagging_radius_cap);

hydrosim::verbose = 0;
pp.query("v", hydrosim::verbose);
pp.query("v", HydroSim::params.hydrosim.verbose);

hydrosim::dump_old = false;
pp.query("dump_old", hydrosim::dump_old);
pp.query("dump_old", HydroSim::params.hydrosim.dump_old);

hydrosim::plane_parallel_domain = false;
pp.query("plane_parallel_domain", hydrosim::plane_parallel_domain);
pp.query("plane_parallel_domain", HydroSim::params.hydrosim.plane_parallel_domain);

#ifdef AMREX_DEBUG
hydrosim::log_source_changes = true;
#else
hydrosim::log_source_changes = false;
#endif
pp.query("log_source_changes", hydrosim::log_source_changes);
pp.query("log_source_changes", HydroSim::params.hydrosim.log_source_changes);

hydrosim::sum_step_period = -1;
pp.query("sum_step_period", hydrosim::sum_step_period);
pp.query("sum_step_period", HydroSim::params.hydrosim.sum_step_period);

hydrosim::sum_time_period = -1.0e0_rt;
pp.query("sum_time_period", hydrosim::sum_time_period);
pp.query("sum_time_period", HydroSim::params.hydrosim.sum_time_period);

hydrosim::job_label = "HydroSim";
pp.query("job_label", hydrosim::job_label);
pp.query("job_label", HydroSim::params.hydrosim.job_label);

hydrosim::write_on_finish = true;
pp.query("write_on_finish", hydrosim::write_on_finish);
pp.query("write_on_finish", HydroSim::params.hydrosim.write_on_finish);

hydrosim::restart_time_override = -1.e200_rt;
pp.query("restart_time_override", hydrosim::restart_time_override);
pp.query("restart_time_override", HydroSim::params.hydrosim.restart_time_override);

hydrosim::restart_step_override = -1;
pp.query("restart_step_override", hydrosim::restart_step_override);
pp.query("restart_step_override", HydroSim::params.hydrosim.restart_step_override);

hydrosim::keep_omegadot = false;
pp.query("keep_omegadot", hydrosim::keep_omegadot);
pp.query("keep_omegadot", HydroSim::params.hydrosim.keep_omegadot);

hydrosim::keep_burn_weights = false;
pp.query("keep_burn_weights", hydrosim::keep_burn_weights);
pp.query("keep_burn_weights", HydroSim::params.hydrosim.keep_burn_weights);

hydrosim::stop_on_bad_params = false;
pp.query("stop_on_bad_params", hydrosim::stop_on_bad_params);
pp.query("stop_on_bad_params", HydroSim::params.hydrosim.stop_on_bad_params);

hydrosim::do_radiative_transfer = true;
pp.query("do_radiative_transfer", hydrosim::do_radiative_transfer);
pp.query("do_radiative_transfer", HydroSim::params.hydrosim.do_radiative_transfer);


