
// This file is automatically created by parse_hydrosim_params.py at build time.
// To update or add runtime parameters, please edit _cpp_parameters and rebuild.

gravity::gravity_model_name = "unset_gravity_type";
pp.query("gravity_model_name", gravity::gravity_model_name);
pp.query("gravity_model_name", HydroSim::params.gravity.gravity_model_name);

gravity::uniform_accel = 0.0_rt;
pp.query("uniform_accel", gravity::uniform_accel);
pp.query("uniform_accel", HydroSim::params.gravity.uniform_accel);

gravity::boundary_direct_sum = false;
pp.query("boundary_direct_sum", gravity::boundary_direct_sum);
pp.query("boundary_direct_sum", HydroSim::params.gravity.boundary_direct_sum);

gravity::radial_bin_ratio = 1;
pp.query("radial_bin_ratio", gravity::radial_bin_ratio);
pp.query("radial_bin_ratio", HydroSim::params.gravity.radial_bin_ratio);

gravity::lnum = 0;
pp.query("multipole_order_cap", gravity::lnum);
pp.query("multipole_order_cap", HydroSim::params.gravity.lnum);

gravity::verbose = 0;
pp.query("v", gravity::verbose);
pp.query("v", HydroSim::params.gravity.verbose);

gravity::no_sync = false;
pp.query("no_sync", gravity::no_sync);
pp.query("no_sync", HydroSim::params.gravity.no_sync);

gravity::composite_phi_fixup = true;
pp.query("composite_phi_fixup", gravity::composite_phi_fixup);
pp.query("composite_phi_fixup", HydroSim::params.gravity.composite_phi_fixup);

gravity::solve_level_cap = LEVEL_LIMIT-1;
pp.query("solve_level_cap", gravity::solve_level_cap);
pp.query("solve_level_cap", HydroSim::params.gravity.solve_level_cap);

gravity::accel_from_phi_grad = false;
pp.query("accel_from_phi_grad", gravity::accel_from_phi_grad);
pp.query("accel_from_phi_grad", HydroSim::params.gravity.accel_from_phi_grad);

gravity::mlmg_fmg_cycles = 0;
pp.query("mlmg_fmg_cycles", gravity::mlmg_fmg_cycles);
pp.query("mlmg_fmg_cycles", HydroSim::params.gravity.mlmg_fmg_cycles);

gravity::mlmg_grid_merge = true;
pp.query("mlmg_grid_merge", gravity::mlmg_grid_merge);
pp.query("mlmg_grid_merge", HydroSim::params.gravity.mlmg_grid_merge);

gravity::mlmg_proc_gather = true;
pp.query("mlmg_proc_gather", gravity::mlmg_proc_gather);
pp.query("mlmg_proc_gather", HydroSim::params.gravity.mlmg_proc_gather);

gravity::mlmg_coarse_nsolve = false;
pp.query("mlmg_coarse_nsolve", gravity::mlmg_coarse_nsolve);
pp.query("mlmg_coarse_nsolve", HydroSim::params.gravity.mlmg_coarse_nsolve);


