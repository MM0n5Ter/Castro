
// This file is automatically created by parse_hydrosim_params.py at build time.
// To update or add runtime parameters, please edit _cpp_parameters and rebuild.

radiation_solve::level_solver_kind = 1;
pp.query("level_solver_kind", radiation_solve::level_solver_kind);
pp.query("level_solver_kind", HydroSim::params.radiation_solve.level_solver_kind);

radiation_solve::use_nonsym_solver_terms = false;
pp.query("use_nonsym_solver_terms", radiation_solve::use_nonsym_solver_terms);
pp.query("use_nonsym_solver_terms", HydroSim::params.radiation_solve.use_nonsym_solver_terms);

radiation_solve::reltol = 1.e-10_rt;
pp.query("reltol", radiation_solve::reltol);
pp.query("reltol", HydroSim::params.radiation_solve.reltol);

radiation_solve::abstol = 1.e-10_rt;
pp.query("abstol", radiation_solve::abstol);
pp.query("abstol", HydroSim::params.radiation_solve.abstol);

radiation_solve::maxiter = 40;
pp.query("maxiter", radiation_solve::maxiter);
pp.query("maxiter", HydroSim::params.radiation_solve.maxiter);

radiation_solve::alpha = 1.0_rt;
pp.query("alpha", radiation_solve::alpha);
pp.query("alpha", HydroSim::params.radiation_solve.alpha);

radiation_solve::beta = 1.0_rt;
pp.query("beta", radiation_solve::beta);
pp.query("beta", HydroSim::params.radiation_solve.beta);

radiation_solve::verbose = 0;
pp.query("v", radiation_solve::verbose);
pp.query("v", HydroSim::params.radiation_solve.verbose);


