
// This file is automatically created by parse_castro_params.py at build time.
// To update or add runtime parameters, please edit _cpp_parameters and rebuild.

radsolve::level_solver_flag = 1;
pp.query("level_solver_flag", radsolve::level_solver_flag);
pp.query("level_solver_flag", Castro::params.radsolve.level_solver_flag);

radsolve::use_hypre_nonsymmetric_terms = false;
pp.query("use_hypre_nonsymmetric_terms", radsolve::use_hypre_nonsymmetric_terms);
pp.query("use_hypre_nonsymmetric_terms", Castro::params.radsolve.use_hypre_nonsymmetric_terms);

radsolve::reltol = 1.e-10_rt;
pp.query("reltol", radsolve::reltol);
pp.query("reltol", Castro::params.radsolve.reltol);

radsolve::abstol = 1.e-10_rt;
pp.query("abstol", radsolve::abstol);
pp.query("abstol", Castro::params.radsolve.abstol);

radsolve::maxiter = 40;
pp.query("maxiter", radsolve::maxiter);
pp.query("maxiter", Castro::params.radsolve.maxiter);

radsolve::alpha = 1.0_rt;
pp.query("alpha", radsolve::alpha);
pp.query("alpha", Castro::params.radsolve.alpha);

radsolve::beta = 1.0_rt;
pp.query("beta", radsolve::beta);
pp.query("beta", Castro::params.radsolve.beta);

radsolve::verbose = 0;
pp.query("v", radsolve::verbose);
pp.query("v", Castro::params.radsolve.verbose);


