# `subchandra`

This is the double detonation Type Ia supernova setup.  It takes a 1-d
initial model representing a sub-Chandra WD and a surface He layer and
initializes a small perturbation at the pole in the He layer.


## Initial model generation

This problem setup generates the initial model internally (see
`initial_model.H`).

The main parameters are:

* `problem.M_WD` : the mass of the underlying WD (solar masses)

* `problem.M_He` : the mass of the He layer (solar masses)

* `problem.delta` : the transition width between the core and He layer
  (cm)

* `problem.temp_core` : the WD isothermal temperature

* `problem.temp_base` : the temperature at the base of the He layer

* `problem.model_r_max` : the maximum radius used for the model
  generation.  If you are doing a low mass WD, then this may need to
  be increased.

Generating the model requires a 2-dimensional root find (on the
central density and density at the base of the He layer).  Convergence
can be tricky if you ask for too much accuracy on the He layer mass.
The following tolerances can be tweaked to help convergence:

* `problem.tol_hse` : the relative tolerance with which we satisfy HSE

* `problem.tol_WD` : the relative tolerance on the WD mass

* `problem.tol_He` : the relative tolerance on the He layer mass

Also if `problem.delta` is made too thin, convergence can be tough if
not enough resolution is used.  The model will be generated with a
resolution that is dx/2, where dx is the fine grid spacing.  But a
minimum resolution is enforced, set via:

* `problem.model_min_res` : minimum resolution of the initial model
