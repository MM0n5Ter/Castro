# ------------------  INPUTS TO MAIN PROGRAM  -------------------
max_step = 500000

stop_time    = 1.e-7

# PROBLEM SIZE & GEOMETRY
geometry.is_periodic = 0
geometry.coord_sys   = 0        # 0 => cart, 1 => RZ  2=>spherical
geometry.prob_lo     = 0
geometry.prob_hi     = 0.16
amr.n_cell           = 512

# >>>>>>>>>>>>>  BC FLAGS <<<<<<<<<<<<<<<<
# 0 = Interior           3 = Symmetry
# 1 = Inflow             4 = SlipWall
# 2 = Outflow            5 = NoSlipWall
# >>>>>>>>>>>>>  BC FLAGS <<<<<<<<<<<<<<<<
hydrosim.lo_bc       =  3
hydrosim.hi_bc       =  2

# WHICH PHYSICS
hydrosim.do_hydro = 1
hydrosim.do_react = 1

hydrosim.small_temp = 1.e6
hydrosim.small_dens = 1.e-5

hydrosim.ppm_type = 1

hydrosim.diffuse_temp = 1
hydrosim.diffuse_cutoff_density = 1.e-2

hydrosim.time_integration_method = 3


# TIME STEP CONTROL
hydrosim.cfl            = 0.6     # cfl number for hyperbolic system
hydrosim.init_shrink    = 0.1     # scale back initial timestep
hydrosim.change_max     = 1.1     # max time step growth


# DIAGNOSTICS & VERBOSITY
hydrosim.sum_interval   = 1       # timesteps between computing mass
hydrosim.v              = 1       # verbosity in HydroSim.cpp
amr.v                 = 1       # verbosity in Amr.cpp

# REFINEMENT / REGRIDDING
amr.max_level       = 2       # maximum level number allowed
amr.ref_ratio       = 4 4  2 2 2 # refinement ratio
amr.regrid_int      = 2 2 2 2 # how often to regrid
amr.blocking_factor = 64      # block factor in grid generation
amr.max_grid_size   = 64
amr.n_error_buf     = 2 2 2 2 # number of buffer cells in error est

# CHECKPOINT FILES
amr.checkpoint_files_output = 1
amr.check_file      = chk        # root name of checkpoint file
amr.check_int       = 2500       # number of timesteps between checkpoints

# PLOTFILES
amr.plot_file       = plt        # root name of plotfile
amr.plot_per        = 1.e-9
amr.derive_plot_vars = ALL
hydrosim.store_omegadot = 1

# problem initialization

problem.rho_fuel = 5.e8
problem.T_fuel = 5.e7
problem.fuel1_name = "carbon-12"

problem.T_ash = 6.e9
problem.ash1_name = "nickel-56"

problem.pert_frac = 0.15
problem.pert_delta = 0.025

# refinement

amr.refinement_indicators = tempgrad

amr.refine.tempgrad.max_level = 5
amr.refine.tempgrad.relative_gradient = 0.025
amr.refine.tempgrad.field_name = Temp

# Microphysics

integrator.rtol_spec = 1.e-6
integrator.atol_spec = 1.e-8
integrator.jacobian = 1
