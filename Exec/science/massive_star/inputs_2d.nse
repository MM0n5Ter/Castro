# ------------------  INPUTS TO MAIN PROGRAM  -------------------

amr.plot_files_output = 1
amr.checkpoint_files_output = 1

max_step = 500000
stop_time = 360000

geometry.is_periodic = 0 0
geometry.coord_sys = 1         # r-z coordinates

geometry.prob_lo   =  0.    0.
geometry.prob_hi   =  8.192e9  1.6384e10

amr.n_cell         = 1024 2048

amr.max_level      = 2      # maximum level number allowed

hydrosim.lo_bc       =  3 2
hydrosim.hi_bc       =  2 2

# >>>>>>>>>>>>>  BC FLAGS <<<<<<<<<<<<<<<<
# 0 = Interior           3 = Symmetry
# 1 = Inflow             4 = SlipWall
# 2 = Outflow            5 = NoSlipWall
# >>>>>>>>>>>>>  BC FLAGS <<<<<<<<<<<<<<<<

hydrosim.do_hydro = 1
hydrosim.do_gravity_flag  = 1
hydrosim.do_react = 1
hydrosim.do_damping_zone = 1

hydrosim.ppm_recon_kind = 1
hydrosim.ppm_temp_mode = 0
hydrosim.use_pressure_slope = 1

hydrosim.use_shock_flatten = 1

hydrosim.riemann_method = 1

gravity.gravity_model_name = MonopoleGrav
gravity.radial_bin_ratio = 2
hydrosim.gravity_source_mode = 4

hydrosim.damping_upper_density = 1.e3
hydrosim.damping_lower_density = 1.e2
hydrosim.damping_timescale     = 1.e-3

hydrosim.cfl            = 0.4     # cfl number for hyperbolic system
hydrosim.init_dt_shrink    = 0.01     # scale back initial timestep by this factor
hydrosim.change_max     = 1.2    # factor by which dt is allowed to change each timestep
hydrosim.sum_step_period   = 1       # timesteps between computing and printing volume averages

#hydrosim.burn_dt_energy = 0.25
#hydrosim.burn_dt_massfrac = 0.25

amr.ref_ratio       = 2 2 2 2 # refinement ratio
amr.regrid_int      = 10000   # how often to regrid
amr.n_error_buf     = 2 2 2 2 # number of buffer cells in error est
amr.grid_eff        = 0.7     # what constitutes an efficient grid

amr.check_file      = massive_star_chk     # root name of checkpoint file
amr.check_int       = 50     # number of timesteps between checkpoints

amr.plot_file       = massive_star_plt     # root name of plot file
amr.plot_per = 5.0
amr.derive_plot_vars = ALL
hydrosim.keep_burn_weights = 0

amr.small_plot_file       = massive_star_smallplt     # root name of plot file
amr.small_plot_per = 0.5
amr.small_plot_vars = density Temp in_nse
amr.derive_small_plot_vars = abar Ye enuc MachNumber magvel magvort

fab.format = NATIVE_32

hydrosim.plot_interval_exact = 0


amr.max_grid_size   = 512       # maximum grid size allowed -- used to control parallelism
amr.blocking_factor = 32       # block factor in grid generation

amr.v               = 1       # control verbosity in Amr.cpp
hydrosim.v            = 1       # control verbosity in HydroSim.cpp


hydrosim.small_dens   = 1.0
hydrosim.small_temp   = 1.e6

hydrosim.time_scheme = 3
hydrosim.allow_step_retry = 1
hydrosim.subcycle_cap = 16

# problem initialization

problem.model_name =  "15m_500_sec.aprox19.hse.5.00km"

problem.perturb_model = 1
problem.velpert_amplitude = 5.e6

problem.interpolate_pres = 1

# convection

hydrosim.drive_initial_convection = 1
hydrosim.convect_drive_reset_period = 2
hydrosim.convect_drive_end_time = 100

# refinement

amr.refine_criteria = denerr

amr.refine.denerr.max_level = 3
amr.refine.denerr.value_greater = 1.e4
amr.refine.denerr.field_name = density


# Microphysics

integrator.rtol_spec = 1.e-5
integrator.atol_spec = 1.e-5
integrator.rtol_enuc = 1.e-5
integrator.atol_enuc = 1.e-5

integrator.jacobian = 1

network.rho_nse = 1.e7
network.T_nse = 3.e9
network.Si_nse = 0.02
network.C_nse = 1.0
network.O_nse = 1.0

integrator.ode_max_steps = 500000

network.small_x = 1.e-10

network.nse_relax_factor = 0.9

# use cubic interpolation from the NSE table
network.nse_table_interp_linear = 0

# disable jacobian caching in VODE
integrator.use_jacobian_caching = 0

# do we include weak rate neutrino losses in the energy?
integrator.nse_include_enu_weak = 0
