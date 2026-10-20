
// This file is automatically created by parse_hydrosim_params.py at build time.
// To update or add runtime parameters, please edit _cpp_parameters and rebuild.

#ifndef GRAVITY_PARAMS_H
#define GRAVITY_PARAMS_H

namespace gravity {
extern std::string gravity_model_name;
extern AMREX_GPU_MANAGED amrex::Real uniform_accel;
extern AMREX_GPU_MANAGED bool boundary_direct_sum;
extern AMREX_GPU_MANAGED int radial_bin_ratio;
extern AMREX_GPU_MANAGED int lnum;
extern AMREX_GPU_MANAGED int verbose;
extern AMREX_GPU_MANAGED bool no_sync;
extern AMREX_GPU_MANAGED bool composite_phi_fixup;
extern AMREX_GPU_MANAGED int solve_level_cap;
extern AMREX_GPU_MANAGED bool accel_from_phi_grad;
extern AMREX_GPU_MANAGED int mlmg_fmg_cycles;
extern AMREX_GPU_MANAGED bool mlmg_grid_merge;
extern AMREX_GPU_MANAGED bool mlmg_proc_gather;
extern AMREX_GPU_MANAGED bool mlmg_coarse_nsolve;
}

#endif
