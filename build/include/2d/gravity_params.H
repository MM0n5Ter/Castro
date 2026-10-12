
// This file is automatically created by parse_castro_params.py at build time.
// To update or add runtime parameters, please edit _cpp_parameters and rebuild.

#ifndef GRAVITY_PARAMS_H
#define GRAVITY_PARAMS_H

namespace gravity {
extern std::string gravity_type;
extern AMREX_GPU_MANAGED amrex::Real const_grav;
extern AMREX_GPU_MANAGED bool direct_sum_bcs;
extern AMREX_GPU_MANAGED int drdxfac;
extern AMREX_GPU_MANAGED int lnum;
extern AMREX_GPU_MANAGED int verbose;
extern AMREX_GPU_MANAGED bool no_sync;
extern AMREX_GPU_MANAGED bool do_composite_phi_correction;
extern AMREX_GPU_MANAGED int max_solve_level;
extern AMREX_GPU_MANAGED bool get_g_from_phi;
extern AMREX_GPU_MANAGED int mlmg_max_fmg_iter;
extern AMREX_GPU_MANAGED bool mlmg_agglomeration;
extern AMREX_GPU_MANAGED bool mlmg_consolidation;
extern AMREX_GPU_MANAGED bool mlmg_nsolve;
}

#endif
