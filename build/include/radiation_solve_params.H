
// This file is automatically created by parse_hydrosim_params.py at build time.
// To update or add runtime parameters, please edit _cpp_parameters and rebuild.

#ifndef RADIATION_SOLVE_PARAMS_H
#define RADIATION_SOLVE_PARAMS_H

namespace radiation_solve {
extern AMREX_GPU_MANAGED int level_solver_kind;
extern AMREX_GPU_MANAGED bool use_nonsym_solver_terms;
extern AMREX_GPU_MANAGED amrex::Real reltol;
extern AMREX_GPU_MANAGED amrex::Real abstol;
extern AMREX_GPU_MANAGED int maxiter;
extern AMREX_GPU_MANAGED amrex::Real alpha;
extern AMREX_GPU_MANAGED amrex::Real beta;
extern AMREX_GPU_MANAGED int verbose;
}

#endif
