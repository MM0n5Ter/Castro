
// This file is automatically created by parse_hydrosim_params.py at build time.
// To update or add runtime parameters, please edit _cpp_parameters and rebuild.

#ifndef RADSOLVE_PARAMS_H
#define RADSOLVE_PARAMS_H

namespace radsolve {
extern AMREX_GPU_MANAGED int level_solver_flag;
extern AMREX_GPU_MANAGED bool use_hypre_nonsymmetric_terms;
extern AMREX_GPU_MANAGED amrex::Real reltol;
extern AMREX_GPU_MANAGED amrex::Real abstol;
extern AMREX_GPU_MANAGED int maxiter;
extern AMREX_GPU_MANAGED amrex::Real alpha;
extern AMREX_GPU_MANAGED amrex::Real beta;
extern AMREX_GPU_MANAGED int verbose;
}

#endif
