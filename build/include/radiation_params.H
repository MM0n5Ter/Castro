
// This file is automatically created by parse_castro_params.py at build time.
// To update or add runtime parameters, please edit _cpp_parameters and rebuild.

#ifndef RADIATION_PARAMS_H
#define RADIATION_PARAMS_H

namespace radiation {
extern AMREX_GPU_MANAGED amrex::Real prop_temp_floor;
extern AMREX_GPU_MANAGED amrex::Real flatten_pp_threshold;
extern AMREX_GPU_MANAGED bool comoving;
extern AMREX_GPU_MANAGED int closure;
extern AMREX_GPU_MANAGED int limiter;
extern AMREX_GPU_MANAGED int fspace_advection_type;
extern AMREX_GPU_MANAGED bool plot_lambda;
extern AMREX_GPU_MANAGED bool plot_kappa_p;
extern AMREX_GPU_MANAGED bool plot_kappa_r;
extern AMREX_GPU_MANAGED bool plot_lab_Er;
extern AMREX_GPU_MANAGED bool plot_lab_flux;
extern AMREX_GPU_MANAGED bool plot_com_flux;
}

#endif
