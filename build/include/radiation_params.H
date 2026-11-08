
// This file is automatically created by parse_hydrosim_params.py at build time.
// To update or add runtime parameters, please edit _cpp_parameters and rebuild.

#ifndef RADIATION_PARAMS_H
#define RADIATION_PARAMS_H

namespace radiation {
extern AMREX_GPU_MANAGED amrex::Real prop_temp_min;
extern AMREX_GPU_MANAGED amrex::Real radiation_flatten_cutoff;
extern AMREX_GPU_MANAGED bool moving_frame;
extern AMREX_GPU_MANAGED int closure;
extern AMREX_GPU_MANAGED int limiter;
extern AMREX_GPU_MANAGED int frequency_advection_mode;
extern AMREX_GPU_MANAGED bool plot_flux_limiter;
extern AMREX_GPU_MANAGED bool plot_planck_opacity;
extern AMREX_GPU_MANAGED bool plot_rosseland_opacity;
extern AMREX_GPU_MANAGED bool plot_lab_rad_energy;
extern AMREX_GPU_MANAGED bool plot_lab_rad_flux;
extern AMREX_GPU_MANAGED bool plot_moving_flux;
}

#endif
