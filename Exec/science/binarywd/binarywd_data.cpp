#include <binarywd_data.H>

using namespace amrex;

AMREX_GPU_MANAGED Real binarywd::mdot_P = 0.0;
AMREX_GPU_MANAGED Real binarywd::mdot_S = 0.0;

AMREX_GPU_MANAGED Real binarywd::rad_P[7] = { 0.0 };
AMREX_GPU_MANAGED Real binarywd::rad_S[7] = { 0.0 };

AMREX_GPU_MANAGED Real binarywd::vol_P[7] = { 0.0 };
AMREX_GPU_MANAGED Real binarywd::vol_S[7] = { 0.0 };

AMREX_GPU_MANAGED Real binarywd::rho_avg_P = 0.0;
AMREX_GPU_MANAGED Real binarywd::rho_avg_S = 0.0;
