#ifndef binarywd_data_H
#define binarywd_data_H

#include <AMReX.H>
#include <AMReX_REAL.H>

#include <prob_parameters.H>

namespace binarywd
{


    extern AMREX_GPU_MANAGED amrex::Real mdot_P;
    extern AMREX_GPU_MANAGED amrex::Real mdot_S;


    extern AMREX_GPU_MANAGED amrex::Real rad_P[7];
    extern AMREX_GPU_MANAGED amrex::Real rad_S[7];


    extern AMREX_GPU_MANAGED amrex::Real vol_P[7];
    extern AMREX_GPU_MANAGED amrex::Real vol_S[7];


    extern AMREX_GPU_MANAGED amrex::Real rho_avg_P;
    extern AMREX_GPU_MANAGED amrex::Real rho_avg_S;
}

#endif
