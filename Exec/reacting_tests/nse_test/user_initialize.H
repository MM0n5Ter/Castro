#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>
#include <network.H>

AMREX_INLINE
void user_initialize ()
{
    const Geometry& default_geom = DefaultGeometry();

    const Real* problo = default_geom.ProbLo();
    const Real* probhi = default_geom.ProbHi();

    for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
        problem::center[idir] = 0.5_rt * (problo[idir] + probhi[idir]);
    }

    for (int idir = AMREX_SPACEDIM; idir < 3; idir++) {
        problem::center[idir] = 0.0_rt;
    }


#if !defined(NSE_TABLE) && !defined(NSE_NET)
    amrex::Error("Error: this problem requires USE_NSE_NET=TRUE or USE_NSE_TABLE=TRUE");
#endif

}

#endif
