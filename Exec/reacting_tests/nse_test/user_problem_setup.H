#ifndef user_problem_setup_H
#define user_problem_setup_H

#include <prob_parameters.H>
#include <eos.H>
#include <network.H>

AMREX_INLINE
void user_problem_setup ()
{
    const Geometry& dgeom = DefaultGeometry();

    const Real* problo = dgeom.ProbLo();
    const Real* probhi = dgeom.ProbHi();

    for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
        problem::center[idir] = 0.5_rt * (problo[idir] + probhi[idir]);
    }

    for (int idir = AMREX_SPACEDIM; idir < 3; idir++) {
        problem::center[idir] = 0.0_rt;
    }

    // we only work in NSE mode, so the mass fractions are not used by
    // the EOS

#if !defined(NSE_TABLE) && !defined(NSE_NET)
    amrex::Error("Error: this problem requires USE_NSE_NET=TRUE or USE_NSE_TABLE=TRUE");
#endif

}

#endif
