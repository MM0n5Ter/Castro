#ifndef user_problem_setup_H
#define user_problem_setup_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_INLINE
void user_problem_setup ()
{
    const Geometry& dgeom = DefaultGeometry();

    const Real* problo = dgeom.ProbLo();
    const Real* probhi = dgeom.ProbHi();

    // set local variable defaults

    problem::split[0] = problem::frac * (problo[0] + probhi[0]);
#if AMREX_SPACEDIM >= 2
    problem::split[1] = problem::frac * (problo[1] + probhi[1]);
#else
    problem::split[1] = 0.0_rt;
#endif
#if AMREX_SPACEDIM == 3
    problem::split[2] = problem::frac * (problo[2] + probhi[2]);
#else
    problem::split[2] = 0.0_rt;
#endif

    problem::L_x = probhi[0] - problo[0];

}

#endif
