
#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_INLINE
void user_initialize ()
{
    const Geometry& default_geom = DefaultGeometry();

    const Real* problo = default_geom.ProbLo();
    const Real* probhi = default_geom.ProbHi();


    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        problem::center[n] = 0.5_rt * (problo[n] + probhi[n]);
    }

    problem::x_r = probhi[0] - problo[0];
    problem::q_r = 0.4_rt * M_PI * problem::x_r / problem::t_r;


    problem::p0 = problem::rho0 * std::pow(problem::q_r, 2.0) /
        (eos_rp::eos_gamma * std::pow(problem::M0, 2.0)) - 0.5_rt;

}

#endif
