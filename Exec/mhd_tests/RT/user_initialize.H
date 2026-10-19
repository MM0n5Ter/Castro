#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_INLINE
void user_initialize ()
{

    const Geometry& default_geom = DefaultGeometry();

    const int coord_type = default_geom.Coord();
    const Real* problo = default_geom.ProbLo();
    const Real* probhi = default_geom.ProbHi();

    problem::split[0] = problem::frac * (problo[0] + probhi[0]);
#if AMREX_SPACEDIM >= 2
    problem::split[1] = problem::frac * (problo[1] + probhi[1]);
#else
    problem::split[1] = 0.0;
#endif
#if AMREX_SPACEDIM == 3
    problem::split[2] = problem::frac * (problo[2] + probhi[2]);
#else
    problem::split[2] = 0.0;
#endif

    problem::L_x = probhi[0] - problo[0];

}
#endif
