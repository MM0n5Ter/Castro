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


    if (coord_type == 0) {
        problem::center[0] = 0.5_rt * (problo[0] + probhi[0]);
    } else if (coord_type >= 1) {
        problem::center[0] = 0.0_rt;
    }

#if AMREX_SPACEDIM == 2
    problem::center[1] = 0.5_rt * (problo[1] + probhi[1]);
#endif
#if AMREX_SPACEDIM == 3
    problem::center[2] = 0.5_rt * (problo[2] + probhi[2]);
#endif
}
#endif
