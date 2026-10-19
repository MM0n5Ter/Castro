#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>
#include <prob_util.H>
#include <conductivity.H>

AMREX_INLINE
void user_initialize ()
{

    const Geometry& default_geom = DefaultGeometry();

    const int coord_type = default_geom.Coord();
    const Real* problo = default_geom.ProbLo();
    const Real* probhi = default_geom.ProbHi();


    if ((problem::center[0] == 0.0_rt) && (coord_type == 0)) {
        problem::center[0] = 0.5_rt * (problo[0] + probhi[0]);
    }
#if AMREX_SPACEDIM >= 2
    if ((problem::center[1] == 0.0_rt) && (coord_type <= 1)) {
        problem::center[1] = 0.5_rt * (problo[1] + probhi[1]);
    }
#endif
#if AMREX_SPACEDIM == 3
    if (problem::center[2] == 0.0_rt) {
        problem::center[2] = 0.5_rt * (problo[2] + probhi[2]);
    }
#endif

    Real X[NumSpec] = {0.0_rt};
    X[0] = 1.0_rt;

    eos_t eos_state;
    eos_state.T = problem::T1;
    eos_state.rho = 1.0_rt;
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = X[n];
    }

    eos(eos_input_rt, eos_state);

#ifdef DIFFUSION
    conductivity(eos_state);

    problem::rho0 = eos_state.conductivity / (problem::diff_coeff * eos_state.cv);
#else
    problem::rho0 = 1.0_rt;
#endif

}

#endif
