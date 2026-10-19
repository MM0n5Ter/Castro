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

    Real xn[NumSpec] = {0.0};
    xn[0] = 1.0_rt;

    eos_t eos_state;

    if (problem::use_Tinit) {

        eos_state.rho = problem::rho_l;
        eos_state.T = problem::T_l;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n] = xn[n];
        }

        eos(eos_input_rt, eos_state);

        problem::rhoe_l = problem::rho_l * eos_state.e;
        problem::p_l = eos_state.p;

        eos_state.rho = problem::rho_r;
        eos_state.T = problem::T_r;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n] = xn[n];
        }

        eos(eos_input_rt, eos_state);

        problem::rhoe_r = problem::rho_r * eos_state.e;
        problem::p_r = eos_state.p;

    } else {

        eos_state.rho = problem::rho_l;
        eos_state.p = problem::p_l;
        eos_state.T = 100000.0_rt;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n] = xn[n];
        }

        eos(eos_input_rp, eos_state);

        problem::rhoe_l = problem::rho_l * eos_state.e;
        problem::T_l = eos_state.T;

        eos_state.rho = problem::rho_r;
        eos_state.p = problem::p_r;
        eos_state.T = 100000.0_rt;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n] = xn[n];
        }

        eos(eos_input_rp, eos_state);

        problem::rhoe_r = problem::rho_r * eos_state.e;
        problem::T_r = eos_state.T;

    }

}

#endif
