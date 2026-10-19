#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_INLINE
void user_initialize ()
{

    const Real M_SQRT_PI = std::sqrt(M_PI);
    problem::B_0 = 1.0 / (2.0_rt * M_SQRT_PI);

    Real xn[NumSpec] = {0.0_rt};
    xn[0] = 1.0_rt;

    eos_t eos_state;
    eos_state.rho = problem::rho_0;
    eos_state.p = problem::p_0;
    eos_state.T = 100000.0_rt;
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = xn[n];
    }

    eos(eos_input_rp, eos_state);

    problem::rhoe_0 = problem::rho_0 * eos_state.e;
    problem::T_0 = eos_state.T;

}

#endif
