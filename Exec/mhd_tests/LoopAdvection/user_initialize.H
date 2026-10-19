#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_INLINE
void user_initialize ()
{


    eos_t eos_state;


      eos_state.rho = problem::rho_0;
      eos_state.p = problem::p_0;
      eos_state.T = 100000.0_rt;
      for (auto & X : eos_state.xn) {
          X = 0.0_rt;
      }
      eos_state.xn[0] = 1.0_rt;

      eos(eos_input_rp, eos_state);

      problem::rhoe_0 = problem::rho_0 * eos_state.e;
      problem::T_0 = eos_state.T;
}
#endif
