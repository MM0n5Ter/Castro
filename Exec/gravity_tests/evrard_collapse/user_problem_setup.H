#ifndef user_problem_setup_H
#define user_problem_setup_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_INLINE
void user_problem_setup ()
{
    const Geometry& dgeom = DefaultGeometry();

    const int coord_type = dgeom.Coord();
    const Real* problo = dgeom.ProbLo();
    const Real* probhi = dgeom.ProbHi();

    eos_t eos_state;

    // Given the inputs of small_dens and small_temp, figure out small_pres.

    if (small_dens > 0.0_rt && small_temp > 0.0_rt) {
        eos_state.rho = small_dens;
        eos_state.T   = small_temp;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n]  = 1.0_rt / static_cast<Real>(NumSpec);
        }

        eos(eos_input_rt, eos_state);

        small_pres = eos_state.p;
        small_ener = eos_state.e;
    }
}
#endif
