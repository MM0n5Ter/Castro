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

    eos_t eos_state;


    if (small_dens > 0.0_rt && small_temp > 0.0_rt) {
        eos_state.rho = small_dens;
        eos_state.T   = small_temp;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n]  = 1.0_rt / static_cast<Real>(NumSpec);
        }

        eos(eos_input_rt, eos_state);

        pressure_cutoff = eos_state.p;
        eint_floor = eos_state.e;
    }
}
#endif
