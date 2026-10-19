#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>
#include <model_parser.H>

AMREX_INLINE
void user_initialize ()
{

    const Geometry& default_geom = DefaultGeometry();

    const Real* problo = default_geom.ProbLo();
    const Real* probhi = default_geom.ProbHi();



    read_model_file(problem::model_name);


    for (int k = 0; k < model::npts; k++) {

        burn_t burn_state;

        burn_state.rho = model::profile(0).state(k, model::idens);
        burn_state.T = model::profile(0).state(k, model::itemp);
        burn_state.e = 0.0_rt;

        for (int n = 0; n < NumSpec; n++) {
            burn_state.xn[n] = model::profile(0).state(k, model::ispec+n);
        }

        burn_state.success = true;

        burner(burn_state, problem::burn_timestep);

        if (! burn_state.success) {
            std::cout << "burning failed for zone " << k << std::endl;
        } else {
            std::cout << "zone, enuc = " << k << " " << burn_state.e / problem::burn_timestep << std::endl;
        }
    }

    amrex::Error("done burning");
}
#endif
