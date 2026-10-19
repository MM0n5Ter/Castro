#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>
#include <network.H>

AMREX_INLINE
void user_initialize ()
{


    problem::ihe4 = network_spec_index("helium-4");
    problem::ic12 = network_spec_index("carbon-12");
    problem::in14 = network_spec_index("nitrogen-14");
    problem::io16 = network_spec_index("oxygen-16");
    problem::ine20 = network_spec_index("neon-20");

    if (problem::ihe4 < 0 || problem::ic12 < 0 || problem::io16 < 0 || problem::ine20 < 0) {
        amrex::Error("ERROR: species indices not found");
    }


    if (problem::cfrac > 1.0_rt || problem::cfrac < 0.0_rt) {
        amrex::Error("ERROR: cfrac must fall between 0 and 1");
    }


    if (problem::nfrac > 1.0_rt || problem::nfrac < 0.0_rt) {
        amrex::Error("ERROR: nfrac must fall between 0 and 1");
    }


    if (problem::ofrac > 1.0_rt || problem::ofrac < 0.0_rt) {
        amrex::Error("ERROR: ofrac must fall between 0 and 1");
    }


    if (problem::nefrac > 1.0_rt || problem::nefrac < 0.0_rt) {
        amrex::Error("ERROR: nefrac must fall between 0 and 1");
    }


    if (problem::cfrac + problem::nfrac + problem::ofrac + problem::nefrac > 1.0_rt) {
        amrex::Error("ERROR: cfrac + nfrac + ofrac + nefrac cannot exceed 1.");
    }


    for (auto& X : problem::xn) {
        X = problem::smallx;
    }

    problem::xn[problem::ic12] = amrex::max(problem::cfrac, problem::smallx);
    problem::xn[problem::io16] = amrex::max(problem::ofrac, problem::smallx);
    problem::xn[problem::ine20] = amrex::max(problem::nefrac, problem::smallx);

    if (problem::in14 >= 0) {
       problem::xn[problem::in14] = amrex::max(problem::nfrac, problem::smallx);
       problem::xn[problem::ihe4] = 1.0_rt - problem::xn[problem::ic12]
                                           - problem::xn[problem::in14]
                                           - problem::xn[problem::io16]
                                           - problem::xn[problem::ine20]
                                           - (NumSpec - 5) * problem::smallx;
    }
    else {
       problem::xn[problem::ihe4] = 1.0_rt - problem::xn[problem::ic12]
                                           - problem::xn[problem::io16]
                                           - problem::xn[problem::ine20]
                                           - (NumSpec - 4) * problem::smallx;
    }


    problem::ambient_dens = problem::dens;
    for (int n = 0; n < NumSpec; n++) {
        problem::ambient_comp[n] = problem::xn[n];
    }

    eos_t eos_state;
    eos_state.rho = problem::ambient_dens;
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n]  = problem::ambient_comp[n];
    }

#ifdef AUX_THERMO
    set_aux_comp_from_X(eos_state);
#endif

    eos_state.T = problem::T_l;

    eos(eos_input_rt, eos_state);

    problem::ambient_e_l = eos_state.e;

    eos_state.T = problem::T_r;

    eos(eos_input_rt, eos_state);

    problem::ambient_e_r = eos_state.e;

}

#endif
