#ifndef INITIAL_MODEL_H
#define INITIAL_MODEL_H

#include <prob_parameters.H>
#include <network.H>
#include <eos.H>

struct model_t {
    Real density_base = -1;
    Real p_base = -1;
    Real xn[NumSpec] = {0.0};
};



AMREX_INLINE
void
generate_initial_model(const int npts_model, const Real xmin, const Real xmax,
                       const model_t model_params) {

    model::npts = npts_model;
    model::initialized = true;

    if (npts_model > NPTS_MODEL) {
        amrex::Error("Error: model has more than NPTS_MODEL points,  Increase MAX_NPTS_MODEL");
    }


    Real H = model_params.p_base / model_params.density_base / std::abs(gravity::uniform_accel);


    Real dx = (xmax - xmin) / npts_model;

    for (int i = 0; i < npts_model; i++) {
        model::profile(0).r(i) = xmin + (static_cast<Real>(i) + 0.5_rt) * dx;
    }

    for (int i = 0; i < npts_model; i++) {


        Real temp = eos_temp_guess;

        Real density;


        if (problem::do_isentropic == 1) {


            Real z = xmin + static_cast<Real>(i) * dx;

            density = model_params.density_base *
                std::pow(gravity::uniform_accel * model_params.density_base *
                 (eos_rp::eos_gamma - 1.0) * z /
                 (eos_rp::eos_gamma * model_params.p_base) + 1.0_rt, 1.0_rt/(eos_rp::eos_gamma - 1.0_rt));

        } else {
            Real z = xmin + (static_cast<Real>(i) + 0.5_rt) * dx;

            density = model_params.density_base * std::exp(-z/H);
        }


        Real pressure;

        if (i == 0) {
            pressure = model_params.p_base;

        } else {
            pressure = model::profile(0).state(i-1, model::ipres) -
                dx * 0.5_rt * (density + model::profile(0).state(i-1, model::idens)) * std::abs(gravity::uniform_accel);
        }

        eos_t eos_state;
        eos_state.p = pressure;
        eos_state.T = temp;
        eos_state.rho = density;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n] = model_params.xn[n];
        }

        eos(eos_input_rp, eos_state);

        model::profile(0).state(i, model::idens) = density;
        model::profile(0).state(i, model::ipres) = pressure;
        model::profile(0).state(i, model::itemp) = eos_state.T;
        for (int n = 0; n < NumSpec; n++) {
            model::profile(0).state(i, model::ispec+n) = model_params.xn[n];
        }

    }

}

#endif
