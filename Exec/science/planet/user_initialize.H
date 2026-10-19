#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>
#include <model_parser.H>
#include <ambient.H>

AMREX_INLINE
void user_initialize ()
{

    const Geometry& default_geom = DefaultGeometry();

    const Real* problo = default_geom.ProbLo();
    const Real* probhi = default_geom.ProbHi();

    if (problem::num_vortices > problem::max_num_vortices) {
        amrex::Error("num_vortices too large, please increase max_num_vortices and the size of xloc_vortices");
    }


    read_model_file(problem::model_name);


    Real max_hse_err = -1.e30_rt;
    Real dr_model = model::profile(0).r(1) - model::profile(0).r(0);

    for (int i = 1; i < model::npts-2; i++) {
        Real dpdr = (model::profile(0).state(i, model::ipres) -
                     model::profile(0).state(i-1, model::ipres)) / dr_model;
        Real rhog = 0.5_rt * (model::profile(0).state(i, model::idens) +
                              model::profile(0).state(i-1, model::idens)) * gravity::uniform_accel;
        Real hse_err = std::abs(dpdr - rhog);


        if (model::profile(0).state(i+1, model::ipres) /=
            model::profile(0).state(i, model::ipres)) {
            max_hse_err = amrex::max(max_hse_err, hse_err);
        }
    }

    std::cout << "maximum hse error = " << max_hse_err << std::endl;


    for (int d = 0; d < AMREX_SPACEDIM; d++) {
        problem::center[d] = 0.5_rt * (problo[d] + probhi[d]);
    }


    Real offset = (probhi[0] - problo[0]) / problem::num_vortices;

    for (int i = 0; i < problem::num_vortices; i++) {
        problem::xloc_vortices[i] = (static_cast<Real>(i) + 0.5_rt) * offset + problo[0];
    }

}
#endif
