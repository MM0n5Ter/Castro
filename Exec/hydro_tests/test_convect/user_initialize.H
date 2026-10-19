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

    if (problem::num_vortices > problem::max_num_vortices) {
        amrex::Error("num_vortices too large, please increase max_num_vortices and the size of xloc_vortices");
    }

    read_model_file(problem::model_name);

    if (ParallelDescriptor::IOProcessor()) {
        for (int i = 0; i < model::npts; i++) {
            std::cout << i << " " << model::profile(0).r(i) << " " << model::profile(0).state(i,model::idens) << std::endl;
        }
    }


    Real offset = (probhi[0] - problo[0]) / problem::num_vortices;

    for (int i = 0; i < problem::num_vortices; i++) {
        problem::xloc_vortices[i] = (static_cast<Real>(i) + 0.5_rt) * offset + problo[0];
    }


    if (hydrosim::sum_step_period > 0 && amrex::ParallelDescriptor::IOProcessor()) {

        HydroSim::user_run_logs.resize(1);

        HydroSim::user_run_logs[0].reset(new std::fstream);
        HydroSim::user_run_logs[0]->open("problem_diag.out", std::ios::out | std::ios::app);
        if (!HydroSim::user_run_logs[0]->good()) {
            amrex::FileOpenFailed("problem_diag.out");
        }

    }

}
#endif
