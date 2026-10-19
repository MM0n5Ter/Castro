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
        HydroSim::user_run_logs[0]->open("xrb_diag.out", std::ios::out | std::ios::app);
        if (!HydroSim::user_run_logs[0]->good()) {
            amrex::FileOpenFailed("xrb_diag.out");
        }

    }


    ambient::ambient_reference_state[URHO] = model::profile(0).state(model::npts-1, model::idens);
    ambient::ambient_reference_state[UTEMP] = model::profile(0).state(model::npts-1, model::itemp);
    for (int n = 0; n < NumSpec; n++) {
        ambient::ambient_reference_state[UFS+n] =
            ambient::ambient_reference_state[URHO] * model::profile(0).state(model::npts-1, model::ispec+n);
    }

    ambient::ambient_reference_state[UMX] = 0.0_rt;
    ambient::ambient_reference_state[UMY] = 0.0_rt;
    ambient::ambient_reference_state[UMZ] = 0.0_rt;


    eos_t eos_state;
    eos_state.rho = ambient::ambient_reference_state[URHO];
    eos_state.T = ambient::ambient_reference_state[UTEMP];
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = ambient::ambient_reference_state[UFS+n] / eos_state.rho;
    }

    eos(eos_input_rt, eos_state);

    ambient::ambient_reference_state[UEINT] = eos_state.rho * eos_state.e;
    ambient::ambient_reference_state[UEDEN] = eos_state.rho * eos_state.e;

}
#endif

