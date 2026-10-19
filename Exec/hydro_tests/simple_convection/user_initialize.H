#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>
#include <model_parser.H>
#include <initial_model.H>
#include <global.H>
#include <ambient.H>

AMREX_INLINE
void user_initialize ()
{

    const Geometry& default_geom = DefaultGeometry();

    const Real* problo = default_geom.ProbLo();
    const Real* probhi = default_geom.ProbHi();

    problem::center[0] = 0.5_rt * (problo[0] + probhi[0]);
#if AMREX_SPACEDIM >= 2
    problem::center[1] = 0.5_rt * (problo[1] + probhi[1]);
#endif
#if AMREX_SPACEDIM == 3
    problem::center[2] = 0.5_rt * (problo[2] + probhi[2]);
#endif


    auto fine_geom = global::amr_instance->Geom(global::amr_instance->maxLevel());

    auto dx = fine_geom.CellSizeArray();
    auto dx_model = dx[AMREX_SPACEDIM-1];

    int nx = (2.0_rt * problem::center[AMREX_SPACEDIM-1] + 1.e-8_rt) /
        dx_model;

    model_t model_params;
    model_params.density_base = problem::density_base;
    model_params.scale_height = problem::scale_height;
    for (int n = 0; n < NumSpec; ++n) {
        model_params.xn[n] = 0.0_rt;
    }
    model_params.xn[0] = 1.0_rt;

    generate_initial_model(nx, problo[AMREX_SPACEDIM-1], probhi[AMREX_SPACEDIM-1],
                           model_params);


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
