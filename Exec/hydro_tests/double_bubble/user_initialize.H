#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>
#include <model_parser.H>
#include <initial_model.H>
#include <global.H>

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
    model_params.p_base = problem::pres_base;
    for (int n = 0; n < NumSpec; ++n) {
        model_params.xn[n] = 0.0_rt;
    }
    model_params.xn[0] = 1.0_rt;

    generate_initial_model(nx, problo[AMREX_SPACEDIM-1], probhi[AMREX_SPACEDIM-1],
                           model_params);

    if (problem::single == 1) {
        problem::left_bubble_x_center = problo[0] + 0.5_rt * (probhi[0] - problo[0]);
    } else {
        problem::left_bubble_x_center = problo[0] + (probhi[0] - problo[0]) / 3.0_rt;
        problem::right_bubble_x_center = problo[0] + 2.0_rt * (probhi[0] - problo[0]) / 3.0_rt;
    }

}

#endif
