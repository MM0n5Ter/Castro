#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>
#include <model_parser.H>
#include <initial_model.H>
#include <network.H>
#include <global.H>

AMREX_INLINE
void user_initialize ()
{
    const Geometry& default_geom = DefaultGeometry();

    const Real* problo = default_geom.ProbLo();
    const Real* probhi = default_geom.ProbHi();

    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        problem::center[n] = 0.5_rt * (problo[n] + probhi[n]);
    }


    int ihe4 = network_spec_index("helium-4");
    if (ihe4 < 0) {
        amrex::Error("Error: helium-4 not present");
    }

    model_t model_params;
    model_params.T_base = problem::temp_base;
    model_params.density_base = problem::density_base;
    for (auto& X : model_params.xn) {
        X = 100 * network_rp::small_x;
    }
    model_params.xn[ihe4] = 1.0_rt - (NumSpec - 1) * 100 * network_rp::small_x;


    int nbuf = 8;

    auto fine_geom = global::amr_instance->Geom(global::amr_instance->maxLevel());

    auto dx = fine_geom.CellSizeArray();
    auto dx_model = dx[AMREX_SPACEDIM-1];

    int nx_model = static_cast<int>((probhi[AMREX_SPACEDIM-1] -
                                     problo[AMREX_SPACEDIM-1] + 1.e-8_rt) / dx_model);


    generate_initial_model(nx_model,
                           problo[AMREX_SPACEDIM-1], probhi[AMREX_SPACEDIM-1],
                           model_params, nbuf);

}

#endif
