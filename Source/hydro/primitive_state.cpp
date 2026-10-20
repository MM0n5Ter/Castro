#include <HydroSim.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

using namespace amrex;

#ifdef TRUE_SDC
void
HydroSim::make_primitives(const Real time)
{

    BL_PROFILE("HydroSim::make_primitives()");

#ifdef RADIATION
    AmrLevel::FillPatch(*this, RadEnergyGrown, NUM_GROW, time, Radiation_Type, 0, Radiation::nGroups);

    MultiFab rad_lambda_grown(grids, dmap, Radiation::nGroups, NUM_GROW);
    if (radiation->hydro_only) {
      rad_lambda_grown.setVal(0.0, NUM_GROW);
    }
    else {
      radiation->build_flux_limiter(level, grids, Sborder, RadEnergyGrown, rad_lambda_grown);
    }
#endif

    MultiFab& S_new = get_new_data(State_Type);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(S_new, hydro_tile_size); mfi.isValid(); ++mfi) {

        const Box& qbx = mfi.growntilebox(NUM_GROW);

        Array4<Real> const prim_arr = q.array(mfi);
        Array4<Real> const qthermo_arr = qthermo.array(mfi);
        Array4<Real> const StateBorder_arr = Sborder.array(mfi);
#ifdef RADIATION
        Array4<Real> const RadEnergyBorder_arr = RadEnergyGrown.array(mfi);
        Array4<Real> const flux_limiter_border_arr = rad_lambda_grown.array(mfi);
#endif

        ctoprim(qbx,
                time,
                StateBorder_arr,
#ifdef RADIATION
                RadEnergyBorder_arr,
                flux_limiter_border_arr,
#endif
                prim_arr,
                qthermo_arr);

    }

}
#endif

#ifdef TRUE_SDC
void
HydroSim::make_primitives_fourth(const Real time)
{

    BL_PROFILE("HydroSim::make_primitives_fourth()");

    auto domain_lo = geom.Domain().loVect3d();
    auto domain_hi = geom.Domain().hiVect3d();

    MultiFab& S_new = get_new_data(State_Type);

#ifdef RADIATION
    amrex::Abort("radiation not supported to fourth order");
#else
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        FArrayBox centered_state;

        for (MFIter mfi(S_new, hydro_tile_size); mfi.isValid(); ++mfi) {

            const Box& qbx = mfi.growntilebox(NUM_GROW);
            const Box& grown_box_minus1 = mfi.growntilebox(NUM_GROW-1);

            centered_state.resize(qbx, NUM_STATE);
            Elixir u_center_hold = centered_state.elixir();
            auto const centered_state_arr = centered_state.array();

            build_center_state(grown_box_minus1, Sborder.array(mfi), centered_state_arr, domain_lo, domain_hi);

            clamp_state_density(grown_box_minus1, centered_state.array(), verbose);

            repair_internal_energy(grown_box_minus1, centered_state.array());

            auto StateBorder_arr = Sborder.array(mfi);
            auto q_mean_arr = q_second.array(mfi);
            auto qthermo_mean_arr = qthermo_second.array(mfi);

            ctoprim(qbx,
                    time,
                    StateBorder_arr,
                    q_mean_arr,
                    qthermo_mean_arr);

            auto prim_arr = q.array(mfi);
            auto qthermo_arr = qthermo.array(mfi);

            ctoprim(grown_box_minus1,
                    time,
                    centered_state_arr,
                    prim_arr,
                    qthermo_arr);
        }
    }

#ifndef AMREX_USE_GPU
    scan_for_nan(q);
    scan_for_nan(q_second);
#endif

#ifdef DIFFUSION
    MultiFab::Copy(Temperature_cc, q, QTEMP, 0, 1, NUM_GROW-1);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(S_new, hydro_tile_size); mfi.isValid(); ++mfi) {

      const Box& grown_box_minus1 = mfi.growntilebox(NUM_GROW-1);

      build_fourth_average(grown_box_minus1, q.array(mfi), q_second.array(mfi), domain_lo, domain_hi);

      build_fourth_average(grown_box_minus1, qthermo.array(mfi), qthermo_second.array(mfi), domain_lo, domain_hi);

    }

#endif
}
#endif

void
HydroSim::reset_edge_thermo(const Box& bx,
                                Array4<Real> const& edge_state)
{

    int apply_eos = transverse_eos_pressure;
    int reset_eint_flag = transverse_rebuild_rhoe;

    Real temperature_floor = small_temp;
    Real pressure_floor = pressure_cutoff;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

#ifdef RADIATION
        Real prior_pressure = edge_state(i,j,k,QPRES);
#endif

        eos_rep_t eos_state;

        if (reset_eint_flag == 1) {
            if (edge_state(i,j,k,QEINTDENS) < 0.0_rt) {

                eos_state.rho = edge_state(i,j,k,QRHO);
                eos_state.T = temperature_floor;
                for (int n = 0; n < NumSpec; ++n) {
                    eos_state.xn[n] = edge_state(i,j,k,QFS+n);
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; ++n) {
                    eos_state.aux[n] = edge_state(i,j,k,QFX+n);
                }
#endif

                eos(eos_input_rt, eos_state);

                edge_state(i,j,k,QEINTDENS) = edge_state(i,j,k,QRHO) * eos_state.e;
                edge_state(i,j,k,QPRES) = eos_state.p;
            }
        }

        if (apply_eos == 1) {
            eos_state.rho = edge_state(i,j,k,QRHO);
            eos_state.e   = edge_state(i,j,k,QEINTDENS) / edge_state(i,j,k,QRHO);
            eos_state.T   = temperature_floor;
            for (int n = 0; n < NumSpec; ++n) {
                eos_state.xn[n] = edge_state(i,j,k,QFS+n);
            }
#if NAUX_NET > 0
            for (int n = 0; n < NumAux; ++n) {
                eos_state.aux[n] = edge_state(i,j,k,QFX+n);
            }
#endif

            eos(eos_input_re, eos_state);

            edge_state(i,j,k,QEINTDENS) = eos_state.e * eos_state.rho;
            edge_state(i,j,k,QPRES) = amrex::max(eos_state.p, pressure_floor);
        }

#ifdef RADIATION
        edge_state(i,j,k,QPTOTAL) = edge_state(i,j,k,QPTOTAL) + (edge_state(i,j,k,QPRES) - prior_pressure);
#endif

    });

}

void
HydroSim::edge_temp_to_pressure(const Box& bx,
                                Array4<Real> const& qm,
                                Array4<Real> const& qp)
{

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

        eos_rep_t eos_state;

        eos_state.rho    = qp(i,j,k,QRHO);
        eos_state.T      = qp(i,j,k,QTEMP);
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = qp(i,j,k,QFS+n);
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; ++n) {
            eos_state.aux[n] = qp(i,j,k,QFX+n);
        }
#endif

        eos(eos_input_rt, eos_state);

        qp(i,j,k,QPRES) = eos_state.p;
        qp(i,j,k,QEINTDENS) = qp(i,j,k,QRHO) * eos_state.e;

        eos_state.rho    = qm(i,j,k,QRHO);
        eos_state.T      = qm(i,j,k,QTEMP);
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = qm(i,j,k,QFS+n);
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; ++n) {
            eos_state.aux[n] = qm(i,j,k,QFX+n);
        }
#endif

        eos(eos_input_rt, eos_state);

        qm(i,j,k,QPRES) = eos_state.p;
        qm(i,j,k,QEINTDENS) = qm(i,j,k,QRHO) * eos_state.e;

    });

}
