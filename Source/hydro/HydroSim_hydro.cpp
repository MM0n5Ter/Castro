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
HydroSim::verify_courant_limit(const MultiFab& State, const Real dt)
{
    BL_PROFILE("HydroSim::verify_courant_limit()");

    int cfl_breach = 0;

    auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
    const auto coord = geom.Coord();
    amrex::ignore_unused(problo, coord);

    Real dtdx = dt / dx[0];

    Real dtdy = 0.0_rt;
    if (AMREX_SPACEDIM >= 2) {
      dtdy = dt / dx[1];
      if (coord == 2) {
          dtdy /= problo[0] + 0.5_rt * dx[0];
      }
    }

    Real dtdz = 0.0_rt;
    if (AMREX_SPACEDIM == 3) {
      dtdz = dt / dx[2];
    }

    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(State, hydro_tile_size); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        auto U = State.array(mfi);

        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
        {

            Real rho = U(i,j,k,URHO);
            Real rhoInv = 1.0 / rho;

            Real u = U(i,j,k,UMX) * rhoInv;
            Real v = U(i,j,k,UMY) * rhoInv;
            Real w = U(i,j,k,UMZ) * rhoInv;

            eos_rep_t eos_state;

            eos_state.rho = U(i,j,k,URHO);
            eos_state.T = U(i,j,k,UTEMP);
            eos_state.e = U(i,j,k,UEINT) * rhoInv;
            for (int n = 0; n < NumSpec; n++) {
                eos_state.xn[n] = U(i,j,k,UFS+n) * rhoInv;
            }
#if NAUX_NET > 0
            for (int n = 0; n < NumAux; n++) {
                eos_state.aux[n] = U(i,j,k,UFX+n) * rhoInv;
            }
#endif

            eos(eos_input_re, eos_state);

            Real cs = eos_state.cs;

            Real courant_x = (cs + std::abs(u)) * dtdx;
            Real courant_y = (cs + std::abs(v)) * dtdy;
            Real courant_z = (cs + std::abs(w)) * dtdz;

            if (hydrosim::time_scheme == 0) {

#ifndef AMREX_USE_GPU
                if (verbose == 1) {

                    if (courant_x > 1.0_rt) {
                        std::cout << std::endl;
                        std::cout << Font::Bold << FGColor::Yellow;
                        std::cout << "[WARNING] CFL violation in verify_courant_limit" << std::endl;
                        std::cout << ">>> ... (u+c) * dt / dx > 1 " << courant_x << std::endl;
                        std::cout << ">>> ... at cell i = " << i << " j = " << j << " k = " << k << std::endl;
                        std::cout << ">>> ... u = " << u << " c = " << cs << std::endl;
                        std::cout << ">>> ... density = " << rho << std::endl;
                        std::cout << ResetDisplay;
                    }

                    if (courant_y > 1.0_rt) {
                        std::cout << std::endl;
                        std::cout << Font::Bold << FGColor::Yellow;
                        std::cout << "[WARNING] CFL violation in verify_courant_limit" << std::endl;
                        std::cout << ">>> ... (v+c) * dt / dx > 1 " << courant_y << std::endl;
                        std::cout << ">>> ... at cell i = " << i << " j = " << j << " k = " << k << std::endl;
                        std::cout << ">>> ... v = " << v << " c = " << cs << std::endl;
                        std::cout << ">>> ... density = " << rho << std::endl;
                        std::cout << ResetDisplay;
                    }

                    if (courant_z > 1.0_rt) {
                        std::cout << std::endl;
                        std::cout << Font::Bold << FGColor::Yellow;
                        std::cout << "[WARNING] CFL violation in verify_courant_limit" << std::endl;
                        std::cout << ">>> ... (w+c) * dt / dx > 1 " << courant_z << std::endl;
                        std::cout << ">>> ... at cell i = " << i << " j = " << j << " k = " << k << std::endl;
                        std::cout << ">>> ... w = " << w << " c = " << cs << std::endl;
                        std::cout << ">>> ... density = " << rho << std::endl;
                        std::cout << ResetDisplay;
                    }

                }
#endif


                return {amrex::max(courant_x, courant_y, courant_z)};

            }
            else {

                Real courant_combined = courant_x;
                if (AMREX_SPACEDIM >= 2) {
                    courant_combined += courant_y;
                }
                if (AMREX_SPACEDIM == 3) {
                    courant_combined += courant_z;
                }

#ifndef AMREX_USE_GPU
                if (verbose == 1) {

                    if (courant_combined > 1.0_rt) {
                        std::cout << std::endl;
                        std::cout << Font::Bold << FGColor::Yellow;
                        std::cout << "[WARNING] CFL violation in verify_courant_limit" << std::endl;
                        std::cout << ">>> ... at cell i = " << i << " j = " << j << " k = " << k << std::endl;
                        std::cout << ">>> ... u = " << u << " v = " << v
                                  << " w = " << w << " c = " << cs << std::endl;
                        std::cout << ">>> ... density = " << rho << std::endl;
                        std::cout << ResetDisplay;
                    }

                }
#endif

                return {courant_combined};

            }

        });

    }

    ReduceTuple hv = reduce_data.value();
    Real courant_max = amrex::get<0>(hv);

    ParallelDescriptor::ReduceRealMax(courant_max);

    if (courant_max > 1.0) {
        amrex::Print() << Font::Bold << FGColor::Yellow << "[WARNING] EFFECTIVE CFL AT LEVEL " << level << " IS " << courant_max << ResetDisplay << std::endl << std::endl;
        cfl_breach = 1;
    }

    if (cfl_breach) {
        amrex::Abort("CFL is too high at this level; go back to a checkpoint and restart with lower CFL number");
    }
}
