#include <HydroSim.H>
#include <advection_util.H>


using namespace amrex;




void
HydroSim::check_for_cfl_violation(const MultiFab& State, const Real dt)
{
    BL_PROFILE("HydroSim::check_for_cfl_violation()");

    int cfl_violation = 0;

    auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
    const auto coord = geom.Coord();
    amrex::ignore_unused(problo, coord);

    Real dtdx = dt / dx[0];

    Real dtdy = 0.0_rt;
    if (AMREX_SPACEDIM >= 2) {
      dtdy = dt / dx[1];
      if (coord == 2) {
          // dx[1] in Spherical2D is just rdtheta, need rdtheta for physical length
          // Just choose to divide by the smallest r
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
            // Compute running max of Courant number over grids

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
                eos_state.xn[n] = U(i,j,k,USPEC+n) * rhoInv;
            }
#if NAUX_NET > 0
            for (int n = 0; n < NumAux; n++) {
                eos_state.aux[n] = U(i,j,k,UAUXV+n) * rhoInv;
            }
#endif

            eos(eos_input_re, eos_state);

            Real cs = eos_state.cs;

            Real courx = (cs + std::abs(u)) * dtdx;
            Real coury = (cs + std::abs(v)) * dtdy;
            Real courz = (cs + std::abs(w)) * dtdz;

            if (hydrosim::time_integration_method == 0) {

#ifndef AMREX_USE_GPU
                if (verbose == 1) {

                    if (courx > 1.0_rt) {
                        std::cout << std::endl;
                        std::cout << Font::Bold << FGColor::Yellow;
                        std::cout << "[WARNING] CFL violation in check_for_cfl_violation" << std::endl;
                        std::cout << ">>> ... (u+c) * dt / dx > 1 " << courx << std::endl;
                        std::cout << ">>> ... at cell i = " << i << " j = " << j << " k = " << k << std::endl;
                        std::cout << ">>> ... u = " << u << " c = " << cs << std::endl;
                        std::cout << ">>> ... density = " << rho << std::endl;
                        std::cout << ResetDisplay;
                    }

                    if (coury > 1.0_rt) {
                        std::cout << std::endl;
                        std::cout << Font::Bold << FGColor::Yellow;
                        std::cout << "[WARNING] CFL violation in check_for_cfl_violation" << std::endl;
                        std::cout << ">>> ... (v+c) * dt / dx > 1 " << coury << std::endl;
                        std::cout << ">>> ... at cell i = " << i << " j = " << j << " k = " << k << std::endl;
                        std::cout << ">>> ... v = " << v << " c = " << cs << std::endl;
                        std::cout << ">>> ... density = " << rho << std::endl;
                        std::cout << ResetDisplay;
                    }

                    if (courz > 1.0_rt) {
                        std::cout << std::endl;
                        std::cout << Font::Bold << FGColor::Yellow;
                        std::cout << "[WARNING] CFL violation in check_for_cfl_violation" << std::endl;
                        std::cout << ">>> ... (w+c) * dt / dx > 1 " << courz << std::endl;
                        std::cout << ">>> ... at cell i = " << i << " j = " << j << " k = " << k << std::endl;
                        std::cout << ">>> ... w = " << w << " c = " << cs << std::endl;
                        std::cout << ">>> ... density = " << rho << std::endl;
                        std::cout << ResetDisplay;
                    }

                }
#endif

                // CTU integration constraint

                return {amrex::max(courx, coury, courz)};

            }
            else {

                // method-of-lines constraint
                Real courtmp = courx;
                if (AMREX_SPACEDIM >= 2) {
                    courtmp += coury;
                }
                if (AMREX_SPACEDIM == 3) {
                    courtmp += courz;
                }

#ifndef AMREX_USE_GPU
                if (verbose == 1) {

                    // note: it might not be 1 for all RK integrators
                    if (courtmp > 1.0_rt) {
                        std::cout << std::endl;
                        std::cout << Font::Bold << FGColor::Yellow;
                        std::cout << "[WARNING] CFL violation in check_for_cfl_violation" << std::endl;
                        std::cout << ">>> ... at cell i = " << i << " j = " << j << " k = " << k << std::endl;
                        std::cout << ">>> ... u = " << u << " v = " << v
                                  << " w = " << w << " c = " << cs << std::endl;
                        std::cout << ">>> ... density = " << rho << std::endl;
                        std::cout << ResetDisplay;
                    }

                }
#endif

                return {courtmp};

            }

        });

    }

    ReduceTuple hv = reduce_data.value();
    Real courno = amrex::get<0>(hv);

    ParallelDescriptor::ReduceRealMax(courno);

    if (courno > 1.0) {
        amrex::Print() << Font::Bold << FGColor::Yellow << "[WARNING] EFFECTIVE CFL AT LEVEL " << level << " IS " << courno << ResetDisplay << std::endl << std::endl;
        cfl_violation = 1;
    }

    // If we detect a CFL violation, abort.
    if (cfl_violation) {
        amrex::Abort("CFL is too high at this level; go back to a checkpoint and restart with lower CFL number");
    }
}
