
#include <HydroSim.H>
#include <advection_util.H>
#ifdef MODEL_PARSER
#include <model_parser.H>
#endif
#include <sdc_cons_to_burn.H>

using std::string;
using namespace amrex;


advance_status
HydroSim::handle_split_reactions (Real time, Real dt, int strang_half)
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    advance_status status {};

    int burn_success{1};

    MultiFab& R_old = get_old_data(Reactions_Type);
    MultiFab& R_new = get_new_data(Reactions_Type);
    MultiFab& S_new = get_new_data(State_Type);


    if (time_integration_method == SimplifiedSpectralDeferredCorrections) {
        return status;
    }

    // The first half burns Sborder in place, starting at the old time;
    // the second half burns the new-time state over the last half of
    // the timestep.

    if (strang_half == 0) {
        burn_success = react_state(Sborder, R_old, time, 0.5 * dt, 0);
    } else {
        burn_success = react_state(S_new, R_new, time - 0.5 * dt, 0.5 * dt, 1);
    }

    if (burn_success != 1) {
        status.success = false;
        status.reason = "burn unsuccessful";

        return status;
    }

    if (strang_half == 0) {
        clean_state(
                    Sborder, time, Sborder.nGrow());

        MultiFab::Copy(R_new, R_old, 0, 0, R_new.nComp(), R_new.nGrow());
    } else {
        clean_state(
                    S_new, time, S_new.nGrow());
    }


    return status;
}

// Strang version

int
HydroSim::react_state(MultiFab& s, MultiFab& r, Real time, Real dt, const int strang_half)
{

    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::react_state()");

    // Sanity check: should only be in here if we're doing CTU.

    if (time_integration_method != CornerTransportUpwind) {
        amrex::Error("Strang reactions are only supported for the CTU advance.");
    }

    const Real strt_time = ParallelDescriptor::second();

    // Start off by assuming a successful burn.

    int burn_success = 1;

    if (do_react != 1) {

        // Ensure we always have valid data, even if we don't do the burn.
        r.setVal(0.0, r.nGrow());

        return burn_success;

    }

    // Check if we have any zones to burn.

    if (!valid_zones_to_burn(s)) {

        // Ensure we always have valid data, even if we don't do the burn.
        r.setVal(0.0, r.nGrow());

        return burn_success;

    }

    const int ng = s.nGrow();

    if (verbose) {
        amrex::Print() << "... Entering burner on level " << level << " and doing half-timestep of burning." << std::endl << std::endl;
    }

    // If we're not subcycling, we only need to do the burn on leaf cells.

    bool mask_covered_zones = false;

    if (level < parent->finestLevel() && parent->subcyclingMode() == "None") {
        mask_covered_zones = true;
    }

    MultiFab tmp_mask_mf;
    const MultiFab& mask_mf = mask_covered_zones ? getLevel(level+1).build_fine_mask() : tmp_mask_mf;

    // Only zones inside this (rho, T) range are burned.

    const Real rho_min = hydrosim::react_rho_min;
    const Real rho_max = hydrosim::react_rho_max;
    const Real T_min = hydrosim::react_T_min;
    const Real T_max = hydrosim::react_T_max;

#ifdef AMREX_USE_GPU
    Gpu::Buffer<int> d_num_failed({0});
    auto* p_num_failed = d_num_failed.data();
#endif
    int num_failed = 0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:num_failed)
#endif
    for (MFIter mfi(s, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {

        const Box& bx = mfi.growntilebox(ng);

        auto U = s.array(mfi);
        auto reactions = r.array(mfi);
        auto weights = store_burn_weights ? burn_weights.array(mfi) : Array4<Real>{};
        Array4<Real> empty_arr{};
        const auto& mask = mask_covered_zones ? mask_mf.array(mfi) : empty_arr;

        const auto dx = geom.CellSizeArray();
#ifdef MODEL_PARSER
        const auto problo = geom.ProbLoArray();
        const auto geomdata = geom.data();
#endif

#ifdef AMREX_USE_GPU
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
#else
        LoopOnCpu(bx, [&] (int i, int j, int k)
#endif
        {

            burn_t burn_state;

#if AMREX_SPACEDIM == 1
            burn_state.dx = dx[0];
#else
            burn_state.dx = amrex::min(AMREX_D_DECL(dx[0], dx[1], dx[2]));
#endif

            // Initialize some data for later.

            bool do_burn = true;
            burn_state.success = true;
            int burn_failed = 0;

            // Don't burn on zones inside shock regions, if the relevant option is set.

#ifdef SHOCK_VAR
            if (U(i,j,k,USHOCK) > 0.0_rt && disable_shock_burning == 1) {
                do_burn = false;
            }
#endif
            // Don't burn on zones that are masked out.

            if (mask_covered_zones && mask.contains(i,j,k)) {
                if (mask(i,j,k) == 0.0_rt) {
                    do_burn = false;
                }
            }

            Real rhoInv = 1.0_rt / U(i,j,k,URHO);

            burn_state.rho = U(i,j,k,URHO);

            // e is used as an input for some NSE solvers

            burn_state.e = U(i,j,k,UEINT) * rhoInv;

            // this T is consistent with UEINT because we did an EOS call before
            // calling this function

            burn_state.T = U(i,j,k,UTEMP);

            burn_state.T_fixed = -1.e30_rt;

#ifdef MODEL_PARSER
            if (drive_initial_convection) {
                GpuArray<Real, 3> rr = {0.0_rt};

                rr[0] = problo[0] + dx[0] * (static_cast<Real>(i) + 0.5_rt) - problem::center[0];
#if AMREX_SPACEDIM >= 2
                rr[1] = problo[1] + dx[1] * (static_cast<Real>(j) + 0.5_rt) - problem::center[1];
#endif
#if AMREX_SPACEDIM == 3
                rr[2] = problo[2] + dx[2] * (static_cast<Real>(k) + 0.5_rt) - problem::center[2];
#endif

                Real dist;

                if (domain_is_plane_parallel) {
                    dist = rr[AMREX_SPACEDIM-1];
                } else {
                    dist = distance(geomdata, rr);
                }

                burn_state.T_fixed = interpolate(dist, model::itemp);

            }
#endif

            for (int n = 0; n < NumSpec; ++n) {
                burn_state.xn[n] = U(i,j,k,USPEC+n) * rhoInv;
            }

#if NAUX_NET > 0
            for (int n = 0; n < NumAux; ++n) {
                burn_state.aux[n] = U(i,j,k,UAUXV+n) * rhoInv;
            }
#endif

            // Ensure we start with no RHS or Jacobian calls registered.

            burn_state.n_rhs = 0;
            burn_state.n_jac = 0;

            // for diagnostics

            burn_state.i = i;
            burn_state.j = j;
            burn_state.k = k;

#ifdef NONAKA_PLOT
            burn_state.level = level;
            burn_state.reference_time = time;
#ifdef STRANG
            burn_state.strang_half = strang_half;
#endif
#endif

            // Don't burn if we're outside of the relevant (rho, T) range.

            if (burn_state.T < T_min || burn_state.T > T_max ||
                burn_state.rho < rho_min || burn_state.rho > rho_max) {
                do_burn = false;
            }

            if (do_burn) {
                burner(burn_state, dt);

                // If we were unsuccessful, update the failure count.

                if (!burn_state.success) {
                    burn_failed = 1;
                }

                // Add burning rates to reactions MultiFab, but be
                // careful because the reactions and state MFs may
                // not have the same number of ghost cells.

                if (reactions.contains(i,j,k)) {

                    reactions(i,j,k,0) = (U(i,j,k,URHO) * burn_state.e - U(i,j,k,UEINT)) / dt;

                    if (store_omegadot == 1) {
                        if (reactions.contains(i,j,k)) {
                            for (int n = 0; n < NumSpec; ++n) {
                                reactions(i,j,k,1+n) = U(i,j,k,URHO) * (burn_state.xn[n] - U(i,j,k,USPEC+n) * rhoInv) / dt;
                            }
#if NAUX_NET > 0
                            for (int n = 0; n < NumAux; ++n) {
                                reactions(i,j,k,1+n+NumSpec) = U(i,j,k,URHO) * (burn_state.aux[n] - U(i,j,k,UAUXV+n) * rhoInv) / dt;
                            }
#endif
                        }
                    }

                    if (store_burn_weights) {

                        if (integrator_rp::jacobian == 1) {
                            weights(i,j,k,strang_half) = amrex::max(1.0_rt, static_cast<Real>(burn_state.n_rhs + 2 * burn_state.n_jac));
                        } else {
                            // the RHS evals for the numerical differencing in the Jacobian are already accounted for in n_rhs
                            weights(i,j,k,strang_half) = amrex::max(1.0_rt, static_cast<Real>(burn_state.n_rhs));
                        }
                    }
                }

                // update the state
                for (int n = 0; n < NumSpec; ++n) {
                    U(i,j,k,USPEC+n) = U(i,j,k,URHO) * burn_state.xn[n];
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; ++n) {
                    U(i,j,k,UAUXV+n) = U(i,j,k,URHO) * burn_state.aux[n];
                }
#endif
                Real reint_old = U(i,j,k,UEINT);
                U(i,j,k,UEINT) = U(i,j,k,URHO) * burn_state.e;
                U(i,j,k,UEDEN) += U(i,j,k,UEINT) - reint_old;

            } else {  // do_burn = false

                if (reactions.contains(i,j,k)) {
                    for (int n = 0; n < reactions.nComp(); n++) {
                        reactions(i,j,k,n) = 0.0_rt;
                    }
                }

            }

#ifdef AMREX_USE_GPU
            if (burn_failed) {
                Gpu::Atomic::Add(p_num_failed, burn_failed);
            }
#else
            num_failed += burn_failed;
#endif
        });

#ifdef AMREX_USE_HIP
        Gpu::streamSynchronize(); // otherwise HIP may fail to allocate the necessary resources.
#endif

#ifdef ALLOW_GPU_PRINTF
        std::fflush(nullptr);
#endif

    }

#ifdef AMREX_USE_GPU
    num_failed = *(d_num_failed.copyToHost());
#endif

    burn_success = !num_failed;

    ParallelDescriptor::ReduceIntMin(burn_success);

    if (print_update_diagnostics) {

        Real e_added = r.sum(0);

        if (e_added != 0.0) {
            amrex::Print() << "... (rho e) added from burning: " << e_added * dt << std::endl << std::endl;
        }

    }

    if (verbose) {
        amrex::Print() << "... Leaving burner on level " << level << " after completing half-timestep of burning." << std::endl << std::endl;
    }

    if (verbose > 0) {
        print_stage_time("HydroSim::react_state()", strt_time, true);
    }

    return burn_success;

}



namespace {

// The reduced limiters hold the density entry first (if we limit on
// density), followed by the temperature entry (if we limit on T).

void
unpack_burn_limiters (const amrex::Vector<Real>& limiters, bool limit_rho, bool limit_T,
                      Real& rho_limit, Real& T_limit)
{
    if (limit_rho) {
        rho_limit = limiters[0];
        if (limit_T) {
            T_limit = limiters[1];
        }
    } else {
        T_limit = limiters[0];
    }
}

}

bool
HydroSim::valid_zones_to_burn(const MultiFab& State)
{

    // The default values of the limiters are 0 and 1.e200, respectively.

    Real small = 1.e-10;
    Real large = 1.e199;

    // Check whether we are limiting on either rho or T.

    bool limit_small_rho = react_rho_min >= small;
    bool limit_large_rho = react_rho_max <= large;

    bool limit_rho = limit_small_rho || limit_large_rho;

    bool limit_small_T = react_T_min >= small;
    bool limit_large_T = react_T_max <= large;

    bool limit_T = limit_small_T || limit_large_T;

    bool limit = limit_rho || limit_T;

    if (!limit) {
      return true;
    }

    // if we're limiting on rho, collect the minimum and/or maximum

    amrex::Vector<Real> small_limiters;
    amrex::Vector<Real> large_limiters;

    bool local = true;

    Real smalldens = small;
    Real largedens = large;

    if (limit_small_rho) {
      smalldens = State.min(URHO, 0, local);
      small_limiters.push_back(smalldens);
    }

    if (limit_large_rho) {
      largedens = State.max(URHO, 0, local);
      large_limiters.push_back(largedens);
    }

    // if we're limiting on T, collect the minimum and/or maximum

    Real small_T = small;
    Real large_T = large;

    if (limit_small_T) {
      small_T = State.min(UTEMP, 0, local);
      small_limiters.push_back(small_T);
    }

    if (limit_large_T) {
      large_T = State.max(UTEMP, 0, local);
      large_limiters.push_back(large_T);
    }

    // Now do the reductions. We're being careful here
    // to limit the amount of work and communication,
    // because regularly doing this check only makes sense
    // if it is negligible compared to the amount of work
    // needed to just do the burn as normal.

    int small_size = static_cast<int>(small_limiters.size());

    if (small_size > 0) {
        amrex::ParallelDescriptor::ReduceRealMin(small_limiters.dataPtr(), small_size);

        unpack_burn_limiters(small_limiters, limit_small_rho, limit_small_T, smalldens, small_T);
    }

    int large_size = static_cast<int>(large_limiters.size());

    if (large_size > 0) {
        amrex::ParallelDescriptor::ReduceRealMax(large_limiters.dataPtr(), large_size);

        unpack_burn_limiters(large_limiters, limit_large_rho, limit_large_T, largedens, large_T);
    }

    // Finally check on whether min <= rho <= max
    // and min <= T <= max. The defaults are small
    // and large respectively, so if the limiters
    // are not on, these checks will not be triggered.

    if (largedens >= react_rho_min &&
        smalldens <= react_rho_max &&
        large_T >= react_T_min &&
        small_T <= react_T_max) {
        return true;
    }

    // If we got to this point, we did not survive the limiters,
    // so there are no zones to burn.

    if (verbose > 1) {
        amrex::Print() << "  No valid zones to burn, skipping react_state()." << std::endl;
    }

    return false;

}

