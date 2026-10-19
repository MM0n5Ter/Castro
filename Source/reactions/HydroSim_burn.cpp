
#include <HydroSim.H>
#include <transport_util.H>
#ifdef MODEL_PARSER
#include <model_parser.H>
#endif
#include <sdc_conserved_to_burn.H>

using std::string;
using namespace amrex;

#ifndef TRUE_SDC

step_result
HydroSim::apply_old_reactions (Real time, Real dt) {  // NOLINT(readability-convert-member-functions-to-static)

    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    step_result status {};

#ifndef SIMPLIFIED_SDC
    int burn_outcome{1};

    MultiFab& Reactions_old = get_old_data(Burning_Type);
    MultiFab& react_new = get_new_data(Burning_Type);

    if (time_scheme != SimplifiedIterCorrection) {
        // The result of the reactions is added directly to Sborder.
        burn_outcome = burn_over_state(Sborder, Reactions_old, time, 0.5 * dt, 0);

        if (burn_outcome != 1) {
            status.success = false;
            status.reason = "burn unsuccessful";

            return status;
        }

        clean_state(
#ifdef MHD
                    Bx_old_grown, By_old_grown, Bz_old_grown,
#endif
                    Sborder, time, Sborder.nGrow());

        MultiFab::Copy(react_new, Reactions_old, 0, 0, react_new.nComp(), react_new.nGrow());
    }
#endif

    return status;
}

step_result
HydroSim::apply_new_reactions (Real time, Real dt)
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    step_result status {};

    int burn_outcome{1};

    MultiFab& react_new = get_new_data(Burning_Type);
    MultiFab& S_new = get_new_data(State_Type);

#ifdef SIMPLIFIED_SDC
    MultiFab& Reactions_old = get_old_data(Burning_Type);

    if (time_scheme == SimplifiedIterCorrection) {

        if (do_react) {

            // Do the ODE integration to capture the reaction source terms.

            burn_outcome = burn_over_state(time, dt);

            if (burn_outcome != 1) {
                status.success = false;
                status.reason = "burn unsuccessful";

                return status;
            }

            clean_state(S_new, time + dt, S_new.nGrow());

            // Check for NaN's.

            scan_for_nan(S_new);

        }
        else {

            // If we're not burning, just initialize the reactions data to zero.

            MultiFab& SDC_burn_new = get_new_data(Burn_Corr_Type);
            SDC_burn_new.setVal(0.0, SDC_burn_new.nGrow());

            Reactions_old.setVal(0.0, Reactions_old.nGrow());
            react_new.setVal(0.0, react_new.nGrow());

        }

    }

#else // SIMPLIFIED_SDC

    if (time_scheme != SimplifiedIterCorrection) {

        burn_outcome = burn_over_state(S_new, react_new, time - 0.5 * dt, 0.5 * dt, 1);

        if (burn_outcome != 1) {
            status.success = false;
            status.reason = "burn unsuccessful";

            return status;
        }

        clean_state(
#ifdef MHD
                    magfield_x_new, magfield_y_new, magfield_z_new,
#endif
                    S_new, time, S_new.nGrow());

    }

#endif // SIMPLIFIED_SDC

    return status;
}

// Strang version

int
HydroSim::burn_over_state(MultiFab& s, MultiFab& r, Real time, Real dt, const int strang_half)
{

    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::burn_over_state()");

    // Sanity check: should only be in here if we're doing CTU.

    if (time_scheme != CornerTrace) {
        amrex::Error("Strang reactions are only supported for the CTU advance.");
    }

    const Real strt_time = ParallelDescriptor::second();

    // Start off by assuming a successful burn.

    int burn_outcome = 1;

    if (do_react != 1) {

        // Ensure we always have valid data, even if we don't do the burn.
        r.setVal(0.0, r.nGrow());

        return burn_outcome;

    }

    // Check if we have any zones to burn.

    if (!zones_are_burnable(s)) {

        // Ensure we always have valid data, even if we don't do the burn.
        r.setVal(0.0, r.nGrow());

        return burn_outcome;

    }

    const int ng = s.nGrow();

    if (verbose) {
        amrex::Print() << "... Entering burner on level " << level << " and doing half-timestep of burning." << std::endl << std::endl;
    }

    // If we're not subcycling, we only need to do the burn on leaf cells.

    bool exclude_covered_zones = false;

    if (level < parent->finestLevel() && parent->subcyclingMode() == "None") {
        exclude_covered_zones = true;
    }

    MultiFab scratch_cover_mask;
    const MultiFab& fine_cover_mf = exclude_covered_zones ? getLevel(level+1).make_fine_mask() : scratch_cover_mask;

#ifdef AMREX_USE_GPU
    Gpu::Buffer<int> device_fail_count({0});
    auto* device_fail_count_ptr = device_fail_count.data();
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
        auto weights = keep_burn_weights ? burn_split_data.array(mfi) : Array4<Real>{};
        Array4<Real> null_view{};
        const auto& mask = exclude_covered_zones ? fine_cover_mf.array(mfi) : null_view;

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
#ifdef NSE_NET
            burn_state.mu_p = U(i,j,k,UPROTON_MU);
            burn_state.mu_n = U(i,j,k,UNEUTRON_MU);

            burn_state.y_e = -1.0_rt;
#endif

#if AMREX_SPACEDIM == 1
            burn_state.dx = dx[0];
#else
            burn_state.dx = amrex::min(AMREX_D_DECL(dx[0], dx[1], dx[2]));
#endif

            // Initialize some data for later.

            bool burn_allowed = true;
            burn_state.success = true;
            int burn_fault_flag = 0;

            // Don't burn on zones inside shock regions, if the relevant option is set.

#ifdef SHOCK_VAR
            if (U(i,j,k,UFLAG_SHOCK) > 0.0_rt && skip_burning_in_shocks == 1) {
                burn_allowed = false;
            }
#endif
            // Don't burn on zones that are masked out.

            if (exclude_covered_zones && mask.contains(i,j,k)) {
                if (mask(i,j,k) == 0.0_rt) {
                    burn_allowed = false;
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

                if (plane_parallel_domain) {
                    dist = rr[AMREX_SPACEDIM-1];
                } else {
                    dist = distance(geomdata, rr);
                }

                burn_state.T_fixed = interpolate(dist, model::itemp);

            }
#endif

            for (int n = 0; n < NumSpec; ++n) {
                burn_state.xn[n] = U(i,j,k,UFS+n) * rhoInv;
            }

#if NAUX_NET > 0
            for (int n = 0; n < NumAux; ++n) {
                burn_state.aux[n] = U(i,j,k,UFX+n) * rhoInv;
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

            if (burn_state.T < hydrosim::burn_temp_min || burn_state.T > hydrosim::burn_temp_max ||
                burn_state.rho < hydrosim::burn_rho_min || burn_state.rho > hydrosim::burn_rho_max) {
                burn_allowed = false;
            }

            if (burn_allowed) {
                burner(burn_state, dt);

                // If we were unsuccessful, update the failure count.

                if (!burn_state.success) {
                    burn_fault_flag = 1;
                }

                // Add burning rates to reactions MultiFab, but be
                // careful because the reactions and state MFs may
                // not have the same number of ghost cells.

                if (reactions.contains(i,j,k)) {

                    reactions(i,j,k,0) = (U(i,j,k,URHO) * burn_state.e - U(i,j,k,UEINT)) / dt;

                    if (keep_omegadot == 1) {
                        if (reactions.contains(i,j,k)) {
                            for (int n = 0; n < NumSpec; ++n) {
                                reactions(i,j,k,1+n) = U(i,j,k,URHO) * (burn_state.xn[n] - U(i,j,k,UFS+n) * rhoInv) / dt;
                            }
#if NAUX_NET > 0
                            for (int n = 0; n < NumAux; ++n) {
                                reactions(i,j,k,1+n+NumSpec) = U(i,j,k,URHO) * (burn_state.aux[n] - U(i,j,k,UFX+n) * rhoInv) / dt;
                            }
#endif
                        }
                    }

                    if (keep_burn_weights) {

                        if (integrator_rp::jacobian == 1) {
                            weights(i,j,k,strang_half) = amrex::max(1.0_rt, static_cast<Real>(burn_state.n_rhs + 2 * burn_state.n_jac));
                        } else {
                            // the RHS evals for the numerical differencing in the Jacobian are already accounted for in n_rhs
                            weights(i,j,k,strang_half) = amrex::max(1.0_rt, static_cast<Real>(burn_state.n_rhs));
                        }
                    }
#ifdef NSE
                    if (keep_omegadot == 1) {
                        reactions(i,j,k,NumSpec+NumAux+1) = burn_state.nse;
                    }
                    else {
                        reactions(i,j,k,1) = burn_state.nse;
                    }
#endif
                }

                // update the state
#ifdef NSE_NET
                U(i,j,k,UPROTON_MU) = burn_state.mu_p;
                U(i,j,k,UNEUTRON_MU) = burn_state.mu_n;
#endif
                for (int n = 0; n < NumSpec; ++n) {
                    U(i,j,k,UFS+n) = U(i,j,k,URHO) * burn_state.xn[n];
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; ++n) {
                    U(i,j,k,UFX+n) = U(i,j,k,URHO) * burn_state.aux[n];
                }
#endif
                Real internal_energy_old = U(i,j,k,UEINT);
                U(i,j,k,UEINT) = U(i,j,k,URHO) * burn_state.e;
                U(i,j,k,UEDEN) += U(i,j,k,UEINT) - internal_energy_old;

            } else {  // burn_allowed = false

                if (reactions.contains(i,j,k)) {
                    for (int n = 0; n < reactions.nComp(); n++) {
                        reactions(i,j,k,n) = 0.0_rt;
                    }
                }

            }

#ifdef AMREX_USE_GPU
            if (burn_fault_flag) {
                Gpu::Atomic::Add(device_fail_count_ptr, burn_fault_flag);
            }
#else
            num_failed += burn_fault_flag;
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
    num_failed = *(device_fail_count.copyToHost());
#endif

    burn_outcome = !num_failed;

    ParallelDescriptor::ReduceIntMin(burn_outcome);

    if (log_source_changes) {

        Real energy_added = r.sum(0);

        if (energy_added != 0.0) {
            amrex::Print() << "... (rho e) added from burning: " << energy_added * dt << std::endl << std::endl;
        }

    }

    if (verbose) {
        amrex::Print() << "... Leaving burner on level " << level << " after completing half-timestep of burning." << std::endl << std::endl;
    }

    if (verbose > 0)
    {
        const int IOProc   = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;
        amrex::Real level_real = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::burn_over_state() time = " << run_time
                       << " on level " << level_real << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }

    return burn_outcome;

}

#ifdef SIMPLIFIED_SDC
// Simplified SDC version

int
HydroSim::burn_over_state(Real time, Real dt)
{

    amrex::ignore_unused(time);

    // The goal is to update S_old to S_new with the effects of both
    // advection and reactions.  We come into this S_new having seen
    // the effects of advection and sources.  We create an advective
    // update of the form: -div{F} + 0.5 (S^n + S^{n+1} and pass this
    // to the reaction integrator where it is applied together with
    // the reactions to update the full state.

    // Note: S_new actually is already updated with just advection, so
    // in the event that we do not react on a zone (e.g., because it
    // doesn't meet the thermodynamic thresholds) we don't have to do
    // anything.  If we do react, then we overwrite what is stored in
    // S_new with the combined effects of advection and reactions.

    BL_PROFILE("HydroSim::burn_over_state()");

    // Sanity check: should only be in here if we're doing simplified SDC.

    if (time_scheme != SimplifiedIterCorrection) {
        amrex::Error("This burn_over_state interface is only supported for simplified SDC.");
    }

    const Real strt_time = ParallelDescriptor::second();

    if (verbose) {
        amrex::Print() << "... Entering burner on level " << level << " and doing full timestep of burning." << std::endl << std::endl;
    }

    MultiFab& S_old = get_old_data(State_Type);
    MultiFab& S_new = get_new_data(State_Type);

#ifdef MHD
    MultiFab& magfield_x_new = get_new_data(Bfield_Type_x);
    MultiFab& magfield_y_new = get_new_data(Bfield_Type_y);
    MultiFab& magfield_z_new = get_new_data(Bfield_Type_z);
#endif

    const int ng = S_new.nGrow();

    MultiFab& reactions = get_new_data(Burning_Type);
    MultiFab& SDC_burn_increment = get_new_data(Burn_Corr_Type);

    reactions.setVal(0.0, reactions.nGrow());

    // If we're not subcycling, we only need to do the burn on leaf cells.

    bool exclude_covered_zones = false;

    if (level < parent->finestLevel() && parent->subcyclingMode() == "None") {
        exclude_covered_zones = true;
    }

    MultiFab scratch_cover_mask;
    const MultiFab& fine_cover_mf = exclude_covered_zones ? getLevel(level+1).make_fine_mask() : scratch_cover_mask;

    // Start off assuming a successful burn.

    int burn_outcome = 1;

#ifdef AMREX_USE_GPU
    Gpu::Buffer<int> device_fail_count({0});
    auto* device_fail_count_ptr = device_fail_count.data();
#endif
    int num_failed = 0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:num_failed)
#endif
    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(ng);

        auto Conserved_old = S_old.array(mfi);
        auto cons_new = S_new.array(mfi);
#ifdef MHD
        auto Bx    = magfield_x_new.array(mfi);
        auto By    = magfield_y_new.array(mfi);
        auto Bz    = magfield_z_new.array(mfi);
#endif
        auto I     = SDC_burn_increment.array(mfi);
        auto reaction_source_arr = reactions.array(mfi);
        auto weights = keep_burn_weights ? burn_split_data.array(mfi) : Array4<Real>{};
        Array4<Real> null_view{};
        const auto& mask = exclude_covered_zones ? fine_cover_mf.array(mfi) : null_view;

        int sdc_pass_number = correction_iter;

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

#ifdef NSE_NET
            burn_state.mu_p = Conserved_old(i,j,k,UPROTON_MU);
            burn_state.mu_n = Conserved_old(i,j,k,UNEUTRON_MU);

            burn_state.y_e = -1.0_rt;
#endif
            // Initialize some data for later.

            bool burn_allowed = true;
            burn_state.success = true;
            int burn_fault_flag = 0;

            // Don't burn on zones inside shock regions, if the
            // relevant option is set.

#ifdef SHOCK_VAR
            if (cons_new(i,j,k,UFLAG_SHOCK) > 0.0_rt && skip_burning_in_shocks == 1) {
                burn_allowed = false;
            }
#endif

            // Don't burn on zones that are masked out.

            if (exclude_covered_zones && mask.contains(i,j,k)) {
                if (mask(i,j,k) == 0.0_rt) {
                    burn_allowed = false;
                }
            }

            // Feed in the old-time state data.
            // this also sets burn_state.{rho,T}

            fill_burn_type_from_conserved(i, j, k, Conserved_old, burn_state);

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

                if (plane_parallel_domain) {
                    dist = rr[AMREX_SPACEDIM-1];
                } else {
                    dist = distance(geomdata, rr);
                }

                burn_state.T_fixed = interpolate(dist, model::itemp);

            }
#endif

            // Don't burn if we're outside of the relevant (rho, T) range.

            if (Conserved_old(i,j,k,UTEMP) < hydrosim::burn_temp_min || Conserved_old(i,j,k,UTEMP) > hydrosim::burn_temp_max ||
                Conserved_old(i,j,k,URHO) < hydrosim::burn_rho_min || Conserved_old(i,j,k,URHO) > hydrosim::burn_rho_max) {
                burn_allowed = false;
            }

            // Tell the integrator about the non-reacting source terms,
            // which are advective_source + 1/2 (old source + new source).
            //
            // We have the following:
            //
            // cons_new:      U^n - dt div{F} + dt/2 (S^n + S^{n+1})
            // Conserved_old:      U^n
            //
            // So we can compute:
            //
            //   (1 / dt) * (cons_new - Conserved_old)
            //
            // To get the non-reacting sources:
            //
            //   -div{F} + (1/2) (S^n + S^{n+1})

            Real invTimestep = 1.0_rt / dt;

            burn_state.ydot_a[SRHO] = (cons_new(i,j,k,URHO) - Conserved_old(i,j,k,URHO)) * invTimestep;
            burn_state.ydot_a[SMX] = (cons_new(i,j,k,UMX) - Conserved_old(i,j,k,UMX)) * invTimestep;
            burn_state.ydot_a[SMY] = (cons_new(i,j,k,UMY) - Conserved_old(i,j,k,UMY)) * invTimestep;
            burn_state.ydot_a[SMZ] = (cons_new(i,j,k,UMZ) - Conserved_old(i,j,k,UMZ)) * invTimestep;
            burn_state.ydot_a[SEDEN] = (cons_new(i,j,k,UEDEN) - Conserved_old(i,j,k,UEDEN)) * invTimestep;
            burn_state.ydot_a[SEINT] = (cons_new(i,j,k,UEINT) - Conserved_old(i,j,k,UEINT)) * invTimestep;
            for (int n = 0; n < NumSpec; n++) {
                burn_state.ydot_a[SFS+n] = (cons_new(i,j,k,UFS+n) - Conserved_old(i,j,k,UFS+n)) * invTimestep;
            }
#if NAUX_NET > 0
            for (int n = 0; n < NumAux; n++) {
                burn_state.ydot_a[SFX+n] = (cons_new(i,j,k,UFX+n) - Conserved_old(i,j,k,UFX+n)) * invTimestep;
            }
#endif

            // Convert the current state to primitive data.
            // This state is U* = Conserved_old + dt A where A = -div U + S_hydro.

            Array1D<Real, 0, NQ-1> prim_pre_react;
            Array1D<Real, 0, NQEXTRAQ-1> qthermo_dummy;

            hydro::cell_to_primitive(i, j, k, cons_new,
#ifdef MHD
                                             Bx, By, Bz,
#endif
                                             prim_pre_react, qthermo_dummy, prim_pre_react.len() == NQ);

            // dual energy formalism: in doing EOS calls in the burn,
            // switch between e and (E - K) depending on (E - K) / E.

            burn_state.i = i;
            burn_state.j = j;
            burn_state.k = k;

#ifdef NONAKA_PLOT
            burn_state.level = level;
            burn_state.reference_time = time;
#endif

            burn_state.sdc_iter = sdc_pass_number;
            burn_state.num_sdc_iters = deferred_iters;

            if (burn_allowed) {
                burner(burn_state, dt);

                // If we were unsuccessful, update the failure count.

                if (!burn_state.success) {
                    burn_fault_flag = 1;
                }

                // update the state data.
#ifdef NSE_NET
                cons_new(i,j,k,UPROTON_MU) = burn_state.mu_p;
                cons_new(i,j,k,UNEUTRON_MU) = burn_state.mu_n;
#endif
                cons_new(i,j,k,UEDEN) = burn_state.y[SEDEN];
                cons_new(i,j,k,UEINT) = burn_state.y[SEINT];
                for (int n = 0; n < NumSpec; n++) {
                    cons_new(i,j,k,UFS+n) = burn_state.y[SFS+n];
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                    cons_new(i,j,k,UFX+n) = burn_state.y[SFX+n];
                }
#endif

                if (reaction_source_arr.contains(i,j,k)) {
                    // store the reaction data for the plotfile.
                    // Note, we want to just capture the reaction
                    // portion here, so we subtract off the advective
                    // part.

                    // rho enuc
                    reaction_source_arr(i,j,k,0) = (cons_new(i,j,k,UEINT) - Conserved_old(i,j,k,UEINT)) * invTimestep - burn_state.ydot_a[SEINT];

                    if (keep_omegadot) {
                        // rho omegadot_k
                        for (int n = 0; n < NumSpec; ++n) {
                            reaction_source_arr(i,j,k,1+n) = (cons_new(i,j,k,UFS+n) - Conserved_old(i,j,k,UFS+n)) * invTimestep - burn_state.ydot_a[SFS+n];
                        }
#if NAUX_NET > 0
                        // rho auxdot_k
                        for (int n = 0; n < NumAux; ++n) {
                            reaction_source_arr(i,j,k,1+n+NumSpec) = (cons_new(i,j,k,UFX+n) - Conserved_old(i,j,k,UFX+n)) * invTimestep - burn_state.ydot_a[SFX+n];
                        }
#endif
                    }

                    // burn weights

                    if (keep_burn_weights) {

                         if (integrator_rp::jacobian == 1) {
                             weights(i,j,k,sdc_pass_number) = amrex::max(1.0_rt, static_cast<Real>(burn_state.n_rhs + 2 * burn_state.n_jac));
                         } else {
                             // the RHS evals for the numerical differencing in the Jacobian are already accounted for in n_rhs
                             weights(i,j,k,sdc_pass_number) = amrex::max(1.0_rt, static_cast<Real>(burn_state.n_rhs));
                         }
                     }
#ifdef NSE
                    if (keep_omegadot == 1) {
                        reaction_source_arr(i,j,k,NumSpec+NumAux+1) = burn_state.nse;
                    }
                    else {
                        reaction_source_arr(i,j,k,1) = burn_state.nse;
                    }
#endif
                 }

            }

            // Convert the updated state (with the contribution from burning) to primitive data.

            Array1D<Real, 0, NQ-1> q_new;

            hydro::cell_to_primitive(i, j, k, cons_new,
#ifdef MHD
                                             Bx, By, Bz,
#endif
                                             q_new, qthermo_dummy, q_new.len() == NQ);

            // Compute the reaction source term.

            // I_q = (q^{n+1} - q^n) / dt - A(q)
            //
            // but A(q) = (q* - q^n) / dt -- that's the effect of advection w/o burning
            //
            // so I_q = (q^{n+1} - q*) / dt

            if (hydrosim::include_react_in_sdc_advect) {
                for (int n = 0; n < NQ; ++n) {
                    I(i,j,k,n) = (q_new(n) - prim_pre_react(n)) * invTimestep;
                }
            }
            else {
                for (int n = 0; n < NQ; ++n) {
                    I(i,j,k,n) = 0.0_rt;
                }
            }

#ifdef AMREX_USE_GPU
            if (burn_fault_flag) {
                Gpu::Atomic::Add(device_fail_count_ptr, burn_fault_flag);
            }
#else
            num_failed += burn_fault_flag;
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
    num_failed = *(device_fail_count.copyToHost());
#endif

    burn_outcome = !num_failed;

    ParallelDescriptor::ReduceIntMin(burn_outcome);

    if (ng > 0) {
        S_new.FillBoundary(geom.periodicity());
    }

    Real cur_time = get_state_data(Burn_Corr_Type).curTime();
    AmrLevel::FillPatch(*this, SDC_burn_increment, SDC_burn_increment.nGrow(), cur_time, Burn_Corr_Type, 0, SDC_burn_increment.nComp());

    if (log_source_changes) {

        Real energy_added = reactions.sum(0);

        if (energy_added != 0.0)
            amrex::Print() << "... (rho e) added from burning: " << energy_added * dt << std::endl << std::endl;

    }

    if (verbose) {

        amrex::Print() << "... Leaving burner on level " << level << " after completing full timestep of burning." << std::endl << std::endl;

        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;
        amrex::Real level_real = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time, IOProc);

        amrex::Print() << "HydroSim::burn_over_state() time = " << run_time
                       << " on level " << level_real << std::endl << std::endl;
#ifdef BL_LAZY
        });
#endif

    }

    return burn_outcome;

}
#endif


bool
HydroSim::zones_are_burnable(const MultiFab& State)
{

    // The default values of the limiters are 0 and 1.e200, respectively.

    Real small = 1.e-10;
    Real large = 1.e199;

    // Check whether we are limiting on either rho or T.

    bool limit_dens_lower = burn_rho_min >= small;
    bool limit_dens_upper = burn_rho_max <= large;

    bool limit_dens = limit_dens_lower || limit_dens_upper;

    bool limit_temp_lower = burn_temp_min >= small;
    bool limit_temp_upper = burn_temp_max <= large;

    bool limit_temp = limit_temp_lower || limit_temp_upper;

    bool limit = limit_dens || limit_temp;

    if (!limit) {
      return true;
    }

    // if we're limiting on rho, collect the minimum and/or maximum

    amrex::Vector<Real> lower_limit_values;
    amrex::Vector<Real> upper_limit_values;

    bool local = true;

    Real dens_lower = small;
    Real dens_upper = large;

    if (limit_dens_lower) {
      dens_lower = State.min(URHO, 0, local);
      lower_limit_values.push_back(dens_lower);
    }

    if (limit_dens_upper) {
      dens_upper = State.max(URHO, 0, local);
      upper_limit_values.push_back(dens_upper);
    }

    // if we're limiting on T, collect the minimum and/or maximum

    Real temp_lower = small;
    Real temp_upper = large;

    if (limit_temp_lower) {
      temp_lower = State.min(UTEMP, 0, local);
      lower_limit_values.push_back(temp_lower);
    }

    if (limit_temp_upper) {
      temp_upper = State.max(UTEMP, 0, local);
      upper_limit_values.push_back(temp_upper);
    }

    // Now do the reductions. We're being careful here
    // to limit the amount of work and communication,
    // because regularly doing this check only makes sense
    // if it is negligible compared to the amount of work
    // needed to just do the burn as normal.

    int small_size = static_cast<int>(lower_limit_values.size());

    if (small_size > 0) {
        amrex::ParallelDescriptor::ReduceRealMin(lower_limit_values.dataPtr(), small_size);

        if (limit_dens_lower) {
            dens_lower = lower_limit_values[0];
            if (limit_temp_lower) {
                temp_lower = lower_limit_values[1];
            }
        } else {
            temp_lower = lower_limit_values[0];
        }
    }

    int large_size = static_cast<int>(upper_limit_values.size());

    if (large_size > 0) {
        amrex::ParallelDescriptor::ReduceRealMax(upper_limit_values.dataPtr(), large_size);

        if (limit_dens_upper) {
            dens_upper = upper_limit_values[0];
            if (limit_temp_upper) {
                temp_upper = upper_limit_values[1];
            }
        } else {
            temp_upper = upper_limit_values[0];
        }
    }

    // Finally check on whether min <= rho <= max
    // and min <= T <= max. The defaults are small
    // and large respectively, so if the limiters
    // are not on, these checks will not be triggered.

    if (dens_upper >= burn_rho_min &&
        dens_lower <= burn_rho_max &&
        temp_upper >= burn_temp_min &&
        temp_lower <= burn_temp_max) {
        return true;
    }

    // If we got to this point, we did not survive the limiters,
    // so there are no zones to burn.

    if (verbose > 1) {
        amrex::Print() << "  No valid zones to burn, skipping burn_over_state()." << std::endl;
    }

    return false;

}

#endif
