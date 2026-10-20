#include <HydroSim.H>
#include <primitive_conversion.H>
#ifdef MODEL_PARSER
#include <model_parser.H>
#endif
#include <burn_state_fill.H>

using std::string;
using namespace amrex;

#ifndef TRUE_SDC
static AMREX_GPU_HOST_DEVICE AMREX_INLINE
void store_burned_zone (Array4<Real> const& U, int i, int j, int k, const burn_t& burn_state)
{
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
}

bool
HydroSim::zones_are_burnable(const MultiFab& State)
{

    Real small = 1.e-10;
    Real large = 1.e199;

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

    if (dens_upper >= burn_rho_min &&
        dens_lower <= burn_rho_max &&
        temp_upper >= burn_temp_min &&
        temp_lower <= burn_temp_max) {
        return true;
    }

    if (verbose > 1) {
        amrex::Print() << "  No valid zones to burn, skipping burn_over_state()." << std::endl;
    }

    return false;

}

int
HydroSim::burn_over_state(MultiFab& s, MultiFab& r, Real time, Real dt, const int strang_half)
{

    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::burn_over_state()");

    if (time_scheme != CornerTrace) {
        amrex::Error("Strang reactions are only supported for the CTU advance.");
    }

    const Real strt_time = ParallelDescriptor::second();

    int burn_outcome = 1;

    if (do_react != 1) {

        r.setVal(0.0, r.nGrow());

        return burn_outcome;

    }

    if (!zones_are_burnable(s)) {

        r.setVal(0.0, r.nGrow());

        return burn_outcome;

    }

    const int ng = s.nGrow();

    if (verbose) {
        amrex::Print() << "... Entering burner on level " << level << " and doing half-timestep of burning." << std::endl << std::endl;
    }

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

            bool burn_allowed = true;
            burn_state.success = true;
            int burn_fault_flag = 0;

#ifdef SHOCK_VAR
            if (U(i,j,k,UFLAG_SHOCK) > 0.0_rt && skip_burning_in_shocks == 1) {
                burn_allowed = false;
            }
#endif

            if (exclude_covered_zones && mask.contains(i,j,k)) {
                if (mask(i,j,k) == 0.0_rt) {
                    burn_allowed = false;
                }
            }

            Real rhoInv = 1.0_rt / U(i,j,k,URHO);

            burn_state.rho = U(i,j,k,URHO);

            burn_state.e = U(i,j,k,UEINT) * rhoInv;

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

            burn_state.n_rhs = 0;
            burn_state.n_jac = 0;

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

            if (burn_state.T < hydrosim::burn_temp_min || burn_state.T > hydrosim::burn_temp_max ||
                burn_state.rho < hydrosim::burn_rho_min || burn_state.rho > hydrosim::burn_rho_max) {
                burn_allowed = false;
            }

            if (burn_allowed) {
                burner(burn_state, dt);

                if (!burn_state.success) {
                    burn_fault_flag = 1;
                }

                if (reactions.contains(i,j,k)) {

                    reactions(i,j,k,0) = (U(i,j,k,URHO) * burn_state.e - U(i,j,k,UEINT)) / dt;

                    if (keep_omegadot == 1) {
                        for (int n = 0; n < NumSpec; ++n) {
                            reactions(i,j,k,1+n) = U(i,j,k,URHO) * (burn_state.xn[n] - U(i,j,k,UFS+n) * rhoInv) / dt;
                        }
#if NAUX_NET > 0
                        for (int n = 0; n < NumAux; ++n) {
                            reactions(i,j,k,1+n+NumSpec) = U(i,j,k,URHO) * (burn_state.aux[n] - U(i,j,k,UFX+n) * rhoInv) / dt;
                        }
#endif
                    }

                    if (keep_burn_weights) {

                        if (integrator_rp::jacobian == 1) {
                            weights(i,j,k,strang_half) = amrex::max(1.0_rt, static_cast<Real>(burn_state.n_rhs + 2 * burn_state.n_jac));
                        } else {
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

                store_burned_zone(U, i, j, k, burn_state);

            } else {

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
        Gpu::streamSynchronize();
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

step_result
HydroSim::apply_old_reactions (Real time, Real dt) {

    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    step_result status {};

#ifndef SIMPLIFIED_SDC
    int burn_outcome{1};

    MultiFab& Reactions_old = get_old_data(Burning_Type);
    MultiFab& react_new = get_new_data(Burning_Type);

    if (time_scheme != SimplifiedIterCorrection) {
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

            burn_outcome = burn_over_state(time, dt);

            if (burn_outcome != 1) {
                status.success = false;
                status.reason = "burn unsuccessful";

                return status;
            }

            clean_state(S_new, time + dt, S_new.nGrow());

            scan_for_nan(S_new);

        }
        else {

            MultiFab& SDC_burn_new = get_new_data(Burn_Corr_Type);
            SDC_burn_new.setVal(0.0, SDC_burn_new.nGrow());

            Reactions_old.setVal(0.0, Reactions_old.nGrow());
            react_new.setVal(0.0, react_new.nGrow());

        }

    }

#else

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

#endif

    return status;
}
#endif
