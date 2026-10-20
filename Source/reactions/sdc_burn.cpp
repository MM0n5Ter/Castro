#include <HydroSim.H>
#include <primitive_conversion.H>
#ifdef MODEL_PARSER
#include <model_parser.H>
#endif
#include <burn_state_fill.H>

using std::string;
using namespace amrex;

#ifndef TRUE_SDC
#ifdef SIMPLIFIED_SDC
int
HydroSim::burn_over_state(Real time, Real dt)
{

    amrex::ignore_unused(time);



    BL_PROFILE("HydroSim::burn_over_state()");


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


    bool exclude_covered_zones = false;

    if (level < parent->finestLevel() && parent->subcyclingMode() == "None") {
        exclude_covered_zones = true;
    }

    MultiFab scratch_cover_mask;
    const MultiFab& fine_cover_mf = exclude_covered_zones ? getLevel(level+1).make_fine_mask() : scratch_cover_mask;


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

            bool burn_allowed = true;
            burn_state.success = true;
            int burn_fault_flag = 0;


#ifdef SHOCK_VAR
            if (cons_new(i,j,k,UFLAG_SHOCK) > 0.0_rt && skip_burning_in_shocks == 1) {
                burn_allowed = false;
            }
#endif


            if (exclude_covered_zones && mask.contains(i,j,k)) {
                if (mask(i,j,k) == 0.0_rt) {
                    burn_allowed = false;
                }
            }


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


            if (Conserved_old(i,j,k,UTEMP) < hydrosim::burn_temp_min || Conserved_old(i,j,k,UTEMP) > hydrosim::burn_temp_max ||
                Conserved_old(i,j,k,URHO) < hydrosim::burn_rho_min || Conserved_old(i,j,k,URHO) > hydrosim::burn_rho_max) {
                burn_allowed = false;
            }


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


            Array1D<Real, 0, NQ-1> prim_pre_react;
            Array1D<Real, 0, NQEXTRAQ-1> qthermo_dummy;

            hydro::cell_to_primitive(i, j, k, cons_new,
#ifdef MHD
                                             Bx, By, Bz,
#endif
                                             prim_pre_react, qthermo_dummy, prim_pre_react.len() == NQ);


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


                if (!burn_state.success) {
                    burn_fault_flag = 1;
                }

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

                    reaction_source_arr(i,j,k,0) = (cons_new(i,j,k,UEINT) - Conserved_old(i,j,k,UEINT)) * invTimestep - burn_state.ydot_a[SEINT];

                    if (keep_omegadot) {
                        for (int n = 0; n < NumSpec; ++n) {
                            reaction_source_arr(i,j,k,1+n) = (cons_new(i,j,k,UFS+n) - Conserved_old(i,j,k,UFS+n)) * invTimestep - burn_state.ydot_a[SFS+n];
                        }
#if NAUX_NET > 0
                        for (int n = 0; n < NumAux; ++n) {
                            reaction_source_arr(i,j,k,1+n+NumSpec) = (cons_new(i,j,k,UFX+n) - Conserved_old(i,j,k,UFX+n)) * invTimestep - burn_state.ydot_a[SFX+n];
                        }
#endif
                    }


                    if (keep_burn_weights) {

                         if (integrator_rp::jacobian == 1) {
                             weights(i,j,k,sdc_pass_number) = amrex::max(1.0_rt, static_cast<Real>(burn_state.n_rhs + 2 * burn_state.n_jac));
                         } else {
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


            Array1D<Real, 0, NQ-1> q_new;

            hydro::cell_to_primitive(i, j, k, cons_new,
#ifdef MHD
                                             Bx, By, Bz,
#endif
                                             q_new, qthermo_dummy, q_new.len() == NQ);



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
#endif
