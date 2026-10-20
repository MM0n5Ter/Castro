#ifndef IMPLICIT_BURN_SOLVERS_H
#define IMPLICIT_BURN_SOLVERS_H

#include <burn_rhs.H>
#include <linpack.H>
#include <network.H>

#include <burn_type.H>

#include <extern_parameters.H>

#ifdef NEW_NETWORK_IMPLEMENTATION
#include <rhs.H>
#endif

#ifndef NEW_NETWORK_IMPLEMENTATION
#include <actual_network.H>

#include <actual_rhs.H>
#endif

#include <burn_state_fill.H>

#include <burner.H>
#include <hybrj.H>

#include <hybrj_type.H>

namespace newton_outcome {
    enum status : std::int8_t {
        success = 0,
        singular_jacobian = -1,
        no_convergence = -2,
        species_fraction_violation = -3
    };
}

#ifdef REACTIONS
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
begin_implicit_burn(const Real substep_dt,
                    GpuArray<Real, NUM_STATE> const& Conserved_old,
                    GpuArray<Real, NUM_STATE> & cons_new,
                    GpuArray<Real, NUM_STATE> const& C,
                    burn_t& burn_state) {

    cons_new[URHO] = Conserved_old[URHO] + substep_dt * C[URHO];

    for (int n = 0; n < 3; ++n) {
        cons_new[UMX+n] = Conserved_old[UMX+n] + substep_dt * C[UMX+n];
    }

    fill_burn_type_from_conserved(cons_new, burn_state);
    burn_state.rho = cons_new[URHO];

    for (int n = 0; n < NumSpec; ++n) {
        burn_state.ydot_a[SFS + n] = Conserved_old[UFS + n] + substep_dt * C[UFS + n];
    }
    burn_state.ydot_a[SEINT] = Conserved_old[UEINT] + substep_dt * C[UEINT];
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
finish_implicit_burn(const Real substep_dt,
                     GpuArray<Real, NUM_STATE> const& Conserved_old,
                     GpuArray<Real, NUM_STATE> & cons_new,
                     GpuArray<Real, NUM_STATE> const& C) {

    Real rho_Sdot = (cons_new[UEINT] - Conserved_old[UEINT]) / substep_dt - C[UEINT];

    cons_new[UEDEN] = Conserved_old[UEDEN] + substep_dt * (C[UEDEN] + rho_Sdot);
}
#endif

namespace newton {
    constexpr int MAX_ITER = 100;
    constexpr int MAX_SUBSTEPS = 64;
    constexpr Real species_failure_tolerance = 1.e-2_rt;

#ifdef REACTIONS
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    residual_and_jacobian(const Real substep_dt,
              burn_t& burn_state,
              Array1D<Real, 1, NumSpec+1>& f,
              JacNetArray2D& Jac) {



        GpuArray<Real, NUM_STATE> React_full;

        zone_react_source(burn_state, React_full);


        for (int n = 1; n <= NumSpec; ++n) {
            f(n) = -burn_state.y[SFS-1+n] +
                substep_dt * React_full[UFS-1+n] + burn_state.ydot_a[SFS-1+n];
        }
        f(NumSpec+1) = -burn_state.y[SEINT] +
            substep_dt * React_full[UEINT] + burn_state.ydot_a[SEINT];




        zone_jacobian(burn_state, substep_dt, Jac);


        Jac.mul(-substep_dt);
        Jac.add_identity();
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    int
    deferred_newton_solve(const Real substep_dt,
                     GpuArray<Real, NUM_STATE> const& Conserved_old,
                     GpuArray<Real, NUM_STATE> & cons_new,
                     GpuArray<Real, NUM_STATE> const& C,
                     const int correction_iter,
                     Real& err_out) {



        amrex::ignore_unused(correction_iter);

        JacNetArray2D Jac;


        Array1D<Real, 1, NumSpec+1> f;


        burn_t burn_state;

        begin_implicit_burn(substep_dt, Conserved_old, cons_new, C, burn_state);


        int iter = 0;

        Real err = 1.e30_rt;
        bool converged = false;

        while (!converged && iter < MAX_ITER) {


            burn_state.T = Conserved_old[UTEMP];

            int info = 0;
            residual_and_jacobian(substep_dt, burn_state, f, Jac);

#ifdef NEW_NETWORK_IMPLEMENTATION
            info = RHS::dgefa(Jac);
#else
            IArray1D pivot_index;
            constexpr bool allow_pivot{true};
            dgefa<NumSpec+1, allow_pivot>(Jac, pivot_index, info);
#endif
            if (info != 0) {
                return newton_outcome::singular_jacobian;
            }

#ifdef NEW_NETWORK_IMPLEMENTATION
            RHS::dgesl(Jac, f);
#else
            dgesl<NumSpec+1, allow_pivot>(Jac, pivot_index, f);
#endif

            for (int n = 0; n < NumSpec; ++n) {
                burn_state.y[SFS+n] += f(n + 1);
            }
            burn_state.y[SEINT] += f(NumSpec+1);


            auto weighted_error_sum = 0.0_rt;
            for (int n = 1; n <= NumSpec+1; ++n) {
                Real eps;
                if (n < NumSpec+1) {
                    eps = integrator_rp::rtol_spec * std::abs(burn_state.y[SFS-1+n]) +
                        integrator_rp::atol_spec * std::abs(cons_new[URHO]);
                } else {
                    eps = integrator_rp::rtol_enuc * std::abs(burn_state.y[SEINT]) +
                        integrator_rp::atol_enuc;
                }
                weighted_error_sum += f(n) * f(n) / (eps * eps);
            }
            err = std::sqrt(weighted_error_sum / static_cast<Real>(NumSpec+1));

            if (err < 1.0_rt) {
                converged = true;
            }
            iter++;
        }

        err_out = err;

        if (!converged) {
            return newton_outcome::no_convergence;
        }


        for (int n = 0; n < NumSpec; ++n) {
            cons_new[UFS+n] = burn_state.y[SFS+n];
        }

        cons_new[UEINT] = burn_state.y[SEINT];


        finish_implicit_burn(substep_dt, Conserved_old, cons_new, C);

        return newton_outcome::success;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    int
    deferred_newton_subdivide(const Real substep_dt,
                         GpuArray<Real, NUM_STATE> const& Conserved_old,
                         GpuArray<Real, NUM_STATE>& cons_new,
                         GpuArray<Real, NUM_STATE> const& C,
                         const int correction_iter,
                         Real& err_out) {


        GpuArray<Real, NUM_STATE> State_begin;


        int nsub = 1;
        int ierr = newton_outcome::no_convergence;

        while (nsub <= MAX_SUBSTEPS && ierr != newton_outcome::success) {


            for (int n = 0; n < NUM_STATE; ++n) {
                State_begin[n] = Conserved_old[n];
            }


            if (nsub > 1) {
                for (int n = 0; n < NUM_STATE; ++n) {
                    cons_new[n] = Conserved_old[n];
                }
            }
            Real dt_sub = substep_dt / nsub;

            for (int isub = 0; isub < nsub; ++isub) {


                Real species_density_sum = 0.0_rt;
                for (int n = 0; n < NumSpec; ++n) {
                    State_begin[UFS + n] = amrex::max(State_begin[URHO] * network_rp::small_x, State_begin[UFS + n]);
                    species_density_sum += State_begin[UFS + n];
                }
                for (int n = 0; n < NumSpec; ++n) {
                    State_begin[UFS + n] *= State_begin[URHO] / species_density_sum;
                }

                ierr = deferred_newton_solve(dt_sub, State_begin, cons_new, C, correction_iter, err_out);

                for (int n = 0; n < NumSpec; ++n) {
                    if (cons_new[UFS+n] < -species_failure_tolerance * cons_new[URHO] ||
                        cons_new[UFS+n] > (1.0_rt + species_failure_tolerance) * cons_new[URHO]) {
                        ierr = newton_outcome::species_fraction_violation;
                    }
                }

                if (ierr != newton_outcome::success) {
                    break;
                }

                for (int n = 0; n < NUM_STATE; ++n) {
                    State_begin[n] = cons_new[n];
                }
            }
            nsub *= 2;
        }

        return ierr;
    }
#endif
}

namespace powell {
#ifdef REACTIONS
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    constraint_jacobian(amrex::Array1D<amrex::Real, 1, NumSpec+1>& x,
            JacNetArray2D& Jac,
            const burn_t& burn_state_in,
            int& iflag) {

        amrex::ignore_unused(iflag);

        auto burn_state = burn_state_in;

        for (int n = 0; n < NumSpec; ++n) {
            burn_state.y[SFS+n] = x(n+1);
        }
        burn_state.y[SEINT] = x(NumSpec+1);

#if NAUX_NET > 0
        amrex::Error("error: aux data not currently supported in true SDC");
#endif

        {
            GpuArray<Real, NUM_STATE> React_full;
            zone_react_source(burn_state, React_full);
        }

        auto substep_dt = burn_state.time;

        zone_jacobian(burn_state, substep_dt, Jac);

        Jac.mul(-substep_dt);
        Jac.add_identity();

    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    Constraint_residual(amrex::Array1D<amrex::Real, 1, NumSpec+1>& x,
          Array1D<Real, 1, NumSpec+1>& Z,
          const burn_t& burn_state_in,
          int& iflag) {

        amrex::ignore_unused(iflag);

        GpuArray<Real, NUM_STATE> React_full;

        auto burn_state = burn_state_in;

        for (int n = 0; n < NumSpec; ++n) {
            burn_state.y[SFS+n] = x(n+1);
        }
        burn_state.y[SEINT] = x(NumSpec+1);

        zone_react_source(burn_state, React_full);

        auto substep_dt = burn_state.time;

        for (int n = 1; n <= NumSpec; ++n) {
            Z(n) = burn_state.y[SFS-1+n] -
                substep_dt * React_full[UFS-1+n] - burn_state.ydot_a[SFS-1+n];
        }
        Z(NumSpec+1) = burn_state.y[SEINT] -
            substep_dt * React_full[UEINT] - burn_state.ydot_a[SEINT];

    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    deferred_dogleg_solve(const Real substep_dt,
                     GpuArray<Real, NUM_STATE> const& Conserved_old,
                     GpuArray<Real, NUM_STATE> & cons_new,
                     GpuArray<Real, NUM_STATE> const& C,
                     const int correction_iter) {

        amrex::ignore_unused(correction_iter);

        burn_t burn_state;

        begin_implicit_burn(substep_dt, Conserved_old, cons_new, C, burn_state);

        burn_state.time = substep_dt;

        hybrj_t<NumSpec+1, JacNetArray2D> hj;
        hj.xtol = 1.e-5;
        hj.mode = 1;

        for (int n = 0; n < NumSpec; ++n) {
            hj.x(n+1) = cons_new[UFS+n];
        }
        hj.x(NumSpec+1) = cons_new[UEINT];

        hybrj<NumSpec+1, burn_t>(hj, burn_state, Constraint_residual, constraint_jacobian);

        if (hj.info != 1) {
            std::cout << "failure in the Powell solver, error code is: " << hj.info << std::endl;
            amrex::Error("hybrid Powell solve failed");
        }

        for (int n = 0; n < NumSpec; ++n) {
            cons_new[UFS+n] = hj.x(n+1);
        }

        cons_new[UEINT] = hj.x(NumSpec+1);

        finish_implicit_burn(substep_dt, Conserved_old, cons_new, C);

    }
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
deferred_stiff_solve(const Real substep_dt,
               GpuArray<Real, NUM_STATE> const& Conserved_old,
               GpuArray<Real, NUM_STATE>& cons_new,
               GpuArray<Real, NUM_STATE> const& C,
               const int correction_iter) {

    cons_new[URHO] = Conserved_old[URHO] + substep_dt * C[URHO];

    for (int n = 0; n < 3; ++n) {
        cons_new[UMX+n] = Conserved_old[UMX+n] + substep_dt * C[UMX+n];
    }

    burn_t burn_state;

    burn_state.success = true;

    fill_burn_type_from_conserved(Conserved_old, burn_state);

    burn_state.T = Conserved_old[UTEMP];

    burn_state.T_fixed = -1.e30_rt;
    burn_state.rho = burn_state.y[SRHO];

    burn_state.ydot_a[SRHO] = C[URHO];
    burn_state.ydot_a[SMX] = C[UMX];
    burn_state.ydot_a[SMY] = C[UMY];
    burn_state.ydot_a[SMZ] = C[UMZ];
    burn_state.ydot_a[SEDEN] = C[UEDEN];
    burn_state.ydot_a[SEINT] = C[UEINT];
    for (int n = 0; n < NumSpec; n++) {
        burn_state.ydot_a[SFS+n] = C[UFS+n];
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
        burn_state.ydot_a[SFX+n] = C[UFX+n];
    }
#endif

    burn_state.sdc_iter = correction_iter;
    burn_state.num_sdc_iters = sdc_accuracy_order + sdc_extra_passes;

    burner(burn_state, substep_dt);

    if (! burn_state.success) {
        amrex::Error("Error: integration failed");
    }

    cons_new[UEDEN] = burn_state.y[SEDEN];
    cons_new[UEINT] = burn_state.y[SEINT];
    for (int n = 0; n < NumSpec; n++) {
        cons_new[UFS+n] = burn_state.y[SFS+n];
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
        cons_new[UFX+n] = burn_state.y[SFX+n];
    }
#endif

    cons_new[UTEMP] = Conserved_old[UTEMP];

}

#endif
