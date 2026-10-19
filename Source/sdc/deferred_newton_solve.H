#ifndef DEFERRED_NEWTON_SOLVE_H
#define DEFERRED_NEWTON_SOLVE_H

#include <deferred_react_util.H>
#include <linpack.H>

// error codes
namespace newton_outcome {
    enum status : std::int8_t {
        success = 0,
        singular_jacobian = -1,
        no_convergence = -2,
        species_fraction_violation = -3
    };

}

namespace newton {
    constexpr Real species_failure_tolerance = 1.e-2_rt;
    constexpr int MAX_ITER = 100;
    constexpr int MAX_SUBSTEPS = 64;

#ifdef REACTIONS

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    residual_and_jacobian(const Real substep_dt,
              burn_t& burn_state,
              Array1D<Real, 1, NumSpec+1>& f,
              JacNetArray2D& Jac) {

        // This is used with the Newton solve and returns f and the
        // Jacobian

        // zone_react_source will use burn_state.y[] to fill
        // burn_state.rho, burn_state.e, and burn_state.xn[] and then call
        // the EOS to get the temperature.

        GpuArray<Real, NUM_STATE> React_full;

        zone_react_source(burn_state, React_full);

        // we are solving J dU = -f
        // where f is evaluated with the current guess for U
        // note: we store -f

        for (int n = 1; n <= NumSpec; ++n) {
            f(n) = -burn_state.y[SFS-1+n] +
                substep_dt * React_full[UFS-1+n] + burn_state.ydot_a[SFS-1+n];
        }
        f(NumSpec+1) = -burn_state.y[SEINT] +
            substep_dt * React_full[UEINT] + burn_state.ydot_a[SEINT];

        // get the Jacobian.

        // Instead of the decomposition into dw/dU and dR/dw, we instead use the form from
        // the simplified-SDC.  Note: we are
        // not including density anymore.

        // note: zone_jacobian assumes that the burn_state is in sync
        // already with the current solution guess.  In this case, that
        // was done in zone_react_source.

        zone_jacobian(burn_state, substep_dt, Jac);

        // Our Jacobian has the form:  J = I - dt dR/dw dwdU
        // so now we fix that

        Jac.mul(-substep_dt);
        Jac.add_identity();
    }


    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    deferred_newton_solve(const Real substep_dt,
                     GpuArray<Real, NUM_STATE> const& Conserved_old,
                     GpuArray<Real, NUM_STATE> & cons_new,
                     GpuArray<Real, NUM_STATE> const& C,
                     const int correction_iter,
                     Real& err_out,
                     int& ierr) {

        // the purpose of this function is to solve the system
        //  U - dt R(U) = Conserved_old + dt C
        // using a Newton solve.
        //
        // here, cons_new should come in as a guess for the new U for
        // iterations > 0, it will be the solution from the previous
        // iteration initially.
        //
        // upon exit, cons_new will be returned with the value that satisfied
        // the nonlinear function


        amrex::ignore_unused(correction_iter);

        JacNetArray2D Jac;

        // we will do the implicit update of only the terms that
        // have reactive sources
        //
        // 1:NumSpec : species
        // NumSpec+1 : (rho e)

        Array1D<Real, 1, NumSpec+1> f;

        ierr = newton_outcome::success;

        // update the density and momenta for this zone -- they don't react

        cons_new[URHO] = Conserved_old[URHO] + substep_dt * C[URHO];

        for (int n = 0; n < 3; ++n) {
            cons_new[UMX+n] = Conserved_old[UMX+n] + substep_dt * C[UMX+n];
        }

        burn_t burn_state;

        fill_burn_type_from_conserved(cons_new, burn_state);
        burn_state.rho = cons_new[URHO];

        // for the Jacobian solve, we are solving
        //   f(U) = U - dt R(U) - Conserved_old - dt C = 0
        // we define f_source = Conserved_old + dt C so we are solving
        //   f(U) = U - dt R(U) - f_source = 0
        //
        // we'll store this in the burn_t ydot_a[]

        for (int n = 0; n < NumSpec; ++n) {
            burn_state.ydot_a[SFS + n] = Conserved_old[UFS + n] + substep_dt * C[UFS + n];
        }
        burn_state.ydot_a[SEINT] = Conserved_old[UEINT] + substep_dt * C[UEINT];

        // do a simple Newton solve

        // iterative loop
        int iter = 0;

        Real err = 1.e30_rt;
        bool converged = false;

        while (!converged && iter < MAX_ITER) {

            // burn_state.y[] will always contain the current guess
            // for the solution only the species and internal energy
            // are updated though

            // initial guess
            burn_state.T = Conserved_old[UTEMP];

            int info = 0;
            residual_and_jacobian(substep_dt, burn_state, f, Jac);

            // solve the linear system: Jac dU = -f
#ifdef NEW_NETWORK_IMPLEMENTATION
            info = RHS::dgefa(Jac);
#else
            IArray1D pivot_index;
            constexpr bool allow_pivot{true};
            dgefa<NumSpec+1, allow_pivot>(Jac, pivot_index, info);
#endif
            if (info != 0) {
                ierr = newton_outcome::singular_jacobian;
                return;
            }

#ifdef NEW_NETWORK_IMPLEMENTATION
            RHS::dgesl(Jac, f);
#else
            dgesl<NumSpec+1, allow_pivot>(Jac, pivot_index, f);
#endif

            // on output, f is the solution (dU)
            for (int n = 0; n < NumSpec; ++n) {
                burn_state.y[SFS+n] += f(n + 1);
            }
            burn_state.y[SEINT] += f(NumSpec+1);

            // compute the norm of the weighted error, where the
            // weights are 1/eps_tot

            auto weighted_error_sum = 0.0_rt;
            for (int n = 1; n <= NumSpec+1; ++n) {
                Real eps;
                if (n < NumSpec+1) {
                    // for species, atol is the mass fraction limit, so we
                    // multiply by density to get a partial density limit
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
            ierr = newton_outcome::no_convergence;
            return;
        }

        // update the full cons_new

        for (int n = 0; n < NumSpec; ++n) {
            cons_new[UFS+n] = burn_state.y[SFS+n];
        }

        cons_new[UEINT] = burn_state.y[SEINT];

        // we want to do a conservative update for (rho E), so first
        // figure out the energy generation rate

        Real rho_Sdot = (cons_new[UEINT] - Conserved_old[UEINT]) / substep_dt - C[UEINT];

        cons_new[UEDEN] = Conserved_old[UEDEN] + substep_dt * (C[UEDEN] + rho_Sdot);

    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    deferred_newton_subdivide(const Real substep_dt,
                         GpuArray<Real, NUM_STATE> const& Conserved_old,
                         GpuArray<Real, NUM_STATE>& cons_new,
                         GpuArray<Real, NUM_STATE> const& C,
                         const int correction_iter,
                         Real& err_out,
                         int& ierr) {

        // This is the driver for solving the nonlinear update for the
        // reating/advecting system using Newton's method. It attempts
        // to do the solution for the full substep_dt requested, but if it
        // fails, will subdivide the domain until it converges or
        // reaches our limit on the number of subintervals.

        GpuArray<Real, NUM_STATE> State_begin;

        // subdivide the timestep and do multiple Newtons.

        int nsub = 1;
        ierr = newton_outcome::no_convergence;

        while (nsub <= MAX_SUBSTEPS && ierr != newton_outcome::success) {

            // For the first interval, we set the starting state to
            // the old-time solution.

            for (int n = 0; n < NUM_STATE; ++n) {
                State_begin[n] = Conserved_old[n];
            }

            // We come in here with an initial guess for the new
            // solution stored in cons_new. That only really makes sense
            // for the case where we have 1 substep. Otherwise, we
            // should just use the previous subinterval solution.

            if (nsub > 1) {
                for (int n = 0; n < NUM_STATE; ++n) {
                    cons_new[n] = Conserved_old[n];
                }
            }
            Real dt_sub = substep_dt / nsub;

            for (int isub = 0; isub < nsub; ++isub) {

                // integrate from State_begin to cons_new over a step substep_dt / nsub

                // normalize species -- we are working with rho X_k here
                Real species_density_sum = 0.0_rt;
                for (int n = 0; n < NumSpec; ++n) {
                    State_begin[UFS + n] = amrex::max(State_begin[URHO] * network_rp::small_x, State_begin[UFS + n]);
                    species_density_sum += State_begin[UFS + n];
                }
                for (int n = 0; n < NumSpec; ++n) {
                    State_begin[UFS + n] *= State_begin[URHO] / species_density_sum;
                }

                deferred_newton_solve(dt_sub, State_begin, cons_new, C, correction_iter, err_out, ierr);

                // our solve may have resulted in mass fractions outside
                // of [0, 1] -- reject if this is the case
                for (int n = 0; n < NumSpec; ++n) {
                    if (cons_new[UFS+n] < -species_failure_tolerance * cons_new[URHO] ||
                        cons_new[UFS+n] > (1.0_rt + species_failure_tolerance) * cons_new[URHO]) {
                        ierr = newton_outcome::species_fraction_violation;
                    }
                }

                if (ierr != newton_outcome::success) {
                    // no point in continuing this subdivision if one stage failed
                    break;
                }

                for (int n = 0; n < NUM_STATE; ++n) {
                    State_begin[n] = cons_new[n];
                }
            }
            nsub *= 2;
        }
    }
#endif
}

#endif
