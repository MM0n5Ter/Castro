#ifndef SDC_NEWTON_SOLVE_H
#define SDC_NEWTON_SOLVE_H

#include <sdc_react_util.H>
#include <linpack.H>

// error codes
namespace newton_status {
    enum status : std::int8_t {
        success = 0,
        singular_matrix = -1,
        convergence_failure = -2,
        bad_mass_fractions = -3
    };

}

namespace newton {
    constexpr Real species_failure_tolerance = 1.e-2_rt;
    constexpr int MAX_ITER = 100;
    constexpr int MAX_NSUB = 64;

    // outcome of a zone solve: the weighted norm of the last Newton
    // correction and the newton_status code

    struct zone_solve {
        Real err;
        int ierr;
    };

#ifdef REACTIONS

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    f_sdc_jac(const Real dt_m,
              burn_t& burn_state,
              Array1D<Real, 1, NumSpec+1>& f,
              JacNetArray2D& Jac) {

        // This is used with the Newton solve and returns f and the
        // Jacobian

        // single_zone_react_source will use burn_state.y[] to fill
        // burn_state.rho, burn_state.e, and burn_state.xn[] and then call
        // the EOS to get the temperature.

        const auto R_full = single_zone_react_source(burn_state);

        // we are solving J dU = -f
        // where f is evaluated with the current guess for U
        // note: we store -f

        for (int n = 1; n <= NumSpec; ++n) {
            f(n) = -burn_state.y[SFS-1+n] +
                dt_m * R_full[USPEC-1+n] + burn_state.ydot_a[SFS-1+n];
        }
        f(NumSpec+1) = -burn_state.y[SEINT] +
            dt_m * R_full[UEINT] + burn_state.ydot_a[SEINT];

        // get the Jacobian.

        // Instead of the decomposition into dw/dU and dR/dw, we instead use the form from
        // the simplified-SDC.  Note: we are
        // not including density anymore.

        // note: single_zone_jac assumes that the burn_state is in sync
        // already with the current solution guess.  In this case, that
        // was done in single_zone_react_source.

        single_zone_jac(burn_state, dt_m, Jac);

        // Our Jacobian has the form:  J = I - dt dR/dw dwdU
        // so now we fix that

        Jac.mul(-dt_m);
        Jac.add_identity();
    }


    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    zone_solve
    sdc_newton_solve(GpuArray<Real, NUM_STATE> const& U_old,
                     GpuArray<Real, NUM_STATE> const& C,
                     const Real dt_m,
                     const int sdc_iteration,
                     const Real err_prev,
                     GpuArray<Real, NUM_STATE> & U_new) {

        // the purpose of this function is to solve the system
        //  U - dt R(U) = U_old + dt C
        // using a Newton solve.
        //
        // here, U_new should come in as a guess for the new U for
        // iterations > 0, it will be the solution from the previous
        // iteration initially.
        //
        // upon exit, U_new will be returned with the value that satisfied
        // the nonlinear function.  The error norm is only measured once
        // the linear system has been solved, so if the Jacobian is
        // singular we return err_prev unchanged


        amrex::ignore_unused(sdc_iteration);

        JacNetArray2D Jac;

        // we will do the implicit update of only the terms that
        // have reactive sources
        //
        // 1:NumSpec : species
        // NumSpec+1 : (rho e)

        Array1D<Real, 1, NumSpec+1> f;

        zone_solve result{err_prev, newton_status::success};

        // update the density and momenta for this zone -- they don't react

        U_new[URHO] = U_old[URHO] + dt_m * C[URHO];

        for (int n = 0; n < 3; ++n) {
            U_new[UMX+n] = U_old[UMX+n] + dt_m * C[UMX+n];
        }

        burn_t burn_state;

        copy_cons_to_burn_type(U_new, burn_state);
        burn_state.rho = U_new[URHO];

        // for the Jacobian solve, we are solving
        //   f(U) = U - dt R(U) - U_old - dt C = 0
        // we define f_source = U_old + dt C so we are solving
        //   f(U) = U - dt R(U) - f_source = 0
        //
        // we'll store this in the burn_t ydot_a[]

        for (int n = 0; n < NumSpec; ++n) {
            burn_state.ydot_a[SFS + n] = U_old[USPEC + n] + dt_m * C[USPEC + n];
        }
        burn_state.ydot_a[SEINT] = U_old[UEINT] + dt_m * C[UEINT];

        // do a simple Newton solve

        // iterative loop
        int iter = 0;

        Real err = 1.e30_rt;
        auto converged = false;

        while (!converged && iter < MAX_ITER) {

            // burn_state.y[] will always contain the current guess
            // for the solution only the species and internal energy
            // are updated though

            // initial guess
            burn_state.T = U_old[UTEMP];

            int info = 0;
            f_sdc_jac(dt_m, burn_state, f, Jac);

            // solve the linear system: Jac dU = -f
#ifdef NEW_NETWORK_IMPLEMENTATION
            info = RHS::dgefa(Jac);
#else
            IArray1D ipvt;
            constexpr bool allow_pivot{true};
            dgefa<NumSpec+1, allow_pivot>(Jac, ipvt, info);
#endif
            if (info != 0) {
                result.ierr = newton_status::singular_matrix;
                return result;
            }

#ifdef NEW_NETWORK_IMPLEMENTATION
            RHS::dgesl(Jac, f);
#else
            dgesl<NumSpec+1, allow_pivot>(Jac, ipvt, f);
#endif

            // on output, f is the solution (dU)
            for (int n = 0; n < NumSpec; ++n) {
                burn_state.y[SFS+n] += f(n + 1);
            }
            burn_state.y[SEINT] += f(NumSpec+1);

            // compute the norm of the weighted error, where the
            // weights are 1/eps_tot

            auto err_sum = 0.0_rt;
            for (int n = 1; n <= NumSpec+1; ++n) {
                Real eps;
                if (n < NumSpec+1) {
                    // for species, atol is the mass fraction limit, so we
                    // multiply by density to get a partial density limit
                    eps = integrator_rp::rtol_spec * std::abs(burn_state.y[SFS-1+n]) +
                        integrator_rp::atol_spec * std::abs(U_new[URHO]);
                } else {
                    eps = integrator_rp::rtol_enuc * std::abs(burn_state.y[SEINT]) +
                        integrator_rp::atol_enuc;
                }
                err_sum += f(n) * f(n) / (eps * eps);
            }
            err = std::sqrt(err_sum / static_cast<Real>(NumSpec+1));

            if (err < 1.0_rt) {
                converged = true;
            }
            iter++;
        }

        result.err = err;

        if (!converged) {
            result.ierr = newton_status::convergence_failure;
            return result;
        }

        // update the full U_new

        for (int n = 0; n < NumSpec; ++n) {
            U_new[USPEC+n] = burn_state.y[SFS+n];
        }

        U_new[UEINT] = burn_state.y[SEINT];

        // we want to do a conservative update for (rho E), so first
        // figure out the energy generation rate

        Real rho_Sdot = (U_new[UEINT] - U_old[UEINT]) / dt_m - C[UEINT];

        U_new[UEDEN] = U_old[UEDEN] + dt_m * (C[UEDEN] + rho_Sdot);

        return result;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    zone_solve
    sdc_newton_subdivide(GpuArray<Real, NUM_STATE> const& U_old,
                         GpuArray<Real, NUM_STATE> const& C,
                         const Real dt_m,
                         const int sdc_iteration,
                         GpuArray<Real, NUM_STATE>& U_new) {

        // This is the driver for solving the nonlinear update for the
        // reating/advecting system using Newton's method. It attempts
        // to do the solution for the full dt_m requested, but if it
        // fails, will subdivide the domain until it converges or
        // reaches our limit on the number of subintervals.

        GpuArray<Real, NUM_STATE> U_begin;

        // subdivide the timestep and do multiple Newtons.

        int nsub = 1;
        zone_solve result{0.0_rt, newton_status::convergence_failure};

        while (nsub <= MAX_NSUB && result.ierr != newton_status::success) {

            // For the first interval, we set the starting state to
            // the old-time solution.

            for (int n = 0; n < NUM_STATE; ++n) {
                U_begin[n] = U_old[n];
            }

            // We come in here with an initial guess for the new
            // solution stored in U_new. That only really makes sense
            // for the case where we have 1 substep. Otherwise, we
            // should just use the previous subinterval solution.

            if (nsub > 1) {
                for (int n = 0; n < NUM_STATE; ++n) {
                    U_new[n] = U_old[n];
                }
            }
            Real dt_sub = dt_m / nsub;

            for (int isub = 0; isub < nsub; ++isub) {

                // integrate from U_begin to U_new over a step dt_m / nsub

                // normalize species -- we are working with rho X_k here
                Real sum_rhoX = 0.0_rt;
                for (int n = 0; n < NumSpec; ++n) {
                    U_begin[USPEC + n] = amrex::max(U_begin[URHO] * network_rp::small_x, U_begin[USPEC + n]);
                    sum_rhoX += U_begin[USPEC + n];
                }
                for (int n = 0; n < NumSpec; ++n) {
                    U_begin[USPEC + n] *= U_begin[URHO] / sum_rhoX;
                }

                result = sdc_newton_solve(U_begin, C, dt_sub, sdc_iteration, result.err, U_new);

                // our solve may have resulted in mass fractions outside
                // of [0, 1] -- reject if this is the case
                for (int n = 0; n < NumSpec; ++n) {
                    if (U_new[USPEC+n] < -species_failure_tolerance * U_new[URHO] ||
                        U_new[USPEC+n] > (1.0_rt + species_failure_tolerance) * U_new[URHO]) {
                        result.ierr = newton_status::bad_mass_fractions;
                    }
                }

                if (result.ierr != newton_status::success) {
                    // no point in continuing this subdivision if one stage failed
                    break;
                }

                for (int n = 0; n < NUM_STATE; ++n) {
                    U_begin[n] = U_new[n];
                }
            }
            nsub *= 2;
        }

        return result;
    }
#endif
}

#endif
