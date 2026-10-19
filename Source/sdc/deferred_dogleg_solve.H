#ifndef DEFERRED_DOGLEG_SOLVE_H
#define DEFERRED_DOGLEG_SOLVE_H

#include <deferred_react_util.H>
#include <hybrj.H>
#include <hybrj_type.H>

namespace powell {

#ifdef REACTIONS

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    Constraint_residual(amrex::Array1D<amrex::Real, 1, NumSpec+1>& x,  // solution vector
          Array1D<Real, 1, NumSpec+1>& Z,  // constraint equations to zero
          const burn_t& burn_state_in,
          int& iflag) {

        amrex::ignore_unused(iflag);

        // This is used with the hybrid Powell solver to compute Z,
        // evaluated with x
        //
        // x is {rho X_1, rho X_2, ... rho X_N, rho e}
        //
        // using 1-based indexing

        GpuArray<Real, NUM_STATE> React_full;

        // we'll use the passed-in burn_t as our working state.  We need to
        // sync it with the input state x.  The zone_react_source[]
        // routine works off of burn_state.y[]

        auto burn_state = burn_state_in;

        for (int n = 0; n < NumSpec; ++n) {
            burn_state.y[SFS+n] = x(n+1);
        }
        burn_state.y[SEINT] = x(NumSpec+1);

        // get the reactive term for this state

        zone_react_source(burn_state, React_full);

        // now Z is evaluated with the current guess for U
        // we use ydot_a here to be U^n + substep_dt C

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
    constraint_jacobian(amrex::Array1D<amrex::Real, 1, NumSpec+1>& x,  // solution vector
            JacNetArray2D& Jac,  // constraint equations to zero
            const burn_t& burn_state_in,
            int& iflag) {

        amrex::ignore_unused(iflag);

        // This is used with the hybrid Powell solver to compute J
        // evaluated with x.
        //
        // x is {rho X_1, rho X_2, ... rho X_N, rho e}
        //
        // using 1-based indexing

        // we'll use the passed-in burn_t as our working state.  We need to
        // sync it with the input state x

        auto burn_state = burn_state_in;

        for (int n = 0; n < NumSpec; ++n) {
            burn_state.y[SFS+n] = x(n+1);
        }
        burn_state.y[SEINT] = x(NumSpec+1);

#if NAUX_NET > 0
        amrex::Error("error: aux data not currently supported in true SDC");
#endif

        // we don't actually need this reactive term, but it will
        // ensure that the state is all synced.

        {
            GpuArray<Real, NUM_STATE> React_full;
            zone_react_source(burn_state, React_full);
        }

        // Instead of the decomposition into dw/dU and dR/dw, we instead use the
        // form from the simplified-SDC.  Note:
        // we are not including density anymore.

        auto substep_dt = burn_state.time;

        zone_jacobian(burn_state, substep_dt, Jac);

        Jac.mul(-substep_dt);
        Jac.add_identity();

    }


    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    deferred_dogleg_solve(const Real substep_dt,
                     GpuArray<Real, NUM_STATE> const& Conserved_old,
                     GpuArray<Real, NUM_STATE> & cons_new,
                     GpuArray<Real, NUM_STATE> const& C,
                     const int correction_iter) {

        // the purpose of this function is to solve the system
        //  U - dt R(U) = Conserved_old + dt C
        // using the hybrid-Powell solver.
        //
        // here, cons_new should come in as a guess for the new U for
        // iterations > 0, it will be the solution from the previous
        // iteration initially.
        //
        // upon exit, cons_new will be returned with the value that satisfied
        // the nonlinear function

        amrex::ignore_unused(correction_iter);

        // we will do the implicit update of only the terms that
        // have reactive sources
        //
        // 1:NumSpec : species
        // NumSpec+1 : (rho e)


        // update the density and momenta for this zone -- they don't react

        cons_new[URHO] = Conserved_old[URHO] + substep_dt * C[URHO];

        for (int n = 0; n < 3; ++n) {
            cons_new[UMX+n] = Conserved_old[UMX+n] + substep_dt * C[UMX+n];
        }

        burn_t burn_state;

        fill_burn_type_from_conserved(cons_new, burn_state);
        burn_state.rho = cons_new[URHO];

        // we are solving
        //   f(U) = U - dt R(U) - Conserved_old - dt C = 0
        // we define f_source = Conserved_old + dt C so we are solving
        //   f(U) = U - dt R(U) - f_source = 0
        //
        // we'll store this in the burn_t ydot_a[]

        for (int n = 0; n < NumSpec; ++n) {
            burn_state.ydot_a[SFS + n] = Conserved_old[UFS + n] + substep_dt * C[UFS + n];
        }
        burn_state.ydot_a[SEINT] = Conserved_old[UEINT] + substep_dt * C[UEINT];

        // we need dt in the RHS, pass this through using the time field
        burn_state.time = substep_dt;

        // The Powell solver stores the solution vector in its own type
        hybrj_t<NumSpec+1, JacNetArray2D> hj;
        hj.xtol = 1.e-5;
        hj.mode = 1;

        // locate the initial guess
        for (int n = 0; n < NumSpec; ++n) {
            hj.x(n+1) = cons_new[UFS+n];
        }
        hj.x(NumSpec+1) = cons_new[UEINT];

        // call the powell solver

        hybrj<NumSpec+1, burn_t>(hj, burn_state, Constraint_residual, constraint_jacobian);

        if (hj.info != 1) {
            std::cout << "failure in the Powell solver, error code is: " << hj.info << std::endl;
            amrex::Error("hybrid Powell solve failed");
        }

        // update the full cons_new

        for (int n = 0; n < NumSpec; ++n) {
            cons_new[UFS+n] = hj.x(n+1);
        }

        cons_new[UEINT] = hj.x(NumSpec+1);

        // we want to do a conservative update for (rho E), so first
        // figure out the energy generation rate

        Real rho_Sdot = (cons_new[UEINT] - Conserved_old[UEINT]) / substep_dt - C[UEINT];

        cons_new[UEDEN] = Conserved_old[UEDEN] + substep_dt * (C[UEDEN] + rho_Sdot);

    }

#endif
}

#endif
