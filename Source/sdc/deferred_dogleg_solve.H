#ifndef DEFERRED_DOGLEG_SOLVE_H
#define DEFERRED_DOGLEG_SOLVE_H

#include <deferred_react_util.H>
#include <hybrj.H>
#include <hybrj_type.H>

namespace powell {

#ifdef REACTIONS

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
    deferred_dogleg_solve(const Real substep_dt,
                     GpuArray<Real, NUM_STATE> const& Conserved_old,
                     GpuArray<Real, NUM_STATE> & cons_new,
                     GpuArray<Real, NUM_STATE> const& C,
                     const int correction_iter) {


        amrex::ignore_unused(correction_iter);




        cons_new[URHO] = Conserved_old[URHO] + substep_dt * C[URHO];

        for (int n = 0; n < 3; ++n) {
            cons_new[UMX+n] = Conserved_old[UMX+n] + substep_dt * C[UMX+n];
        }

        burn_t burn_state;

        fill_burn_type_from_conserved(cons_new, burn_state);
        burn_state.rho = cons_new[URHO];


        for (int n = 0; n < NumSpec; ++n) {
            burn_state.ydot_a[SFS + n] = Conserved_old[UFS + n] + substep_dt * C[UFS + n];
        }
        burn_state.ydot_a[SEINT] = Conserved_old[UEINT] + substep_dt * C[UEINT];

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


        Real rho_Sdot = (cons_new[UEINT] - Conserved_old[UEINT]) / substep_dt - C[UEINT];

        cons_new[UEDEN] = Conserved_old[UEDEN] + substep_dt * (C[UEDEN] + rho_Sdot);

    }

#endif
}

#endif
