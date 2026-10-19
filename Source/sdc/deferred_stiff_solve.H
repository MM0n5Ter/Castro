#ifndef DEFERRED_STIFF_SOLVE_H
#define DEFERRED_STIFF_SOLVE_H

#include <network.H>
#include <burn_type.H>
#include <extern_parameters.H>

#ifdef NEW_NETWORK_IMPLEMENTATION
#include <rhs.H>
#else
#include <actual_network.H>
#include <actual_rhs.H>
#endif
#include <sdc_conserved_to_burn.H>
#include <burner.H>


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
