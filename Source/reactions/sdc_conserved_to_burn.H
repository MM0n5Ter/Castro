#ifndef SDC_CONSERVED_TO_BURN_H
#define SDC_CONSERVED_TO_BURN_H

#include <burn_type.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
fill_burn_type_from_conserved(const int i, const int j, const int k,
                       Array4<const Real> const& state,
                       burn_t& burn_state) {

    const auto cons_cell = state.cellData(i,j,k);
    burn_state.y[SRHO] = cons_cell[URHO];
    burn_state.y[SMX] = cons_cell[UMX];
    burn_state.y[SMY] = cons_cell[UMY];
    burn_state.y[SMZ] = cons_cell[UMZ];
    burn_state.y[SEDEN] = cons_cell[UEDEN];
    burn_state.y[SEINT] = cons_cell[UEINT];
    amrex::constexpr_for<0, NumSpec>([&] (auto n) {
        burn_state.y[SFS+n] = cons_cell[UFS+n];
    });
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
        burn_state.y[SFX+n] = state(i,j,k,UFX+n);
    }
#endif

    burn_state.T = cons_cell[UTEMP];
    burn_state.rho = burn_state.y[SRHO];

}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
fill_burn_type_from_conserved(GpuArray<Real, NUM_STATE> const& state,
                       burn_t& burn_state) {

    burn_state.y[SRHO] = state[URHO];
    burn_state.y[SMX] = state[UMX];
    burn_state.y[SMY] = state[UMY];
    burn_state.y[SMZ] = state[UMZ];
    burn_state.y[SEDEN] = state[UEDEN];
    burn_state.y[SEINT] = state[UEINT];
    for (int n = 0; n < NumSpec; n++) {
        burn_state.y[SFS+n] = state[UFS+n];
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
        burn_state.y[SFX+n] = state[UFX+n];
    }
#endif

    burn_state.T = state[UTEMP];
    burn_state.rho = state[URHO];

}



#endif
