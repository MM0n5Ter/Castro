#ifndef BURN_STATE_FILL_H
#define BURN_STATE_FILL_H

#include <burn_type.H>

template <typename ComponentAt>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
load_burn_state(const ComponentAt& at, burn_t& burn_state) {

    burn_state.y[SRHO] = at(URHO);
    burn_state.y[SMX] = at(UMX);
    burn_state.y[SMY] = at(UMY);
    burn_state.y[SMZ] = at(UMZ);
    burn_state.y[SEDEN] = at(UEDEN);
    burn_state.y[SEINT] = at(UEINT);
    for (int n = 0; n < NumSpec; n++) {
        burn_state.y[SFS+n] = at(UFS+n);
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
        burn_state.y[SFX+n] = at(UFX+n);
    }
#endif

    burn_state.T = at(UTEMP);
    burn_state.rho = at(URHO);

}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
fill_burn_type_from_conserved(const int i, const int j, const int k,
                       Array4<const Real> const& state,
                       burn_t& burn_state) {

    const auto cons_cell = state.cellData(i,j,k);
    load_burn_state([&] (int n) { return cons_cell[n]; }, burn_state);

}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
fill_burn_type_from_conserved(GpuArray<Real, NUM_STATE> const& state,
                       burn_t& burn_state) {

    load_burn_state([&] (int n) { return state[n]; }, burn_state);

}



#endif
