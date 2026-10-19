#ifndef user_initialize_state_data_H
#define user_initialize_state_data_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_initialize_state_data (int i, int j, int k,
                                    Array4<Real> const& state,
                                    const GeometryData& geomdata)
{

    state(i,j,k,URHO) = 1.0_rt;
    state(i,j,k,UMX) = 0.0_rt;
    state(i,j,k,UMY) = 0.0_rt;
    state(i,j,k,UMZ) = 0.0_rt;

    for (int n = 0; n < NumSpec; n++) {
        state(i,j,k,UFS+n) = 0.0_rt;
    }
    state(i,j,k,UFS) = state(i,j,k,URHO);

    state(i,j,k,UTEMP) = 1.0e-50_rt;
    state(i,j,k,UEINT) = 1.0e-50_rt;
    state(i,j,k,UEDEN) = 1.0e-50_rt;
}
#endif
