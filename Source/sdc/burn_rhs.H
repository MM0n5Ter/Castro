#ifndef BURN_RHS_H
#define BURN_RHS_H

#include <burn_type.H>
#include <burn_state_fill.H>
#include <numerical_jacobian.H>

using ConservedZone = amrex::GpuArray<amrex::Real, NUM_STATE>;

AMREX_GPU_HOST_DEVICE AMREX_INLINE
bool
burning_permitted(const int i, const int j, const int k,
             Array4<const Real> const& state) {

    if (state(i,j,k,UTEMP) < burn_temp_min || state(i,j,k,UTEMP) > burn_temp_max ||
        state(i,j,k,URHO) < burn_rho_min || state(i,j,k,URHO) > burn_rho_max) {
        return false;
    }

#ifdef SHOCK_VAR
    if (state(i,j,k,UFLAG_SHOCK) > 0.0_rt && skip_burning_in_shocks == 1) {
        return false;
    }
#endif

    return true;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
bool
burning_permitted(ConservedZone const& state) {

    if (state[UTEMP] < burn_temp_min || state[UTEMP] > burn_temp_max ||
        state[URHO] < burn_rho_min || state[URHO] > burn_rho_max) {
        return false;
    }

#ifdef SHOCK_VAR
    if (state[UFLAG_SHOCK] > 0.0_rt && skip_burning_in_shocks == 1) {
        return false;
    }
#endif

    return true;
}

template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
zone_jacobian(burn_t& burn_state, const Real dt, T& Jac) {

#ifdef SIMPLIFIED_SDC
#ifndef AMREX_USE_GPU
    amrex::Error("we shouldn't be here with the simplified SDC method (USE_SIMPLIFIED_SDC=TRUE)");
#endif
#else


    if (integrator_rp::jacobian == 2) {

        jac_info_t jac_info;
        jac_info.h = dt;


        numerical_jac(burn_state, jac_info, Jac);

    } else {
        actual_jac(burn_state, Jac);


        for (int n = 1; n <= NumSpec; n++) {
            for (int m = 1; m <= neqs; m++) {
                Jac(n,m) = Jac(n,m) * aion[n-1];
            }
        }

        for (int m = 1; m <= neqs; m++) {
            for (int n = 1; n <= NumSpec; n++) {
                Jac(m,n) = Jac(m,n) * aion_inv[n-1];
            }
        }



        eos_xderivs_t eos_xderivs = composition_derivatives(burn_state);

        for (int m = 1; m <= neqs; m++) {
            for (int n = 1; n <= NumSpec; n++) {
                Jac(m, n) -= eos_xderivs.dedX[n-1] * Jac(m, net_ienuc);
            }
        }
    }
#endif


}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
zone_react_source(burn_t& burn_state,
                         ConservedZone& R) {



    auto rhoInv = 1.0_rt / burn_state.y[SRHO];

    burn_state.rho = burn_state.y[SRHO];
    burn_state.e = burn_state.y[SEINT] * rhoInv;

    for (int n = 0; n < NumSpec; ++n) {
       burn_state.xn[n] = amrex::Clamp(burn_state.y[SFS+n] * rhoInv, small_x, 1.0_rt);
    }

#if NAUX_NET > 0
    for (int n = 0; n < NumAux; ++n) {
       burn_state.aux[n] = burn_state.y[SFX+n] * rhoInv;
    }
#endif

    static_assert(NAUX_NET == 0, "Error: aux data not currently supported in true SDC");

    eos(eos_input_re, burn_state);

    burn_state.T = amrex::Clamp(burn_state.T, small_temp, integrator_rp::MAX_TEMP);

    Array1D<Real, 1, neqs> ydot;

    actual_rhs(burn_state, ydot);

    for (int n = 0; n < NUM_STATE; ++n) {
        R[n] = 0.0_rt;
    }

    for (int n = 0; n < NumSpec; ++n) {
        R[UFS+n] = burn_state.rho * aion[n] * ydot(1+n);
    }

    R[UEDEN] = burn_state.rho * ydot(net_ienuc);
    R[UEINT] = burn_state.rho * ydot(net_ienuc);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
zone_react_source(const int i, const int j, const int k,
                         Array4<const Real> const& state,
                         Array4<Real> const& R) {

    GpuArray<Real, NUM_STATE> scaled_grad_arr;

    burn_t burn_state;

    fill_burn_type_from_conserved(i, j, k, state, burn_state);

    zone_react_source(burn_state, scaled_grad_arr);

    for (int n = 0; n < NUM_STATE; ++n) {
        R(i,j,k,n) = scaled_grad_arr[n];
    }
}

#endif
