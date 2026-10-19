#ifndef flattening_H
#define flattening_H

namespace hydro {

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real flatten(int i, int j, int k,
             Array4<Real const> const& prim_arr,
             const int pressure_index)
{
    constexpr Real pressure_cutoff = 1.e-200_rt;

    constexpr Real shock_threshold = 0.33_rt;
    constexpr Real zcut_low = 0.75_rt;
    constexpr Real zcut_high = 0.85_rt;
    constexpr Real zcut_inv_width = 1.0_rt / (zcut_high-zcut_low);


    Real dp = prim_arr(i+1,j,k,pressure_index) - prim_arr(i-1,j,k,pressure_index);

    int upwind_shift = dp > 0.0_rt ? 1 : -1;

    Real denom = std::max(pressure_cutoff, std::abs(prim_arr(i+2,j,k,pressure_index) - prim_arr(i-2,j,k,pressure_index)));
    Real zeta = std::abs(dp) / denom;
    Real z = amrex::Clamp(zcut_inv_width * (zeta - zcut_low), 0.0_rt, 1.0_rt);

    Real tst = 0.0_rt;
    if (prim_arr(i-1,j,k,QU) - prim_arr(i+1,j,k,QU) >= 0.0_rt) {
      tst = 1.0_rt;
    }

    Real tmp = std::min(prim_arr(i+1,j,k,pressure_index), prim_arr(i-1,j,k,pressure_index));

    Real chi = 0.0_rt;
    if (std::abs(dp) > shock_threshold*tmp) {
      chi = tst;
    }


    dp = prim_arr(i+1-upwind_shift,j,k,pressure_index) - prim_arr(i-1-upwind_shift,j,k,pressure_index);

    denom = std::max(pressure_cutoff, std::abs(prim_arr(i+2-upwind_shift,j,k,pressure_index)-prim_arr(i-2-upwind_shift,j,k,pressure_index)));
    zeta = std::abs(dp) / denom;
    Real z2 = amrex::Clamp(zcut_inv_width * (zeta - zcut_low), 0.0_rt, 1.0_rt);

    tst = 0.0_rt;
    if (prim_arr(i-1-upwind_shift,j,k,QU) - prim_arr(i+1-upwind_shift,j,k,QU) >= 0.0_rt) {
      tst = 1.0_rt;
    }

    tmp = std::min(prim_arr(i+1-upwind_shift,j,k,pressure_index), prim_arr(i-1-upwind_shift,j,k,pressure_index));

    Real chi2 = 0.0_rt;
    if (std::abs(dp) > shock_threshold*tmp) {
      chi2 = tst;
    }

    Real flatten_factor = 1.0_rt - std::max(chi2 * z2, chi * z);


#if AMREX_SPACEDIM >= 2

    dp = prim_arr(i,j+1,k,pressure_index) - prim_arr(i,j-1,k,pressure_index);

    upwind_shift = dp > 0.0_rt ? 1 : -1;

    denom = std::max(pressure_cutoff, std::abs(prim_arr(i,j+2,k,pressure_index) - prim_arr(i,j-2,k,pressure_index)));
    zeta = std::abs(dp) / denom;
    z = amrex::Clamp(zcut_inv_width * (zeta - zcut_low), 0.0_rt, 1.0_rt);

    tst = 0.0_rt;
    if (prim_arr(i,j-1,k,QV) - prim_arr(i,j+1,k,QV) >= 0.0_rt) {
      tst = 1.0_rt;
    }

    tmp = std::min(prim_arr(i,j+1,k,pressure_index), prim_arr(i,j-1,k,pressure_index));

    chi = 0.0_rt;
    if (std::abs(dp) > shock_threshold*tmp) {
      chi = tst;
    }


    dp = prim_arr(i,j+1-upwind_shift,k,pressure_index) - prim_arr(i,j-1-upwind_shift,k,pressure_index);

    denom = std::max(pressure_cutoff, std::abs(prim_arr(i,j+2-upwind_shift,k,pressure_index) - prim_arr(i,j-2-upwind_shift,k,pressure_index)));
    zeta = std::abs(dp) / denom;
    z2 = amrex::Clamp(zcut_inv_width * (zeta - zcut_low), 0.0_rt, 1.0_rt);

    tst = 0.0_rt;
    if (prim_arr(i,j-1-upwind_shift,k,QV) - prim_arr(i,j+1-upwind_shift,k,QV) >= 0.0_rt) {
      tst = 1.0_rt;
    }

    tmp = std::min(prim_arr(i,j+1-upwind_shift,k,pressure_index), prim_arr(i,j-1-upwind_shift,k,pressure_index));

    chi2 = 0.0_rt;
    if (std::abs(dp) > shock_threshold*tmp) {
      chi2 = tst;
    }

    flatten_factor = std::min(flatten_factor, 1.0_rt - std::max(chi2 * z2, chi * z));
#endif


#if AMREX_SPACEDIM == 3

    dp = prim_arr(i,j,k+1,pressure_index) - prim_arr(i,j,k-1,pressure_index);

    upwind_shift = dp > 0.0_rt ? 1: -1;

    denom = std::max(pressure_cutoff, std::abs(prim_arr(i,j,k+2,pressure_index) - prim_arr(i,j,k-2,pressure_index)));
    zeta = std::abs(dp) / denom;
    z = amrex::Clamp(zcut_inv_width * (zeta - zcut_low), 0.0_rt, 1.0_rt);

    tst = 0.0_rt;
    if (prim_arr(i,j,k-1,QW) - prim_arr(i,j,k+1,QW) >= 0.0_rt) {
      tst = 1.0_rt;
    }

    tmp = std::min(prim_arr(i,j,k+1,pressure_index), prim_arr(i,j,k-1,pressure_index));

    chi = 0.0_rt;
    if (std::abs(dp) > shock_threshold*tmp) {
      chi = tst;
    }


    dp = prim_arr(i,j,k+1-upwind_shift,pressure_index) - prim_arr(i,j,k-1-upwind_shift,pressure_index);

    denom = std::max(pressure_cutoff, std::abs(prim_arr(i,j,k+2-upwind_shift,pressure_index) - prim_arr(i,j,k-2-upwind_shift,pressure_index)));
    zeta = std::abs(dp) / denom;
    z2 = amrex::Clamp(zcut_inv_width * (zeta - zcut_low), 0.0_rt, 1.0_rt);

    tst = 0.0_rt;
    if (prim_arr(i,j,k-1-upwind_shift,QW) - prim_arr(i,j,k+1-upwind_shift,QW) >= 0.0_rt) {
      tst = 1.0_rt;
    }

    tmp = std::min(prim_arr(i,j,k+1-upwind_shift,pressure_index), prim_arr(i,j,k-1-upwind_shift,pressure_index));

    chi2 = 0.0_rt;
    if (std::abs(dp) > shock_threshold*tmp) {
      chi2 = tst;
    }

    flatten_factor = std::min(flatten_factor,  1.0_rt - std::max(chi2 * z2, chi * z));
#endif

    return flatten_factor;
}

}

#endif
