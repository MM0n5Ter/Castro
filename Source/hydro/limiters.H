#ifndef LIMITERS_H
#define LIMITERS_H

#include <HydroSim.H>



#include <cmath>

#include <stencil_tools.H>

using namespace amrex;
using namespace reconstruction;

namespace hydro {
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real shock_flattening_along(int i, int j, int k, int di, int dj, int dk,
                            Array4<Real const> const& prim_arr,
                            const int vel_index, const int pressure_index)
{
    constexpr Real pressure_cutoff = 1.e-200_rt;

    constexpr Real shock_threshold = 0.33_rt;
    constexpr Real zcut_low = 0.75_rt;
    constexpr Real zcut_high = 0.85_rt;
    constexpr Real zcut_inv_width = 1.0_rt / (zcut_high-zcut_low);

    Real dp = prim_arr(i+di,j+dj,k+dk,pressure_index) - prim_arr(i-di,j-dj,k-dk,pressure_index);

    int upwind_shift = dp > 0.0_rt ? 1 : -1;

    Real denom = std::max(pressure_cutoff, std::abs(prim_arr(i+2*di,j+2*dj,k+2*dk,pressure_index) - prim_arr(i-2*di,j-2*dj,k-2*dk,pressure_index)));
    Real zeta = std::abs(dp) / denom;
    Real z = amrex::Clamp(zcut_inv_width * (zeta - zcut_low), 0.0_rt, 1.0_rt);

    Real tst = 0.0_rt;
    if (prim_arr(i-di,j-dj,k-dk,vel_index) - prim_arr(i+di,j+dj,k+dk,vel_index) >= 0.0_rt) {
      tst = 1.0_rt;
    }

    Real tmp = std::min(prim_arr(i+di,j+dj,k+dk,pressure_index), prim_arr(i-di,j-dj,k-dk,pressure_index));

    Real chi = 0.0_rt;
    if (std::abs(dp) > shock_threshold*tmp) {
      chi = tst;
    }

    const int up_hi = (1-upwind_shift);
    const int up_lo = (-1-upwind_shift);

    dp = prim_arr(i+up_hi*di,j+up_hi*dj,k+up_hi*dk,pressure_index) - prim_arr(i+up_lo*di,j+up_lo*dj,k+up_lo*dk,pressure_index);

    denom = std::max(pressure_cutoff, std::abs(prim_arr(i+(2-upwind_shift)*di,j+(2-upwind_shift)*dj,k+(2-upwind_shift)*dk,pressure_index)
                                              -prim_arr(i+(-2-upwind_shift)*di,j+(-2-upwind_shift)*dj,k+(-2-upwind_shift)*dk,pressure_index)));
    zeta = std::abs(dp) / denom;
    Real z2 = amrex::Clamp(zcut_inv_width * (zeta - zcut_low), 0.0_rt, 1.0_rt);

    tst = 0.0_rt;
    if (prim_arr(i+up_lo*di,j+up_lo*dj,k+up_lo*dk,vel_index) - prim_arr(i+up_hi*di,j+up_hi*dj,k+up_hi*dk,vel_index) >= 0.0_rt) {
      tst = 1.0_rt;
    }

    tmp = std::min(prim_arr(i+up_hi*di,j+up_hi*dj,k+up_hi*dk,pressure_index), prim_arr(i+up_lo*di,j+up_lo*dj,k+up_lo*dk,pressure_index));

    Real chi2 = 0.0_rt;
    if (std::abs(dp) > shock_threshold*tmp) {
      chi2 = tst;
    }

    return 1.0_rt - std::max(chi2 * z2, chi * z);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real flatten(int i, int j, int k,
             Array4<Real const> const& prim_arr,
             const int pressure_index)
{
    constexpr int velocity_comp[3] = {QU, QV, QW};
    constexpr int unit[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    Real flatten_factor = shock_flattening_along(i, j, k, unit[0][0], unit[0][1], unit[0][2], prim_arr, velocity_comp[0], pressure_index);

    for (int d = 1; d < AMREX_SPACEDIM; ++d) {
        flatten_factor = std::min(flatten_factor, shock_flattening_along(i, j, k, unit[d][0], unit[d][1], unit[d][2], prim_arr, velocity_comp[d], pressure_index));
    }

    return flatten_factor;
}
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real
mc_limited_difference(Real dl, Real dr) {

  Real dcen = 0.25_rt * (dl + dr);
  Real dsgn = std::copysign(1.0_rt, dcen);
  Real slop = amrex::min(std::abs(dl), std::abs(dr));
  Real dlim = dl*dr >= 0.0_rt ? slop : 0.0_rt;

  return dsgn*amrex::min(dlim, std::abs(dcen));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real
limited_slope(const Real* q, const Real flatten_factor,
       bool reflect_lo_edge, bool reflect_hi_edge) {

  Real dq;

  if (plm_iorder == 1) {

    dq = 0.0;

  } else {

    if (plm_slope_limiter == 1) {

      Real dlft = 2.0_rt*(q[i0] - q[im1]);
      Real drgt = 2.0_rt*(q[ip1] - q[i0]);
      Real dcen = 0.25_rt * (dlft + drgt);
      Real dsgn = std::copysign(1.0_rt, dcen);
      Real slop = amrex::min(std::abs(dlft), std::abs(drgt));
      Real dlim = dlft*drgt >= 0.0_rt ? slop : 0.0_rt;

      dq = flatten_factor * dsgn * amrex::min(dlim, std::abs(dcen));

    } else {


      Real qm2 = q[im2];
      Real qm1 = q[im1];
      Real q0 = q[i0];
      Real qp1 = q[ip1];
      Real qp2 = q[ip2];

      if (reflect_lo_edge) {
        qm2 = -qp1;
        qm1 = -3.0_rt*q0 + qp1 - 0.125_rt*(qp2 + qp1);
      }

      if (reflect_hi_edge) {
        qp2 = -qm1;
        qp1 = -3.0_rt*q0 + qm1 - 0.125_rt*(qm2 + qm1);
      }


      Real left_diff_p1 = 2.0_rt*(qp1 - q0);
      Real right_diff_p1 = 2.0_rt*(qp2 - qp1);
      Real limited_slope_p1 = mc_limited_difference(left_diff_p1, right_diff_p1);

      Real left_diff_m1 = 2.0_rt*(qm1 - qm2);
      Real right_diff_m1 = 2.0_rt*(q0 - qm1);
      Real limited_slope_m1 = mc_limited_difference(left_diff_m1, right_diff_m1);

      Real dlft = right_diff_m1;
      Real drgt = left_diff_p1;
      Real dcen = 0.25_rt * (dlft + drgt);
      Real dsgn = std::copysign(1.0_rt, dcen);
      Real slop = amrex::min(std::abs(dlft), std::abs(drgt));
      Real dlim = dlft*drgt >= 0.0_rt ? slop : 0.0_rt;

      Real dq1 = (4.0_rt/3.0_rt)*dcen - (1.0_rt/6.0_rt)*(limited_slope_p1 + limited_slope_m1);
      dq = flatten_factor * dsgn * amrex::min(dlim, std::abs(dq1));
    }
  }

  return dq;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
pressure_slope(const Real* rho, const Real* p, const Real* src, const Real flatten_factor,
       bool at_lo_symmetry, bool at_hi_symmetry,
       const Real dx, Real& dp) {


  if (plm_iorder == 1) {

    dp = 0.0_rt;

  } else {

    if (rho[i0] < slope_rho_cutoff) {
      return;
    }

    Real p0;
    Real pp1;
    Real pm1;
    Real pp2;
    Real pm2;

    Real p0_hydrostatic;
    Real pp1_hydrostatic;
    Real pp2_hydrostatic;
    Real pm1_hydrostatic;
    Real pm2_hydrostatic;

    p0_hydrostatic = p[i0];

    pp1_hydrostatic = p0_hydrostatic + 0.25_rt*dx * (rho[i0] + rho[ip1]) * (src[i0] + src[ip1]);
    pp2_hydrostatic = pp1_hydrostatic + 0.25_rt*dx * (rho[ip1] + rho[ip2]) * (src[ip1] + src[ip2]);

    pm1_hydrostatic = p0_hydrostatic - 0.25_rt*dx * (rho[i0] + rho[im1]) * (src[i0] + src[im1]);
    pm2_hydrostatic = pm1_hydrostatic - 0.25_rt*dx * (rho[im1] + rho[im2]) * (src[im1] + src[im2]);

    p0 = 0.0_rt;
    pp1 = p[ip1] - pp1_hydrostatic;
    pp2 = p[ip2] - pp2_hydrostatic;

    pm1 = p[im1] - pm1_hydrostatic;
    pm2 = p[im2] - pm2_hydrostatic;

    if (at_lo_symmetry) {
      pm1 = 0.0_rt;
      pm2 = 0.0_rt;
    }

    if (at_hi_symmetry) {
      pp1 = 0.0_rt;
      pp2 = 0.0_rt;
    }





    Real left_diff_p1 = pp1 - p0;
    Real right_diff_p1 = pp2 - pp1;

    Real dcen = 0.5_rt*(left_diff_p1 + right_diff_p1);
    Real dsgn = std::copysign(1.0_rt, dcen);
    Real dlim = left_diff_p1*right_diff_p1 >= 0.0_rt ? 2.0_rt * amrex::min(std::abs(left_diff_p1), std::abs(right_diff_p1)) : 0.0_rt;
    Real limited_slope_p1 = dsgn*amrex::min(dlim, std::abs(dcen));



    Real left_diff_m1 = pm1 - pm2;
    Real right_diff_m1 = p0 - pm1;

    dcen = 0.5_rt*(left_diff_m1 + right_diff_m1);
    dsgn = std::copysign(1.0_rt, dcen);
    dlim = left_diff_m1*right_diff_m1 >= 0.0_rt ? 2.0_rt * amrex::min(std::abs(left_diff_m1), std::abs(right_diff_m1)) : 0.0_rt;

    Real limited_slope_m1 = dsgn*amrex::min(dlim, std::abs(dcen));



    Real dlft = right_diff_m1;

    Real drgt = left_diff_p1;

    dcen = 0.5_rt*(dlft + drgt);
    dsgn = std::copysign(1.0_rt, dcen);
    dlim = dlft*drgt >= 0.0_rt ? 2.0_rt * amrex::min(std::abs(dlft), std::abs(drgt)) : 0.0_rt;

    Real dp1 = (4.0_rt/3.0_rt)*dcen - (1.0_rt/6.0_rt)*(limited_slope_p1 + limited_slope_m1);
    dp = flatten_factor * dsgn*amrex::min(dlim, std::abs(dp1));

    dp += rho[i0] * src[i0] * dx;
  }

}

#endif
