#ifndef LIMITER_H
#define LIMITER_H

#include <HydroSim.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef GRAVITY
#include <Gravity.H>
#endif

#include <cmath>

#include <reconstruction.H>

using namespace amrex;
using namespace reconstruction;

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
      Real dcen = 0.25_rt * (left_diff_p1 + right_diff_p1);
      Real dsgn = std::copysign(1.0_rt, dcen);
      Real slop = amrex::min(std::abs(left_diff_p1), std::abs(right_diff_p1));
      Real dlim = left_diff_p1*right_diff_p1 >= 0.0_rt ? slop : 0.0_rt;
      Real limited_slope_p1 = dsgn*amrex::min(dlim, std::abs(dcen));

      Real left_diff_m1 = 2.0_rt*(qm1 - qm2);
      Real right_diff_m1 = 2.0_rt*(q0 - qm1);
      dcen = 0.25_rt * (left_diff_m1 + right_diff_m1);
      dsgn = std::copysign(1.0_rt, dcen);
      slop = amrex::min(std::abs(left_diff_m1), std::abs(right_diff_m1));
      dlim = left_diff_m1*right_diff_m1 >= 0.0_rt ? slop : 0.0_rt;
      Real limited_slope_m1 = dsgn*amrex::min(dlim, std::abs(dcen));

      Real dlft = right_diff_m1;
      Real drgt = left_diff_p1;
      dcen = 0.25_rt * (dlft + drgt);
      dsgn = std::copysign(1.0_rt, dcen);
      slop = amrex::min(std::abs(dlft), std::abs(drgt));
      dlim = dlft*drgt >= 0.0_rt ? slop : 0.0_rt;

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
