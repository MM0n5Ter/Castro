#ifndef PARABOLIC_EDGE_H
#define PARABOLIC_EDGE_H

#include <cmath>

#include <AMReX_REAL.H>
#include <stencil_tools.H>

using namespace amrex::literals;
using namespace reconstruction;

template <typename RealT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
integrate_parabolic_profile_single(const RealT sm, const RealT sp, const RealT sc,
                       const RealT lam, const RealT dtdx,
                       RealT& Ip, RealT& Im) {

  RealT s6 = 6.0_rt * sc - 3.0_rt * (sm + sp);

  RealT sigma = std::abs(lam) * dtdx;

  if (lam <= 0.0_rt) {
    Ip = sp;
    Im = sm + 0.5_rt * sigma * (sp - sm + (1.0_rt - (2.0_rt/3.0_rt) * sigma) * s6);
  } else {
    Ip = sp - 0.5_rt * sigma * (sp - sm - (1.0_rt - (2.0_rt/3.0_rt) * sigma) * s6);
    Im = sm;
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
integrate_parabolic_profile(const amrex::Real sm, const amrex::Real sp, const amrex::Real sc,
                const amrex::Real u, const amrex::Real c, const amrex::Real dtdx,
                amrex::Real* Ip, amrex::Real* Im) {

  const amrex::Real speeds[3] = {u - c, u, u + c};

  for (int n = 0; n < 3; n++) {
    integrate_parabolic_profile_single(sm, sp, sc, speeds[n], dtdx, Ip[n], Im[n]);
  }

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
van_leer_slope(amrex::Real sl, amrex::Real sc, amrex::Real sr) {

    amrex::Real dsl = 2.0_rt * (sc - sl);
    amrex::Real dsr = 2.0_rt * (sr - sc);

    amrex::Real slope = 0.0_rt;
    if (dsl*dsr > 0.0_rt) {
      amrex::Real dsc = 0.5_rt * (sr - sl);
      slope = std::copysign(1.0_rt, dsc) * amrex::min(std::abs(dsc), std::abs(dsl), std::abs(dsr));
    }

    return slope;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
parabolic_reconstruct(const amrex::Real* s,
                const amrex::Real flatten_factor, amrex::Real& sm, amrex::Real& sp) {

    if (ppm_parabola_clip) {

        amrex::Real van_leer_slope_left = van_leer_slope(s[im2], s[im1], s[i0]);
        amrex::Real van_leer_slope_right = van_leer_slope(s[im1], s[i0], s[ip1]);

        sm = 0.5_rt * (s[i0] + s[im1]) - (1.0_rt/6.0_rt) * (van_leer_slope_right - van_leer_slope_left);

        sm = amrex::Clamp(sm, std::min(s[i0], s[im1]), std::max(s[i0], s[im1]));

        van_leer_slope_left = van_leer_slope(s[im1], s[i0], s[ip1]);
        van_leer_slope_right = van_leer_slope(s[i0], s[ip1], s[ip2]);

        sp = 0.5_rt * (s[ip1] + s[i0]) - (1.0_rt/6.0_rt) * (van_leer_slope_right - van_leer_slope_left);

        sp = amrex::Clamp(sp, std::min(s[ip1], s[i0]), std::max(s[ip1], s[i0]));

        sm = flatten_factor * sm + (1.0_rt - flatten_factor) * s[i0];
        sp = flatten_factor * sp + (1.0_rt - flatten_factor) * s[i0];

        if ((sp - s[i0]) * (s[i0] - sm) <= 0.0_rt) {
          sp = s[i0];
          sm = s[i0];

        } else if (std::abs(sp - s[i0]) >= 2.0_rt * std::abs(sm - s[i0])) {
          sp = 3.0_rt * s[i0] - 2.0_rt * sm;

        } else if (std::abs(sm - s[i0]) >= 2.0_rt * std::abs(sp - s[i0])) {
          sm = 3.0_rt * s[i0] - 2.0_rt * sp;
        }

    } else {

        sm = (7.0_rt/12.0_rt) * (s[i0] + s[im1]) - (1.0_rt/12.0_rt) * (s[im2] + s[ip1]);
        sp = (7.0_rt/12.0_rt) * (s[ip1] + s[i0]) - (1.0_rt/12.0_rt) * (s[im1] + s[ip2]);

    }

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool
reconstruct_parabolic_hydrostatic(const amrex::Real* rho, const amrex::Real* p, const amrex::Real* src, const amrex::Real flatten_factor,
                       const amrex::Real dx,
                       amrex::Real& sm, amrex::Real& sp) {

    amrex::Real tp[nslp];

    amrex::Real p0_hydrostatic = p[i0];

    amrex::Real pp1_hydrostatic = p0_hydrostatic + 0.25_rt*dx * (rho[i0] + rho[ip1]) * (src[i0] + src[ip1]);
    amrex::Real pp2_hydrostatic = pp1_hydrostatic + 0.25_rt*dx * (rho[ip1] + rho[ip2]) * (src[ip1] + src[ip2]);

    amrex::Real pm1_hydrostatic = p0_hydrostatic - 0.25_rt*dx * (rho[i0] + rho[im1]) * (src[i0] + src[im1]);
    amrex::Real pm2_hydrostatic = pm1_hydrostatic - 0.25_rt*dx * (rho[im1] + rho[im2]) * (src[im1] + src[im2]);

    amrex::Real p_plus_hydrostatic = p0_hydrostatic + 0.5_rt*dx * rho[i0] * src[i0];
    amrex::Real p_minus_hydrostatic = p0_hydrostatic - 0.5_rt*dx * rho[i0] * src[i0];

    bool ptest =  p0_hydrostatic < 0.0 ||
                 pp1_hydrostatic < 0.0 ||
                 pp2_hydrostatic < 0.0 ||
                 pm1_hydrostatic < 0.0 ||
                 pm2_hydrostatic < 0.0 ||
                 p_plus_hydrostatic < 0.0 ||
                 p_minus_hydrostatic < 0.0;

    bool use_hydrostatic_edges = !ptest && rho[i0] >= slope_rho_cutoff;

    if (use_hydrostatic_edges) {

        tp[i0] = p[i0] - p0_hydrostatic;

        tp[ip1] = p[ip1] - pp1_hydrostatic;
        tp[ip2] = p[ip2] - pp2_hydrostatic;

        tp[im1] = p[im1] - pm1_hydrostatic;
        tp[im2] = p[im2] - pm2_hydrostatic;

    } else {

        tp[im2] = p[im2];
        tp[im1] = p[im1];
        tp[i0] = p[i0];
        tp[ip1]= p[ip1];
        tp[ip2] = p[ip2];
    }

    parabolic_reconstruct(tp, flatten_factor, sm, sp);

    if (use_hydrostatic_edges && !hydrosim::ppm_equil_recon) {
        sp += p_plus_hydrostatic;
        sm += p_minus_hydrostatic;
    }

    return use_hydrostatic_edges;

}

#endif
