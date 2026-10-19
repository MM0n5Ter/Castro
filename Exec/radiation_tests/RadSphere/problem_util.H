#ifndef PROBLEM_UTIL_H
#define PROBLEM_UTIL_H

#include "fundamental_constants.H"


constexpr Real nu_0 = 3.6e14_rt;
constexpr Real kappa_0 = 1.e13_rt;

constexpr Real R_sphere = 0.02_rt;
constexpr Real r_obs = 0.06_rt;

constexpr Real T_sphere = 1500.0_rt * C::ev2erg / C::k_B;


AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real
sferfc(const Real arg1) {
  return std::erfc(amrex::max(-26.0_rt, amrex::min(26.0_rt, arg1)));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real
sfexp(const Real arg1) {
  return std::exp(amrex::max(-650.0_rt, amrex::min(650.0_rt, arg1)));
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real
planck(const Real nu, const Real T) {


  Real B = (8.0_rt * M_PI * C::hplanck * nu * nu * nu  / (C::c_light * C::c_light * C::c_light)) /
    (std::exp(C::hplanck * nu / (C::k_B * T)) - 1.0_rt);
  B = amrex::min(amrex::max(B, 1.e-50_rt), 1.e200_rt);

  return B;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real
F_radsphere(const Real r, const Real t, const Real nu) {



  Real kappa = kappa_0 * (nu_0 / nu) * (nu_0 / nu) * (nu_0 / nu);

  Real erfc_term1 = sferfc(std::sqrt(3.0_rt * kappa / (4.0_rt * C::c_light * amrex::max(t, 1.e-50_rt))) *
                           (r - R_sphere) - std::sqrt(C::c_light * t * kappa));

  Real erfc_term2 = sferfc(std::sqrt(3.0_rt * kappa / (4.0_rt * C::c_light * amrex::max(t, 1.e-50_rt))) *
                           (r - R_sphere) + std::sqrt(C::c_light * t * kappa));

  Real F = 0.5_rt * (sfexp(-std::sqrt(3.0_rt) * kappa * (r - R_sphere)) * erfc_term1 +
                     sfexp( std::sqrt(3.0_rt) * kappa * (r - R_sphere)) * erfc_term2);

  return amrex::min(amrex::max(F, 1.e-50_rt), 1.e200_rt);
}

#endif
