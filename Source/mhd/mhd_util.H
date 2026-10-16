#ifndef HYDROSIM_MHD_SOUND_SPEED_H
#define HYDROSIM_MHD_SOUND_SPEED_H

#include <HydroSim_util.H>
#include <mhd_sizes.H>

using namespace amrex;

// sound speed calc for ideal MHD
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real
fast_magnetosonic_speed(Real as, Real ca, Real bd) {

  // inputs:
  // as = P_g * gam1/rho
  // ca = B^2/rho
  // bd = B_direction^2/rho

  // Fast Magneto-Sonic Wave
  Real c = 0.5_rt * ((as + ca) + std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*bd));
  return std::sqrt(c);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
qflux(GpuArray<Real, NQ>& qflx,
      GpuArray<Real, NUM_STATE+3> const& flx,
      GpuArray<Real, NQ> const& q_zone) {

  // Calculate the C to P Jacobian applied to the fluxes

  // this implements dW/dU . qflux, where dW/dU is the Jacobian of
  // the primitive quantities (W) with respect to conserved quantities (U)

  qflx[QRHO] = flx[URHO];
  qflx[QU] = (flx[UMX] - flx[URHO] * q_zone[QU]) / q_zone[QRHO];
  qflx[QV] = (flx[UMY] - flx[URHO] * q_zone[QV]) / q_zone[QRHO];
  qflx[QW] = (flx[UMZ] - flx[URHO] * q_zone[QW]) / q_zone[QRHO];

  for (int n = 0; n < NumSpec; n++) {
    qflx[QSPEC+n] = (flx[USPEC+n] - flx[URHO] * q_zone[QSPEC+n]) / q_zone[QRHO];
  }

  eos_t eos_state;
  eos_state.rho = q_zone[QRHO];
  eos_state.p = q_zone[QPRES];
  eos_state.T = 100.0_rt; // dummy initial guess
  for (int n = 0; n < NumSpec; n++) {
    eos_state.xn[n] = q_zone[QSPEC+n];
  }
#if NAUX_NET > 0
  for (int n = 0; n < NumAux; n++) {
    eos_state.aux[n] = q_zone[QAUXV+n];
  }
#endif

  eos(eos_input_rp, eos_state);

  Real dedrho = eos_state.dedr - eos_state.dedT * eos_state.dpdr * 1.0_rt / eos_state.dpdT;
  Real dedp = eos_state.dedT * 1.0_rt / eos_state.dpdT;

  qflx[QPRES] = (-q_zone[QMAGX] * flx[UMAGX] - q_zone[QMAGY] * flx[UMAGY] -
                  q_zone[QMAGZ] * flx[UMAGZ] + flx[UEDEN] -
                 flx[UMX] * q_zone[QU] - flx[UMY] * q_zone[QV] - flx[UMZ] * q_zone[QW] +
                 flx[URHO] * (0.5_rt * (q_zone[QU] * q_zone[QU] +
                                        q_zone[QV] * q_zone[QV] +
                                        q_zone[QW] * q_zone[QW]) -
                              eos_state.e - q_zone[QRHO] * dedrho)) /
    (dedp * q_zone[QRHO]);

  qflx[QMAGX] = flx[UMAGX];
  qflx[QMAGY] = flx[UMAGY];
  qflx[QMAGZ] = flx[UMAGZ];

  qflx[QTEMP] = 0.0_rt;
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real
electric(GpuArray<Real, NQ> const& q_zone, const int comp) {

  // this takes the cell-center primitive state, q_zone, and returns component
  // comp of the cell-center electric field, using Faraday's law:
  // E = -v X B

  if (comp == 0) {
    return -q_zone[QV] * q_zone[QMAGZ] + q_zone[QW] * q_zone[QMAGY];

  } else if (comp == 1) {
    return -q_zone[QW] * q_zone[QMAGX] + q_zone[QU] * q_zone[QMAGZ];

  } else {
    return -q_zone[QU] * q_zone[QMAGY] + q_zone[QV] * q_zone[QMAGX];
  }

}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
ConsToPrim(GpuArray<Real, NQ>& q_zone, GpuArray<Real, NUM_STATE+3> const& U_zone) {

  // calculate the primitive variables from the conserved
  // U has NUM_STATE+3 components
  // q has NQ components


  q_zone[QRHO] = U_zone[URHO];
  q_zone[QU] = U_zone[UMX] / q_zone[QRHO];
  q_zone[QV] = U_zone[UMY] / q_zone[QRHO];
  q_zone[QW] = U_zone[UMZ] / q_zone[QRHO];
  q_zone[QRHOEI] = U_zone[UEDEN] -
    0.5_rt * q_zone[QRHO] * (q_zone[QU] * q_zone[QU] +
                             q_zone[QV] * q_zone[QV] +
                             q_zone[QW] * q_zone[QW]) -
    0.5_rt * (U_zone[UMAGX] * U_zone[UMAGX] +
              U_zone[UMAGY] * U_zone[UMAGY] +
              U_zone[UMAGZ] * U_zone[UMAGZ]);

  // species

  for (int n = 0; n < NumSpec; n++) {
    q_zone[QSPEC+n] = U_zone[USPEC+n] / U_zone[URHO];
  }

  eos_t eos_state;

  eos_state.rho = q_zone[QRHO];
  eos_state.e = q_zone[QRHOEI] / eos_state.rho;
  for (int n = 0; n < NumSpec; n++) {
    eos_state.xn[n] = q_zone[QSPEC+n];
  }
#if NAUX_NET > 0
  for (int n = 0; n < NumAux; n++) {
    eos_state.aux[n] = q_zone[QAUXV+n];
  }
#endif
  eos_state.T = 100.0_rt; // initial guess

  eos(eos_input_re, eos_state);

  q_zone[QTEMP] = eos_state.T;
  q_zone[QPRES] = eos_state.p;

  q_zone[QMAGX] = U_zone[UMAGX];
  q_zone[QMAGY] = U_zone[UMAGY];
  q_zone[QMAGZ] = U_zone[UMAGZ];

}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real
PToC(Array1D<Real, 0, NQ-1>& q,
     Array1D<Real, 0, NUM_STATE+2>& U) {

  // fills the conserved state U from the primitive state q and
  // returns the first adiabatic exponent from the EOS call

  // first the conserved state
  U(URHO) = q(QRHO);
  U(UMX) = q(QRHO) * q(QU);
  U(UMY) = q(QRHO) * q(QV);
  U(UMZ) = q(QRHO) * q(QW);

  eos_t eos_state;

  eos_state.rho = q(QRHO);
  eos_state.p = q(QPRES);
  for (int n = 0; n < NumSpec; n++) {
    eos_state.xn[n] = q(QSPEC+n);
  }
#if NAUX_NET > 0
  for (int n = 0; n < NumAux; n++) {
    eos_state.aux[n] = q(QAUXV+n);
  }
#endif
  eos_state.T = 100.0;  // dummy initial guess

  eos(eos_input_rp, eos_state);

  Real B2 = q(QMAGX) * q(QMAGX) + q(QMAGY) * q(QMAGY) + q(QMAGZ) * q(QMAGZ);

  U(UEINT) = eos_state.rho * eos_state.e;
  U(UEDEN) = U(UEINT) +
    0.5_rt * q(QRHO) * (q(QU) * q(QU) + q(QV) * q(QV) + q(QW) * q(QW)) + 0.5_rt * B2;

  U(UMAGX) = q(QMAGX);
  U(UMAGY) = q(QMAGY);
  U(UMAGZ) = q(QMAGZ);

  for (int n = 0; n < NumSpec; n++) {
    U(USPEC+n) = U(URHO) * q(QSPEC+n);
  }

  U(UTEMP) = eos_state.T;

  return eos_state.gam1;

}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int
epsilon_ijk(const int i, const int j, const int k) {

  int val;

  if (i == j || j == k || i == k) {
    val = 0;
  } else if ((i == 0 && j == 1 && k == 2) ||
             (i == 1 && j == 2 && k == 0) ||
             (i == 2 && j == 0 && k == 1)) {
    val = 1;
  } else {
    val = -1;
  }

  return val;
}

namespace mhd {

// conserved state from the primitive state, zone by zone
void
PrimToCons(const amrex::Box& bx,
           amrex::Array4<amrex::Real const> const& q_arr,
           amrex::Array4<amrex::Real> const& u_arr);

// edge-centered electric field components from the cell-centered
// primitive state and the face fluxes in the two transverse directions

void
electric_edge_x(const amrex::Box& bx,
                amrex::Array4<amrex::Real const> const& q_arr,
                amrex::Array4<amrex::Real> const& E,
                amrex::Array4<amrex::Real const> const& flxy,
                amrex::Array4<amrex::Real const> const& flxz);

void
electric_edge_y(const amrex::Box& bx,
                amrex::Array4<amrex::Real const> const& q_arr,
                amrex::Array4<amrex::Real> const& E,
                amrex::Array4<amrex::Real const> const& flxx,
                amrex::Array4<amrex::Real const> const& flxz);

void
electric_edge_z(const amrex::Box& bx,
                amrex::Array4<amrex::Real const> const& q_arr,
                amrex::Array4<amrex::Real> const& E,
                amrex::Array4<amrex::Real const> const& flxx,
                amrex::Array4<amrex::Real const> const& flxy);

// HLLD fluxes in direction dir from the left and right interface states
void
hlld(const amrex::Box& bx,
     amrex::Array4<amrex::Real const> const& qleft,
     amrex::Array4<amrex::Real const> const& qright,
     amrex::Array4<amrex::Real> const& flx,
     const int dir);

}

#endif

