#ifndef HYDROSIM_MHD_UTIL_H
#define HYDROSIM_MHD_UTIL_H

#include <HydroSim_helpers.H>

using namespace amrex;

constexpr int UBX = NUM_STATE;
constexpr int UBY = NUM_STATE+1;
constexpr int UBZ = NUM_STATE+2;
constexpr int EIGVEC_RHO = 0;
constexpr int EIGVEC_U = 1;
constexpr int EIGVEC_V = 2;
constexpr int EIGVEC_W = 3;
constexpr int EIGVEC_PRESSURE = 4;
constexpr int EIGVEC_BPERP1 = 5;
constexpr int EIGVEC_BPERP2 = 6;
constexpr int NUM_EIGVEC = 7;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int
levi_civita_sign(const int i, const int j, const int k) {

  constexpr int eps[3][3][3] = {{{0, 0, 0}, {0, 0, 1}, {0, -1, 0}},
                                {{0, 0, -1}, {0, 0, 0}, {1, 0, 0}},
                                {{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}}};

  return eps[i][j][k];
}

template <typename RealT>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
fast_magnetosonic_speed(RealT& c, RealT as, RealT ca, RealT bd) {


  c = 0.5_rt * ((as + ca) + std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*bd));
  c = std::sqrt(c);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
ConservedToPrimitive(Real* zone_prim, const Real* cons_zone) {



  zone_prim[QRHO] = cons_zone[URHO];
  zone_prim[QU] = cons_zone[UMX] / zone_prim[QRHO];
  zone_prim[QV] = cons_zone[UMY] / zone_prim[QRHO];
  zone_prim[QW] = cons_zone[UMZ] / zone_prim[QRHO];
  zone_prim[QEINTDENS] = cons_zone[UEDEN] -
    0.5_rt * zone_prim[QRHO] * (zone_prim[QU] * zone_prim[QU] +
                             zone_prim[QV] * zone_prim[QV] +
                             zone_prim[QW] * zone_prim[QW]) -
    0.5_rt * (cons_zone[UBX] * cons_zone[UBX] +
              cons_zone[UBY] * cons_zone[UBY] +
              cons_zone[UBZ] * cons_zone[UBZ]);


  for (int n = 0; n < NumSpec; n++) {
    zone_prim[QFS+n] = cons_zone[UFS+n] / cons_zone[URHO];
  }

  eos_t eos_state;

  eos_state.rho = zone_prim[QRHO];
  eos_state.e = zone_prim[QEINTDENS] / eos_state.rho;
  for (int n = 0; n < NumSpec; n++) {
    eos_state.xn[n] = zone_prim[QFS+n];
  }
#if NAUX_NET > 0
  for (int n = 0; n < NumAux; n++) {
    eos_state.aux[n] = zone_prim[QFX+n];
  }
#endif
  eos_state.T = 100.0_rt;

  eos(eos_input_re, eos_state);

  zone_prim[QTEMP] = eos_state.T;
  zone_prim[QPRES] = eos_state.p;

  zone_prim[QBFIELDX] = cons_zone[UBX];
  zone_prim[QBFIELDY] = cons_zone[UBY];
  zone_prim[QBFIELDZ] = cons_zone[UBZ];

}

using MhdConsVector = Array1D<Real, 0, NUM_STATE+2>;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
PrimToConsAndGamma(Array1D<Real, 0, NQ-1>& q,
     MhdConsVector& U,
     Real& gam1) {

  U(URHO) = q(QRHO);
  U(UMX) = q(QRHO) * q(QU);
  U(UMY) = q(QRHO) * q(QV);
  U(UMZ) = q(QRHO) * q(QW);

  eos_t eos_state;

  eos_state.rho = q(QRHO);
  eos_state.p = q(QPRES);
  for (int n = 0; n < NumSpec; n++) {
    eos_state.xn[n] = q(QFS+n);
  }
#if NAUX_NET > 0
  for (int n = 0; n < NumAux; n++) {
    eos_state.aux[n] = q(QFX+n);
  }
#endif
  eos_state.T = 100.0;

  eos(eos_input_rp, eos_state);

  Real B2 = q(QBFIELDX) * q(QBFIELDX) + q(QBFIELDY) * q(QBFIELDY) + q(QBFIELDZ) * q(QBFIELDZ);

  U(UEINT) = eos_state.rho * eos_state.e;
  U(UEDEN) = U(UEINT) +
    0.5_rt * q(QRHO) * (q(QU) * q(QU) + q(QV) * q(QV) + q(QW) * q(QW)) + 0.5_rt * B2;

  U(UBX) = q(QBFIELDX);
  U(UBY) = q(QBFIELDY);
  U(UBZ) = q(QBFIELDZ);

  for (int n = 0; n < NumSpec; n++) {
    U(UFS+n) = U(URHO) * q(QFS+n);
  }

  U(UTEMP) = eos_state.T;

  gam1 = eos_state.gam1;

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
to_prim_flux(GpuArray<Real, NQ>& prim_flux, GpuArray<Real, NUM_STATE+3> const& flx, GpuArray<Real, NQ> const& zone_prim) {



  prim_flux[QRHO] = flx[URHO];
  prim_flux[QU] = (flx[UMX] - flx[URHO] * zone_prim[QU]) / zone_prim[QRHO];
  prim_flux[QV] = (flx[UMY] - flx[URHO] * zone_prim[QV]) / zone_prim[QRHO];
  prim_flux[QW] = (flx[UMZ] - flx[URHO] * zone_prim[QW]) / zone_prim[QRHO];

  for (int n = 0; n < NumSpec; n++) {
    prim_flux[QFS+n] = (flx[UFS+n] - flx[URHO] * zone_prim[QFS+n]) / zone_prim[QRHO];
  }

  eos_t eos_state;
  eos_state.rho = zone_prim[QRHO];
  eos_state.p = zone_prim[QPRES];
  eos_state.T = 100.0_rt;
  for (int n = 0; n < NumSpec; n++) {
    eos_state.xn[n] = zone_prim[QFS+n];
  }
#if NAUX_NET > 0
  for (int n = 0; n < NumAux; n++) {
    eos_state.aux[n] = zone_prim[QFX+n];
  }
#endif

  eos(eos_input_rp, eos_state);

  Real e_wrt_rho = eos_state.dedr - eos_state.dedT * eos_state.dpdr * 1.0_rt / eos_state.dpdT;
  Real e_wrt_p = eos_state.dedT * 1.0_rt / eos_state.dpdT;

  prim_flux[QPRES] = (-zone_prim[QBFIELDX] * flx[UBX] - zone_prim[QBFIELDY] * flx[UBY] -
                  zone_prim[QBFIELDZ] * flx[UBZ] + flx[UEDEN] -
                 flx[UMX] * zone_prim[QU] - flx[UMY] * zone_prim[QV] - flx[UMZ] * zone_prim[QW] +
                 flx[URHO] * (0.5_rt * (zone_prim[QU] * zone_prim[QU] +
                                        zone_prim[QV] * zone_prim[QV] +
                                        zone_prim[QW] * zone_prim[QW]) -
                              eos_state.e - zone_prim[QRHO] * e_wrt_rho)) /
    (e_wrt_p * zone_prim[QRHO]);

  prim_flux[QBFIELDX] = flx[UBX];
  prim_flux[QBFIELDY] = flx[UBY];
  prim_flux[QBFIELDZ] = flx[UBZ];

  prim_flux[QTEMP] = 0.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real
electric(const Real* zone_prim, const int comp) {

  if (comp == 0) {
    return -zone_prim[QV] * zone_prim[QBFIELDZ] + zone_prim[QW] * zone_prim[QBFIELDY];

  } else if (comp == 1) {
    return -zone_prim[QW] * zone_prim[QBFIELDX] + zone_prim[QU] * zone_prim[QBFIELDZ];

  }

  return -zone_prim[QU] * zone_prim[QBFIELDY] + zone_prim[QV] * zone_prim[QBFIELDX];
}

#endif
