#include <HydroSim.H>

#include <mhd_zone_ops.H>

using namespace amrex;

struct MhdAxis {
  int QBNORM, QBPERP1, QBPERP2;
  int QVNORM, QVPERP1, QVPERP2;
  int UMN, UMPERP1, UMPERP2;
  int UBNORM, UBPERP1, UBPERP2;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
mhd_normal_flux(const Array1D<Real, 0, NQ-1>& q, const MhdConsVector& u,
                const Real B2, const Real UB, const MhdAxis& axis,
                MhdConsVector& F) {

    F(URHO) = q(QRHO) * q(axis.QVNORM);
    F(axis.UMN) = q(QRHO) * q(axis.QVNORM) * q(axis.QVNORM) + (q(QPRES) + 0.5_rt * B2) - q(axis.QBNORM) * q(axis.QBNORM);
    F(axis.UMPERP1) = q(QRHO) * q(axis.QVNORM) * q(axis.QVPERP1) - q(axis.QBNORM) * q(axis.QBPERP1);
    F(axis.UMPERP2) = q(QRHO) * q(axis.QVNORM) * q(axis.QVPERP2) - q(axis.QBNORM) * q(axis.QBPERP2);
    F(UEDEN) = q(axis.QVNORM) * (u(UEDEN) + (q(QPRES) + 0.5_rt * B2)) - q(axis.QBNORM) * UB;
    F(axis.UBNORM) = 0.0;
    F(axis.UBPERP1) = q(axis.QVNORM) * q(axis.QBPERP1) - q(axis.QVPERP1) * q(axis.QBNORM);
    F(axis.UBPERP2) = q(axis.QVNORM) * q(axis.QBPERP2) - q(axis.QVPERP2) * q(axis.QBNORM);
    for (int n = 0; n < NumSpec; n++) {
      F(UFS+n) = q(axis.QVNORM) * u(UFS+n);
    }
    F(UEINT) = q(axis.QVNORM) * u(UEINT);
    F(UTEMP) = 0.0_rt;
}

void
HydroSim::mhd_riemann_flux(const Box& bx,
             Array4<Real const> const& prim_left,
             Array4<Real const> const& prim_right,
             Array4<Real> const& flx,
             const int dir) {





  constexpr int bfield_comp[3] = {QBFIELDX, QBFIELDY, QBFIELDZ};
  constexpr int vel_comp[3] = {QU, QV, QW};
  constexpr int mom_comp[3] = {UMX, UMY, UMZ};
  constexpr int bflux_comp[3] = {UBX, UBY, UBZ};

  const int perp1 = (dir + 1) % 3;
  const int perp2 = (dir + 2) % 3;

  const int QBNORM  = bfield_comp[dir];
  const int QBPERP1 = bfield_comp[perp1];
  const int QBPERP2 = bfield_comp[perp2];
  const int QVNORM  = vel_comp[dir];
  const int QVPERP1 = vel_comp[perp1];
  const int QVPERP2 = vel_comp[perp2];
  const int UMN     = mom_comp[dir];
  const int UMPERP1 = mom_comp[perp1];
  const int UMPERP2 = mom_comp[perp2];
  const int UBNORM  = bflux_comp[dir];
  const int UBPERP1 = bflux_comp[perp1];
  const int UBPERP2 = bflux_comp[perp2];

  const MhdAxis axis{QBNORM, QBPERP1, QBPERP2,
                     QVNORM, QVPERP1, QVPERP2,
                     UMN, UMPERP1, UMPERP2,
                     UBNORM, UBPERP1, UBPERP2};

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {


    Array1D<Real, 0, NQ-1> qL;
    Array1D<Real, 0, NQ-1> qR;

    for (int n = 0; n < NQ; n++) {
      qL(n) = prim_left(i,j,k,n);
      qR(n) = prim_right(i,j,k,n);
    }

    qL(QRHO) = amrex::max(small_dens, qL(QRHO));
    qR(QRHO) = amrex::max(small_dens, qR(QRHO));
    qL(QPRES) = amrex::max(pressure_cutoff, qL(QPRES));
    qR(QPRES) = amrex::max(pressure_cutoff, qR(QPRES));

    MhdConsVector uL;
    Real gamma1_L;

    PrimToConsAndGamma(qL, uL, gamma1_L);

    Real BL2 = qL(QBFIELDX) * qL(QBFIELDX) + qL(QBFIELDY) * qL(QBFIELDY) + qL(QBFIELDZ) * qL(QBFIELDZ);
    Real UBL = qL(QBFIELDX) * qL(QU) + qL(QBFIELDY) * qL(QV) + qL(QBFIELDZ) * qL(QW);
    MhdConsVector FL;

    mhd_normal_flux(qL, uL, BL2, UBL, axis, FL);

    MhdConsVector uR;
    Real gamma1_R;

    PrimToConsAndGamma(qR, uR, gamma1_R);

    Real BR2 = qR(QBFIELDX) * qR(QBFIELDX) + qR(QBFIELDY) * qR(QBFIELDY) + qR(QBFIELDZ) * qR(QBFIELDZ);
    Real UBR = qR(QBFIELDX) * qR(QU) + qR(QBFIELDY) * qR(QV) + qR(QBFIELDZ) * qR(QW);

    MhdConsVector FR;

    mhd_normal_flux(qR, uR, BR2, UBR, axis, FR);

    Real asL = gamma1_L * qL(QPRES) / qL(QRHO);
    Real asR = gamma1_R * qR(QPRES) / qR(QRHO);

    Real caL  = BL2 / qL(QRHO);
    Real caR  = BR2 / qR(QRHO);

    Real normalAlfvenSqL = qL(QBNORM) * qL(QBNORM) / qL(QRHO);
    Real normalAlfvenSqR = qR(QBNORM) * qR(QBNORM) / qR(QRHO);


    Real fastSpeedL = std::sqrt(0.5_rt * ((asL + caL) + std::sqrt((asL + caL)*(asL + caL) - 4.0_rt * asL * normalAlfvenSqL)));
    Real cfR = std::sqrt(0.5_rt * ((asR + caR) + std::sqrt((asR + caR)*(asR + caR) - 4.0_rt * asR * normalAlfvenSqR)));


    Real sL = amrex::min(qL(QVNORM) - fastSpeedL, qR(QVNORM) - cfR);
    Real sR = amrex::max(qL(QVNORM) + fastSpeedL, qR(QVNORM) + cfR);


    Real ptL = qL(QPRES) + 0.5_rt * BL2;
    Real ptR = qR(QPRES) + 0.5_rt * BR2;


    Real sM  = ((sR - qR(QVNORM)) * qR(QRHO) * qR(QVNORM) - (sL - qL(QVNORM)) * qL(QRHO) * qL(QVNORM) -
           ptR + ptL) /
      ((sR - qR(QVNORM)) * qR(QRHO) - (sL - qL(QVNORM)) * qL(QRHO));



    Real pst  = (sR - qR(QVNORM)) * qR(QRHO) * ptL - (sL - qL(QVNORM)) * qL(QRHO) * ptR +
      qL(QRHO) * qR(QRHO) * (sR - qR(QVNORM)) * (sL - qL(QVNORM)) * (qR(QVNORM) - qL(QVNORM));
    pst  = pst / ((sR - qR(QVNORM)) * qR(QRHO) - (sL - qL(QVNORM)) * qL(QRHO));


    MhdConsVector UsL;
    MhdConsVector UsR;


    UsL(URHO) = amrex::max(small_dens, qL(QRHO) * ((sL - qL(QVNORM)) / (sL - sM)));
    UsR(URHO) = amrex::max(small_dens, qR(QRHO) * ((sR - qR(QVNORM)) / (sR - sM)));


    for (int n = 0; n < NumSpec; n++) {
      UsL(UFS+n) = qL(QFS+n) * UsL(URHO);
      UsR(UFS+n) = qR(QFS+n) * UsR(URHO);
    }


    UsL(UEINT) = uL(UEINT) / uL(URHO) * UsL(URHO);
    UsR(UEINT) = uR(UEINT) / uR(URHO) * UsR(URHO);



    UsL(UMN) = sM;
    UsR(UMN) = sM;


    Real denom_vperp_L = qL(QRHO) * (sL - qL(QVNORM)) * (sL - sM) - qL(QBNORM) * qL(QBNORM);
    Real denom_vperp_R = qR(QRHO) * (sR - qR(QVNORM)) * (sR - sM) - qR(QBNORM)*qR(QBNORM);

    if (std::abs(denom_vperp_L) < 1.e-14_rt) {
      UsL(UMPERP1) = qL(QVPERP1);
      UsL(UMPERP2) = qL(QVPERP2);

    } else {
      UsL(UMPERP1) = qL(QVPERP1) - qL(QBNORM)*qL(QBPERP1) *
        ((sM - qL(QVNORM)) / denom_vperp_L);
      UsL(UMPERP2) = qL(QVPERP2) - qL(QBNORM) * qL(QBPERP2) *
        ((sM - qL(QVNORM)) / denom_vperp_L);

    }

    if (std::abs(denom_vperp_R) < 1.e-14_rt) {
      UsR(UMPERP1) = qR(QVPERP1);
      UsR(UMPERP2) = qR(QVPERP2);

    } else {
      UsR(UMPERP1) = qR(QVPERP1) - qR(QBNORM)*qR(QBPERP1) *
        ((sM - qR(QVNORM)) / denom_vperp_R);
      UsR(UMPERP2) = qR(QVPERP2) - qR(QBNORM) * qR(QBPERP2) *
        ((sM - qR(QVNORM)) / denom_vperp_R);

    }


    UsL(UMX) = UsL(UMX) * UsL(URHO);
    UsL(UMY) = UsL(UMY) * UsL(URHO);
    UsL(UMZ) = UsL(UMZ) * UsL(URHO);

    UsR(UMX) = UsR(UMX) * UsR(URHO);
    UsR(UMY) = UsR(UMY) * UsR(URHO);
    UsR(UMZ) = UsR(UMZ) * UsR(URHO);



    UsL(UBNORM) = qL(QBNORM);
    UsR(UBNORM) = qR(QBNORM);


    Real denom_bperp_L = qL(QRHO) * (sL - qL(QVNORM)) * (sL - sM) - qL(QBNORM) * qL(QBNORM);
    Real denom_bperp_R = qR(QRHO) * (sR - qR(QVNORM)) * (sR - sM) - qR(QBNORM) * qR(QBNORM);

    if (std::abs(denom_bperp_L) < 1.e-14_rt) {
      UsL(UBPERP1) = 0.0_rt;
      UsL(UBPERP2) = 0.0_rt;

    } else {
      UsL(UBPERP1) = qL(QBPERP1) * (qL(QRHO) * (sL - qL(QVNORM)) * (sL - qL(QVNORM)) - qL(QBNORM) * qL(QBNORM)) /
                    denom_bperp_L;
      UsL(UBPERP2) = qL(QBPERP2) * (qL(QRHO) * (sL - qL(QVNORM)) * (sL - qL(QVNORM)) - qL(QBNORM) * qL(QBNORM)) /
                    denom_bperp_L;

    }

    if (std::abs(denom_bperp_R) < 1.e-14_rt) {
      UsR(UBPERP1) = 0.0_rt;
      UsR(UBPERP2) = 0.0_rt;

    } else {
      UsR(UBPERP1) = qR(QBPERP1) * (qR(QRHO) * (sR - qR(QVNORM)) * (sR - qR(QVNORM)) - qR(QBNORM) * qR(QBNORM)) /
                    denom_bperp_R;
      UsR(UBPERP2) = qR(QBPERP2) * (qR(QRHO) * (sR - qR(QVNORM)) * (sR - qR(QVNORM)) - qR(QBNORM) * qR(QBNORM)) /
                    denom_bperp_R;

    }



    UsL(UEDEN) = (sL - qL(QVNORM)) * uL(UEDEN) - ptL * qL(QVNORM) + pst * sM +
      qL(QBNORM) * (UBL - (UsL(UMX) * UsL(UBX) + UsL(UMY) * UsL(UBY) + UsL(UMZ) * UsL(UBZ)) / UsL(URHO));
    UsL(UEDEN) = UsL(UEDEN) / (sL - sM);

    UsR(UEDEN) = (sR - qR(QVNORM)) * uR(UEDEN) - ptR * qR(QVNORM) + pst * sM +
      qR(QBNORM) * (UBR - (UsR(UMX) * UsR(UBX) + UsR(UMY) * UsR(UBY) + UsR(UMZ) * UsR(UBZ)) / UsR(URHO));
    UsR(UEDEN) = UsR(UEDEN) / (sR - sM);

    UsL(UTEMP) = 0.0_rt;
    UsR(UTEMP) = 0.0_rt;


    Real ssL = sM - std::abs(qL(QBNORM)) / std::sqrt(UsL(URHO));
    Real ssR = sM + std::abs(qR(QBNORM)) / std::sqrt(UsR(URHO));


    MhdConsVector DoubleStarL;
    MhdConsVector DoubleStarR;


    DoubleStarL(URHO) = UsL(URHO);
    DoubleStarR(URHO) = UsR(URHO);


    for (int n = 0; n < NumSpec; n++) {
      DoubleStarL(UFS+n) = UsL(UFS+n);
      DoubleStarR(UFS+n) = UsR(UFS+n);
    }

    DoubleStarL(UEINT) = UsL(UEINT);
    DoubleStarR(UEINT) = UsR(UEINT);


    DoubleStarL(UMN) = sM;
    DoubleStarR(UMN) = sM;


    DoubleStarL(UMPERP1) = (std::sqrt(UsL(URHO)) * UsL(UMPERP1) / UsL(URHO) +
                  std::sqrt(UsR(URHO)) * UsR(UMPERP1) / UsR(URHO) +
                  (UsR(UBPERP1) - UsL(UBPERP1)) * std::copysign(1.0_rt, qL(QBNORM))) /
      (std::sqrt(UsL(URHO)) + std::sqrt(UsR(URHO)));
    DoubleStarR(UMPERP1) = DoubleStarL(UMPERP1);


    DoubleStarL(UMPERP2) = (std::sqrt(UsL(URHO)) * UsL(UMPERP2) / UsL(URHO) +
                  std::sqrt(UsR(URHO)) * UsR(UMPERP2) / UsR(URHO) +
                  (UsR(UBPERP2) - UsL(UBPERP2)) * std::copysign(1.0_rt, qL(QBNORM))) /
      (std::sqrt(UsL(URHO)) + std::sqrt(UsR(URHO)));
    DoubleStarR(UMPERP2) = DoubleStarL(UMPERP2);

    DoubleStarL(UMX) = DoubleStarL(UMX) * DoubleStarL(URHO);
    DoubleStarL(UMY) = DoubleStarL(UMY) * DoubleStarL(URHO);
    DoubleStarL(UMZ) = DoubleStarL(UMZ) * DoubleStarL(URHO);

    DoubleStarR(UMX) = DoubleStarR(UMX) * DoubleStarR(URHO);
    DoubleStarR(UMY) = DoubleStarR(UMY) * DoubleStarR(URHO);
    DoubleStarR(UMZ) = DoubleStarR(UMZ) * DoubleStarR(URHO);


    DoubleStarL(UBNORM) = UsL(UBNORM);
    DoubleStarR(UBNORM) = UsR(UBNORM);


    DoubleStarL(UBPERP1) = (std::sqrt(UsL(URHO)) * UsR(UBPERP1) + std::sqrt(UsR(URHO)) * UsL(UBPERP1) +
                    std::sqrt(UsL(URHO) * UsR(URHO)) * (UsR(UMPERP1) / UsR(URHO) -
                                                        UsL(UMPERP1) / UsL(URHO)) * std::copysign(1.0_rt, qL(QBNORM))) /
      (std::sqrt(UsL(URHO)) + std::sqrt(UsR(URHO)));

    DoubleStarR(UBPERP1) = DoubleStarL(UBPERP1);


    DoubleStarL(UBPERP2) = (std::sqrt(UsL(URHO)) * UsR(UBPERP2) + std::sqrt(UsR(URHO)) * UsL(UBPERP2) +
                    std::sqrt(UsL(URHO) * UsR(URHO)) * (UsR(UMPERP2) / UsR(URHO) -
                                                        UsL(UMPERP2) / UsL(URHO)) * std::copysign(1.0_rt, qL(QBNORM))) /
      (std::sqrt(UsL(URHO)) + std::sqrt(UsR(URHO)));
    DoubleStarR(UBPERP2) = DoubleStarL(UBPERP2);


    DoubleStarL(UEDEN) = UsL(UEDEN) - std::sqrt(UsL(URHO)) *
      ((UsL(UMX) * UsL(UBX) + UsL(UMY) * UsL(UBY) + UsL(UMZ) * UsL(UBZ)) / UsL(URHO) -
       (DoubleStarL(UMX) * DoubleStarL(UBX) + DoubleStarL(UMY) * DoubleStarL(UBY) + DoubleStarL(UMZ) * DoubleStarL(UBZ)) / DoubleStarL(URHO)) *
      std::copysign(1.0_rt, qL(QBNORM));
    DoubleStarR(UEDEN) = UsR(UEDEN) + std::sqrt(UsR(URHO)) *
      ((UsR(UMX) * UsR(UBX) + UsR(UMY) * UsR(UBY) + UsR(UMZ) * UsR(UBZ)) / UsR(URHO) -
       (DoubleStarR(UMX) * DoubleStarR(UBX) + DoubleStarR(UMY) * DoubleStarR(UBY) + DoubleStarR(UMZ) * DoubleStarR(UBZ)) / DoubleStarR(URHO)) *
      std::copysign(1.0_rt, qR(QBNORM));

    DoubleStarL(UTEMP) = 0.0_rt;
    DoubleStarR(UTEMP) = 0.0_rt;



    if (sL > 0.0) {
      for (int n = 0; n < NUM_STATE+3; n++) {
        flx(i,j,k,n) = FL(n);
      }

    } else if (sL <= 0.0 && ssL > 0.0) {
      for (int n = 0; n < NUM_STATE+3; n++) {
        flx(i,j,k,n) = FL(n) + sL * UsL(n) - sL * uL(n);
      }

    } else if (ssL <= 0.0 && sM > 0.0) {
      for (int n = 0; n < NUM_STATE+3; n++) {
        flx(i,j,k,n) = FL(n) + ssL * DoubleStarL(n) - (ssL - sL) * UsL(n) - sL * uL(n);
      }

    } else if (sM <= 0.0 && ssR > 0.0) {
      for (int n = 0; n < NUM_STATE+3; n++) {
        flx(i,j,k,n) = FR(n) + ssR * DoubleStarR(n) - (ssR - sR) * UsR(n) - sR * uR(n);
      }

    } else if (ssR <= 0.0 && sR > 0.0) {
      for (int n = 0; n < NUM_STATE+3; n++) {
        flx(i,j,k,n) = FR(n) + sR * UsR(n) - sR * uR(n);
      }

    } else {
      for (int n = 0; n < NUM_STATE+3; n++) {
        flx(i,j,k,n) = FR(n);
      }
    }

    flx(i,j,k,UTEMP) = 0.0;
  });
}
