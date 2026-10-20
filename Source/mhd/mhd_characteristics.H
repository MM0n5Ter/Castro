#ifndef HYDROSIM_MHD_CHARACTERISTICS_H
#define HYDROSIM_MHD_CHARACTERISTICS_H

#include <mhd_zone_ops.H>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
eigenbasis_x(Array2D<Real, 0, NUM_EIGVEC-1, 0, NUM_EIGVEC-1>& left_basis,
      Array2D<Real, 0, NUM_EIGVEC-1, 0, NUM_EIGVEC-1>& right_basis,
      const Real sound_speed,
      Array1D<Real, 0, NQ-1>& Q) {



  Real as = sound_speed * sound_speed;


  Real ca = (Q(QBFIELDX)*Q(QBFIELDX) + Q(QBFIELDY)*Q(QBFIELDY) + Q(QBFIELDZ)*Q(QBFIELDZ)) / Q(QRHO);
  Real cax = (Q(QBFIELDX)*Q(QBFIELDX)) / Q(QRHO);


  Real csx = 0.5_rt*((as + ca) - std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*cax));


  Real cfx = 0.5_rt*((as + ca) + std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*cax));


  Real alf;
  Real als;


  if (std::abs(cfx - csx) <= 1e-14){
    alf = 1.0_rt;
    als = 0.0_rt;
  } else {
      if (as - csx < 0.0) {
          alf = 0.0_rt;
      } else {
          alf = std::sqrt((as - csx)/(cfx - csx));
      }

      if (cfx - as < 0.0) {
          als = 0.0_rt;
      } else {
          als = std::sqrt((cfx - as)/(cfx - csx));
      }
  }



  Real dir_cos_y;
  Real dir_cos_z;

  if (std::abs(Q(QBFIELDY)) <= 1.e-14_rt && std::abs(Q(QBFIELDZ)) <= 1.e-14_rt) {
    dir_cos_y = 1.0_rt / std::sqrt(2.0_rt);
    dir_cos_z = dir_cos_y;

  } else {
    dir_cos_y = Q(QBFIELDY) / (std::sqrt(Q(QBFIELDY)*Q(QBFIELDY) + Q(QBFIELDZ)*Q(QBFIELDZ)));
    dir_cos_z = Q(QBFIELDZ) / (std::sqrt(Q(QBFIELDY)*Q(QBFIELDY) + Q(QBFIELDZ)*Q(QBFIELDZ)));
  }

  Real cff = std::sqrt(cfx) * alf;
  Real css = std::sqrt(csx) * als;

  Real S = std::copysign(1.0_rt, Q(QBFIELDX));

  Real Qf = std::sqrt(cfx) * alf * S;
  Real Qs = std::sqrt(csx) * als * S;

  Real N = 0.5_rt / as;

  Real AAf = std::sqrt(as) * alf * std::sqrt(Q(QRHO));
  Real AAs = std::sqrt(as) * als * std::sqrt(Q(QRHO));


  left_basis(0, EIGVEC_RHO) = 0.0;
  left_basis(0, EIGVEC_U) = -N*cff;
  left_basis(0, EIGVEC_V) = N*Qs*dir_cos_y;
  left_basis(0, EIGVEC_W) = N*Qs*dir_cos_z;
  left_basis(0, EIGVEC_PRESSURE) = N*alf/Q(QRHO);
  left_basis(0, EIGVEC_BPERP1) = N*AAs*dir_cos_y/Q(QRHO);
  left_basis(0, EIGVEC_BPERP2) = N*AAs*dir_cos_z/Q(QRHO);

  left_basis(1, EIGVEC_RHO) = 0.0;
  left_basis(1, EIGVEC_U) = 0.0;
  left_basis(1, EIGVEC_V) = -0.5_rt*dir_cos_z;
  left_basis(1, EIGVEC_W) = 0.5_rt*dir_cos_y;
  left_basis(1, EIGVEC_PRESSURE) = 0.0;
  left_basis(1, EIGVEC_BPERP1) = -0.5_rt*dir_cos_z*S/std::sqrt(Q(QRHO));
  left_basis(1, EIGVEC_BPERP2) = 0.5_rt*dir_cos_y*S/std::sqrt(Q(QRHO));

  left_basis(2, EIGVEC_RHO) = 0.0;
  left_basis(2, EIGVEC_U) = -N*css;
  left_basis(2, EIGVEC_V) = -N*Qf*dir_cos_y;
  left_basis(2, EIGVEC_W) = -N*Qf*dir_cos_z;
  left_basis(2, EIGVEC_PRESSURE) = N*als/Q(QRHO);
  left_basis(2, EIGVEC_BPERP1) = -N*AAf*dir_cos_y/Q(QRHO);
  left_basis(2, EIGVEC_BPERP2) = -N*AAf*dir_cos_z/Q(QRHO);

  left_basis(3, EIGVEC_RHO) = 1.0;
  left_basis(3, EIGVEC_U) = 0.0;
  left_basis(3, EIGVEC_V) = 0.0;
  left_basis(3, EIGVEC_W) = 0.0;
  left_basis(3, EIGVEC_PRESSURE) = -1.0_rt/as;
  left_basis(3, EIGVEC_BPERP1) = 0.0;
  left_basis(3, EIGVEC_BPERP2) = 0.0;

  left_basis(4, EIGVEC_RHO) = 0.0;
  left_basis(4, EIGVEC_U) = N*css;
  left_basis(4, EIGVEC_V) = N*Qf*dir_cos_y;
  left_basis(4, EIGVEC_W) = N*Qf*dir_cos_z;
  left_basis(4, EIGVEC_PRESSURE) = N*als/Q(QRHO);
  left_basis(4, EIGVEC_BPERP1) = -N*AAf*dir_cos_y/Q(QRHO);
  left_basis(4, EIGVEC_BPERP2) = -N*AAf*dir_cos_z/Q(QRHO);

  left_basis(5, EIGVEC_RHO) = 0.0;
  left_basis(5, EIGVEC_U) = 0.0;
  left_basis(5, EIGVEC_V) = 0.5_rt*dir_cos_z;
  left_basis(5, EIGVEC_W) = -0.5_rt*dir_cos_y;
  left_basis(5, EIGVEC_PRESSURE) = 0.0;
  left_basis(5, EIGVEC_BPERP1) = -0.5_rt*dir_cos_z*S/std::sqrt(Q(QRHO));
  left_basis(5, EIGVEC_BPERP2) = 0.5_rt*dir_cos_y*S/std::sqrt(Q(QRHO));

  left_basis(6, EIGVEC_RHO) = 0.0;
  left_basis(6, EIGVEC_U) = N*cff;
  left_basis(6, EIGVEC_V) = -N*Qs*dir_cos_y;
  left_basis(6, EIGVEC_W) = -N*Qs*dir_cos_z;
  left_basis(6, EIGVEC_PRESSURE) = N*alf/Q(QRHO);
  left_basis(6, EIGVEC_BPERP1) = N*AAs*dir_cos_y/Q(QRHO);
  left_basis(6, EIGVEC_BPERP2) = N*AAs*dir_cos_z/Q(QRHO);


  right_basis(EIGVEC_RHO, 0) = Q(QRHO)*alf;
  right_basis(EIGVEC_RHO, 1) = 0.0;
  right_basis(EIGVEC_RHO, 2) = Q(QRHO)*als;
  right_basis(EIGVEC_RHO, 3) = 1.0_rt;
  right_basis(EIGVEC_RHO, 4) = Q(QRHO)*als;
  right_basis(EIGVEC_RHO, 5) = 0.0;
  right_basis(EIGVEC_RHO, 6) = Q(QRHO)*alf;

  right_basis(EIGVEC_U, 0) = -cff;
  right_basis(EIGVEC_U, 1) = 0.0;
  right_basis(EIGVEC_U, 2) = -css;
  right_basis(EIGVEC_U, 3) = 0.0;
  right_basis(EIGVEC_U, 4) = css;
  right_basis(EIGVEC_U, 5) = 0.0;
  right_basis(EIGVEC_U, 6) = cff;

  right_basis(EIGVEC_V, 0) = Qs*dir_cos_y;
  right_basis(EIGVEC_V, 1) = -dir_cos_z;
  right_basis(EIGVEC_V, 2) = -Qf*dir_cos_y;
  right_basis(EIGVEC_V, 3) = 0.0;
  right_basis(EIGVEC_V, 4) = Qf*dir_cos_y;
  right_basis(EIGVEC_V, 5) = dir_cos_z;
  right_basis(EIGVEC_V, 6) = -Qs*dir_cos_y;

  right_basis(EIGVEC_W, 0) = Qs*dir_cos_z;
  right_basis(EIGVEC_W, 1) = dir_cos_y;
  right_basis(EIGVEC_W, 2) = -Qf*dir_cos_z;
  right_basis(EIGVEC_W, 3) = 0.0;
  right_basis(EIGVEC_W, 4) = Qf*dir_cos_z;
  right_basis(EIGVEC_W, 5) = -dir_cos_y;
  right_basis(EIGVEC_W, 6) = -Qs*dir_cos_z;

  right_basis(EIGVEC_PRESSURE, 0) = Q(QRHO)*as*alf;
  right_basis(EIGVEC_PRESSURE, 1) = 0.0;
  right_basis(EIGVEC_PRESSURE, 2) = Q(QRHO)*as*als;
  right_basis(EIGVEC_PRESSURE, 3) = 0.0;
  right_basis(EIGVEC_PRESSURE, 4) = Q(QRHO)*as*als;
  right_basis(EIGVEC_PRESSURE, 5) = 0.0;
  right_basis(EIGVEC_PRESSURE, 6) = Q(QRHO)*as*alf;

  right_basis(EIGVEC_BPERP1, 0) = AAs*dir_cos_y;
  right_basis(EIGVEC_BPERP1, 1) = -dir_cos_z*S*std::sqrt(Q(QRHO));
  right_basis(EIGVEC_BPERP1, 2) = -AAf*dir_cos_y;
  right_basis(EIGVEC_BPERP1, 3) = 0.0;
  right_basis(EIGVEC_BPERP1, 4) = -AAf*dir_cos_y;
  right_basis(EIGVEC_BPERP1, 5) = -dir_cos_z*S*std::sqrt(Q(QRHO));
  right_basis(EIGVEC_BPERP1, 6) = AAs*dir_cos_y;

  right_basis(EIGVEC_BPERP2, 0) = AAs*dir_cos_z;
  right_basis(EIGVEC_BPERP2, 1) = dir_cos_y*S*std::sqrt(Q(QRHO));
  right_basis(EIGVEC_BPERP2, 2) = -AAf*dir_cos_z;
  right_basis(EIGVEC_BPERP2, 3) = 0.0;
  right_basis(EIGVEC_BPERP2, 4) = -AAf*dir_cos_z;
  right_basis(EIGVEC_BPERP2, 5) = dir_cos_y*S*std::sqrt(Q(QRHO));
  right_basis(EIGVEC_BPERP2, 6) = AAs*dir_cos_z;

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
eigenbasis_y(Array2D<Real, 0, NUM_EIGVEC-1, 0, NUM_EIGVEC-1>& left_basis,
      Array2D<Real, 0, NUM_EIGVEC-1, 0, NUM_EIGVEC-1>& right_basis,
      const Real sound_speed,
      Array1D<Real, 0, NQ-1>& Q) {




  Real as = sound_speed * sound_speed;


  Real ca = (Q(QBFIELDX)*Q(QBFIELDX) + Q(QBFIELDY)*Q(QBFIELDY) + Q(QBFIELDZ)*Q(QBFIELDZ)) / Q(QRHO);
  Real cay = (Q(QBFIELDY)*Q(QBFIELDY)) / Q(QRHO);


  Real csy = 0.5_rt*((as + ca) - std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*cay));


  Real cfy = 0.5_rt*((as + ca) + std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*cay));


  Real alf;
  Real als;


  if (std::abs(cfy - csy) <= 1e-14){
    alf = 1.0_rt;
    als = 0.0_rt;
  } else {
      if (as - csy < 0.0) {
          alf = 0.0_rt;
      } else {
          alf = std::sqrt((as - csy)/(cfy - csy));
      }

      if (cfy - as < 0.0) {
          als = 0.0_rt;
      } else {
          als = std::sqrt((cfy - as)/(cfy - csy));
      }
  }

  Real dir_cos_x;
  Real dir_cos_z;

  if(std::abs(Q(QBFIELDX)) <= 1.e-14_rt && std::abs(Q(QBFIELDZ)) <= 1.e-14_rt) {
    dir_cos_x = 1.0_rt / std::sqrt(2.0_rt);
    dir_cos_z = dir_cos_x;

  } else {
    dir_cos_x = Q(QBFIELDX) / std::sqrt(Q(QBFIELDX)*Q(QBFIELDX) + Q(QBFIELDZ)*Q(QBFIELDZ));
    dir_cos_z = Q(QBFIELDZ) / std::sqrt(Q(QBFIELDX)*Q(QBFIELDX) + Q(QBFIELDZ)*Q(QBFIELDZ));
  }

  Real cff = std::sqrt(cfy) * alf;
  Real css = std::sqrt(csy) * als;

  Real S = std::copysign(1.0_rt, Q(QBFIELDY));

  Real Qf = std::sqrt(cfy) * alf * S;
  Real Qs = std::sqrt(csy) * als * S;

  Real N = 0.5_rt / as;

  Real AAf = std::sqrt(as) * alf * std::sqrt(Q(QRHO));
  Real AAs = std::sqrt(as) * als * std::sqrt(Q(QRHO));


  left_basis(0, EIGVEC_RHO) = 0.0;
  left_basis(0, EIGVEC_U) = N*Qs*dir_cos_x;
  left_basis(0, EIGVEC_V) = -N*cff;
  left_basis(0, EIGVEC_W) = N*Qs*dir_cos_z;
  left_basis(0, EIGVEC_PRESSURE) = N*alf/Q(QRHO);
  left_basis(0, EIGVEC_BPERP1) = N*AAs*dir_cos_x/Q(QRHO);
  left_basis(0, EIGVEC_BPERP2) = N*AAs*dir_cos_z/Q(QRHO);

  left_basis(1, EIGVEC_RHO) = 0.0;
  left_basis(1, EIGVEC_U) = -0.5_rt*dir_cos_z;
  left_basis(1, EIGVEC_V) = 0.0;
  left_basis(1, EIGVEC_W) = 0.5_rt*dir_cos_x;
  left_basis(1, EIGVEC_PRESSURE) = 0.0;
  left_basis(1, EIGVEC_BPERP1) = -0.5_rt*dir_cos_z*S/std::sqrt(Q(QRHO));
  left_basis(1, EIGVEC_BPERP2) = 0.5_rt*dir_cos_x*S/std::sqrt(Q(QRHO));

  left_basis(2, EIGVEC_RHO) = 0.0;
  left_basis(2, EIGVEC_U) = -N*Qf*dir_cos_x;
  left_basis(2, EIGVEC_V) = -N*css;
  left_basis(2, EIGVEC_W) = -N*Qf*dir_cos_z;
  left_basis(2, EIGVEC_PRESSURE) = N*als/Q(QRHO);
  left_basis(2, EIGVEC_BPERP1) = -N*AAf*dir_cos_x/Q(QRHO);
  left_basis(2, EIGVEC_BPERP2) = -N*AAf*dir_cos_z/Q(QRHO);

  left_basis(3, EIGVEC_RHO) = 1.0_rt;
  left_basis(3, EIGVEC_U) = 0.0;
  left_basis(3, EIGVEC_V) = 0.0;
  left_basis(3, EIGVEC_W) = 0.0;
  left_basis(3, EIGVEC_PRESSURE) = -1.0_rt/as;
  left_basis(3, EIGVEC_BPERP1) = 0.0;
  left_basis(3, EIGVEC_BPERP2) = 0.0;

  left_basis(4, EIGVEC_RHO) = 0.0;
  left_basis(4, EIGVEC_U) = N*Qf*dir_cos_x;
  left_basis(4, EIGVEC_V) = N*css;
  left_basis(4, EIGVEC_W) = N*Qf*dir_cos_z;
  left_basis(4, EIGVEC_PRESSURE) = N*als/Q(QRHO);
  left_basis(4, EIGVEC_BPERP1) = -N*AAf*dir_cos_x/Q(QRHO);
  left_basis(4, EIGVEC_BPERP2) = -N*AAf*dir_cos_z/Q(QRHO);

  left_basis(5, EIGVEC_RHO) = 0.0;
  left_basis(5, EIGVEC_U) = 0.5_rt*dir_cos_z;
  left_basis(5, EIGVEC_V) = 0.0;
  left_basis(5, EIGVEC_W) = -0.5_rt*dir_cos_x;
  left_basis(5, EIGVEC_PRESSURE) = 0.0;
  left_basis(5, EIGVEC_BPERP1) = -0.5_rt*dir_cos_z*S/std::sqrt(Q(QRHO));
  left_basis(5, EIGVEC_BPERP2) = 0.5_rt*dir_cos_x*S/std::sqrt(Q(QRHO));

  left_basis(6, EIGVEC_RHO) = 0.0;
  left_basis(6, EIGVEC_U) = -N*Qs*dir_cos_x;
  left_basis(6, EIGVEC_V) = N*cff;
  left_basis(6, EIGVEC_W) = -N*Qs*dir_cos_z;
  left_basis(6, EIGVEC_PRESSURE) = N*alf/Q(QRHO);
  left_basis(6, EIGVEC_BPERP1) = N*AAs*dir_cos_x/Q(QRHO);
  left_basis(6, EIGVEC_BPERP2) = N*AAs*dir_cos_z/Q(QRHO);


  right_basis(EIGVEC_RHO, 0) = Q(QRHO)*alf;
  right_basis(EIGVEC_RHO, 1) = 0.0;
  right_basis(EIGVEC_RHO, 2) = Q(QRHO)*als;
  right_basis(EIGVEC_RHO, 3) = 1.0_rt;
  right_basis(EIGVEC_RHO, 4) = Q(QRHO)*als;
  right_basis(EIGVEC_RHO, 5) = 0.0;
  right_basis(EIGVEC_RHO, 6) = Q(QRHO)*alf;

  right_basis(EIGVEC_V, 0) = -cff;
  right_basis(EIGVEC_V, 1) = 0.0;
  right_basis(EIGVEC_V, 2) = -css;
  right_basis(EIGVEC_V, 3) = 0.0;
  right_basis(EIGVEC_V, 4) = css;
  right_basis(EIGVEC_V, 5) = 0.0;
  right_basis(EIGVEC_V, 6) = cff;

  right_basis(EIGVEC_U, 0) = Qs*dir_cos_x;
  right_basis(EIGVEC_U, 1) = -dir_cos_z;
  right_basis(EIGVEC_U, 2) = -Qf*dir_cos_x;
  right_basis(EIGVEC_U, 3) = 0.0;
  right_basis(EIGVEC_U, 4) = Qf*dir_cos_x;
  right_basis(EIGVEC_U, 5) = dir_cos_z;
  right_basis(EIGVEC_U, 6) = -Qs*dir_cos_x;

  right_basis(EIGVEC_W, 0) = Qs*dir_cos_z;
  right_basis(EIGVEC_W, 1) = dir_cos_x;
  right_basis(EIGVEC_W, 2) = -Qf*dir_cos_z;
  right_basis(EIGVEC_W, 3) = 0.0;
  right_basis(EIGVEC_W, 4) = Qf*dir_cos_z;
  right_basis(EIGVEC_W, 5) = -dir_cos_x;
  right_basis(EIGVEC_W, 6) = -Qs*dir_cos_z;

  right_basis(EIGVEC_PRESSURE, 0) = Q(QRHO)*as*alf;
  right_basis(EIGVEC_PRESSURE, 1) = 0.0;
  right_basis(EIGVEC_PRESSURE, 2) = Q(QRHO)*as*als;
  right_basis(EIGVEC_PRESSURE, 3) = 0.0;
  right_basis(EIGVEC_PRESSURE, 4) = Q(QRHO)*as*als;
  right_basis(EIGVEC_PRESSURE, 5) = 0.0;
  right_basis(EIGVEC_PRESSURE, 6) = Q(QRHO)*as*alf;

  right_basis(EIGVEC_BPERP1, 0) = AAs*dir_cos_x;
  right_basis(EIGVEC_BPERP1, 1) = -dir_cos_z*S*std::sqrt(Q(QRHO));
  right_basis(EIGVEC_BPERP1, 2) = -AAf*dir_cos_x;
  right_basis(EIGVEC_BPERP1, 3) = 0.0;
  right_basis(EIGVEC_BPERP1, 4) = -AAf*dir_cos_x;
  right_basis(EIGVEC_BPERP1, 5) = -dir_cos_z*S*std::sqrt(Q(QRHO));
  right_basis(EIGVEC_BPERP1, 6) = AAs*dir_cos_x;

  right_basis(EIGVEC_BPERP2, 0) = AAs*dir_cos_z;
  right_basis(EIGVEC_BPERP2, 1) = dir_cos_x*S*std::sqrt(Q(QRHO));
  right_basis(EIGVEC_BPERP2, 2) = -AAf*dir_cos_z;
  right_basis(EIGVEC_BPERP2, 3) = 0.0;
  right_basis(EIGVEC_BPERP2, 4) = -AAf*dir_cos_z;
  right_basis(EIGVEC_BPERP2, 5) = dir_cos_x*S*std::sqrt(Q(QRHO));
  right_basis(EIGVEC_BPERP2, 6) = AAs*dir_cos_z;

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
eigenbasis_z(Array2D<Real, 0, NUM_EIGVEC-1, 0, NUM_EIGVEC-1>& left_basis,
      Array2D<Real, 0, NUM_EIGVEC-1, 0, NUM_EIGVEC-1>& right_basis,
      const Real sound_speed,
      Array1D<Real, 0, NQ-1>& Q) {



  Real as = sound_speed * sound_speed;


  Real ca = (Q(QBFIELDX)*Q(QBFIELDX) + Q(QBFIELDY)*Q(QBFIELDY) + Q(QBFIELDZ)*Q(QBFIELDZ)) / Q(QRHO);
  Real caz = (Q(QBFIELDZ)*Q(QBFIELDZ)) / Q(QRHO);


  Real csz = 0.5_rt*((as + ca) - std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*caz));


  Real cfz = 0.5_rt*((as + ca) + std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*caz));


  Real alf;
  Real als;


  if (std::abs(cfz-csz) <= 1e-14){
    alf = 1.0_rt;
    als = 0.0_rt;
  } else {
      if (as - csz < 0.0) {
          alf = 0.0_rt;
      } else {
          alf = std::sqrt((as - csz)/(cfz - csz));
      }

      if (cfz - as < 0.0) {
          als = 0.0_rt;
      } else {
          als = std::sqrt((cfz - as)/(cfz - csz));
      }
  }

  Real dir_cos_x;
  Real dir_cos_y;

  if (std::abs(Q(QBFIELDX)) <= 1.e-14_rt && std::abs(Q(QBFIELDY)) <= 1.e-14_rt) {
    dir_cos_x = 1.0_rt/std::sqrt(2.0_rt);
    dir_cos_y = dir_cos_x;

  } else {
    dir_cos_x = Q(QBFIELDX) / std::sqrt(Q(QBFIELDX)*Q(QBFIELDX) + Q(QBFIELDY)*Q(QBFIELDY));
    dir_cos_y = Q(QBFIELDY) / std::sqrt(Q(QBFIELDX)*Q(QBFIELDX) + Q(QBFIELDY)*Q(QBFIELDY));
  }

  Real cff = std::sqrt(cfz) * alf;
  Real css = std::sqrt(csz) * als;

  Real S = std::copysign(1.0_rt, Q(QBFIELDZ));

  Real Qf = std::sqrt(cfz) * alf * S;
  Real Qs = std::sqrt(csz) * als * S;

  Real N = 0.5_rt / as;

  Real AAf = std::sqrt(as) * alf * std::sqrt(Q(QRHO));
  Real AAs = std::sqrt(as) * als * std::sqrt(Q(QRHO));


  left_basis(0, EIGVEC_RHO) = 0.0;
  left_basis(0, EIGVEC_U) = N*Qs*dir_cos_x;
  left_basis(0, EIGVEC_V) = N*Qs*dir_cos_y;
  left_basis(0, EIGVEC_W) = -N*cff;
  left_basis(0, EIGVEC_PRESSURE) = N*alf/Q(QRHO);
  left_basis(0, EIGVEC_BPERP1) = N*AAs*dir_cos_x/Q(QRHO);
  left_basis(0, EIGVEC_BPERP2) = N*AAs*dir_cos_y/Q(QRHO);

  left_basis(1, EIGVEC_RHO) = 0.0;
  left_basis(1, EIGVEC_U) = -0.5_rt*dir_cos_y;
  left_basis(1, EIGVEC_V) = 0.5_rt*dir_cos_x;
  left_basis(1, EIGVEC_W) = 0.0;
  left_basis(1, EIGVEC_PRESSURE) = 0.0;
  left_basis(1, EIGVEC_BPERP1) = -0.5_rt*dir_cos_y*S/std::sqrt(Q(QRHO));
  left_basis(1, EIGVEC_BPERP2) = 0.5_rt*dir_cos_x*S/std::sqrt(Q(QRHO));

  left_basis(2, EIGVEC_RHO) = 0.0;
  left_basis(2, EIGVEC_U) = -N*Qf*dir_cos_x;
  left_basis(2, EIGVEC_V) = -N*Qf*dir_cos_y;
  left_basis(2, EIGVEC_W) = -N*css;
  left_basis(2, EIGVEC_PRESSURE) = N*als/Q(QRHO);
  left_basis(2, EIGVEC_BPERP1) = -N*AAf*dir_cos_x/Q(QRHO);
  left_basis(2, EIGVEC_BPERP2) = -N*AAf*dir_cos_y/Q(QRHO);

  left_basis(3, EIGVEC_RHO) = 1.0_rt;
  left_basis(3, EIGVEC_U) = 0.0;
  left_basis(3, EIGVEC_V) = 0.0;
  left_basis(3, EIGVEC_W) = 0.0;
  left_basis(3, EIGVEC_PRESSURE) = -1.0_rt/as;
  left_basis(3, EIGVEC_BPERP1) = 0.0;
  left_basis(3, EIGVEC_BPERP2) = 0.0;

  left_basis(4, EIGVEC_RHO) = 0.0;
  left_basis(4, EIGVEC_U) = N*Qf*dir_cos_x;
  left_basis(4, EIGVEC_V) = N*Qf*dir_cos_y;
  left_basis(4, EIGVEC_W) = N*css;
  left_basis(4, EIGVEC_PRESSURE) = N*als/Q(QRHO);
  left_basis(4, EIGVEC_BPERP1) = -N*AAf*dir_cos_x/Q(QRHO);
  left_basis(4, EIGVEC_BPERP2) = -N*AAf*dir_cos_y/Q(QRHO);

  left_basis(5, EIGVEC_RHO) = 0.0;
  left_basis(5, EIGVEC_U) = 0.5_rt*dir_cos_y;
  left_basis(5, EIGVEC_V) = -0.5_rt*dir_cos_x;
  left_basis(5, EIGVEC_W) = 0.0;
  left_basis(5, EIGVEC_PRESSURE) = 0.0;
  left_basis(5, EIGVEC_BPERP1) = -0.5_rt*dir_cos_y*S/std::sqrt(Q(QRHO));
  left_basis(5, EIGVEC_BPERP2) = 0.5_rt*dir_cos_x*S/std::sqrt(Q(QRHO));

  left_basis(6, EIGVEC_RHO) = 0.0;
  left_basis(6, EIGVEC_U) = -N*Qs*dir_cos_x;
  left_basis(6, EIGVEC_V) = -N*Qs*dir_cos_y;
  left_basis(6, EIGVEC_W) = N*cff;
  left_basis(6, EIGVEC_PRESSURE) = N*alf/Q(QRHO);
  left_basis(6, EIGVEC_BPERP1) = N*AAs*dir_cos_x/Q(QRHO);
  left_basis(6, EIGVEC_BPERP2) = N*AAs*dir_cos_y/Q(QRHO);


  right_basis(EIGVEC_RHO, 0) = Q(QRHO)*alf;
  right_basis(EIGVEC_RHO, 1) = 0.0;
  right_basis(EIGVEC_RHO, 2) = Q(QRHO)*als;
  right_basis(EIGVEC_RHO, 3) = 1.0_rt;
  right_basis(EIGVEC_RHO, 4) = Q(QRHO)*als;
  right_basis(EIGVEC_RHO, 5) = 0.0;
  right_basis(EIGVEC_RHO, 6) = Q(QRHO)*alf;

  right_basis(EIGVEC_W, 0) = -cff;
  right_basis(EIGVEC_W, 1) = 0.0;
  right_basis(EIGVEC_W, 2) = -css;
  right_basis(EIGVEC_W, 3) = 0.0;
  right_basis(EIGVEC_W, 4) = css;
  right_basis(EIGVEC_W, 5) = 0.0;
  right_basis(EIGVEC_W, 6) = cff;

  right_basis(EIGVEC_U, 0) = Qs*dir_cos_x;
  right_basis(EIGVEC_U, 1) = -dir_cos_y;
  right_basis(EIGVEC_U, 2) = -Qf*dir_cos_x;
  right_basis(EIGVEC_U, 3) = 0.0;
  right_basis(EIGVEC_U, 4) = Qf*dir_cos_x;
  right_basis(EIGVEC_U, 5) = dir_cos_y;
  right_basis(EIGVEC_U, 6) = -Qs*dir_cos_x;

  right_basis(EIGVEC_V, 0) = Qs*dir_cos_y;
  right_basis(EIGVEC_V, 1) = dir_cos_x;
  right_basis(EIGVEC_V, 2) = -Qf*dir_cos_y;
  right_basis(EIGVEC_V, 3) = 0.0;
  right_basis(EIGVEC_V, 4) = Qf*dir_cos_y;
  right_basis(EIGVEC_V, 5) = -dir_cos_x;
  right_basis(EIGVEC_V, 6) = -Qs*dir_cos_y;

  right_basis(EIGVEC_PRESSURE, 0) = Q(QRHO)*as*alf;
  right_basis(EIGVEC_PRESSURE, 1) = 0.0;
  right_basis(EIGVEC_PRESSURE, 2) = Q(QRHO)*as*als;
  right_basis(EIGVEC_PRESSURE, 3) = 0.0;
  right_basis(EIGVEC_PRESSURE, 4) = Q(QRHO)*as*als;
  right_basis(EIGVEC_PRESSURE, 5) = 0.0;
  right_basis(EIGVEC_PRESSURE, 6) = Q(QRHO)*as*alf;

  right_basis(EIGVEC_BPERP1, 0) = AAs*dir_cos_x;
  right_basis(EIGVEC_BPERP1, 1) = -dir_cos_y*S*std::sqrt(Q(QRHO));
  right_basis(EIGVEC_BPERP1, 2) = -AAf*dir_cos_x;
  right_basis(EIGVEC_BPERP1, 3) = 0.0;
  right_basis(EIGVEC_BPERP1, 4) = -AAf*dir_cos_x;
  right_basis(EIGVEC_BPERP1, 5) = -dir_cos_y*S*std::sqrt(Q(QRHO));
  right_basis(EIGVEC_BPERP1, 6) = AAs*dir_cos_x;

  right_basis(EIGVEC_BPERP2, 0) = AAs*dir_cos_y;
  right_basis(EIGVEC_BPERP2, 1) = dir_cos_x*S*std::sqrt(Q(QRHO));
  right_basis(EIGVEC_BPERP2, 2) = -AAf*dir_cos_y;
  right_basis(EIGVEC_BPERP2, 3) = 0.0;
  right_basis(EIGVEC_BPERP2, 4) = -AAf*dir_cos_y;
  right_basis(EIGVEC_BPERP2, 5) = dir_cos_x*S*std::sqrt(Q(QRHO));
  right_basis(EIGVEC_BPERP2, 6) = AAs*dir_cos_y;

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
evals(Array1D<Real, 0, NUM_EIGVEC-1>& lam,
      const Real sound_speed,
      Array1D<Real, 0, NQ-1>& Q,
      const int dir) {



  Real ca  = (Q(QBFIELDX)*Q(QBFIELDX) + Q(QBFIELDY)*Q(QBFIELDY) + Q(QBFIELDZ)*Q(QBFIELDZ)) / Q(QRHO);

  int QUN;
  Real cad;
  Real cs;
  Real cf;

  Real as = sound_speed * sound_speed;

  if (dir == 0) {
    QUN = QU;
    cad = (Q(QBFIELDX)*Q(QBFIELDX)) / Q(QRHO);
    cs = 0.5_rt * ((as + ca) - std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*cad));
    cf = 0.5_rt * ((as + ca) + std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*cad));

  } else if (dir == 1) {
    QUN = QV;
    cad = (Q(QBFIELDY)*Q(QBFIELDY)) / Q(QRHO);
    cs = 0.5_rt * ((as + ca) - std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*cad));
    cf = 0.5_rt * ((as + ca) + std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*cad));

  } else {
    QUN = QW;
    cad = (Q(QBFIELDZ)*Q(QBFIELDZ)) / Q(QRHO);
    cs = 0.5_rt * ((as + ca) - std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*cad));
    cf = 0.5_rt * ((as + ca) + std::sqrt((as + ca)*(as + ca) - 4.0_rt*as*cad));

  }

  lam(0) = Q(QUN) - std::sqrt(cf);
  lam(1) = Q(QUN) - std::sqrt(cad);
  lam(2) = Q(QUN) - std::sqrt(cs);
  lam(3) = Q(QUN);
  lam(4) = Q(QUN) + std::sqrt(cs);
  lam(5) = Q(QUN) + std::sqrt(cad);
  lam(6) = Q(QUN) + std::sqrt(cf);

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
verify_eigenbasis(Array2D<Real, 0, NUM_EIGVEC-1, 0, NUM_EIGVEC-1>& left_basis,
            Array2D<Real, 0, NUM_EIGVEC-1, 0, NUM_EIGVEC-1>& right_basis) {


  for (int n = 0; n < NUM_EIGVEC; n++) {
    for (int m = 0; m < NUM_EIGVEC; m++) {

      Real dot = 0.0;
      for (int p = 0; p < NUM_EIGVEC; p++)
      {
        dot += left_basis(m, p) * right_basis(p, n);
      }

#ifndef AMREX_USE_GPU
      std::cout << "l(" << m << ") . r(" << n << ") = " << dot << std::endl;
#endif

    }
#ifndef AMREX_USE_GPU
    std::cout << std::endl;
#endif
  }
}

#endif
