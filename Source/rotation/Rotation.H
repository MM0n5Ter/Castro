#ifndef ROTATING_FRAME_H
#define ROTATING_FRAME_H

#include <numbers>

#include <AMReX_Array.H>
#include <HydroSim_math.H>
#include <HydroSim_helpers.H>
#include <hydrosim_params.H>
#include <angular_velocity.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real
centrifugal_potential(const GpuArray<Real, 3>& r, const GpuArray<Real, 3>& omega,
                     const int coord) {

  Real phi = 0.0_rt;
  auto r_vec = r;

  if (coord == 2) {
      r_vec[1] = 0.0_rt;
      r_vec[2] = 0.0_rt;
  }

  if (hydrosim::rotation_centrifugal_on == 1) {

      GpuArray<Real, 3> omega_cross_position;
      cross_product(omega, r_vec, omega_cross_position);

      for (int idir = 0; idir < 3; idir++) {
          phi -= 0.5_rt * omega_cross_position[idir] * omega_cross_position[idir];
      }

  }

  return phi;

}

enum class coriolis_term : std::uint8_t { omit, include };

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
fictitious_acceleration(const GpuArray<Real, 3>& r, const GpuArray<Real, 3>& v,
                        const GpuArray<Real, 3>& omega, const int coord,
                        const coriolis_term coriolis_mode, GpuArray<Real, 3>& Sr) {

  Sr[0] = 0.0;
  Sr[1] = 0.0;
  Sr[2] = 0.0;

  bool c1 = (hydrosim::rotation_centrifugal_on == 1) ? true : false;

  bool c2 = (!((hydrosim::rotation_coriolis_on == 1 && coriolis_mode == coriolis_term::include)) ? (false) : (true));

  auto r_vec = r;

  if (c1) {

      if (coord == 2) {
          r_vec[1] = 0.0_rt;
          r_vec[2] = 0.0_rt;
      }
      GpuArray<Real, 3> omega_cross_position;
      cross_product(omega, r_vec, omega_cross_position);

      GpuArray<Real, 3> centripetal_term;
      cross_product(omega, omega_cross_position, centripetal_term);

      for (int idir = 0; idir < 3; idir++) {
          Sr[idir] -= centripetal_term[idir];
      }
  }

  if (c2) {
      GpuArray<Real, 3> omega_cross_velocity;
      cross_product(omega, v, omega_cross_velocity);
      for (int idir = 0; idir < 3; idir++) {
          Sr[idir] -= 2.0_rt * omega_cross_velocity[idir];
      }
  }

}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void fill_omega_times_dt_matrix(const Real dt, const GpuArray<Real, 3>& omega,
                          Array2D<Real, 0, 2, 0, 2>& omega_times_dt_matrix)
{

    Real omega_times_dt[3];

    if (!(hydrosim::rotation_coriolis_on == 1)) {

      for (auto& e : omega_times_dt) {
        e = 0.0_rt;
      }

    } else {

        for (int idir = 0; idir < 3; idir++) {
          omega_times_dt[idir] = dt * omega[idir];
        }

    }

    omega_times_dt_matrix(0, 0) = 1.0_rt + omega_times_dt[0] * omega_times_dt[0];
    omega_times_dt_matrix(0, 1) = omega_times_dt[0] * omega_times_dt[1] + omega_times_dt[2];
    omega_times_dt_matrix(0, 2) = omega_times_dt[0] * omega_times_dt[2] - omega_times_dt[1];

    omega_times_dt_matrix(1, 0) = omega_times_dt[1] * omega_times_dt[0] - omega_times_dt[2];
    omega_times_dt_matrix(1, 1) = 1.0_rt + omega_times_dt[1] * omega_times_dt[1];
    omega_times_dt_matrix(1, 2) = omega_times_dt[1] * omega_times_dt[2] + omega_times_dt[0];

    omega_times_dt_matrix(2, 0) = omega_times_dt[2] * omega_times_dt[0] + omega_times_dt[1];
    omega_times_dt_matrix(2, 1) = omega_times_dt[2] * omega_times_dt[1] - omega_times_dt[0];
    omega_times_dt_matrix(2, 2) = 1.0_rt + omega_times_dt[2] * omega_times_dt[2];

    const Real norm = 1.0_rt + omega_times_dt[0] * omega_times_dt[0] +
                               omega_times_dt[1] * omega_times_dt[1] +
                               omega_times_dt[2] * omega_times_dt[2];

    for (int l = 0; l < 3; l++) {
      for (int m = 0; m < 3; m++) {
        omega_times_dt_matrix(l, m) /= norm;
      }
    }

}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
GpuArray<Real, 3> rotate_to_inertial_frame(const GpuArray<Real, 3>& vec,
                                    const GpuArray<Real, 3>& omega, Real time)
{
    GpuArray<Real, 3> inertial_vec{};

    Array1D<Real, 0, 2> theta{};
    Array2D<Real, 0, 2, 0, 2> frame_rotation_matrix{};

    if (hydrosim::do_frame_spin == 1) {
        for (int n = 0; n < 3; ++n) {
            theta(n) = omega[n] * time;
        }
    }

    frame_rotation_matrix(0,0) =  cos(theta(1)) * cos(theta(2));
    frame_rotation_matrix(0,1) = -cos(theta(1)) * sin(theta(2));
    frame_rotation_matrix(0,2) =  sin(theta(1));
    frame_rotation_matrix(1,0) =  cos(theta(0)) * sin(theta(2)) + sin(theta(0)) * sin(theta(1)) * cos(theta(2));
    frame_rotation_matrix(1,1) =  cos(theta(0)) * cos(theta(2)) - sin(theta(0)) * sin(theta(1)) * sin(theta(2));
    frame_rotation_matrix(1,2) = -sin(theta(0)) * cos(theta(1));
    frame_rotation_matrix(2,0) =  sin(theta(0)) * sin(theta(2)) - cos(theta(0)) * sin(theta(1)) * cos(theta(2));
    frame_rotation_matrix(2,1) =  sin(theta(0)) * cos(theta(2)) + cos(theta(0)) * sin(theta(1)) * sin(theta(2));
    frame_rotation_matrix(2,2) =  cos(theta(0)) * cos(theta(1));

    for (int l = 0; l < 3; ++l) {
        for (int m = 0; m < 3; ++m) {
            inertial_vec[l] += frame_rotation_matrix(l,m) * vec[m];
        }
    }

    return inertial_vec;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
frame_to_inertial_velocity (const int i, const int j, const int k,
                                 const GeometryData& geomdata,
                                 const Real time, GpuArray<Real, 3>& v) {

  amrex::ignore_unused(time);

  GpuArray<Real, 3> loc;

  position(i, j, k, geomdata, loc);

  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    loc[dir] -= problem::center[dir];
  }

  if (geomdata.Coord() == 2) {
      loc[1] = 0.0_rt;
      loc[2] = 0.0_rt;
  }

  auto omega = get_angular_velocity(geomdata, j);

  v[0] += (omega[1]*loc[2] - omega[2]*loc[1]);
  v[1] += (omega[2]*loc[0] - omega[0]*loc[2]);
  v[2] += (omega[0]*loc[1] - omega[1]*loc[0]);

}

#endif
