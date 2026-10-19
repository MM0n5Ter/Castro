#include <HydroSim.H>
#include <HydroSim_math.H>
#include <Rotation.H>

void
HydroSim::fill_scaled_centrifugal_potential(const Box& bx,
                            Array4<Real> const& psi,
                            const Real time) {

    amrex::ignore_unused(time);


  const auto omega = get_angular_speed();
  Real denom = omega * omega;

  const auto problo = geom.ProbLoArray();
  const auto dx = geom.CellSizeArray();
  const auto coord = geom.Coord();
  const auto geomdata = geom.data();

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    GpuArray<Real, 3> r;

    r[0] = problo[0] + dx[0] * (static_cast<Real>(i) + 0.5_rt) - problem::center[0];
#if AMREX_SPACEDIM >= 2
    r[1] = problo[1] + dx[1] * (static_cast<Real>(j) + 0.5_rt) - problem::center[1];
#else
    r[1] = 0.0_rt;
#endif
#if AMREX_SPACEDIM == 3
    r[2] = problo[2] + dx[2] * (static_cast<Real>(k) + 0.5_rt) - problem::center[2];
#else
    r[2] = 0.0_rt;
#endif

    if (denom != 0.0) {
        auto angular_velocity_vec = get_angular_velocity(geomdata, j);
        psi(i,j,k) = centrifugal_potential(r, angular_velocity_vec, coord) / denom;
    }
    else {
        psi(i,j,k) = 0.0;
    }

  });
}
