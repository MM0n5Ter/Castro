#include <HydroSim.H>
#include <fourth_order_center.H>

using namespace amrex;

void
HydroSim::build_center_state(const Box& bx,
                         Array4<Real const> const& U,
                         Array4<Real> const& centered_state,
                         GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {

  // Take a cell-average state U and a convert it to a cell-center
  // state centered_state via centered_state = U - 1/24 L U

  [[maybe_unused]] const auto U_lower = lbound(U);
  [[maybe_unused]] const auto U_upper = ubound(U);

  [[maybe_unused]] const auto *lo = bx.loVect();
  [[maybe_unused]] const auto *hi = bx.hiVect();

  AMREX_ASSERT(U_lower.x <= lo[0]-1 && U_upper.x >= hi[0]+1);
#if AMREX_SPACEDIM >= 2
  AMREX_ASSERT(U_lower.y <= lo[1]-1 && U_upper.y >= hi[1]+1);
#endif
#if AMREX_SPACEDIM == 3
  AMREX_ASSERT(U_lower.z <= lo[2]-1 && U_upper.z >= hi[2]+1);
#endif

  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  GpuArray<bool, AMREX_SPACEDIM> lo_wrap;
  GpuArray<bool, AMREX_SPACEDIM> hi_wrap;
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    lo_wrap[idir] = lo_bc[idir] == amrex::PhysBCType::interior;
    hi_wrap[idir] = hi_bc[idir] == amrex::PhysBCType::interior;
  }

  amrex::ParallelFor(bx, U.nComp(),
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {
    Real lap = laplacian_at(i, j, k, n, U,
                                 lo_wrap, hi_wrap, domlo, domhi);

    centered_state(i,j,k,n) = U(i,j,k,n) - (1.0_rt/24.0_rt) * lap;

  });
}

void
HydroSim::build_center_state_in_place(const Box& bx,
                                  Array4<Real> const& U,
                                  Array4<Real> const& tmp,
                                  GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {

  // Take a cell-average state U and make it cell-centered in place
  // via U <- U - 1/24 L U.  Note that this operation is not tile
  // safe.

  // here, tmp is a temporary memory space we use to store one-component's Laplacian

  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  GpuArray<bool, AMREX_SPACEDIM> lo_wrap;
  GpuArray<bool, AMREX_SPACEDIM> hi_wrap;
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    lo_wrap[idir] = lo_bc[idir] == amrex::PhysBCType::interior;
    hi_wrap[idir] = hi_bc[idir] == amrex::PhysBCType::interior;
  }

  for (int n = 0; n < U.nComp(); n++) {

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      tmp(i,j,k) = laplacian_at(i, j, k, n, U,
                                     lo_wrap, hi_wrap, domlo, domhi);
    });

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      U(i,j,k,n) = U(i,j,k,n) - (1.0_rt/24.0_rt) * tmp(i,j,k);
    });
  }
}


void
HydroSim::laplacian_of_state(const Box& bx,
                         Array4<Real const> const& U,
                         Array4<Real> const& lap, const int ncomp,
                         GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {

  // Computes the h**2/24 L U term that is used in correcting
  // cell-center to averages (and back)

  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  GpuArray<bool, AMREX_SPACEDIM> lo_wrap;
  GpuArray<bool, AMREX_SPACEDIM> hi_wrap;
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    lo_wrap[idir] = lo_bc[idir] == amrex::PhysBCType::interior;
    hi_wrap[idir] = hi_bc[idir] == amrex::PhysBCType::interior;
  }

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    lap(i,j,k) = (1.0_rt/24.0_rt) *
      laplacian_at(i, j, k, ncomp, U,
                        lo_wrap, hi_wrap, domlo, domhi);
  });

}


void
HydroSim::build_fourth_average(const Box& bx,
                            Array4<Real> const& prim_arr,
                            Array4<Real const> const& q_mean_arr,
                            GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {

  // Take the cell-center state q and another state q_second (e.g.,
  // constructed from the cell-average U) and replace the cell-center
  // q with a 4th-order accurate cell-average, q <- q + 1/24 L q_second

  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  GpuArray<bool, AMREX_SPACEDIM> lo_wrap;
  GpuArray<bool, AMREX_SPACEDIM> hi_wrap;
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    lo_wrap[idir] = lo_bc[idir] == amrex::PhysBCType::interior;
    hi_wrap[idir] = hi_bc[idir] == amrex::PhysBCType::interior;
  }

  amrex::ParallelFor(bx, prim_arr.nComp(),
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {
    Real lap = laplacian_at(i, j, k, n, q_mean_arr,
                                 lo_wrap, hi_wrap, domlo, domhi);

    prim_arr(i,j,k,n) += (1.0_rt/24.0_rt) * lap;
  });
}


void
HydroSim::build_fourth_in_place(const Box& bx,
                             Array4<Real> const& prim_arr,
                             Array4<Real> const& tmp,
                             GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {

  // Take the cell-center q and makes it a cell-average q, in place
  // (e.g. q is overwritten by its average), q <- q + 1/24 L q.
  // Note: this routine is not tile safe.

  for (int n = 0; n < prim_arr.nComp(); n++) {
    build_fourth_in_place_n(bx, prim_arr, n, tmp, domlo, domhi);
  }
}


void
HydroSim::build_fourth_in_place_n(const Box& bx,
                               Array4<Real> const& prim_arr, const int ncomp,
                               Array4<Real> const& tmp,
                               GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {

  // Take the cell-center q and makes it a cell-average q, in place
  // (e.g. q is overwritten by its average), q <- q + 1/24 L q.
  // Note: this routine is not tile safe.
  //
  // This version operates on a single component.  Here ncomp is the
  // component to update

  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  GpuArray<bool, AMREX_SPACEDIM> lo_wrap;
  GpuArray<bool, AMREX_SPACEDIM> hi_wrap;
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    lo_wrap[idir] = lo_bc[idir] == amrex::PhysBCType::interior;
    hi_wrap[idir] = hi_bc[idir] == amrex::PhysBCType::interior;
  }

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    tmp(i,j,k) = laplacian_at(i, j, k, ncomp, prim_arr,
                                   lo_wrap, hi_wrap, domlo, domhi);
  });

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    prim_arr(i,j,k,ncomp) += (1.0_rt/24.0_rt) * tmp(i,j,k);
  });

}
