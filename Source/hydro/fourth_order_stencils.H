#ifndef FOURTH_ORDER_STENCILS_H
#define FOURTH_ORDER_STENCILS_H

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real
second_difference_along(Array4<Real const> const& a, int i, int j, int k, int n,
                  int di, int dj, int dk, bool lo_edge, bool hi_edge) {

  if (lo_edge) {
    return 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i+di,j+dj,k+dk,n) + 4.0_rt*a(i+2*di,j+2*dj,k+2*dk,n) - a(i+3*di,j+3*dj,k+3*dk,n);

  } else if (hi_edge) {
    return 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i-di,j-dj,k-dk,n) + 4.0_rt*a(i-2*di,j-2*dj,k-2*dk,n) - a(i-3*di,j-3*dj,k-3*dk,n);

  }
  return a(i+di,j+dj,k+dk,n) - 2.0_rt*a(i,j,k,n) + a(i-di,j-dj,k-dk,n);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real
laplacian_at(int i, int j, int k, int n,
                  Array4<Real const> const& a,
                  GpuArray<bool, AMREX_SPACEDIM> const& lo_wrap, GpuArray<bool, AMREX_SPACEDIM> const& hi_wrap,
                  GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {

  Real second_deriv_x = 0.0;
  Real second_deriv_y = 0.0;
  Real second_deriv_z = 0.0;


  second_deriv_x = second_difference_along(a, i, j, k, n, 1, 0, 0,
                                              i == domlo[0] && !lo_wrap[0], i == domhi[0] && !hi_wrap[0]);

#if AMREX_SPACEDIM >= 2
  second_deriv_y = second_difference_along(a, i, j, k, n, 0, 1, 0,
                                              j == domlo[1] && !lo_wrap[1], j == domhi[1] && !hi_wrap[1]);
#endif

#if AMREX_SPACEDIM == 3
  second_deriv_z = second_difference_along(a, i, j, k, n, 0, 0, 1,
                                              k == domlo[2] && !lo_wrap[2], k == domhi[2] && !hi_wrap[2]);
#endif

  return second_deriv_x + second_deriv_y + second_deriv_z;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real
tangential_laplacian(int i, int j, int k, int n,
                int idir,
                Array4<Real const> const& a,
                GpuArray<bool, AMREX_SPACEDIM> const& lo_wrap, GpuArray<bool, AMREX_SPACEDIM> const& hi_wrap,
                GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {

  Real second_deriv_x = 0.0;
  Real second_deriv_y = 0.0;
  Real second_deriv_z = 0.0;

  if (idir != 0) {

    second_deriv_x = second_difference_along(a, i, j, k, n, 1, 0, 0,
                                                i == domlo[0] && !lo_wrap[0], i == domhi[0] && !hi_wrap[0]);
  }

#if AMREX_SPACEDIM >= 2
  if (idir != 1) {

    second_deriv_y = second_difference_along(a, i, j, k, n, 0, 1, 0,
                                                j == domlo[1] && !lo_wrap[1], j == domhi[1] && !hi_wrap[1]);
  }
#endif

#if AMREX_SPACEDIM == 3
  if (idir != 2) {

    second_deriv_z = second_difference_along(a, i, j, k, n, 0, 0, 1,
                                                k == domlo[2] && !lo_wrap[2], k == domhi[2] && !hi_wrap[2]);
  }
#endif

  return second_deriv_x + second_deriv_y + second_deriv_z;
}


#endif
