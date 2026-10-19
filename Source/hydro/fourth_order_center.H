#ifndef FOURTH_ORDER_CENTER_H
#define FOURTH_ORDER_CENTER_H

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real
laplacian_at(int i, int j, int k, int n,
                  Array4<Real const> const& a,
                  GpuArray<bool, AMREX_SPACEDIM> const& lo_wrap, GpuArray<bool, AMREX_SPACEDIM> const& hi_wrap,
                  GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {

  Real second_deriv_x = 0.0;
  Real second_deriv_y = 0.0;
  Real second_deriv_z = 0.0;

  // we use 2nd-order accurate one-sided stencils at the physical
  // boundaries
  // I like the one-sided better.

  if (i == domlo[0] && !lo_wrap[0]) {
    second_deriv_x = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i+1,j,k,n) + 4.0_rt*a(i+2,j,k,n) - a(i+3,j,k,n);

  } else if (i == domhi[0] && !hi_wrap[0]) {
    second_deriv_x = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i-1,j,k,n) + 4.0_rt*a(i-2,j,k,n) - a(i-3,j,k,n);

  } else {
    second_deriv_x = a(i+1,j,k,n) - 2.0_rt*a(i,j,k,n) + a(i-1,j,k,n);
  }

#if AMREX_SPACEDIM >= 2
  if (j == domlo[1] && !lo_wrap[1]) {
    second_deriv_y = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j+1,k,n) + 4.0_rt*a(i,j+2,k,n) - a(i,j+3,k,n);

  } else if (j == domhi[1] && !hi_wrap[1]) {
    second_deriv_y = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j-1,k,n) + 4.0_rt*a(i,j-2,k,n) - a(i,j-3,k,n);

  } else {
    second_deriv_y = a(i,j+1,k,n) - 2.0_rt*a(i,j,k,n) + a(i,j-1,k,n);
  }
#endif

#if AMREX_SPACEDIM == 3
  if (k == domlo[2] && !lo_wrap[2]) {
    second_deriv_z = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j,k+1,n) + 4.0_rt*a(i,j,k+2,n) - a(i,j,k+3,n);

  } else if (k == domhi[2] && !hi_wrap[2]) {
    second_deriv_z = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j,k-1,n) + 4.0_rt*a(i,j,k-2,n) - a(i,j,k-3,n);

  } else {
    second_deriv_z = a(i,j,k+1,n) - 2.0_rt*a(i,j,k,n) + a(i,j,k-1,n);
  }
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

  // we use 2nd-order accurate one-sided stencils at the physical boundaries
  if (idir != 0) {

    if (i == domlo[0] && !lo_wrap[0]) {
      second_deriv_x = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i+1,j,k,n) + 4.0_rt*a(i+2,j,k,n) - a(i+3,j,k,n);

    } else if (i == domhi[0] && !hi_wrap[0]) {
      second_deriv_x = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i-1,j,k,n) + 4.0_rt*a(i-2,j,k,n) - a(i-3,j,k,n);

    } else {
      second_deriv_x = a(i+1,j,k,n) - 2.0_rt*a(i,j,k,n) + a(i-1,j,k,n);
    }
  }

#if AMREX_SPACEDIM >= 2
  if (idir != 1) {

    if (j == domlo[1] && !lo_wrap[1]) {
      second_deriv_y = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j+1,k,n) + 4.0_rt*a(i,j+2,k,n) - a(i,j+3,k,n);

    } else if (j == domhi[1] && !hi_wrap[1]) {
      second_deriv_y = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j-1,k,n) + 4.0_rt*a(i,j-2,k,n) - a(i,j-3,k,n);

    } else {
      second_deriv_y = a(i,j+1,k,n) - 2.0_rt*a(i,j,k,n) + a(i,j-1,k,n);
    }
  }
#endif

#if AMREX_SPACEDIM == 3
  if (idir != 2) {

    if (k == domlo[2] && !lo_wrap[2]) {
      second_deriv_z = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j,k+1,n) + 4.0_rt*a(i,j,k+2,n) - a(i,j,k+3,n);

    } else if (k == domhi[2] && !hi_wrap[2]) {
      second_deriv_z = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j,k-1,n) + 4.0_rt*a(i,j,k-2,n) - a(i,j,k-3,n);

    } else {
      second_deriv_z = a(i,j,k+1,n) - 2.0_rt*a(i,j,k,n) + a(i,j,k-1,n);
    }
  }
#endif

  return second_deriv_x + second_deriv_y + second_deriv_z;
}


#endif
