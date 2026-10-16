#ifndef FOURTH_CENTER_AVERAGE_H
#define FOURTH_CENTER_AVERAGE_H

///
/// The index range of the domain and, in each direction, whether its
/// low and high faces are periodic.  The Laplacians below switch to
/// one-sided stencils on the faces that are not.
///
struct DomainBounds {
  GpuArray<int, 3> lo;
  GpuArray<int, 3> hi;
  GpuArray<bool, AMREX_SPACEDIM> lo_periodic;
  GpuArray<bool, AMREX_SPACEDIM> hi_periodic;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real
compute_laplacian(int i, int j, int k, int n,
                  Array4<Real const> const& a,
                  DomainBounds const& domain) {

  Real lapx = 0.0;
  Real lapy = 0.0;
  Real lapz = 0.0;

  // we use 2nd-order accurate one-sided stencils at the physical
  // boundaries
  // I like the one-sided better.

  if (i == domain.lo[0] && !domain.lo_periodic[0]) {
    lapx = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i+1,j,k,n) + 4.0_rt*a(i+2,j,k,n) - a(i+3,j,k,n);

  } else if (i == domain.hi[0] && !domain.hi_periodic[0]) {
    lapx = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i-1,j,k,n) + 4.0_rt*a(i-2,j,k,n) - a(i-3,j,k,n);

  } else {
    lapx = a(i+1,j,k,n) - 2.0_rt*a(i,j,k,n) + a(i-1,j,k,n);
  }

#if AMREX_SPACEDIM >= 2
  if (j == domain.lo[1] && !domain.lo_periodic[1]) {
    lapy = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j+1,k,n) + 4.0_rt*a(i,j+2,k,n) - a(i,j+3,k,n);

  } else if (j == domain.hi[1] && !domain.hi_periodic[1]) {
    lapy = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j-1,k,n) + 4.0_rt*a(i,j-2,k,n) - a(i,j-3,k,n);

  } else {
    lapy = a(i,j+1,k,n) - 2.0_rt*a(i,j,k,n) + a(i,j-1,k,n);
  }
#endif

#if AMREX_SPACEDIM == 3
  if (k == domain.lo[2] && !domain.lo_periodic[2]) {
    lapz = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j,k+1,n) + 4.0_rt*a(i,j,k+2,n) - a(i,j,k+3,n);

  } else if (k == domain.hi[2] && !domain.hi_periodic[2]) {
    lapz = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j,k-1,n) + 4.0_rt*a(i,j,k-2,n) - a(i,j,k-3,n);

  } else {
    lapz = a(i,j,k+1,n) - 2.0_rt*a(i,j,k,n) + a(i,j,k-1,n);
  }
#endif

  return lapx + lapy + lapz;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real
transverse_laplacian(int i, int j, int k, int n,
                int idir,
                Array4<Real const> const& a,
                DomainBounds const& domain) {

  Real lapx = 0.0;
  Real lapy = 0.0;
  Real lapz = 0.0;

  // we use 2nd-order accurate one-sided stencils at the physical boundaries
  if (idir != 0) {

    if (i == domain.lo[0] && !domain.lo_periodic[0]) {
      lapx = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i+1,j,k,n) + 4.0_rt*a(i+2,j,k,n) - a(i+3,j,k,n);

    } else if (i == domain.hi[0] && !domain.hi_periodic[0]) {
      lapx = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i-1,j,k,n) + 4.0_rt*a(i-2,j,k,n) - a(i-3,j,k,n);

    } else {
      lapx = a(i+1,j,k,n) - 2.0_rt*a(i,j,k,n) + a(i-1,j,k,n);
    }
  }

#if AMREX_SPACEDIM >= 2
  if (idir != 1) {

    if (j == domain.lo[1] && !domain.lo_periodic[1]) {
      lapy = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j+1,k,n) + 4.0_rt*a(i,j+2,k,n) - a(i,j+3,k,n);

    } else if (j == domain.hi[1] && !domain.hi_periodic[1]) {
      lapy = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j-1,k,n) + 4.0_rt*a(i,j-2,k,n) - a(i,j-3,k,n);

    } else {
      lapy = a(i,j+1,k,n) - 2.0_rt*a(i,j,k,n) + a(i,j-1,k,n);
    }
  }
#endif

#if AMREX_SPACEDIM == 3
  if (idir != 2) {

    if (k == domain.lo[2] && !domain.lo_periodic[2]) {
      lapz = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j,k+1,n) + 4.0_rt*a(i,j,k+2,n) - a(i,j,k+3,n);

    } else if (k == domain.hi[2] && !domain.hi_periodic[2]) {
      lapz = 2.0_rt*a(i,j,k,n) - 5.0_rt*a(i,j,k-1,n) + 4.0_rt*a(i,j,k-2,n) - a(i,j,k-3,n);

    } else {
      lapz = a(i,j,k+1,n) - 2.0_rt*a(i,j,k,n) + a(i,j,k-1,n);
    }
  }
#endif

  return lapx + lapy + lapz;
}


#endif
