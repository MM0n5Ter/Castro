#ifndef user_initial_radiation_H
#define user_initial_radiation_H

#include <blackbody.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_initial_radiation (int i, int j, int k,
                                  Array4<Real> const& rad,
                                  GpuArray<Real, NGROUPS+1> const& xnu,
                                  GpuArray<Real, NGROUPS> const& nugroup,
                                  GpuArray<Real, NGROUPS> const& dnugroup,
                                  const GeometryData& geomdata)
{

    for (int g = 0; g < NGROUPS; g++) {
        rad(i,j,k,g) = 0.0_rt;
    }
}
#endif
