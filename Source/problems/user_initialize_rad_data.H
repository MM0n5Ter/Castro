#ifndef user_initialize_rad_data_H
#define user_initialize_rad_data_H

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_initialize_rad_data (int i, int j, int k,
                                  Array4<Real> const& rad,
                                  GpuArray<Real, NGROUPS+1> const& xnu,
                                  GpuArray<Real, NGROUPS> const& freq_center,
                                  GpuArray<Real, NGROUPS> const& freq_width,
                                  const GeometryData& geomdata) {}

#endif
