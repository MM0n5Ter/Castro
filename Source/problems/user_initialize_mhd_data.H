#ifndef user_initialize_mhd_data_H
#define user_initialize_mhd_data_H

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_initialize_mhd_data (int i, int j, int k,
                                  Array4<Real> const& B, const int idir,
                                  const GeometryData& geomdata) {}

#endif
