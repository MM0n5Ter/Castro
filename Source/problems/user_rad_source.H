#ifndef user_rad_source_H
#define user_rad_source_H

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_rad_source (int i, int j, int k,
                         Array4<Real> const& rhs,
                         const GeometryData& geomdata, Real time,
                         Real dt, int igroup) {}

#endif
