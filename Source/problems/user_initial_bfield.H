#ifndef user_initial_bfield_H
#define user_initial_bfield_H

///
/// Initialize the simulation state data
///
/// @param i         x-index
/// @param j         y-index
/// @param k         z-index
/// @param B         face-centered magnetic field in direction idir (Fab)
/// @param idir      direction of the face to fill (0=x, 1=y, 2=z)
/// @param geomdata  geometry data
///
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_initial_bfield (int i, int j, int k,
                                  Array4<Real> const& B, const int idir,
                                  const GeometryData& geomdata) {}

#endif
