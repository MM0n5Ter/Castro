#ifndef user_initial_state_H
#define user_initial_state_H

///
/// Initialize the simulation state data
///
/// @param i         x-index
/// @param j         y-index
/// @param k         z-index
/// @param state     simulation state (Fab)
/// @param geomdata  geometry data
///
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_initial_state (int i, int j, int k, Array4<Real> const& state, const GeometryData& geomdata) {}

#endif
