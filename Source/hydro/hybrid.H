#ifndef HYDROSIM_HYBRID_H
#define HYDROSIM_HYBRID_H

#include <HydroSim_helpers.H>

using namespace amrex;

///
/// For hybrid momentum advection, convert the linear momentum into the
/// hybrid / angular momentum components
///
/// @param loc         spatial location
/// @param cartesian_vector  linear momentum
/// @param hybrid_vector  hybrid / angular momentum
///
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void cartesian_to_hybrid(const GpuArray<Real, 3>& loc,
                      const GpuArray<Real, 3>& cartesian_vector,
                      GpuArray<Real, 3>& hybrid_vector)
{
    // Convert a linear momentum into the "hybrid" scheme
    // that has radial and angular components.

    Real R = amrex::max(std::sqrt(loc[0] * loc[0] + loc[1] * loc[1]),
                        std::numeric_limits<Real>::min());
    Real InvRadius = 1.0_rt / R;

    // Note that we expect the linear momentum to be consistent
    // with which frame we're measuring the fluid quantities in.
    // So we're effectively always using the first form of those
    // equalities, not the second.

    hybrid_vector[0] = cartesian_vector[0] * (loc[0] * InvRadius) + cartesian_vector[1] * (loc[1] * InvRadius);
    hybrid_vector[1] = cartesian_vector[1] * loc[0] - cartesian_vector[0] * loc[1];
    hybrid_vector[2] = cartesian_vector[2];
}


///
/// Convert the hybrid / angular momentum back to linear momentum
///
/// @param loc          spatial location
/// @param hybrid_vector   hybrid / angular momentum
/// @param cartesian_vector   linear momentum
///
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void hybrid_to_cartesian(const GpuArray<Real, 3>& loc,
                      const GpuArray<Real, 3>& hybrid_vector,
                      GpuArray<Real, 3>& cartesian_vector)
{
    // Convert a "hybrid" momentum into a linear one.

    Real R = amrex::max(std::sqrt(loc[0] * loc[0] + loc[1] * loc[1]),
                        std::numeric_limits<Real>::min());
    Real InvRadius = 1.0_rt / R;

    // This is the inverse.

    cartesian_vector[0] = hybrid_vector[0] * (loc[0] * InvRadius) - hybrid_vector[1] * (loc[1] * InvRadius * InvRadius);
    cartesian_vector[1] = hybrid_vector[1] * (loc[0] * InvRadius * InvRadius) + hybrid_vector[0] * (loc[1] * InvRadius);
    cartesian_vector[2] = hybrid_vector[2];
}


///
/// Create the hydrodynamic source term for the hybrid momenta based on the
/// input linear momentum source term
///
/// @param loc             spatial location
/// @param cartesian_source_vector   linear momentum source term
/// @param hybrid_source_vector   hybrid / angular momentum source term
///
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void compute_hybrid_momentum_source(const GpuArray<Real, 3>& loc,
                                const GpuArray<Real, 3>& cartesian_source_vector,
                                GpuArray<Real, 3>& hybrid_source_vector)
{
    Real R = amrex::max(std::sqrt(loc[0] * loc[0] + loc[1] * loc[1]),
                        std::numeric_limits<Real>::min());
    Real InvRadius = 1.0_rt / R;

    // This is analogous to the conversion of linear momentum to hybrid momentum.

    hybrid_source_vector[0] = cartesian_source_vector[0] * (loc[0] * InvRadius) + cartesian_source_vector[1] * (loc[1] * InvRadius);
    hybrid_source_vector[1] = cartesian_source_vector[1] * loc[0] - cartesian_source_vector[0] * loc[1];
    hybrid_source_vector[2] = cartesian_source_vector[2];
}


///
/// Compute and store the flux of the hybrid momentum components
///
/// @param state          the Godunov state
/// @param geomdata       the HydroSim geometry data object
/// @param idir           coordinate direction for flux computation (0 = x, 1 = y, 2 = z)
/// @param i              x zone index
/// @param j              y zone index
/// @param k              z zone index
/// @param flux           hydrodynamic flux in direction idir
/// @param on_cell_center  is this a cell-centered flux or an interface flux?
///
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void hybrid_flux_from_state(const GpuArray<Real, NGDSTATE>& state, const GeometryData& geomdata,
                         int idir, int i, int j, int k,
                         GpuArray<Real, NUM_STATE>& flux, bool on_cell_center = false)
{
    bool ccx = true;
    bool ccy = true;
    bool ccz = true;

    if (idir == 0) {
        ccx = on_cell_center;
    }
    else if (idir == 1) {
        ccy = on_cell_center;
    }
    else {
        ccz = on_cell_center;
    }

    GpuArray<Real, 3> loc;

    position(i, j, k, geomdata, loc, ccx, ccy, ccz);

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        loc[dir] -= problem::center[dir];

    GpuArray<Real, 3> cartesian_vector;

    for (int dir = 0; dir < 3; ++dir)
        cartesian_vector[dir] = state[GDDENSITY] * state[GDU + dir];

    GpuArray<Real, 3> hybrid_vector;

    cartesian_to_hybrid(loc, cartesian_vector, hybrid_vector);

    Real advect_velocity = state[GDU + idir];

    if (idir == 0) {
        flux[UMR] = hybrid_vector[0] * advect_velocity;
        flux[UML] = hybrid_vector[1] * advect_velocity - loc[1] * state[GDPRESSURE];
        flux[UMP] = hybrid_vector[2] * advect_velocity;
    }
    else if (idir == 1) {
        flux[UMR] = hybrid_vector[0] * advect_velocity;
        flux[UML] = hybrid_vector[1] * advect_velocity + loc[0] * state[GDPRESSURE];
        flux[UMP] = hybrid_vector[2] * advect_velocity;
    }
    else {
        flux[UMR] = hybrid_vector[0] * advect_velocity;
        flux[UML] = hybrid_vector[1] * advect_velocity;
        flux[UMP] = hybrid_vector[2] * advect_velocity + state[GDPRESSURE];
    }

}

#endif
