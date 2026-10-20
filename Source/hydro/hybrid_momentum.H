#ifndef HYDROSIM_HYBRID_MOMENTUM_H
#define HYDROSIM_HYBRID_MOMENTUM_H

#include <HydroSim_helpers.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void cartesian_to_hybrid(const GpuArray<Real, 3>& loc,
                      const GpuArray<Real, 3>& cartesian_vector,
                      GpuArray<Real, 3>& hybrid_vector)
{

    Real R = amrex::max(std::sqrt(loc[0] * loc[0] + loc[1] * loc[1]),
                        std::numeric_limits<Real>::min());
    Real InvRadius = 1.0_rt / R;


    hybrid_vector[0] = cartesian_vector[0] * (loc[0] * InvRadius) + cartesian_vector[1] * (loc[1] * InvRadius);
    hybrid_vector[1] = cartesian_vector[1] * loc[0] - cartesian_vector[0] * loc[1];
    hybrid_vector[2] = cartesian_vector[2];
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void hybrid_to_cartesian(const GpuArray<Real, 3>& loc,
                      const GpuArray<Real, 3>& hybrid_vector,
                      GpuArray<Real, 3>& cartesian_vector)
{

    Real R = amrex::max(std::sqrt(loc[0] * loc[0] + loc[1] * loc[1]),
                        std::numeric_limits<Real>::min());
    Real InvRadius = 1.0_rt / R;


    cartesian_vector[0] = hybrid_vector[0] * (loc[0] * InvRadius) - hybrid_vector[1] * (loc[1] * InvRadius * InvRadius);
    cartesian_vector[1] = hybrid_vector[1] * (loc[0] * InvRadius * InvRadius) + hybrid_vector[0] * (loc[1] * InvRadius);
    cartesian_vector[2] = hybrid_vector[2];
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void compute_hybrid_momentum_source(const GpuArray<Real, 3>& loc,
                                const GpuArray<Real, 3>& cartesian_source_vector,
                                GpuArray<Real, 3>& hybrid_source_vector)
{
    Real R = amrex::max(std::sqrt(loc[0] * loc[0] + loc[1] * loc[1]),
                        std::numeric_limits<Real>::min());
    Real InvRadius = 1.0_rt / R;


    hybrid_source_vector[0] = cartesian_source_vector[0] * (loc[0] * InvRadius) + cartesian_source_vector[1] * (loc[1] * InvRadius);
    hybrid_source_vector[1] = cartesian_source_vector[1] * loc[0] - cartesian_source_vector[0] * loc[1];
    hybrid_source_vector[2] = cartesian_source_vector[2];
}


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
