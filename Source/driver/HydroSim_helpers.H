#ifndef HYDROSIM_UTIL_H
#define HYDROSIM_UTIL_H

#include <numbers>

#include <AMReX_Geometry.H>
#include <network_properties.H>
#include <state_indices.H>
#include <cell_geometry.H>

using namespace amrex;

#if AMREX_SPACEDIM == 1
#elif AMREX_SPACEDIM == 2
#else
#endif

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool momentum_flux_has_pressure (const int momentum_axis, const int flux_axis, const int coord)
{

    amrex::ignore_unused(coord);

#if AMREX_SPACEDIM <= 2

    if (momentum_axis == 0 && flux_axis == 0) {

        if (coord != CoordSys::cartesian) {
            return false;
        }
        return true;

    } else if (momentum_axis == 1 && flux_axis == 1) {

        if (coord == CoordSys::SPHERICAL) {
            return false;
        }
        return true;

    } else {
        return (momentum_axis == flux_axis);
    }

#else

    return (momentum_axis == flux_axis);

#endif

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr int passive_prim_index (int passive_index)
{
    if (passive_index < NumAdvected) {
        return QFA + passive_index;
    }

    if (passive_index < NumAdvected + NumSpec) {
        return QFS + passive_index - NumAdvected;
    }

    return QFX + passive_index - NumAdvected - NumSpec;

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr int passive_cons_index (int passive_index)
{
    if (passive_index < NumAdvected) {
        return UFA + passive_index;
    }

    if (passive_index < NumAdvected + NumSpec) {
        return UFS + passive_index - NumAdvected;
    }

    return UFX + passive_index - NumAdvected - NumSpec;

}

#endif
