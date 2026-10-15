#ifndef HYDROSIM_UTIL_H
#define HYDROSIM_UTIL_H

#include <numbers>

#include <AMReX_Geometry.H>
#include <network_properties.H>
#include <state_indices.H>

using namespace amrex;

#if AMREX_SPACEDIM == 1
constexpr int dg0 = 1;
constexpr int dg1 = 0;
constexpr int dg2 = 0;
#elif AMREX_SPACEDIM == 2
constexpr int dg0 = 1;
constexpr int dg1 = 1;
constexpr int dg2 = 0;
#else
constexpr int dg0 = 1;
constexpr int dg1 = 1;
constexpr int dg2 = 1;
#endif

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr int upassmap (int ipassive)
{
    if (ipassive < NumAdv) {
        // passive is one of the advected quantities
        return UADV + ipassive;
    }

    if (ipassive < NumAdv + NumSpec) {
        // passive is one of the species
        return USPEC + ipassive - NumAdv;
    }

    // passive is one of the auxiliary quantities
    return UAUXV + ipassive - NumAdv - NumSpec;

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr int qpassmap (int ipassive)
{
    if (ipassive < NumAdv) {
        // passive is one of the advected quantities
        return QADV + ipassive;
    }

    if (ipassive < NumAdv + NumSpec) {
        // passive is one of the species
        return QSPEC + ipassive - NumAdv;
    }

    // passive is one of the auxiliary quantities
    return QAUXV + ipassive - NumAdv - NumSpec;

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool mom_flux_has_p (const int mom_dir, const int flux_dir, const int coord)
{

    amrex::ignore_unused(coord);

    // Keep track of which components of the momentum flux have pressure

#if AMREX_SPACEDIM <= 2

    if (mom_dir == 0 && flux_dir == 0) {

        if (coord != CoordSys::cartesian) {
            return false;
        }
        return true;

    } else if (mom_dir == 1 && flux_dir == 1) {

        if (coord == CoordSys::SPHERICAL) {
            return false;
        }
        return true;

    } else {
        return (mom_dir == flux_dir);
    }

#else // AMREX_SPACEDIM == 3; Cartesian

    return (mom_dir == flux_dir);

#endif

}



AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void position(int i, int j, int k,
              GeometryData const& geomdata, GpuArray<Real, 3>& loc,
              IndexType itype = IndexType::TheCellType())
{
    // Given 3D indices (i,j,k), return the cell-centered spatial position.
    // In any direction in which itype is nodal we are edge-centered instead.

    const int idx[3] = {i, j, k};

    Real offset[AMREX_SPACEDIM];

    const Real* problo = geomdata.ProbLo();
    const Real* probhi = geomdata.ProbHi();
    const Real* dx     = geomdata.CellSize();
    const int*  domlo  = geomdata.domain.loVect();
    const int*  domhi  = geomdata.domain.hiVect();

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (itype.cellCentered(dir)) {
            // If we're cell-centered, we want to be in the middle of the zone.
            offset[dir] = problo[dir] + 0.5_rt * dx[dir];
        }
        else {
            // We are edge-centered.
            offset[dir] = problo[dir];
        }
    }

    // Be careful when using periodic boundary conditions. In that case,
    // we need to loop around to the other side of the domain.

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {

        if (geomdata.isPeriodic(dir) == 1) {
            if (idx[dir] < domlo[dir]) {
                offset[dir] += probhi[dir] - problo[dir];
            }

            // For the hi end we need to deal correctly with
            // edge-centered indexing.
            if ((itype.cellCentered(dir) && idx[dir] > domhi[dir]) ||
                (itype.nodeCentered(dir) && idx[dir] > domhi[dir]+1)) {
                offset[dir] += problo[dir] - probhi[dir];
            }
        }

    }

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        loc[dir] = offset[dir] + idx[dir] * dx[dir];
    }

    for (int dir = AMREX_SPACEDIM; dir < 3; ++dir) {
        loc[dir] = 0.0_rt;
    }

}


namespace geometry_util
{

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real distance(GeometryData const& geomdata, const GpuArray<Real, 3>& loc)
{
    // Returns the distance from the center provided with loc, the position.
    // loc is the form of [x, y, z,] in Cartesian, [r, z, phi] in cylindrical
    // and [r, theta, phi] in spherical

    const auto coord = geomdata.Coord();

    if (coord == CoordSys::cartesian) {
        return std::sqrt(loc[0]*loc[0] + loc[1]*loc[1] + loc[2]*loc[2]);
    }

    if (coord == CoordSys::RZ) {
        return std::sqrt(loc[0]*loc[0] + loc[1]*loc[1]);
    }

    return std::abs(loc[0]);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real volume(Dim3 zone, const GeometryData& geomdata)
{
    //
    // Given the 3D cell-centered index of a zone, return its volume.
    // this function only provides Cartesian in 1D/2D/3D,
    // Cylindrical (R-Z) in 2D,
    // and Spherical in 1D and 2D (R-THETA).
    //

    amrex::ignore_unused(zone);

    const auto* dx = geomdata.CellSize();

    Real vol;

#if AMREX_SPACEDIM == 1

    auto coord = geomdata.Coord();

    if (coord == 0) {

        // Cartesian

        vol = dx[0];

    } else if (coord == 1) {

        // 1-d cylindrical

        Real rl = geomdata.ProbLo()[0] + static_cast<Real>(zone.x) * dx[0];
        Real rr = rl + dx[0];

        vol = std::abs(std::numbers::pi * dx[0] * (rr + rl));

   } else {

        // Spherical

        Real rl = geomdata.ProbLo()[0] + static_cast<Real>(zone.x) * dx[0];
        Real rr = rl + dx[0];

        vol = std::abs((4.0_rt / 3.0_rt) * std::numbers::pi * dx[0] * (rl * rl + rl * rr + rr * rr));

    }

#elif AMREX_SPACEDIM == 2

    auto coord = geomdata.Coord();

    if (coord == 0) {

        // Cartesian

        vol = dx[0] * dx[1];

    } else if (coord == 1) {

        // Cylindrical

        Real r_l = geomdata.ProbLo()[0] + static_cast<Real>(zone.x) * dx[0];
        Real r_r = geomdata.ProbLo()[0] + static_cast<Real>(zone.x+1) * dx[0];

        vol = std::abs(std::numbers::pi * (r_l + r_r) * dx[0] * dx[1]);

    } else {

        // Spherical

        constexpr Real twoThirdsPi = 2.0_rt * std::numbers::pi / 3.0_rt;

        Real r_l = geomdata.ProbLo()[0] + static_cast<Real>(zone.x) * dx[0];
        Real r_r = geomdata.ProbLo()[0] + static_cast<Real>(zone.x+1) * dx[0];
        Real theta_l = geomdata.ProbLo()[1] + static_cast<Real>(zone.y) * dx[1];
        Real theta_r = geomdata.ProbLo()[1] + static_cast<Real>(zone.y+1) * dx[1];

        vol = std::abs(twoThirdsPi * (std::cos(theta_l) - std::cos(theta_r)) * dx[0] *
                       (r_r * r_r + r_r * r_l + r_l * r_l));

    }

#else

    // Cartesian

    vol = dx[0] * dx[1] * dx[2];

#endif

    return vol;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real area(Dim3 zone, const int& idir, const GeometryData& geomdata)
{

    amrex::ignore_unused(zone);
    amrex::ignore_unused(idir);

    const auto* dx = geomdata.CellSize();

    Real a;

#if AMREX_SPACEDIM == 1

    auto coord = geomdata.Coord();

    if (coord == 0) {

        // Cartesian

        a = 1.0_rt;

    } else if (coord == 1) {

        // 1-D cylindrical

        Real r = geomdata.ProbLo()[0] + static_cast<Real>(zone.x) * dx[0];

        a = std::abs(2.0_rt * std::numbers::pi * r);

    } else {

        // Spherical

        Real r = geomdata.ProbLo()[0] + static_cast<Real>(zone.x) * dx[0];

        a = std::abs(4.0_rt * std::numbers::pi * r * r);

    }

#elif AMREX_SPACEDIM == 2

    auto coord = geomdata.Coord();

    if (coord == 0) {

        // Cartesian

        if (idir == 0) {
            a = dx[1];
        }
        else {
            a = dx[0];
        }

    } else  if (coord == 1) {

        // Cylindrical

        if (idir == 0) {
            Real r = geomdata.ProbLo()[0] + static_cast<Real>(zone.x) * dx[0];
            a = std::abs(2.0_rt * std::numbers::pi * r * dx[1]);
        }
        else {
            Real r = geomdata.ProbLo()[0] + (static_cast<Real>(zone.x) + 0.5_rt) * dx[0];
            a = std::abs(2.0_rt * std::numbers::pi * r * dx[0]);
        }

    } else {

        // Spherical

        if (idir == 0) {
            Real r = geomdata.ProbLo()[0] + static_cast<Real>(zone.x) * dx[0];
            Real theta_l = geomdata.ProbLo()[1] + static_cast<Real>(zone.y) * dx[1];
            Real theta_r = geomdata.ProbLo()[1] + static_cast<Real>(zone.y+1) * dx[1];

            a = std::abs(2.0_rt * std::numbers::pi * r * r * (std::cos(theta_l) - std::cos(theta_r)));
        }
        else {
            Real r = geomdata.ProbLo()[0] + (static_cast<Real>(zone.x) + 0.5_rt) * dx[0];
            Real theta = geomdata.ProbLo()[1] + static_cast<Real>(zone.y) * dx[1];

            a = std::abs(2.0_rt * std::numbers::pi * std::sin(theta) * r * dx[0]);
        }

    }

#else

    // Cartesian

    if (idir == 0) {
        a = dx[1] * dx[2];
    }
    else if (idir == 1) {
        a = dx[0] * dx[2];
    }
    else {
        a = dx[0] * dx[1];
    }

#endif

    return a;
}

}

#endif
