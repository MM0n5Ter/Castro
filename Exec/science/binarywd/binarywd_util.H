#ifndef binarywd_util_H
#define binarywd_util_H

#include <AMReX.H>

#include <HydroSim.H>
#include <HydroSim_math.H>
#include <fundamental_constants.H>

#ifdef ROTATION
#include <Rotation.H>
#endif

void freefall_velocity (Real mass, Real distance, Real& vel);

void kepler_third_law (Real radius_1, Real mass_1, Real radius_2, Real mass_2,
                       Real& period, Real eccentricity, Real phi, Real& a,
                       Real& r_1, Real& r_2, Real& v_1r, Real& v_2r, Real& v_1p, Real& v_2p);

void set_wd_composition (Real mass, Real& envelope_mass, Real core_comp[NumSpec], Real envelope_comp[NumSpec], const std::string& star_type);

void ensure_primary_mass_larger ();

void initialize_problem ();

void finalize_probdata ();

void binary_setup ();

void set_small ();

void update_roche_radii ();


AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real stellar_mask (int i, int j, int k,
                   const amrex::GeometryData& geomdata,
                   Real rho, bool is_primary = true)
{

    Real mask = -1.0_rt;


    if (is_primary) {
        if (problem::mass_P <= 0.0_rt) return mask;
    }
    else {
        if (problem::mass_S <= 0.0_rt) return mask;
    }

    GpuArray<Real, 3> loc;
    loc[0] = geomdata.ProbLo(0) + (static_cast<Real>(i) + 0.5_rt) * geomdata.CellSize(0);

#if AMREX_SPACEDIM >= 2
    loc[1] = geomdata.ProbLo(1) + (static_cast<Real>(j) + 0.5_rt) * geomdata.CellSize(1);
#else
    loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
    loc[2] = geomdata.ProbLo(2) + (static_cast<Real>(k) + 0.5_rt) * geomdata.CellSize(2);
#else
    loc[2] = 0.0_rt;
#endif


    if (rho < problem::stellar_density_threshold) return mask;

    Real r_P = std::sqrt((loc[0] - problem::com_P[0]) * (loc[0] - problem::com_P[0]) +
                         (loc[1] - problem::com_P[1]) * (loc[1] - problem::com_P[1]) +
                         (loc[2] - problem::com_P[2]) * (loc[2] - problem::com_P[2]));

    Real r_S = std::sqrt((loc[0] - problem::com_S[0]) * (loc[0] - problem::com_S[0]) +
                         (loc[1] - problem::com_S[1]) * (loc[1] - problem::com_S[1]) +
                         (loc[2] - problem::com_S[2]) * (loc[2] - problem::com_S[2]));


    for (int iloc = 0; iloc < AMREX_SPACEDIM; ++iloc) {
        loc[iloc] -= problem::center[iloc];
    }

    auto omega = get_angular_velocity(geomdata, j);

    Real phi_rot = centrifugal_potential(loc, omega, geomdata.Coord());

    Real phi_ptrs = -C::Gconst * problem::mass_P / r_P + phi_rot;
    Real phi_s = -C::Gconst * problem::mass_S / r_S + phi_rot;

    if (is_primary) {
        if (phi_ptrs < 0.0_rt && phi_ptrs < phi_s) {
            mask = 1.0_rt;
        }
    }
    else {
        if (phi_s < 0.0_rt && phi_s < phi_ptrs) {
            mask = 1.0_rt;
        }
    }

    return mask;
}

#endif
