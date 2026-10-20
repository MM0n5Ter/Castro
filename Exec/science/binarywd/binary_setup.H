#ifndef binarywd_util_H
#define binarywd_util_H

#include <AMReX.H>

#include <HydroSim.H>
#include <HydroSim_math.H>
#include <fundamental_constants.H>

#ifdef ROTATION
#include <Rotation.H>
#endif
#include <AMReX_REAL.H>

#include <prob_parameters.H>

void freefall_velocity (Real mass, Real distance, Real& vel);

struct KeplerOrbit {
    Real r_1, r_2, v_1r, v_2r, v_1p, v_2p;
};

KeplerOrbit kepler_third_law (Real radius_1, Real mass_1, Real radius_2, Real mass_2,
                              Real& period, Real eccentricity, Real phi, Real& a);

void set_wd_composition (Real mass, Real& envelope_mass, Real core_comp[NumSpec], Real envelope_comp[NumSpec], const std::string& star_type);

void ensure_primary_mass_larger ();

void initialize_problem ();

void finalize_probdata ();

void binary_setup ();

void set_small ();

void update_roche_radii ();

namespace binarywd
{
    extern AMREX_GPU_MANAGED amrex::Real mdot_P;
    extern AMREX_GPU_MANAGED amrex::Real rad_P[7];
    extern AMREX_GPU_MANAGED amrex::Real vol_P[7];
    extern AMREX_GPU_MANAGED amrex::Real rho_avg_P;
    extern AMREX_GPU_MANAGED amrex::Real mdot_S;
    extern AMREX_GPU_MANAGED amrex::Real rad_S[7];
    extern AMREX_GPU_MANAGED amrex::Real vol_S[7];
    extern AMREX_GPU_MANAGED amrex::Real rho_avg_S;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real centrifugal_force (Real M, Real a, Real r)
{
    return C::Gconst * M / (a * a * a) * r;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real centrifugal_slope (Real M, Real a, Real r)
{

    amrex::ignore_unused(r);

    return C::Gconst * M / (a * a * a);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real point_accel (Real M, Real r)
{
    return -C::Gconst * M / (r * r) * std::copysign(1.0_rt, r);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real tidal_gradient (Real M, Real r)
{
    return 2.0_rt * C::Gconst * M / (r * r * r) * std::copysign(1.0_rt, r);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real fL (Real M1, Real M2, Real r1, Real r2, Real r, Real a)
{
    Real g1 = point_accel(M1, r - r1);
    Real g2 = point_accel(M2, r - r2);

    Real c = centrifugal_force(M1 + M2, a, r);

    return g1 + g2 + c;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real netDeriv (Real M1, Real M2, Real r1, Real r2, Real r, Real a)
{
    Real dtidal_1 = tidal_gradient(M1, r - r1);
    Real dtidal_2 = tidal_gradient(M2, r - r2);

    Real dc = centrifugal_slope(M1 + M2, a, r);

    return dtidal_1 + dtidal_2 + dc;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real refine_equilibrium_root (Real mass_1, Real mass_2, Real r1, Real r2, Real a, Real r_min, Real r_max)
{

    const Real tolerance = 1.0e-8_rt;
    const int max_iters = 200;

    Real rm{}, rp{};

    if (r_min == 0.0_rt && r_max == 0.0_rt) {
        amrex::Abort("Lagrange point iteration must have at least one non-zero bound provided.");
    }
    else if (r_min != 0.0_rt && r_max != 0.0_rt) {
        rm = r_min;
        rp = r_max;
    }
    else if (r_min != 0.0_rt && r_max == 0.0_rt) {
        rm = r_min;
        rp = std::abs(r_min) * 1000.0e0_rt;
    }
    else if (r_max != 0.0_rt && r_min == 0.0_rt) {
        rm = -std::abs(r_max) * 1000.0e0_rt;
        rp = r_max;
    }

    Real width = (rp - rm);

    rm = rm + width / 1000.0e0_rt;
    rp = rp - width / 1000.0e0_rt;


    Real rc;

    int i;
    for (i = 1; i <= max_iters; ++i) {

        Real fm = fL(mass_1, mass_2, r1, r2, rm, a);

        rc = 0.5_rt * (rm + rp);
        Real fc = fL(mass_1, mass_2, r1, r2, rc, a);

        if (fm * fc < 0.0_rt) {
            rp = rc;
        } else {
            rm = rc;
        }


        if (std::abs((rp - rm) / a) < tolerance) {
            break;
        }

    }

    if (i > max_iters) {
        return 0.0_rt;
    }

    return rc;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void find_equilibrium_points (Real mass_1, Real mass_2, const Real* center_1, const Real* center_2,
                          GpuArray<Real, 3>& L1, GpuArray<Real, 3>& L2, GpuArray<Real, 3>& L3)
{
    Real r;
    Real a;

    Real r1, r2;


    if (mass_2 < 0.0_rt) return;

    a = std::sqrt((center_2[0] - center_1[0]) * (center_2[0] - center_1[0]) +
                  (center_2[1] - center_1[1]) * (center_2[1] - center_1[1]) +
                  (center_2[2] - center_1[2]) * (center_2[2] - center_1[2]));

    r1 = -std::sqrt(center_1[0] * center_1[0] + center_1[1] * center_1[1] + center_1[2] * center_1[2]);
    r2 = std::sqrt(center_2[0] * center_2[0] + center_2[1] * center_2[1] + center_2[2] * center_2[2]);


    r = refine_equilibrium_root(mass_1, mass_2, r1, r2, a, r1, r2);


    for (int n = 0; n < 3; ++n) {
        L1[n] = r * (center_2[n] - center_1[n]) / a;
    }


    r = refine_equilibrium_root(mass_1, mass_2, r1, r2, a, r2, 0.0_rt);

    for (int n = 0; n < 3; ++n) {
        L2[n] = r * (center_2[n] - center_1[n]) / a;
    }


    r = refine_equilibrium_root(mass_1, mass_2, r1, r2, a, 0.0_rt, r1);

    for (int n = 0; n < 3; ++n) {
        L3[n] = r * (center_2[n] - center_1[n]) / a;
    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void find_lobe_radii (Real roche_q, Real a, Real& r_1, Real& r_2)
{
    const Real c1 = 0.49e0_rt;
    const Real c2 = 0.60e0_rt;

    Real q = roche_q;

    r_1 = a * c1 * std::pow(q, 2.0_rt / 3.0_rt) / (c2 * std::pow(q, 2.0_rt / 3.0_rt) + std::log(1.0_rt + std::pow(q, 1.0_rt / 3.0_rt)));

    q = 1.0_rt / q;

    r_2 = a * c1 * std::pow(q, 2.0_rt / 3.0_rt) / (c2 * std::pow(q, 2.0_rt / 3.0_rt) + std::log(1.0_rt + std::pow(q, 1.0_rt / 3.0_rt)));
}

enum class StarRole { Primary, Secondary };

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real stellar_mask (int i, int j, int k,
                   const amrex::GeometryData& geomdata,
                   Real rho, StarRole role = StarRole::Primary)
{

    Real mask = -1.0_rt;


    if (role == StarRole::Primary) {
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

    if (role == StarRole::Primary) {
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
