#ifndef orbit_props_H
#define orbit_props_H

// This header contains a number of routines that are designed to
// calculate generic properties of binary orbits.

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


// Iterate over the force balance equation to find a Lagrange point.
// r_min and r_max set the domain over which we want to find the answer,
// since in general the equation has multiple roots.
// We assume that r comes in with a valid starting guess value.

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void refine_equilibrium_root (Real& r, Real mass_1, Real mass_2, Real r1, Real r2, Real a, Real r_min, Real r_max)
{
    // Root-find parameters

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

    // Use a bisection search to find the root of the force-balance equation.
    // The reason we don't use something faster like Newton-Raphson is that
    // the force terms have terms like 1/r^2, so evaluating the derivative and
    // then dividing it can be numerically dangerous. Since we only call this routine
    // once per timestep and the force evaluation is cheap, it shouldn't matter
    // that the method itself converges slowly.

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

        // Check to see if we've convernged.

        if (std::abs((rp - rm) / a) < tolerance) {
            break;
        }

    }

    r = rc;

    // If we didn't converge, set the value to zero.

    if (i > max_iters) {
        r = 0.0_rt;
    }
}


// Calculate Lagrange points. In each case we give the zone index
// closest to it (assuming we're on the coarse grid).

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void find_equilibrium_points (Real mass_1, Real mass_2, const Real* center_1, const Real* center_2,
                          GpuArray<Real, 3>& L1, GpuArray<Real, 3>& L2, GpuArray<Real, 3>& L3)
{
    Real r; // Distance from Lagrange point to primary
    Real a; // Distance between secondary and primary

    Real r1, r2;

    // Don't try to calculate the Lagrange points if the secondary
    // is already gone.

    if (mass_2 < 0.0_rt) return;

    a = std::sqrt((center_2[0] - center_1[0]) * (center_2[0] - center_1[0]) +
                  (center_2[1] - center_1[1]) * (center_2[1] - center_1[1]) +
                  (center_2[2] - center_1[2]) * (center_2[2] - center_1[2]));

    r1 = -std::sqrt(center_1[0] * center_1[0] + center_1[1] * center_1[1] + center_1[2] * center_1[2]);
    r2 = std::sqrt(center_2[0] * center_2[0] + center_2[1] * center_2[1] + center_2[2] * center_2[2]);

    // Do a root-find over the quintic equation for L1.

    r = 0.0_rt;

    refine_equilibrium_root(r, mass_1, mass_2, r1, r2, a, r1, r2);

    // Now turn this radial distance into a grid coordinate.

    for (int n = 0; n < 3; ++n) {
        L1[n] = r * (center_2[n] - center_1[n]) / a;
    }

    // Repeat for L2 Lagrange point.

    r = r2 + 0.5_rt * a;

    refine_equilibrium_root(r, mass_1, mass_2, r1, r2, a, r2, 0.0_rt);

    for (int n = 0; n < 3; ++n) {
        L2[n] = r * (center_2[n] - center_1[n]) / a;
    }

    // Repeat for L3 Lagrange point.

    r = r1 - 0.5_rt * a;

    refine_equilibrium_root(r, mass_1, mass_2, r1, r2, a, 0.0_rt, r1);

    for (int n = 0; n < 3; ++n) {
        L3[n] = r * (center_2[n] - center_1[n]) / a;
    }
}

// Given the mass ratio q of two stars (assumed to be q = M_1 / M_2),
// compute the effective Roche radii of the stars, normalized to unity,
// using the approximate formula. Optionally we can
// pass in a distance scale.

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void find_lobe_radii (Real roche_q, Real& r_1, Real& r_2, Real a = 1.0_rt)
{
    const Real c1 = 0.49e0_rt;
    const Real c2 = 0.60e0_rt;

    Real q = roche_q;

    r_1 = a * c1 * std::pow(q, 2.0_rt / 3.0_rt) / (c2 * std::pow(q, 2.0_rt / 3.0_rt) + std::log(1.0_rt + std::pow(q, 1.0_rt / 3.0_rt)));

    q = 1.0_rt / q;

    r_2 = a * c1 * std::pow(q, 2.0_rt / 3.0_rt) / (c2 * std::pow(q, 2.0_rt / 3.0_rt) + std::log(1.0_rt + std::pow(q, 1.0_rt / 3.0_rt)));
}

#endif
