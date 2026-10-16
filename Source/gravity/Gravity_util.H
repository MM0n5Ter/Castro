#ifndef GRAVITY_UTIL_H
#define GRAVITY_UTIL_H

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real factorial (int n)
{
    Real fact = 1.0_rt;

    for (int i = 2; i <= n; ++i) {
        fact *= static_cast<Real>(i);
    }

    return fact;
}

// Value of a Legendre (or associated Legendre) polynomial at the current
// degree, together with the two lower degrees that the recurrence needs.

struct LegendreRecurrence
{
    Real val;
    Real prev1;
    Real prev2;
};

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void evalAssocLegendre (int l, int m, LegendreRecurrence& plm, Real x)
{
    // Calculate the associated Legendre polynomials. There are a number of
    // recurrence relations, but many are unstable. We'll use one that is known
    // to be stable for the reasonably low values of l we care about in a simulation:
    // (l-m)P_l^m(x) = x(2l-1)P_{l-1}^m(x) - (l+m-1)P_{l-2}^m(x).
    // This uses the following two expressions as initial conditions:
    // P_m^m(x) = (-1)^m (2m-1)! (1-x^2)^(m/2)
    // P_{m+1}^m(x) = x (2m+1) P_m^m (x)

    if (l == m) {

        // P_m^m

        plm.val = std::pow(-1, m) * std::pow((1.0_rt - x) * (1.0_rt + x), m * 0.5_rt);

        for (int n = (2*m-1); n >= 3; n = n - 2) {

            plm.val *= n;

        }

    }
    else if (l == m + 1) {

        // P_{m+1}^m

        plm.prev1 = plm.val;
        plm.val  = x * (2*m + 1) * plm.prev1;

    }
    else {

        plm.prev2 = plm.prev1;
        plm.prev1 = plm.val;
        plm.val  = (x * (2*l - 1) * plm.prev1 - (l + m - 1) * plm.prev2) / (l-m);

    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void evalLegendre (int l, LegendreRecurrence& pl, Real x)
{
    // Calculate the Legendre polynomials. We use a stable recurrence relation:
    // (l+1) P_{l+1}(x) = (2l+1) x P_l(x) - l P_{l-1}(x).
    // This uses initial conditions:
    // P_0(x) = 1
    // P_1(x) = x

    if (l == 0) {

        pl.val = 1.0_rt;

    }
    else if (!(l == 1)) {

        pl.prev2 = pl.prev1;
        pl.prev1 = pl.val;
        pl.val  = ((2*l - 1) * x * pl.prev1 - (l-1) * pl.prev2) / l;

    } else {

        pl.prev1 = pl.val;
        pl.val  = x;

    }
}

// Storage for the multipole moments, indexed (l, m, radial bin): the
// interior and exterior moments of the m = 0, cosine and sine terms.

struct MultipoleMomentArrays
{
    Array4<Real> in0;
    Array4<Real> inC;
    Array4<Real> inS;
    Array4<Real> out0;
    Array4<Real> outC;
    Array4<Real> outS;
};

AMREX_GPU_DEVICE AMREX_INLINE
void accumulate_moments(Real cosTheta, Real phiAngle, Real r, Real rho, Real vol,
                   MultipoleMomentArrays const& moments,
                   int npts, int nlo, int index,
                   amrex::Gpu::Handler const& handler,
                   bool parity = false)
{
    LegendreRecurrence pl;
    LegendreRecurrence plm;

    Real rho_r_in, rho_r_out;

    for (int n = nlo; n <= npts-1; ++n) {

        for (int l = 0; l <= gravity::l_max_used; ++l) {

            evalLegendre(l, pl, cosTheta);

            Real dMIn0 = 0.0_rt;
            Real dMOut0 = 0.0_rt;

            if (!(index <= n)) {

                rho_r_out = rho * (std::pow(r, -l-1));

                dMOut0 = pl.val * rho_r_out * vol * multipole::volumeScale;
                if (parity) {
                    dMOut0 = dMOut0 * multipole::mirror_m0(l);
                }

            } else {

                rho_r_in = rho * (std::pow(r, l));

                dMIn0 = pl.val * rho_r_in * vol * multipole::volumeScale;
                if (parity) {
                    dMIn0 = dMIn0 * multipole::mirror_m0(l);
                }

            }

            amrex::Gpu::deviceReduceSum(&moments.in0(l,0,n), dMIn0, handler);
            amrex::Gpu::deviceReduceSum(&moments.out0(l,0,n), dMOut0, handler);

        }

        // For the associated Legendre polynomial loop, we loop over m and then l.
        // It means that we have to recompute rho_r_in or rho_r_out again, but the
        // recursion relation we use for the polynomials depends on l being the
        // innermost loop index.

        for (int m = 1; m <= gravity::l_max_used; ++m) {
            for (int l = 1; l <= gravity::l_max_used; ++l) {

                if (m > l) {
                    continue;
                }

                evalAssocLegendre(l, m, plm, cosTheta);

                Real dMInC = 0.0_rt;
                Real dMInS = 0.0_rt;

                Real dMOutC = 0.0_rt;
                Real dMOutS = 0.0_rt;

                if (!(index <= n)) {

                    rho_r_out = rho * (std::pow(r, -l-1));

                    dMOutC = plm.val * std::cos(m * phiAngle) * rho_r_out * vol * multipole::legNormTable(l,m);
                    if (parity) {
                        dMOutC = dMOutC * multipole::mirror_mC_mS(l,m);
                    }

                    dMOutS = plm.val * std::sin(m * phiAngle) * rho_r_out * vol * multipole::legNormTable(l,m);
                    if (parity) {
                        dMOutS = dMOutS * multipole::mirror_mC_mS(l,m);
                    }

                } else {

                    rho_r_in = rho * (std::pow(r, l));

                    dMInC = plm.val * std::cos(m * phiAngle) * rho_r_in * vol * multipole::legNormTable(l,m);
                    if (parity) {
                        dMInC = dMInC * multipole::mirror_mC_mS(l,m);
                    }

                    dMInS = plm.val * std::sin(m * phiAngle) * rho_r_in * vol * multipole::legNormTable(l,m);
                    if (parity) {
                        dMInS = dMInS * multipole::mirror_mC_mS(l,m);
                    }

                }

                amrex::Gpu::deviceReduceSum(&moments.inC(l,m,n), dMInC, handler);
                amrex::Gpu::deviceReduceSum(&moments.inS(l,m,n), dMInS, handler);
                amrex::Gpu::deviceReduceSum(&moments.outC(l,m,n), dMOutC, handler);
                amrex::Gpu::deviceReduceSum(&moments.outS(l,m,n), dMOutS, handler);

            }
        }

    }
}

AMREX_GPU_DEVICE AMREX_INLINE
void accumulate_moments_mirrored(Real x, Real y, Real z,
                             const GpuArray<Real, AMREX_SPACEDIM>& problo,
                             const GpuArray<Real, AMREX_SPACEDIM>& probhi,
                             Real rho, Real vol,
                             MultipoleMomentArrays const& moments,
                             int npts, int nlo, int index,
                             amrex::Gpu::Handler const& handler)
{

    amrex::ignore_unused(probhi);

    Real xLo = (2.0_rt * (problo[0] - problem::center[0])) / multipole::rmax - x;

#if AMREX_SPACEDIM >= 2
    Real yLo = (2.0_rt * (problo[1] - problem::center[1])) / multipole::rmax - y;
#else
    Real yLo = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
    Real zLo = (2.0_rt * (problo[2] - problem::center[2]) ) / multipole::rmax - z;
#else
    Real zLo = 0.0_rt;
#endif

    Real r, phiAngle, cosTheta;

    if (multipole::mirrorAddLo(0)) {

        r        = std::sqrt(xLo * xLo + y * y + z * z);
        phiAngle = std::atan2(y, xLo);
        cosTheta = z / r;

        accumulate_moments(cosTheta, phiAngle, r, rho, vol, moments, npts, nlo, index, handler);

        if (multipole::mirrorAddLo(1)) {

            r        = std::sqrt(xLo * xLo + yLo * yLo + z * z);
            phiAngle = std::atan2(yLo, xLo);
            cosTheta = z / r;

            accumulate_moments(cosTheta, phiAngle, r, rho, vol, moments, npts, nlo, index, handler);

        }

        if (multipole::mirrorAddLo(2)) {

            r        = std::sqrt(xLo * xLo + y * y + zLo * zLo);
            phiAngle = std::atan2(y, xLo);
            cosTheta = zLo / r;

            accumulate_moments(cosTheta, phiAngle, r, rho, vol, moments, npts, nlo, index, handler);

        }

        if (multipole::mirrorAddLo(1) && multipole::mirrorAddLo(2)) {

            r        = std::sqrt(xLo * xLo + yLo * yLo + zLo * zLo);
            phiAngle = std::atan2(yLo, xLo);
            cosTheta = zLo / r;

            accumulate_moments(cosTheta, phiAngle, r, rho, vol, moments, npts, nlo, index, handler);

        }

    }

    if (multipole::mirrorAddLo(1)) {

        r        = std::sqrt(x * x + yLo * yLo + z * z);
        phiAngle = std::atan2(yLo, x);
        cosTheta = z / r;

        accumulate_moments(cosTheta, phiAngle, r, rho, vol, moments, npts, nlo, index, handler);

        if (multipole::mirrorAddLo(2)) {

            r        = std::sqrt(x * x + yLo * yLo + zLo * zLo);
            phiAngle = std::atan2(yLo, x);
            cosTheta = zLo / r;

            accumulate_moments(cosTheta, phiAngle, r, rho, vol, moments, npts, nlo, index, handler);

        }

    }

    if (multipole::mirrorAddLo(2)) {

        r        = std::sqrt(x * x + y * y + zLo * zLo);
        phiAngle = std::atan2(y, x);
        cosTheta = zLo / r;

        accumulate_moments(cosTheta, phiAngle, r, rho, vol, moments, npts, nlo, index, handler);

    }
}

// The domain edges, and which of them are symmetry boundaries whose
// mirrored mass adds to the direct-sum boundary potential.

struct MirrorDomain
{
    GpuArray<Real, 3> lo;
    GpuArray<Real, 3> hi;
    GpuArray<bool, 3> mirror_lo;
    GpuArray<bool, 3> mirror_hi;
};

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real direct_sum_mirrored(const GpuArray<Real, 3>& loc, const GpuArray<Real, 3>& locb,
                              const MirrorDomain& domain,
                              Real rho, Real dV)
{
    // Add contributions from any symmetric boundaries.

    Real bcTerm = 0.0_rt;

    Real x, y, z, r;

    if (domain.mirror_lo[0]) {

        x = 2.0_rt * domain.lo[0] - loc[0];
        y = loc[1];
        z = loc[2];

        r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

        bcTerm -= C::Gconst * rho * dV / r;

        if (domain.mirror_lo[1]) {

            x = 2.0_rt * domain.lo[0] - loc[0];
            y = 2.0_rt * domain.lo[1] - loc[1];
            z = loc[2];

            r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

            bcTerm -= C::Gconst * rho * dV / r;

        }

        if (domain.mirror_lo[2]) {

            x = 2.0_rt * domain.lo[0] - loc[0];
            y = loc[1];
            z = 2.0_rt * domain.lo[2] - loc[2];

            r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

            bcTerm -= C::Gconst * rho * dV / r;

        }

        if (domain.mirror_lo[1] && domain.mirror_lo[2]) {

            x = 2.0_rt * domain.lo[0] - loc[0];
            y = 2.0_rt * domain.lo[1] - loc[1];
            z = 2.0_rt * domain.lo[2] - loc[2];

            r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

            bcTerm -= C::Gconst * rho * dV / r;

        }

    }

    if (domain.mirror_lo[1]) {

        x = loc[0];
        y = 2.0_rt * domain.lo[1] - loc[1];
        z = loc[2];

        r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

        bcTerm -= C::Gconst * rho * dV / r;

        if (domain.mirror_lo[2]) {

            x = loc[0];
            y = 2.0_rt * domain.lo[1] - loc[1];
            z = 2.0_rt * domain.lo[2] - loc[2];

            r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

            bcTerm -= C::Gconst * rho * dV / r;

        }

    }

    if (domain.mirror_lo[2]) {

        x = loc[0];
        y = loc[1];
        z = 2.0_rt * domain.lo[2] - loc[2];

        r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

        bcTerm -= C::Gconst * rho * dV / r;

    }



    if (domain.mirror_hi[0]) {

        x = 2.0_rt * domain.hi[0] - loc[0];
        y = loc[1];
        z = loc[2];

        r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

        bcTerm -= C::Gconst * rho * dV / r;

        if (domain.mirror_hi[1]) {

            x = 2.0_rt * domain.hi[0] - loc[0];
            y = 2.0_rt * domain.hi[1] - loc[1];
            z = loc[2];

            r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

            bcTerm -= C::Gconst * rho * dV / r;

        }

       if (domain.mirror_hi[2]) {

           x = 2.0_rt * domain.hi[0] - loc[0];
           y = loc[1];
           z = 2.0_rt * domain.hi[2] - loc[2];

           r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

           bcTerm -= C::Gconst * rho * dV / r;

       }

       if (domain.mirror_hi[1] && domain.mirror_hi[2]) {

           x = 2.0_rt * domain.hi[0] - loc[0];
           y = 2.0_rt * domain.hi[1] - loc[1];
           z = 2.0_rt * domain.hi[2] - loc[2];

           r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

           bcTerm -= C::Gconst * rho * dV / r;

       }

    }

    if (domain.mirror_hi[1]) {

        x = loc[0];
        y = 2.0_rt * domain.hi[1] - loc[1];
        z = loc[2];

        r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

        bcTerm -= C::Gconst * rho * dV / r;

        if (domain.mirror_hi[2]) {

            x = loc[0];
            y = 2.0_rt * domain.hi[1] - loc[1];
            z = 2.0_rt * domain.hi[2] - loc[2];

            r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

            bcTerm -= C::Gconst * rho * dV / r;

        }

    }

    if (domain.mirror_hi[2]) {

        x = loc[0];
        y = loc[1];
        z = 2.0_rt * domain.hi[2] - loc[2];

        r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

        bcTerm -= C::Gconst * rho * dV / r;

    }

    return bcTerm;

}

#endif
