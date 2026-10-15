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

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void evalAssocLegendre (int l, int m, Real& plmVal, Real& plmPrev1, Real& plmPrev2, Real x)
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

        plmVal = std::pow(-1, m) * std::pow((1.0_rt - x) * (1.0_rt + x), m * 0.5_rt);

        for (int n = (2*m-1); n >= 3; n = n - 2) {

            plmVal *= n;

        }

    }
    else if (l == m + 1) {

        // P_{m+1}^m

        plmPrev1 = plmVal;
        plmVal  = x * (2*m + 1) * plmPrev1;

    }
    else {

        plmPrev2 = plmPrev1;
        plmPrev1 = plmVal;
        plmVal  = (x * (2*l - 1) * plmPrev1 - (l + m - 1) * plmPrev2) / (l-m);

    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void evalLegendre (int l, Real& plVal, Real& plPrev1, Real& plPrev2, Real x)
{
    // Calculate the Legendre polynomials. We use a stable recurrence relation:
    // (l+1) P_{l+1}(x) = (2l+1) x P_l(x) - l P_{l-1}(x).
    // This uses initial conditions:
    // P_0(x) = 1
    // P_1(x) = x

    if (l == 0) {

        plVal = 1.0_rt;

    }
    else if (!(l == 1)) {

        plPrev2 = plPrev1;
        plPrev1 = plVal;
        plVal  = ((2*l - 1) * x * plPrev1 - (l-1) * plPrev2) / l;

    } else {

        plPrev1 = plVal;
        plVal  = x;

    }
}

AMREX_GPU_DEVICE AMREX_INLINE
void accumulate_moments(Real cosTheta, Real phiAngle, Real r, Real rho, Real vol,
                   Array4<Real> const& mIn0,
                   Array4<Real> const& mInC,
                   Array4<Real> const& mInS,
                   Array4<Real> const& mOut0,
                   Array4<Real> const& mOutC,
                   Array4<Real> const& mOutS,
                   int npts, int nlo, int index,
                   amrex::Gpu::Handler const& handler,
                   bool parity = false)
{
    Real plVal, plPrev1, plPrev2;
    Real plmVal, plmPrev1, plmPrev2;

    Real rho_r_in, rho_r_out;

    for (int n = nlo; n <= npts-1; ++n) {

        for (int l = 0; l <= gravity::l_max_used; ++l) {

            evalLegendre(l, plVal, plPrev1, plPrev2, cosTheta);

            Real dMIn0 = 0.0_rt;
            Real dMOut0 = 0.0_rt;

            if (!(index <= n)) {

                rho_r_out = rho * (std::pow(r, -l-1));

                dMOut0 = plVal * rho_r_out * vol * multipole::volumeScale;
                if (parity) {
                    dMOut0 = dMOut0 * multipole::mirror_m0(l);
                }

            } else {

                rho_r_in = rho * (std::pow(r, l));

                dMIn0 = plVal * rho_r_in * vol * multipole::volumeScale;
                if (parity) {
                    dMIn0 = dMIn0 * multipole::mirror_m0(l);
                }

            }

            amrex::Gpu::deviceReduceSum(&mIn0(l,0,n), dMIn0, handler);
            amrex::Gpu::deviceReduceSum(&mOut0(l,0,n), dMOut0, handler);

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

                evalAssocLegendre(l, m, plmVal, plmPrev1, plmPrev2, cosTheta);

                Real dMInC = 0.0_rt;
                Real dMInS = 0.0_rt;

                Real dMOutC = 0.0_rt;
                Real dMOutS = 0.0_rt;

                if (!(index <= n)) {

                    rho_r_out = rho * (std::pow(r, -l-1));

                    dMOutC = plmVal * std::cos(m * phiAngle) * rho_r_out * vol * multipole::legNormTable(l,m);
                    if (parity) {
                        dMOutC = dMOutC * multipole::mirror_mC_mS(l,m);
                    }

                    dMOutS = plmVal * std::sin(m * phiAngle) * rho_r_out * vol * multipole::legNormTable(l,m);
                    if (parity) {
                        dMOutS = dMOutS * multipole::mirror_mC_mS(l,m);
                    }

                } else {

                    rho_r_in = rho * (std::pow(r, l));

                    dMInC = plmVal * std::cos(m * phiAngle) * rho_r_in * vol * multipole::legNormTable(l,m);
                    if (parity) {
                        dMInC = dMInC * multipole::mirror_mC_mS(l,m);
                    }

                    dMInS = plmVal * std::sin(m * phiAngle) * rho_r_in * vol * multipole::legNormTable(l,m);
                    if (parity) {
                        dMInS = dMInS * multipole::mirror_mC_mS(l,m);
                    }

                }

                amrex::Gpu::deviceReduceSum(&mInC(l,m,n), dMInC, handler);
                amrex::Gpu::deviceReduceSum(&mInS(l,m,n), dMInS, handler);
                amrex::Gpu::deviceReduceSum(&mOutC(l,m,n), dMOutC, handler);
                amrex::Gpu::deviceReduceSum(&mOutS(l,m,n), dMOutS, handler);

            }
        }

    }
}

AMREX_GPU_DEVICE AMREX_INLINE
void accumulate_moments_mirrored(Real x, Real y, Real z,
                             const GpuArray<Real, AMREX_SPACEDIM>& problo,
                             const GpuArray<Real, AMREX_SPACEDIM>& probhi,
                             Real rho, Real vol,
                             Array4<Real> const& mIn0,
                             Array4<Real> const& mInC,
                             Array4<Real> const& mInS,
                             Array4<Real> const& mOut0,
                             Array4<Real> const& mOutC,
                             Array4<Real> const& mOutS,
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

        accumulate_moments(cosTheta, phiAngle, r, rho, vol, mIn0, mInC, mInS, mOut0, mOutC, mOutS, npts, nlo, index, handler);

        if (multipole::mirrorAddLo(1)) {

            r        = std::sqrt(xLo * xLo + yLo * yLo + z * z);
            phiAngle = std::atan2(yLo, xLo);
            cosTheta = z / r;

            accumulate_moments(cosTheta, phiAngle, r, rho, vol, mIn0, mInC, mInS, mOut0, mOutC, mOutS, npts, nlo, index, handler);

        }

        if (multipole::mirrorAddLo(2)) {

            r        = std::sqrt(xLo * xLo + y * y + zLo * zLo);
            phiAngle = std::atan2(y, xLo);
            cosTheta = zLo / r;

            accumulate_moments(cosTheta, phiAngle, r, rho, vol, mIn0, mInC, mInS, mOut0, mOutC, mOutS, npts, nlo, index, handler);

        }

        if (multipole::mirrorAddLo(1) && multipole::mirrorAddLo(2)) {

            r        = std::sqrt(xLo * xLo + yLo * yLo + zLo * zLo);
            phiAngle = std::atan2(yLo, xLo);
            cosTheta = zLo / r;

            accumulate_moments(cosTheta, phiAngle, r, rho, vol, mIn0, mInC, mInS, mOut0, mOutC, mOutS, npts, nlo, index, handler);

        }

    }

    if (multipole::mirrorAddLo(1)) {

        r        = std::sqrt(x * x + yLo * yLo + z * z);
        phiAngle = std::atan2(yLo, x);
        cosTheta = z / r;

        accumulate_moments(cosTheta, phiAngle, r, rho, vol, mIn0, mInC, mInS, mOut0, mOutC, mOutS, npts, nlo, index, handler);

        if (multipole::mirrorAddLo(2)) {

            r        = std::sqrt(x * x + yLo * yLo + zLo * zLo);
            phiAngle = std::atan2(yLo, x);
            cosTheta = zLo / r;

            accumulate_moments(cosTheta, phiAngle, r, rho, vol, mIn0, mInC, mInS, mOut0, mOutC, mOutS, npts, nlo, index, handler);

        }

    }

    if (multipole::mirrorAddLo(2)) {

        r        = std::sqrt(x * x + y * y + zLo * zLo);
        phiAngle = std::atan2(y, x);
        cosTheta = zLo / r;

        accumulate_moments(cosTheta, phiAngle, r, rho, vol, mIn0, mInC, mInS, mOut0, mOutC, mOutS, npts, nlo, index, handler);

    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real direct_sum_mirrored(const GpuArray<Real, 3>& loc, const GpuArray<Real, 3>& locb,
                              const GpuArray<Real, 3>& problo, const GpuArray<Real, 3>& probhi,
                              Real rho, Real dV,
                              const GpuArray<bool, 3>& mirrorAddLo, const GpuArray<bool, 3>& mirrorAddHi)
{
    // Add contributions from any symmetric boundaries.

    Real bcTerm = 0.0_rt;

    Real x, y, z, r;

    if (mirrorAddLo[0]) {

        x = 2.0_rt * problo[0] - loc[0];
        y = loc[1];
        z = loc[2];

        r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

        bcTerm -= C::Gconst * rho * dV / r;

        if (mirrorAddLo[1]) {

            x = 2.0_rt * problo[0] - loc[0];
            y = 2.0_rt * problo[1] - loc[1];
            z = loc[2];

            r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

            bcTerm -= C::Gconst * rho * dV / r;

        }

        if (mirrorAddLo[2]) {

            x = 2.0_rt * problo[0] - loc[0];
            y = loc[1];
            z = 2.0_rt * problo[2] - loc[2];

            r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

            bcTerm -= C::Gconst * rho * dV / r;

        }

        if (mirrorAddLo[1] && mirrorAddLo[2]) {

            x = 2.0_rt * problo[0] - loc[0];
            y = 2.0_rt * problo[1] - loc[1];
            z = 2.0_rt * problo[2] - loc[2];

            r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

            bcTerm -= C::Gconst * rho * dV / r;

        }

    }

    if (mirrorAddLo[1]) {

        x = loc[0];
        y = 2.0_rt * problo[1] - loc[1];
        z = loc[2];

        r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

        bcTerm -= C::Gconst * rho * dV / r;

        if (mirrorAddLo[2]) {

            x = loc[0];
            y = 2.0_rt * problo[1] - loc[1];
            z = 2.0_rt * problo[2] - loc[2];

            r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

            bcTerm -= C::Gconst * rho * dV / r;

        }

    }

    if (mirrorAddLo[2]) {

        x = loc[0];
        y = loc[1];
        z = 2.0_rt * problo[2] - loc[2];

        r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

        bcTerm -= C::Gconst * rho * dV / r;

    }



    if (mirrorAddHi[0]) {

        x = 2.0_rt * probhi[0] - loc[0];
        y = loc[1];
        z = loc[2];

        r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

        bcTerm -= C::Gconst * rho * dV / r;

        if (mirrorAddHi[1]) {

            x = 2.0_rt * probhi[0] - loc[0];
            y = 2.0_rt * probhi[1] - loc[1];
            z = loc[2];

            r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

            bcTerm -= C::Gconst * rho * dV / r;

        }

       if (mirrorAddHi[2]) {

           x = 2.0_rt * probhi[0] - loc[0];
           y = loc[1];
           z = 2.0_rt * probhi[2] - loc[2];

           r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

           bcTerm -= C::Gconst * rho * dV / r;

       }

       if (mirrorAddHi[1] && mirrorAddHi[2]) {

           x = 2.0_rt * probhi[0] - loc[0];
           y = 2.0_rt * probhi[1] - loc[1];
           z = 2.0_rt * probhi[2] - loc[2];

           r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

           bcTerm -= C::Gconst * rho * dV / r;

       }

    }

    if (mirrorAddHi[1]) {

        x = loc[0];
        y = 2.0_rt * probhi[1] - loc[1];
        z = loc[2];

        r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

        bcTerm -= C::Gconst * rho * dV / r;

        if (mirrorAddHi[2]) {

            x = loc[0];
            y = 2.0_rt * probhi[1] - loc[1];
            z = 2.0_rt * probhi[2] - loc[2];

            r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

            bcTerm -= C::Gconst * rho * dV / r;

        }

    }

    if (mirrorAddHi[2]) {

        x = loc[0];
        y = loc[1];
        z = 2.0_rt * probhi[2] - loc[2];

        r = std::sqrt((x - locb[0]) * (x - locb[0]) + (y - locb[1]) * (y - locb[1]) + (z - locb[2]) * (z - locb[2]));

        bcTerm -= C::Gconst * rho * dV / r;

    }

    return bcTerm;

}

#endif
