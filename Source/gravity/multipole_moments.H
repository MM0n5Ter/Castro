#ifndef MULTIPOLE_MOMENTS_H
#define MULTIPOLE_MOMENTS_H

#include <HydroSim_math.H>

namespace multipole {
    const int max_moment_order = 30;
    extern AMREX_GPU_MANAGED amrex::Array2D<amrex::Real, 0, max_moment_order, 0, max_moment_order> factorialTable;
    extern AMREX_GPU_MANAGED amrex::Array1D<amrex::Real, 0, max_moment_order> axis_parity_table;
    extern AMREX_GPU_MANAGED amrex::Array2D<amrex::Real, 0, max_moment_order, 0, max_moment_order> azimuth_parity_table;
    extern AMREX_GPU_MANAGED amrex::Real volumeScale;
    extern AMREX_GPU_MANAGED amrex::Real parityScale;
    extern AMREX_GPU_MANAGED amrex::Real rmax;
    extern AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> doMirrorLo;
    extern AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> doMirrorHi;
    extern AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> doFoldedAddLo;
    extern AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> doFoldedAddHi;
    extern AMREX_GPU_MANAGED bool doFoldedAdd;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void evaluateAssociatedLegendre (int l, int m, Real& assocLegendreValue, Real& assocLegendreValue1, Real& assocLegendreValue2, Real x)
{

    if (l == m) {

        assocLegendreValue = std::pow(-1, m) * std::pow((1.0_rt - x) * (1.0_rt + x), m * 0.5_rt);

        for (int n = (2*m-1); n >= 3; n = n - 2) {

            assocLegendreValue *= n;

        }

    }
    else if (l == m + 1) {

        assocLegendreValue1 = assocLegendreValue;
        assocLegendreValue  = x * (2*m + 1) * assocLegendreValue1;

    }
    else {

        assocLegendreValue2 = assocLegendreValue1;
        assocLegendreValue1 = assocLegendreValue;
        assocLegendreValue  = (x * (2*l - 1) * assocLegendreValue1 - (l + m - 1) * assocLegendreValue2) / (l-m);

    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void evaluateLegendreByDegree (int l, Real& legendreValue, Real& legendreValue1, Real& legendreValue2, Real x)
{

    if (l == 0) {

        legendreValue = 1.0_rt;

    }
    else if (l == 1) {

        legendreValue1 = legendreValue;
        legendreValue  = x;

    }
    else {

        legendreValue2 = legendreValue1;
        legendreValue1 = legendreValue;
        legendreValue  = ((2*l - 1) * x * legendreValue1 - (l-1) * legendreValue2) / l;

    }
}

AMREX_GPU_DEVICE AMREX_INLINE
void accumulate_moments(Real cosPolar, Real sampleAzimuth, Real r, Real rho, Real vol,
                   Array4<Real> const& qL0,
                   Array4<Real> const& qLC,
                   Array4<Real> const& qLS,
                   Array4<Real> const& qU0,
                   Array4<Real> const& qUC,
                   Array4<Real> const& qUS,
                   int npts, int nlo, int index,
                   amrex::Gpu::Handler const& handler,
                   bool parity = false)
{
    Real legendreValue, legendreValue1, legendreValue2;
    Real assocLegendreValue, assocLegendreValue1, assocLegendreValue2;

    Real density_rpow_pos, density_rpow_neg;

    for (int n = nlo; n <= npts-1; ++n) {

        for (int l = 0; l <= gravity::lnum; ++l) {

            evaluateLegendreByDegree(l, legendreValue, legendreValue1, legendreValue2, cosPolar);

            Real dMomLower0 = 0.0_rt;
            Real dMomUpper0 = 0.0_rt;

            if (index <= n) {

                density_rpow_pos = rho * (std::pow(r, l));

                dMomLower0 = legendreValue * density_rpow_pos * vol * multipole::volumeScale;
                if (parity) {
                    dMomLower0 = dMomLower0 * multipole::axis_parity_table(l);
                }

            }
            else {

                density_rpow_neg = rho * (std::pow(r, -l-1));

                dMomUpper0 = legendreValue * density_rpow_neg * vol * multipole::volumeScale;
                if (parity) {
                    dMomUpper0 = dMomUpper0 * multipole::axis_parity_table(l);
                }

            }

            amrex::Gpu::deviceReduceSum(&qL0(l,0,n), dMomLower0, handler);
            amrex::Gpu::deviceReduceSum(&qU0(l,0,n), dMomUpper0, handler);

        }

        for (int m = 1; m <= gravity::lnum; ++m) {
            for (int l = 1; l <= gravity::lnum; ++l) {

                if (m > l) {
                    continue;
                }

                evaluateAssociatedLegendre(l, m, assocLegendreValue, assocLegendreValue1, assocLegendreValue2, cosPolar);

                Real dMomLowerCos = 0.0_rt;
                Real dMomLowerSin = 0.0_rt;

                Real dMomUpperCos = 0.0_rt;
                Real dMomUpperSin = 0.0_rt;

                if (index <= n) {

                    density_rpow_pos = rho * (std::pow(r, l));

                    dMomLowerCos = assocLegendreValue * std::cos(m * sampleAzimuth) * density_rpow_pos * vol * multipole::factorialTable(l,m);
                    if (parity) {
                        dMomLowerCos = dMomLowerCos * multipole::azimuth_parity_table(l,m);
                    }

                    dMomLowerSin = assocLegendreValue * std::sin(m * sampleAzimuth) * density_rpow_pos * vol * multipole::factorialTable(l,m);
                    if (parity) {
                        dMomLowerSin = dMomLowerSin * multipole::azimuth_parity_table(l,m);
                    }

                }
                else {

                    density_rpow_neg = rho * (std::pow(r, -l-1));

                    dMomUpperCos = assocLegendreValue * std::cos(m * sampleAzimuth) * density_rpow_neg * vol * multipole::factorialTable(l,m);
                    if (parity) {
                        dMomUpperCos = dMomUpperCos * multipole::azimuth_parity_table(l,m);
                    }

                    dMomUpperSin = assocLegendreValue * std::sin(m * sampleAzimuth) * density_rpow_neg * vol * multipole::factorialTable(l,m);
                    if (parity) {
                        dMomUpperSin = dMomUpperSin * multipole::azimuth_parity_table(l,m);
                    }

                }

                amrex::Gpu::deviceReduceSum(&qLC(l,m,n), dMomLowerCos, handler);
                amrex::Gpu::deviceReduceSum(&qLS(l,m,n), dMomLowerSin, handler);
                amrex::Gpu::deviceReduceSum(&qUC(l,m,n), dMomUpperCos, handler);
                amrex::Gpu::deviceReduceSum(&qUS(l,m,n), dMomUpperSin, handler);

            }
        }

    }
}

AMREX_GPU_DEVICE AMREX_INLINE
void accumulate_mirror_moments(Real x, Real y, Real z,
                             const GpuArray<Real, AMREX_SPACEDIM>& problo,
                             const GpuArray<Real, AMREX_SPACEDIM>& probhi,
                             Real rho, Real vol,
                             Array4<Real> const& qL0,
                             Array4<Real> const& qLC,
                             Array4<Real> const& qLS,
                             Array4<Real> const& qU0,
                             Array4<Real> const& qUC,
                             Array4<Real> const& qUS,
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

    Real r, sampleAzimuth, cosPolar;

    if (multipole::doFoldedAddLo(0)) {

        r        = std::sqrt(xLo * xLo + y * y + z * z);
        sampleAzimuth = std::atan2(y, xLo);
        cosPolar = z / r;

        accumulate_moments(cosPolar, sampleAzimuth, r, rho, vol, qL0, qLC, qLS, qU0, qUC, qUS, npts, nlo, index, handler);

        if (multipole::doFoldedAddLo(1)) {

            r        = std::sqrt(xLo * xLo + yLo * yLo + z * z);
            sampleAzimuth = std::atan2(yLo, xLo);
            cosPolar = z / r;

            accumulate_moments(cosPolar, sampleAzimuth, r, rho, vol, qL0, qLC, qLS, qU0, qUC, qUS, npts, nlo, index, handler);

        }

        if (multipole::doFoldedAddLo(2)) {

            r        = std::sqrt(xLo * xLo + y * y + zLo * zLo);
            sampleAzimuth = std::atan2(y, xLo);
            cosPolar = zLo / r;

            accumulate_moments(cosPolar, sampleAzimuth, r, rho, vol, qL0, qLC, qLS, qU0, qUC, qUS, npts, nlo, index, handler);

        }

        if (multipole::doFoldedAddLo(1) && multipole::doFoldedAddLo(2)) {

            r        = std::sqrt(xLo * xLo + yLo * yLo + zLo * zLo);
            sampleAzimuth = std::atan2(yLo, xLo);
            cosPolar = zLo / r;

            accumulate_moments(cosPolar, sampleAzimuth, r, rho, vol, qL0, qLC, qLS, qU0, qUC, qUS, npts, nlo, index, handler);

        }

    }

    if (multipole::doFoldedAddLo(1)) {

        r        = std::sqrt(x * x + yLo * yLo + z * z);
        sampleAzimuth = std::atan2(yLo, x);
        cosPolar = z / r;

        accumulate_moments(cosPolar, sampleAzimuth, r, rho, vol, qL0, qLC, qLS, qU0, qUC, qUS, npts, nlo, index, handler);

        if (multipole::doFoldedAddLo(2)) {

            r        = std::sqrt(x * x + yLo * yLo + zLo * zLo);
            sampleAzimuth = std::atan2(yLo, x);
            cosPolar = zLo / r;

            accumulate_moments(cosPolar, sampleAzimuth, r, rho, vol, qL0, qLC, qLS, qU0, qUC, qUS, npts, nlo, index, handler);

        }

    }

    if (multipole::doFoldedAddLo(2)) {

        r        = std::sqrt(x * x + y * y + zLo * zLo);
        sampleAzimuth = std::atan2(y, x);
        cosPolar = zLo / r;

        accumulate_moments(cosPolar, sampleAzimuth, r, rho, vol, qL0, qLC, qLS, qU0, qUC, qUS, npts, nlo, index, handler);

    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real accumulate_mirror_contribution(const GpuArray<Real, 3>& loc, const GpuArray<Real, 3>& boundaryLoc,
                              const GpuArray<Real, 3>& problo, const GpuArray<Real, 3>& probhi,
                              Real rho, Real dV,
                              const GpuArray<bool, 3>& doFoldedAddLo, const GpuArray<bool, 3>& doFoldedAddHi)
{

    Real boundaryTerm = 0.0_rt;

    Real x, y, z, r;

    if (doFoldedAddLo[0]) {

        x = 2.0_rt * problo[0] - loc[0];
        y = loc[1];
        z = loc[2];

        r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

        boundaryTerm -= C::Gconst * rho * dV / r;

        if (doFoldedAddLo[1]) {

            x = 2.0_rt * problo[0] - loc[0];
            y = 2.0_rt * problo[1] - loc[1];
            z = loc[2];

            r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

            boundaryTerm -= C::Gconst * rho * dV / r;

        }

        if (doFoldedAddLo[2]) {

            x = 2.0_rt * problo[0] - loc[0];
            y = loc[1];
            z = 2.0_rt * problo[2] - loc[2];

            r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

            boundaryTerm -= C::Gconst * rho * dV / r;

        }

        if (doFoldedAddLo[1] && doFoldedAddLo[2]) {

            x = 2.0_rt * problo[0] - loc[0];
            y = 2.0_rt * problo[1] - loc[1];
            z = 2.0_rt * problo[2] - loc[2];

            r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

            boundaryTerm -= C::Gconst * rho * dV / r;

        }

    }

    if (doFoldedAddLo[1]) {

        x = loc[0];
        y = 2.0_rt * problo[1] - loc[1];
        z = loc[2];

        r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

        boundaryTerm -= C::Gconst * rho * dV / r;

        if (doFoldedAddLo[2]) {

            x = loc[0];
            y = 2.0_rt * problo[1] - loc[1];
            z = 2.0_rt * problo[2] - loc[2];

            r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

            boundaryTerm -= C::Gconst * rho * dV / r;

        }

    }

    if (doFoldedAddLo[2]) {

        x = loc[0];
        y = loc[1];
        z = 2.0_rt * problo[2] - loc[2];

        r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

        boundaryTerm -= C::Gconst * rho * dV / r;

    }

    if (doFoldedAddHi[0]) {

        x = 2.0_rt * probhi[0] - loc[0];
        y = loc[1];
        z = loc[2];

        r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

        boundaryTerm -= C::Gconst * rho * dV / r;

        if (doFoldedAddHi[1]) {

            x = 2.0_rt * probhi[0] - loc[0];
            y = 2.0_rt * probhi[1] - loc[1];
            z = loc[2];

            r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

            boundaryTerm -= C::Gconst * rho * dV / r;

        }

       if (doFoldedAddHi[2]) {

           x = 2.0_rt * probhi[0] - loc[0];
           y = loc[1];
           z = 2.0_rt * probhi[2] - loc[2];

           r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

           boundaryTerm -= C::Gconst * rho * dV / r;

       }

       if (doFoldedAddHi[1] && doFoldedAddHi[2]) {

           x = 2.0_rt * probhi[0] - loc[0];
           y = 2.0_rt * probhi[1] - loc[1];
           z = 2.0_rt * probhi[2] - loc[2];

           r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

           boundaryTerm -= C::Gconst * rho * dV / r;

       }

    }

    if (doFoldedAddHi[1]) {

        x = loc[0];
        y = 2.0_rt * probhi[1] - loc[1];
        z = loc[2];

        r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

        boundaryTerm -= C::Gconst * rho * dV / r;

        if (doFoldedAddHi[2]) {

            x = loc[0];
            y = 2.0_rt * probhi[1] - loc[1];
            z = 2.0_rt * probhi[2] - loc[2];

            r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

            boundaryTerm -= C::Gconst * rho * dV / r;

        }

    }

    if (doFoldedAddHi[2]) {

        x = loc[0];
        y = loc[1];
        z = 2.0_rt * probhi[2] - loc[2];

        r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

        boundaryTerm -= C::Gconst * rho * dV / r;

    }

    return boundaryTerm;

}

#endif
