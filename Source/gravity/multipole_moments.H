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

    struct MomentViews {
        amrex::Array4<amrex::Real> lower0, lowerC, lowerS, upper0, upperC, upperS;
    };
}

struct LegendreRecurrence {
    Real value, prev, prev2;
};

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void evaluateLegendreByDegree (int l, LegendreRecurrence& p, Real x)
{

    if (l == 0) {

        p.value = 1.0_rt;

    }
    else if (!(l == 1)) {

        p.prev2 = p.prev;
        p.prev = p.value;
        p.value = ((2*l - 1) * x * p.prev - (l-1) * p.prev2) / l;

    } else {

        p.prev = p.value;
        p.value = x;

    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void evaluateAssociatedLegendre (int l, int m, LegendreRecurrence& p, Real x)
{

    if (l == m) {

        p.value = std::pow(-1, m) * std::pow((1.0_rt - x) * (1.0_rt + x), m * 0.5_rt);

        for (int n = (2*m-1); n >= 3; n = n - 2) {

            p.value *= n;

        }

    }
    else if (!(l == m + 1)) {

        p.prev2 = p.prev;
        p.prev = p.value;
        p.value = (x * (2*l - 1) * p.prev - (l + m - 1) * p.prev2) / (l-m);

    } else {

        p.prev = p.value;
        p.value = x * (2*m + 1) * p.prev;

    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real image_point_potential(Real x, Real y, Real z, const GpuArray<Real, 3>& boundaryLoc,
                           Real rho, Real dV)
{
    Real r = std::sqrt((x - boundaryLoc[0]) * (x - boundaryLoc[0]) + (y - boundaryLoc[1]) * (y - boundaryLoc[1]) + (z - boundaryLoc[2]) * (z - boundaryLoc[2]));

    return C::Gconst * rho * dV / r;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real accumulate_mirror_contribution(const GpuArray<Real, 3>& loc, const GpuArray<Real, 3>& boundaryLoc,
                              const GpuArray<Real, 3>& problo, const GpuArray<Real, 3>& probhi,
                              Real rho, Real dV,
                              const GpuArray<bool, 3>& doFoldedAddLo, const GpuArray<bool, 3>& doFoldedAddHi)
{

    Real boundaryTerm = 0.0_rt;

    if (doFoldedAddLo[0]) {

        boundaryTerm -= image_point_potential(2.0_rt * problo[0] - loc[0], loc[1], loc[2], boundaryLoc, rho, dV);

        if (doFoldedAddLo[1]) {

            boundaryTerm -= image_point_potential(2.0_rt * problo[0] - loc[0], 2.0_rt * problo[1] - loc[1], loc[2], boundaryLoc, rho, dV);

        }

        if (doFoldedAddLo[2]) {

            boundaryTerm -= image_point_potential(2.0_rt * problo[0] - loc[0], loc[1], 2.0_rt * problo[2] - loc[2], boundaryLoc, rho, dV);

        }

        if (doFoldedAddLo[1] && doFoldedAddLo[2]) {

            boundaryTerm -= image_point_potential(2.0_rt * problo[0] - loc[0], 2.0_rt * problo[1] - loc[1], 2.0_rt * problo[2] - loc[2], boundaryLoc, rho, dV);

        }

    }

    if (doFoldedAddLo[1]) {

        boundaryTerm -= image_point_potential(loc[0], 2.0_rt * problo[1] - loc[1], loc[2], boundaryLoc, rho, dV);

        if (doFoldedAddLo[2]) {

            boundaryTerm -= image_point_potential(loc[0], 2.0_rt * problo[1] - loc[1], 2.0_rt * problo[2] - loc[2], boundaryLoc, rho, dV);

        }

    }

    if (doFoldedAddLo[2]) {

        boundaryTerm -= image_point_potential(loc[0], loc[1], 2.0_rt * problo[2] - loc[2], boundaryLoc, rho, dV);

    }

    if (doFoldedAddHi[0]) {

        boundaryTerm -= image_point_potential(2.0_rt * probhi[0] - loc[0], loc[1], loc[2], boundaryLoc, rho, dV);

        if (doFoldedAddHi[1]) {

            boundaryTerm -= image_point_potential(2.0_rt * probhi[0] - loc[0], 2.0_rt * probhi[1] - loc[1], loc[2], boundaryLoc, rho, dV);

        }

       if (doFoldedAddHi[2]) {

           boundaryTerm -= image_point_potential(2.0_rt * probhi[0] - loc[0], loc[1], 2.0_rt * probhi[2] - loc[2], boundaryLoc, rho, dV);

       }

       if (doFoldedAddHi[1] && doFoldedAddHi[2]) {

           boundaryTerm -= image_point_potential(2.0_rt * probhi[0] - loc[0], 2.0_rt * probhi[1] - loc[1], 2.0_rt * probhi[2] - loc[2], boundaryLoc, rho, dV);

       }

    }

    if (doFoldedAddHi[1]) {

        boundaryTerm -= image_point_potential(loc[0], 2.0_rt * probhi[1] - loc[1], loc[2], boundaryLoc, rho, dV);

        if (doFoldedAddHi[2]) {

            boundaryTerm -= image_point_potential(loc[0], 2.0_rt * probhi[1] - loc[1], 2.0_rt * probhi[2] - loc[2], boundaryLoc, rho, dV);

        }

    }

    if (doFoldedAddHi[2]) {

        boundaryTerm -= image_point_potential(loc[0], loc[1], 2.0_rt * probhi[2] - loc[2], boundaryLoc, rho, dV);

    }

    return boundaryTerm;

}

AMREX_GPU_DEVICE AMREX_INLINE
void accumulate_moments(Real cosPolar, Real sampleAzimuth, Real r, Real rho, Real vol,
                   const multipole::MomentViews& views,
                   int npts, int nlo, int index,
                   amrex::Gpu::Handler const& handler,
                   bool parity = false)
{
    LegendreRecurrence legendre;
    LegendreRecurrence assoc_legendre;

    Real density_rpow_pos, density_rpow_neg;

    for (int n = nlo; n <= npts-1; ++n) {

        for (int l = 0; l <= gravity::lnum; ++l) {

            evaluateLegendreByDegree(l, legendre, cosPolar);

            Real dMomLower0 = 0.0_rt;
            Real dMomUpper0 = 0.0_rt;

            if (index <= n) {

                density_rpow_pos = rho * (std::pow(r, l));

                dMomLower0 = legendre.value * density_rpow_pos * vol * multipole::volumeScale;
                if (parity) {
                    dMomLower0 = dMomLower0 * multipole::axis_parity_table(l);
                }

            }
            else {

                density_rpow_neg = rho * (std::pow(r, -l-1));

                dMomUpper0 = legendre.value * density_rpow_neg * vol * multipole::volumeScale;
                if (parity) {
                    dMomUpper0 = dMomUpper0 * multipole::axis_parity_table(l);
                }

            }

            amrex::Gpu::deviceReduceSum(&views.lower0(l,0,n), dMomLower0, handler);
            amrex::Gpu::deviceReduceSum(&views.upper0(l,0,n), dMomUpper0, handler);

        }

        for (int m = 1; m <= gravity::lnum; ++m) {
            for (int l = 1; l <= gravity::lnum; ++l) {

                if (m > l) {
                    continue;
                }

                evaluateAssociatedLegendre(l, m, assoc_legendre, cosPolar);

                Real dMomLowerCos = 0.0_rt;
                Real dMomLowerSin = 0.0_rt;

                Real dMomUpperCos = 0.0_rt;
                Real dMomUpperSin = 0.0_rt;

                if (index <= n) {

                    density_rpow_pos = rho * (std::pow(r, l));

                    dMomLowerCos = assoc_legendre.value * std::cos(m * sampleAzimuth) * density_rpow_pos * vol * multipole::factorialTable(l,m);
                    if (parity) {
                        dMomLowerCos = dMomLowerCos * multipole::azimuth_parity_table(l,m);
                    }

                    dMomLowerSin = assoc_legendre.value * std::sin(m * sampleAzimuth) * density_rpow_pos * vol * multipole::factorialTable(l,m);
                    if (parity) {
                        dMomLowerSin = dMomLowerSin * multipole::azimuth_parity_table(l,m);
                    }

                }
                else {

                    density_rpow_neg = rho * (std::pow(r, -l-1));

                    dMomUpperCos = assoc_legendre.value * std::cos(m * sampleAzimuth) * density_rpow_neg * vol * multipole::factorialTable(l,m);
                    if (parity) {
                        dMomUpperCos = dMomUpperCos * multipole::azimuth_parity_table(l,m);
                    }

                    dMomUpperSin = assoc_legendre.value * std::sin(m * sampleAzimuth) * density_rpow_neg * vol * multipole::factorialTable(l,m);
                    if (parity) {
                        dMomUpperSin = dMomUpperSin * multipole::azimuth_parity_table(l,m);
                    }

                }

                amrex::Gpu::deviceReduceSum(&views.lowerC(l,m,n), dMomLowerCos, handler);
                amrex::Gpu::deviceReduceSum(&views.lowerS(l,m,n), dMomLowerSin, handler);
                amrex::Gpu::deviceReduceSum(&views.upperC(l,m,n), dMomUpperCos, handler);
                amrex::Gpu::deviceReduceSum(&views.upperS(l,m,n), dMomUpperSin, handler);

            }
        }

    }
}

AMREX_GPU_DEVICE AMREX_INLINE
void accumulate_image_moments(Real xi, Real yi, Real zi, Real rho, Real vol,
                              const multipole::MomentViews& views,
                              int npts, int nlo, int index,
                              amrex::Gpu::Handler const& handler)
{
    Real r        = std::sqrt(xi * xi + yi * yi + zi * zi);
    Real sampleAzimuth = std::atan2(yi, xi);
    Real cosPolar = zi / r;

    accumulate_moments(cosPolar, sampleAzimuth, r, rho, vol, views, npts, nlo, index, handler);
}

AMREX_GPU_DEVICE AMREX_INLINE
void accumulate_mirror_moments(Real x, Real y, Real z,
                             const GpuArray<Real, AMREX_SPACEDIM>& problo,
                             const GpuArray<Real, AMREX_SPACEDIM>& probhi,
                             Real rho, Real vol,
                             const multipole::MomentViews& views,
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

    if (multipole::doFoldedAddLo(0)) {

        accumulate_image_moments(xLo, y, z, rho, vol, views, npts, nlo, index, handler);

        if (multipole::doFoldedAddLo(1)) {

            accumulate_image_moments(xLo, yLo, z, rho, vol, views, npts, nlo, index, handler);

        }

        if (multipole::doFoldedAddLo(2)) {

            accumulate_image_moments(xLo, y, zLo, rho, vol, views, npts, nlo, index, handler);

        }

        if (multipole::doFoldedAddLo(1) && multipole::doFoldedAddLo(2)) {

            accumulate_image_moments(xLo, yLo, zLo, rho, vol, views, npts, nlo, index, handler);

        }

    }

    if (multipole::doFoldedAddLo(1)) {

        accumulate_image_moments(x, yLo, z, rho, vol, views, npts, nlo, index, handler);

        if (multipole::doFoldedAddLo(2)) {

            accumulate_image_moments(x, yLo, zLo, rho, vol, views, npts, nlo, index, handler);

        }

    }

    if (multipole::doFoldedAddLo(2)) {

        accumulate_image_moments(x, y, zLo, rho, vol, views, npts, nlo, index, handler);

    }
}

#endif
