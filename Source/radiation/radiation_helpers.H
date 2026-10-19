#ifndef HYDROSIM_RAD_UTIL_H
#define HYDROSIM_RAD_UTIL_H

#include <cmath>

#include <HydroSim_helpers.H>
#include <radiation_params.H>
#include <flux_limiter.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void cell_center_geometry_factors(int i, int j, int k, const GeometryData& geomdata,
                        Real& r, Real& s)
{
    if (geomdata.Coord() == 0)
    {
        r = 1.0_rt;
        s = 1.0_rt;
    }
    else if (geomdata.Coord() == 1)
    {
        GpuArray<Real, 3> loc;
        position(i, j, k, geomdata, loc);

        r = loc[0];
        s = 1.0_rt;
    }
    else if (geomdata.Coord() == 2)
    {
       GpuArray<Real, 3> loc;
       position(i, j, k, geomdata, loc);

       Real h1 = 0.5e0_rt * geomdata.CellSize(0);
       Real d1 = 1.e0_rt / (3.e0_rt * geomdata.CellSize(0));

       r = loc[0];
       r = d1 * (std::pow(r + h1, 3) - std::pow(r - h1, 3));

#if AMREX_SPACEDIM >= 2
       Real h2 = 0.5e0_rt * geomdata.CellSize(1);
       Real d2 = 1.e0_rt / geomdata.CellSize(1);

       s = loc[1];
       s = d2 * (std::cos(s - h2) - std::cos(s + h2));
#else
       s = 1.0_rt;
#endif
    }
}



AMREX_GPU_HOST_DEVICE AMREX_INLINE
void edge_center_geometry_factors (int i, int j, int k, int idir, const GeometryData& geomdata,
                         Real& r, Real& s)
{
    if (geomdata.Coord() == 0)
    {
        r = 1.0_rt;
        s = 1.0_rt;

    }
    else if (geomdata.Coord() == 1)
    {
        GpuArray<Real, 3> loc;

        if (idir == 0)
        {
            bool ccx = false;
            position(i, j, k, geomdata, loc, ccx);
        }
        else
        {
            position(i, j, k, geomdata, loc);
        }

        r = loc[0];
        s = 1.0_rt;
    }
    else if (geomdata.Coord() == 2)
    {
        GpuArray<Real, 3> loc;

        if (idir == 0)
        {
            bool ccx = false;
            position(i, j, k, geomdata, loc, ccx);

            r = loc[0] * loc[0];

#if AMREX_SPACEDIM >= 2
            position(i, j, k, geomdata, loc);
            s = loc[1];

            Real h2 = 0.5e0_rt * geomdata.CellSize(1);
            Real d2 = 1.e0_rt / geomdata.CellSize(1);

            s = d2 * (std::cos(s - h2) - std::cos(s + h2));
#else
            s = 1.0_rt;
#endif
        }
        else
        {
            GpuArray<Real, 3> loc;
            position(i, j, k, geomdata, loc);

            r = loc[0];

            if (AMREX_SPACEDIM >= 2)
            {
                bool ccx = true;
                bool ccy = false;
                position(i, j, k, geomdata, loc, ccx, ccy);
            }
            else
            {
                bool ccx = false;
                position(i, j, k, geomdata, loc, ccx);
            }

            s = loc[1];

            Real h1 = 0.5e0_rt * geomdata.CellSize(0);
            Real d1 = 1.e0_rt / (3.e0_rt * geomdata.CellSize(0));

            r = d1 * (std::pow(r + h1, 3) - std::pow(r - h1, 3));
            s = std::sin(s);
        }
    }
}



AMREX_GPU_HOST_DEVICE AMREX_INLINE
void face_geometry_factors (int i, int j, int k, int lo, int hi,
                  const GeometryData& geomdata, int face_normal_dir, int side_is_low, Real& r)
{
    const Real* problo = geomdata.ProbLo();
    const Real* dx = geomdata.CellSize();

    if (face_normal_dir == 0) {

        if (geomdata.Coord() == 0) {

            r = 1.0_rt;

        }
        else {

            if (side_is_low == 1) {

                r = problo[0] + lo * dx[0];

            }
            else {

                r = problo[0] + (hi + 1) * dx[0];

            }

            if (geomdata.Coord() == 2) {

                r = r * r;

            }

        }

    }
    else {

        if (geomdata.Coord() == 0) {

            r = 1.0_rt;

        }
        else {

            // This must be RZ since we only support spherical coordinates in 1D.

            GpuArray<Real, 3> loc;
            position(i, j, k, geomdata, loc);

            r = loc[0];

        }

    }
}



AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
compute_total_pressure_sound(const Real* lam, const Real* q,
                  Real cg,
                  Real& total_pressure, Real& total_sound_speed, Real& gamma1_total) {

  Real rad_sound_speed_sq = 0.0_rt;
  Real prad = 0.0_rt;

  for (int g = 0; g < NGROUPS; g++) {
    Real radiation_gamma = 0.0;
    if (radiation::moving_frame) {
      Real RadClosure = Eddington_factor(lam[g]);
      radiation_gamma = (3.0_rt - RadClosure)/2.0_rt;
    } else {
      radiation_gamma = lam[g] + 1.0_rt;
    }

    prad += lam[g]*q[QRADENG+g];
    rad_sound_speed_sq += radiation_gamma * (lam[g]*q[QRADENG+g]) / q[QRHO];
  }

  total_pressure = q[QPRES] + prad;

  total_sound_speed = cg*cg + rad_sound_speed_sq;
  gamma1_total = total_sound_speed * q[QRHO] / total_pressure;

  total_sound_speed = std::sqrt(total_sound_speed);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real LimitedDiffusionLambda (Real r)
{
    Real lambda;

    if (radiation::limiter == 0) {
        // no limiter
        lambda = 1.e0_rt / 3.e0_rt;
    }
    else if (radiation::limiter < 10) {
        // approximate LP
        lambda = (2.e0_rt + r) / (6.e0_rt + r * (3.e0_rt + r));
    }
    else if (radiation::limiter < 20) {
        // Bruenn
        lambda = 1.e0_rt / (3.e0_rt + r);
    }
    else if (radiation::limiter < 30) {
        // Larsen's square root
        lambda = 1.e0_rt / std::sqrt(9.e0_rt + r * r);
    }
    else if (radiation::limiter < 40) {
        // Minerbo
        if (r < 1.5e0_rt) {
            lambda = 2.e0_rt / (3.e0_rt + std::sqrt(9.e0_rt + 12.e0_rt * r * r));
        } else {
            lambda = 1.e0_rt / (1.e0_rt + r + std::sqrt(1.e0_rt + 2.e0_rt * r));
        }
    }
    else {
#ifndef AMREX_USE_GPU
        amrex::Error("Unknown limiter type");
#endif
    }

    return lambda;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real face_opacity_average(Real a, Real b, Real d, int opt)
{
#ifndef AMREX_USE_GPU
    if (opt > 2) {
        amrex::Error("face_opacity_average: invalid averaging option");
    }
#endif

    Real k;
    const Real tiny = 1.e-50_rt;

    if (opt == 0)
    {
        // arithmetic average, geometrically correct(?) but underestimates surface flux
        k = 0.5e0_rt * (a + b + tiny);

    }
    else if (opt == 1)
    {
        // harmonic average, overestimates surface flux
        k = (2.e0_rt * a * b) / (a + b + tiny) + tiny;
    }
    else
    {
        // chooses arithmetic where optically thin, harmonic where optically thick,
        // surface flux approximation at a thick/thin boundary
        k = amrex::min(0.5e0_rt * (a + b + tiny),
                       amrex::max((2.e0_rt * a * b) / (a + b + tiny) + tiny,
                                  4.e0_rt / (3.e0_rt * d)));
    }

    return k;
}

AMREX_INLINE
void coarse_to_fine_face (Array4<Real> const fine,
             Array4<Real const> const crse,
             int idim, const IntVect& level_ratio)
{
    const Dim3 flo = amrex::lbound(fine);
    const Dim3 fhi = amrex::ubound(fine);

    const Dim3 clo = amrex::lbound(crse);
    const Dim3 chi = amrex::ubound(crse);

    Real ifac = 1.0_rt;
    Real refine_factor_j = 1.0_rt;
    Real refine_factor_k = 1.0_rt;
    Real tang_avg_scale = 1.0_rt;

    if (idim == 0) {
#if AMREX_SPACEDIM >= 2
        refine_factor_j = level_ratio[1];
#endif
#if AMREX_SPACEDIM == 3
        refine_factor_k = level_ratio[2];
#endif

        Real tang_avg_scale = refine_factor_j * refine_factor_k;

        int i = flo.x;
        for (int k = flo.z; k <= fhi.z; ++k) {
            for (int j = flo.y; j <= fhi.y; ++j) {
                fine(i,j,k) = crse(clo.x, static_cast<int>(j / refine_factor_j), static_cast<int>(k / refine_factor_k)) / tang_avg_scale;
            }
        }
    }
    else if (idim == 1) {
#if AMREX_SPACEDIM >= 2
        ifac = level_ratio[0];
#endif
#if AMREX_SPACEDIM == 3
        refine_factor_k = level_ratio[2];
#endif

        tang_avg_scale = ifac * refine_factor_k;

        int j = flo.y;
        for (int k = flo.z; k <= fhi.z; ++k) {
            for (int i = flo.x; i <= fhi.x; ++i) {
                fine(i,j,k) = crse(static_cast<int>(i / ifac), clo.y, static_cast<int>(k / refine_factor_k)) / tang_avg_scale;
            }
        }
    }
    else {
#if AMREX_SPACEDIM >= 2
        ifac = level_ratio[0];
#endif
#if AMREX_SPACEDIM == 3
        refine_factor_j = level_ratio[1];
#endif

        tang_avg_scale = ifac * refine_factor_j;

        int k = flo.z;
        for (int j = flo.y; j <= fhi.y; ++j) {
            for (int i = flo.x; i <= fhi.x; ++i) {
                fine(i,j,k) = crse(static_cast<int>(i / ifac), static_cast<int>(j / refine_factor_j), clo.z) / tang_avg_scale;
            }
        }
    }
}

#endif
