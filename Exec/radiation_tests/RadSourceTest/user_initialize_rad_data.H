#ifndef user_initialize_rad_data_H
#define user_initialize_rad_data_H

#include <planck_integrals.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_initialize_rad_data (int i, int j, int k,
                                  Array4<Real> const& rad,
                                  GpuArray<Real, NGROUPS+1> const& xnu,
                                  GpuArray<Real, NGROUPS> const& freq_center,
                                  GpuArray<Real, NGROUPS> const& freq_width,
                                  const GeometryData& geomdata)
{

    if (NGROUPS == 1) {
        rad(i,j,k,0) = problem::E_rad;

    } else {

        Real trad = std::pow(problem::E_rad * C::c_light * 0.25_rt / C::sigma_SB, 0.25_rt);

        Real mgfac = 8.0_rt * M_PI * C::hplanck / std::pow(C::c_light, 3);

        for (int n = 0; n < NGROUPS; n++) {
            Real nu = freq_center[n];
            Real radtmp = std::exp(C::hplanck * nu / (C::k_B * trad)) - 1.0_rt;
            radtmp = (mgfac * nu * nu * nu / radtmp) * freq_width[n];

            rad(i,j,k,n) = radtmp;
        }
    }
}
#endif
