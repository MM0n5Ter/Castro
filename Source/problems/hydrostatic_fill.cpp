#include <AMReX_BLFort.H>
#include <HydroSim.H>
#include <runtime_parameters.H>
#include <external_bc_codes.H>

using namespace amrex;




void
hydrostatic_fill(const Box& bx, Array4<Real> const& adv,
              Geometry const& geom, const Vector<BCRec>& bcr,
              const Real time)
{

    amrex::ignore_unused(time);

    auto domlo = geom.Domain().loVect3d();
    auto domhi = geom.Domain().hiVect3d();

    const auto *lo = bx.loVect();
    const auto *hi = bx.hiVect();

    auto adv_region = Box(adv);
    auto adv_lo = adv_region.loVect3d();
    auto adv_hi = adv_region.hiVect3d();

    auto dx = geom.CellSizeArray();



    if (bcr[URHO].lo(0) == amrex::BCType::ext_dir && lo[0] < domlo[0]) {

        if (xl_bc_kind == EXTERNAL_HYDROSTATIC) {


            Box gbx(IntVect(AMREX_D_DECL(domlo[0]-1, lo[1], lo[2])),
                    IntVect(AMREX_D_DECL(domlo[0]-1, hi[1], hi[2])));

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {

                amrex::ignore_unused(i);

                Real density_above = adv(domlo[0],j,k,URHO);
                Real temperature_above = adv(domlo[0],j,k,UTEMP);
                Real zone_mass_frac[NumSpec];
                for (int n = 0; n < NumSpec; n++) {
                    zone_mass_frac[n] = adv(domlo[0],j,k,UFS+n) / density_above;
                }
#if NAUX_NET > 0
                Real zone_aux_vars[NumAux];
                for (int n = 0; n < NumAux; n++) {
                    zone_aux_vars[n] = adv(domlo[0],j,k,UFX+n) / density_above;
                }
#endif


                Real density_base = density_above;


                eos_rep_t eos_state;
                eos_state.rho = density_above;
                eos_state.T = temperature_above;
                for (int n = 0; n < NumSpec; n++) {
                    eos_state.xn[n] = zone_mass_frac[n];
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                    eos_state.aux[n] = zone_aux_vars[n];
                }
#endif

                eos(eos_input_rt, eos_state);

                Real pressure_above = eos_state.p;

                for (int ii = domlo[0]-1; ii >= adv_lo[0]; ii--) {




                    Real dens_zone = density_above;


                    Real temp_zone;
                    if (hse_temp_extrap == 1) {
                        temp_zone = 2*adv(ii+1,j,k,UTEMP) - adv(ii+2,j,k,UTEMP);
                    } else {
                        if (hse_bound_temp > 0.0_rt) {
                            temp_zone = hse_bound_temp;
                        } else {
                            temp_zone = temperature_above;
                        }
                    }

                    [[maybe_unused]] bool hydrostatic_converged = false;

                    Real p_want;
                    Real drho;

                    for (int iter = 0; iter < hse::MAX_ITER; iter++) {


                        p_want = pressure_above -
                            dx[0] * 0.5_rt * (dens_zone + density_above) * gravity::uniform_accel;


                        eos_state.rho = dens_zone;
                        eos_state.T = temp_zone;

                        eos(eos_input_rt, eos_state);

                        Real pressure_zone = eos_state.p;
                        Real dpdr = eos_state.dpdr;

                        Real A = p_want - pressure_zone;
                        drho = A / (dpdr + 0.5_rt * dx[0] * gravity::uniform_accel);

                        dens_zone = amrex::max(0.9_rt*dens_zone,
                                               amrex::min(dens_zone + drho, 1.1_rt*dens_zone));


                        if (std::abs(drho) < hse::TOL * dens_zone) {
                            hydrostatic_converged = true;
                            break;
                        }

                    }

#ifndef AMREX_USE_GPU
                    if (! hydrostatic_converged) {
                        std::cout << "ii, j, k, domlo[0]: " << ii << " " << j << " " << k << " " << domlo[0] << std::endl;
                        std::cout << "p_want:    " << p_want << std::endl;
                        std::cout << "dens_zone: " << dens_zone << std::endl;
                        std::cout << "temp_zone: " << temp_zone << std::endl;
                        std::cout << "drho:      " << drho << std::endl;
                        std::cout << std::endl;
                        std::cout << "column info: " << std::endl;
                        std::cout << "   dens: " << adv(ii,j,k,URHO) << std::endl;
                        std::cout << "   temp: " << adv(ii,j,k,UTEMP) << std::endl;
                        amrex::Error("ERROR in bc_ext_fill_nd: failure to converge in -X BC");
                   }
#endif


                   if (hse_rest_vel == 1) {


                       adv(ii,j,k,UMX) = 0.0_rt;
                       adv(ii,j,k,UMY) = 0.0_rt;
                       adv(ii,j,k,UMZ) = 0.0_rt;

                   } else {

                      if (hse_mirror_vel == 1) {
                          int ioff = domlo[0]-ii-1;
                          adv(ii,j,k,UMX) = -dens_zone * (adv(domlo[0]+ioff,j,k,UMX) / adv(domlo[0]+ioff,j,k,URHO));

                          adv(ii,j,k,UMY) = dens_zone * (adv(domlo[0],j,k,UMY) / density_base);
                          adv(ii,j,k,UMZ) = dens_zone * (adv(domlo[0],j,k,UMZ) / density_base);
                      } else {
                          adv(ii,j,k,UMX) = dens_zone * (adv(domlo[0],j,k,UMX) / density_base);
                          adv(ii,j,k,UMY) = dens_zone * (adv(domlo[0],j,k,UMY) / density_base);
                          adv(ii,j,k,UMZ) = dens_zone * (adv(domlo[0],j,k,UMZ) / density_base);
                      }
                   }

                   eos_state.rho = dens_zone;
                   eos_state.T = temp_zone;

                   eos(eos_input_rt, eos_state);

                   Real pressure_zone = eos_state.p;
                   Real eint = eos_state.e;


                   adv(ii,j,k,URHO) = dens_zone;
                   adv(ii,j,k,UEINT) = dens_zone * eint;
                   adv(ii,j,k,UEDEN) = dens_zone * eint +
                       0.5_rt * (adv(ii,j,k,UMX) * adv(ii,j,k,UMX) +
                                 adv(ii,j,k,UMY) * adv(ii,j,k,UMY) +
                                 adv(ii,j,k,UMZ) * adv(ii,j,k,UMZ)) / dens_zone;
                   adv(ii,j,k,UTEMP) = temp_zone;
                   for (int n = 0; n < NumSpec; n++) {
                       adv(ii,j,k,UFS+n) = dens_zone * zone_mass_frac[n];
                   }
#if NAUX_NET > 0
                   for (int n = 0; n < NumAux; n++) {
                       adv(ii,j,k,UFX+n) = dens_zone * zone_aux_vars[n];
                   }
#endif


                   density_above = dens_zone;
                   pressure_above = pressure_zone;

                }
            });

        }

    }



    if (bcr[URHO].hi(0) == amrex::BCType::ext_dir && hi[0] > domhi[0]) {

       if (xr_bc_kind == EXTERNAL_HYDROSTATIC) {

            Box gbx(IntVect(AMREX_D_DECL(domhi[0]+1, lo[1], lo[2])),
                    IntVect(AMREX_D_DECL(domhi[0]+1, hi[1], hi[2])));

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {

                amrex::ignore_unused(i);

                Real density_below = adv(domhi[0],j,k,URHO);
                Real temperature_below = adv(domhi[0],j,k,UTEMP);
                Real zone_mass_frac[NumSpec];
                for (int n = 0; n < NumSpec; n++) {
                    zone_mass_frac[n] = adv(domhi[0],j,k,UFS+n) / density_below;
                }
#if NAUX_NET > 0
                Real zone_aux_vars[NumAux];
                for (int n = 0; n < NumAux; n++) {
                    zone_aux_vars[n] = adv(domhi[0],j,k,UFX+n) / density_below;
                }
#endif


                Real density_base = density_below;


                eos_rep_t eos_state;
                eos_state.rho = density_below;
                eos_state.T = temperature_below;
                for (int n = 0; n < NumSpec; n++) {
                    eos_state.xn[n] = zone_mass_frac[n];
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                    eos_state.aux[n] = zone_aux_vars[n];
                }
#endif

                eos(eos_input_rt, eos_state);

                Real pressure_below = eos_state.p;

                for (int ii = domhi[0]+1; ii <= adv_hi[0]; ii++) {


                    Real dens_zone = density_below;


                    Real temp_zone;
                    if (hse_temp_extrap == 1) {
                        temp_zone = 2*adv(ii-1,j,k,UTEMP) - adv(ii-2,j,k,UTEMP);
                    } else {
                        if (hse_bound_temp > 0.0_rt) {
                            temp_zone = hse_bound_temp;
                        } else {
                            temp_zone = temperature_below;
                        }
                    }

                    [[maybe_unused]] bool hydrostatic_converged = false;

                    Real p_want;
                    Real drho;

                    for (int iter = 0; iter < hse::MAX_ITER; iter++) {

                        p_want = pressure_below +
                            dx[0] * 0.5_rt * (dens_zone + density_below) * gravity::uniform_accel;


                        eos_state.rho = dens_zone;
                        eos_state.T = temp_zone;

                        eos(eos_input_rt, eos_state);

                        Real pressure_zone = eos_state.p;
                        Real dpdr = eos_state.dpdr;

                        Real A = p_want - pressure_zone;
                        drho = A / (dpdr - 0.5_rt * dx[0] * gravity::uniform_accel);

                        dens_zone = amrex::max(0.9_rt*dens_zone,
                                               amrex::min(dens_zone + drho, 1.1_rt*dens_zone));


                        if (std::abs(drho) < hse::TOL * dens_zone) {
                            hydrostatic_converged = true;
                            break;
                        }

                    }

#ifndef AMREX_USE_GPU
                   if (! hydrostatic_converged) {
                       std::cout << "ii, j, k, domhi[0]: " << ii << " " << j << " " << k << " " << domhi[0] << std::endl;
                       std::cout << "p_want:    " << p_want << std::endl;
                       std::cout << "dens_zone: " << dens_zone << std::endl;
                       std::cout << "temp_zone: " << temp_zone << std::endl;
                       std::cout << "drho:      " << drho << std::endl;
                       std::cout << std::endl;
                       std::cout << "column info: " << std::endl;
                       std::cout << "   dens: " << adv(ii,j,k,URHO) << std::endl;
                       std::cout << "   temp: " << adv(ii,j,k,UTEMP) << std::endl;
                       amrex::Error("ERROR in bc_ext_fill_nd: failure to converge in +X BC");
                   }
#endif


                   if (hse_rest_vel == 1) {


                       adv(ii,j,k,UMX) = 0.0_rt;
                       adv(ii,j,k,UMY) = 0.0_rt;
                       adv(ii,j,k,UMZ) = 0.0_rt;

                   } else {

                       if (hse_mirror_vel == 1) {
                           int ioff = ii-domhi[0]-1;
                           adv(ii,j,k,UMX) = -dens_zone * (adv(domhi[0]-ioff,j,k,UMX) / adv(domhi[0]-ioff,j,k,URHO));

                           adv(ii,j,k,UMY) = dens_zone * (adv(domhi[0],j,k,UMY) / density_base);
                           adv(ii,j,k,UMZ) = dens_zone * (adv(domhi[0],j,k,UMZ) / density_base);
                       } else {
                           adv(ii,j,k,UMX) = dens_zone * (adv(domhi[0],j,k,UMX) / density_base);
                           adv(ii,j,k,UMY) = dens_zone * (adv(domhi[0],j,k,UMY) / density_base);
                           adv(ii,j,k,UMZ) = dens_zone * (adv(domhi[0],j,k,UMZ) / density_base);
                       }
                   }

                   eos_state.rho = dens_zone;
                   eos_state.T = temp_zone;

                   eos(eos_input_rt, eos_state);

                   Real pressure_zone = eos_state.p;
                   Real eint = eos_state.e;


                   adv(ii,j,k,URHO) = dens_zone;
                   adv(ii,j,k,UEINT) = dens_zone * eint;
                   adv(ii,j,k,UEDEN) = dens_zone * eint +
                       0.5_rt * (adv(ii,j,k,UMX) * adv(ii,j,k,UMX) +
                                 adv(ii,j,k,UMY) * adv(ii,j,k,UMY) +
                                 adv(ii,j,k,UMZ) * adv(ii,j,k,UMZ)) / dens_zone;
                   adv(ii,j,k,UTEMP) = temp_zone;
                   for (int n = 0; n < NumSpec; n++) {
                       adv(ii,j,k,UFS+n) = dens_zone * zone_mass_frac[n];
                   }
#if NAUX_NET > 0
                   for (int n = 0; n < NumAux; n++) {
                       adv(ii,j,k,UFX+n) = dens_zone * zone_aux_vars[n];
                   }
#endif


                   density_below = dens_zone;
                   pressure_below = pressure_zone;

                }
            });

       }

    }


#if AMREX_SPACEDIM >= 2


    if (bcr[URHO].lo(1) == amrex::BCType::ext_dir && lo[1] < domlo[1]) {

        if (yl_bc_kind == EXTERNAL_HYDROSTATIC) {

            Box gbx(IntVect(AMREX_D_DECL(lo[0], domlo[1]-1, lo[2])),
                    IntVect(AMREX_D_DECL(hi[0], domlo[1]-1, hi[2])));

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {

                amrex::ignore_unused(j);

                Real density_above = adv(i,domlo[1],k,URHO);
                Real temperature_above = adv(i,domlo[1],k,UTEMP);
                Real zone_mass_frac[NumSpec];
                for (int n = 0; n < NumSpec; n++) {
                    zone_mass_frac[n] = adv(i,domlo[1],k,UFS+n) / density_above;
                }
#if NAUX_NET > 0
                Real zone_aux_vars[NumAux];
                for (int n = 0; n < NumAux; n++) {
                    zone_aux_vars[n] = adv(i,domlo[1],k,UFX+n) / density_above;
                }
#endif


                Real density_base = density_above;


                eos_rep_t eos_state;
                eos_state.rho = density_above;
                eos_state.T = temperature_above;
                for (int n = 0; n < NumSpec; n++) {
                    eos_state.xn[n] = zone_mass_frac[n];
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                    eos_state.aux[n] = zone_aux_vars[n];
                }
#endif

                eos(eos_input_rt, eos_state);

                Real pressure_above = eos_state.p;

                for (int jj = domlo[1]-1; jj >= adv_lo[1]; jj--) {



                    Real dens_zone = density_above;


                    Real temp_zone;
                    if (hse_temp_extrap == 1) {
                        temp_zone = 2*adv(i,jj+1,k,UTEMP) - adv(i,jj+2,k,UTEMP);
                    } else {
                        if (hse_bound_temp > 0.0_rt) {
                            temp_zone = hse_bound_temp;
                        } else {
                            temp_zone = temperature_above;
                        }
                    }

                    [[maybe_unused]] bool hydrostatic_converged = false;

                    Real p_want;
                    Real drho;

                    for (int iter = 0; iter < hse::MAX_ITER; iter++) {


                        p_want = pressure_above -
                            dx[1] * 0.5_rt * (dens_zone + density_above) * gravity::uniform_accel;


                        eos_state.rho = dens_zone;
                        eos_state.T = temp_zone;

                        eos(eos_input_rt, eos_state);

                        Real pressure_zone = eos_state.p;
                        Real dpdr = eos_state.dpdr;

                        Real A = p_want - pressure_zone;
                        drho = A / (dpdr + 0.5_rt * dx[1] * gravity::uniform_accel);

                        dens_zone = amrex::max(0.9_rt*dens_zone,
                                               amrex::min(dens_zone + drho, 1.1_rt*dens_zone));


                        if (std::abs(drho) < hse::TOL * dens_zone) {
                            hydrostatic_converged = true;
                            break;
                        }

                    }

#ifndef AMREX_USE_GPU
                   if (! hydrostatic_converged) {
                       std::cout << "i, jj, k, domlo[1]: " << i << " " << jj << " " << k << " " << domlo[1] << std::endl;
                       std::cout << "p_want:    " << p_want << std::endl;
                       std::cout << "dens_zone: " << dens_zone << std::endl;
                       std::cout << "temp_zone: " << temp_zone << std::endl;
                       std::cout << "drho:      " << drho << std::endl;
                       std::cout << std::endl;
                       std::cout << "column info: " << std::endl;
                       std::cout << "   dens: " << adv(i,jj,k,URHO) << std::endl;
                       std::cout << "   temp: " << adv(i,jj,k,UTEMP) << std::endl;
                       amrex::Error("ERROR in bc_ext_fill_nd: failure to converge in -Y BC");
                   }
#endif


                   if (hse_rest_vel == 1) {


                       adv(i,jj,k,UMX) = 0.0_rt;
                       adv(i,jj,k,UMY) = 0.0_rt;
                       adv(i,jj,k,UMZ) = 0.0_rt;

                   } else {

                       if (hse_mirror_vel == 1) {
                           int joff = domlo[1]-jj-1;
                           adv(i,jj,k,UMY) = -dens_zone*(adv(i,domlo[1]+joff,k,UMY) / adv(i,domlo[1]+joff,k,URHO));

                           adv(i,jj,k,UMX) = dens_zone*(adv(i,domlo[1],k,UMX) / density_base);
                           adv(i,jj,k,UMZ) = dens_zone*(adv(i,domlo[1],k,UMZ) / density_base);
                       } else {
                           adv(i,jj,k,UMX) = dens_zone * (adv(i,domlo[1],k,UMX) / density_base);
                           adv(i,jj,k,UMY) = dens_zone * (adv(i,domlo[1],k,UMY) / density_base);
                           adv(i,jj,k,UMZ) = dens_zone * (adv(i,domlo[1],k,UMZ) / density_base);
                       }
                   }

                   eos_state.rho = dens_zone;
                   eos_state.T = temp_zone;

                   eos(eos_input_rt, eos_state);

                   Real pressure_zone = eos_state.p;
                   Real eint = eos_state.e;


                   adv(i,jj,k,URHO) = dens_zone;
                   adv(i,jj,k,UEINT) = dens_zone * eint;
                   adv(i,jj,k,UEDEN) = dens_zone * eint +
                       0.5_rt * (adv(i,jj,k,UMX) * adv(i,jj,k,UMX) +
                                 adv(i,jj,k,UMY) * adv(i,jj,k,UMY) +
                                 adv(i,jj,k,UMZ) * adv(i,jj,k,UMZ)) / dens_zone;
                   adv(i,jj,k,UTEMP) = temp_zone;
                   for (int n = 0; n < NumSpec; n++) {
                       adv(i,jj,k,UFS+n) = dens_zone * zone_mass_frac[n];
                   }
#if NAUX_NET > 0
                   for (int n = 0; n < NumAux; n++) {
                       adv(i,jj,k,UFX+n) = dens_zone * zone_aux_vars[n];
                   }
#endif


                   density_above = dens_zone;
                   pressure_above = pressure_zone;

                }
            });

       }


    }



    if (bcr[URHO].hi(1) == amrex::BCType::ext_dir && hi[1] > domhi[1]) {

        if (yr_bc_kind == EXTERNAL_HYDROSTATIC) {

            Box gbx(IntVect(AMREX_D_DECL(lo[0], domhi[1]+1, lo[2])),
                    IntVect(AMREX_D_DECL(hi[0], domhi[1]+1, hi[2])));

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {

                amrex::ignore_unused(j);

                Real density_below = adv(i,domhi[1],k,URHO);
                Real temperature_below = adv(i,domhi[1],k,UTEMP);
                Real zone_mass_frac[NumSpec];
                for (int n = 0; n < NumSpec; n++) {
                    zone_mass_frac[n] = adv(i,domhi[1],k,UFS+n) / density_below;
                }
#if NAUX_NET > 0
                Real zone_aux_vars[NumAux];
                for (int n = 0; n < NumAux; n++) {
                    zone_aux_vars[n] = adv(i,domhi[1],k,UFX+n) / density_below;
                }
#endif


                Real density_base = density_below;


                eos_rep_t eos_state;
                eos_state.rho = density_below;
                eos_state.T = temperature_below;
                for (int n = 0; n < NumSpec; n++) {
                    eos_state.xn[n] = zone_mass_frac[n];
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                    eos_state.aux[n] = zone_aux_vars[n];
                }
#endif

                eos(eos_input_rt, eos_state);

                Real pressure_below = eos_state.p;

                for (int jj = domhi[1]+1; jj <= adv_hi[1]; jj++) {


                    Real dens_zone = density_below;


                    Real temp_zone;
                    if (hse_temp_extrap == 1) {
                        temp_zone = 2*adv(i,jj-1,k,UTEMP) - adv(i,jj-2,k,UTEMP);
                    } else {
                        if (hse_bound_temp > 0.0_rt) {
                            temp_zone = hse_bound_temp;
                        } else {
                            temp_zone = temperature_below;
                        }
                    }

                    [[maybe_unused]] bool hydrostatic_converged = false;

                    Real p_want;
                    Real drho;

                    for (int iter = 0; iter < hse::MAX_ITER; iter++) {

                        p_want = pressure_below +
                            dx[1] * 0.5_rt * (dens_zone + density_below) * gravity::uniform_accel;


                        eos_state.rho = dens_zone;
                        eos_state.T = temp_zone;

                        eos(eos_input_rt, eos_state);

                        Real pressure_zone = eos_state.p;
                        Real dpdr = eos_state.dpdr;

                        Real A = p_want - pressure_zone;
                        drho = A / (dpdr - 0.5_rt * dx[1] * gravity::uniform_accel);

                        dens_zone = amrex::max(0.9_rt*dens_zone,
                                               amrex::min(dens_zone + drho, 1.1_rt*dens_zone));


                        if (std::abs(drho) < hse::TOL * dens_zone) {
                            hydrostatic_converged = true;
                            break;
                        }

                    }

#ifndef AMREX_USE_GPU
                   if (! hydrostatic_converged) {
                       std::cout << "i, jj, k, domhi[1]: " << i << " " << jj << " " << k << " " << domhi[1] << std::endl;
                       std::cout << "p_want:    " << p_want << std::endl;
                       std::cout << "dens_zone: " << dens_zone << std::endl;
                       std::cout << "temp_zone: " << temp_zone << std::endl;
                       std::cout << "drho:      " << drho << std::endl;
                       std::cout << std::endl;
                       std::cout << "column info: " << std::endl;
                       std::cout << "   dens: " << adv(i,jj,k,URHO) << std::endl;
                       std::cout << "   temp: " << adv(i,jj,k,UTEMP) << std::endl;
                       amrex::Error("ERROR in bc_ext_fill_nd: failure to converge in +Y BC");
                   }
#endif


                   if (hse_rest_vel == 1) {


                       adv(i,jj,k,UMX) = 0.0_rt;
                       adv(i,jj,k,UMY) = 0.0_rt;
                       adv(i,jj,k,UMZ) = 0.0_rt;

                   } else {

                       if (hse_mirror_vel == 1) {
                           int joff = jj-domhi[1]-1;
                           adv(i,jj,k,UMY) = -dens_zone * (adv(i,domhi[1]-joff,k,UMY) / adv(i,domhi[1]-joff,k,URHO));

                           adv(i,jj,k,UMX) = dens_zone * (adv(i,domhi[1],k,UMX) / density_base);
                           adv(i,jj,k,UMZ) = dens_zone * (adv(i,domhi[1],k,UMZ) / density_base);
                       } else {
                           adv(i,jj,k,UMX) = dens_zone * (adv(i,domhi[1],k,UMX) / density_base);
                           adv(i,jj,k,UMY) = dens_zone * (adv(i,domhi[1],k,UMY) / density_base);
                           adv(i,jj,k,UMZ) = dens_zone * (adv(i,domhi[1],k,UMZ) / density_base);
                       }
                   }

                   eos_state.rho = dens_zone;
                   eos_state.T = temp_zone;

                   eos(eos_input_rt, eos_state);

                   Real pressure_zone = eos_state.p;
                   Real eint = eos_state.e;


                   adv(i,jj,k,URHO) = dens_zone;
                   adv(i,jj,k,UEINT) = dens_zone * eint;
                   adv(i,jj,k,UEDEN) = dens_zone * eint +
                       0.5_rt * (adv(i,jj,k,UMX) * adv(i,jj,k,UMX) +
                                 adv(i,jj,k,UMY) * adv(i,jj,k,UMY) +
                                 adv(i,jj,k,UMZ) * adv(i,jj,k,UMZ)) / dens_zone;
                   adv(i,jj,k,UTEMP) = temp_zone;
                   for (int n = 0; n < NumSpec; n++) {
                       adv(i,jj,k,UFS+n) = dens_zone * zone_mass_frac[n];
                   }
#if NAUX_NET > 0
                   for (int n = 0; n < NumAux; n++) {
                       adv(i,jj,k,UFX+n) = dens_zone * zone_aux_vars[n];
                   }
#endif


                   density_below = dens_zone;
                   pressure_below = pressure_zone;

                }
            });
        }

    }
#endif

#if AMREX_SPACEDIM == 3


    if (bcr[URHO].lo(2) == amrex::BCType::ext_dir && lo[2] < domlo[2]) {

        if (zl_bc_kind == EXTERNAL_HYDROSTATIC) {

            Box gbx(IntVect(AMREX_D_DECL(lo[0], lo[1], domlo[2]-1)),
                    IntVect(AMREX_D_DECL(hi[0], hi[1], domlo[2]-1)));

            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {

                amrex::ignore_unused(k);

                Real density_above = adv(i,j,domlo[2],URHO);
                Real temperature_above = adv(i,j,domlo[2],UTEMP);
                Real zone_mass_frac[NumSpec];
                for (int n = 0; n < NumSpec; n++) {
                    zone_mass_frac[n] = adv(i,j,domlo[2],UFS+n) / density_above;
                }
#if NAUX_NET > 0
                Real zone_aux_vars[NumAux];
                for (int n = 0; n < NumAux; n++) {
                    zone_aux_vars[n] = adv(i,j,domlo[2],UFX+n) / density_above;
                }
#endif


                Real density_base = density_above;


                eos_rep_t eos_state;
                eos_state.rho = density_above;
                eos_state.T = temperature_above;
                for (int n = 0; n < NumSpec; n++) {
                    eos_state.xn[n] = zone_mass_frac[n];
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                    eos_state.aux[n] = zone_aux_vars[n];
                }
#endif

                eos(eos_input_rt, eos_state);

                Real pressure_above = eos_state.p;

                for (int kk = domlo[2]-1; kk >= adv_lo[2]; kk--) {



                    Real dens_zone = density_above;


                    Real temp_zone;
                    if (hse_temp_extrap == 1) {
                        temp_zone = 2*adv(i,j,kk+1,UTEMP) - adv(i,j,kk+2,UTEMP);
                    } else {
                        if (hse_bound_temp > 0.0_rt) {
                            temp_zone = hse_bound_temp;
                        } else {
                            temp_zone = temperature_above;
                        }
                    }

                    [[maybe_unused]] bool hydrostatic_converged = false;

                    Real p_want;
                    Real drho;

                    for (int iter = 0; iter < hse::MAX_ITER; iter++) {


                        p_want = pressure_above -
                            dx[2] * 0.5_rt * (dens_zone + density_above) * gravity::uniform_accel;


                        eos_state.rho = dens_zone;
                        eos_state.T = temp_zone;

                        eos(eos_input_rt, eos_state);

                        Real pressure_zone = eos_state.p;
                        Real dpdr = eos_state.dpdr;

                        Real A = p_want - pressure_zone;
                        drho = A / (dpdr + 0.5_rt * dx[2] * gravity::uniform_accel);

                        dens_zone = amrex::max(0.9_rt*dens_zone,
                                               amrex::min(dens_zone + drho, 1.1_rt*dens_zone));


                        if (std::abs(drho) < hse::TOL * dens_zone) {
                            hydrostatic_converged = true;
                            break;
                        }

                    }

#ifndef AMREX_USE_GPU
                   if (! hydrostatic_converged) {
                       std::cout << "i, j, kk, domlo[2]: " << i << " " << j << " " << kk << " " << domlo[2] << std::endl;
                       std::cout << "p_want:    " << p_want << std::endl;
                       std::cout << "dens_zone: " << dens_zone << std::endl;
                       std::cout << "temp_zone: " << temp_zone << std::endl;
                       std::cout << "drho:      " << drho << std::endl;
                       std::cout << std::endl;
                       std::cout << "column info: " << std::endl;
                       std::cout << "   dens: " << adv(i,j,kk,URHO) << std::endl;
                       std::cout << "   temp: " << adv(i,j,kk,UTEMP) << std::endl;
                       amrex::Error("ERROR in bc_ext_fill_nd: failure to converge in -Z BC");
                   }
#endif


                   if (hse_rest_vel == 1) {


                       adv(i,j,kk,UMX) = 0.0_rt;
                       adv(i,j,kk,UMY) = 0.0_rt;
                       adv(i,j,kk,UMZ) = 0.0_rt;

                   } else {

                       if (hse_mirror_vel == 1) {
                           int koff = domlo[2]-kk-1;
                           adv(i,j,kk,UMZ) = -dens_zone * (adv(i,j,domlo[2]+koff,UMZ) / adv(i,j,domlo[2]+koff,URHO));

                           adv(i,j,kk,UMX) = dens_zone * (adv(i,j,domlo[2],UMX) / density_base);
                           adv(i,j,kk,UMY) = dens_zone * (adv(i,j,domlo[2],UMY) / density_base);
                       } else {
                           adv(i,j,kk,UMX) = dens_zone * (adv(i,j,domlo[2],UMX) / density_base);
                           adv(i,j,kk,UMY) = dens_zone * (adv(i,j,domlo[2],UMY) / density_base);
                           adv(i,j,kk,UMZ) = dens_zone * (adv(i,j,domlo[2],UMZ) / density_base);
                       }
                   }

                   eos_state.rho = dens_zone;
                   eos_state.T = temp_zone;

                   eos(eos_input_rt, eos_state);

                   Real pressure_zone = eos_state.p;
                   Real eint = eos_state.e;


                   adv(i,j,kk,URHO) = dens_zone;
                   adv(i,j,kk,UEINT) = dens_zone * eint;
                   adv(i,j,kk,UEDEN) = dens_zone * eint +
                       0.5_rt * (adv(i,j,kk,UMX) * adv(i,j,kk,UMX) +
                                 adv(i,j,kk,UMY) * adv(i,j,kk,UMY) +
                                 adv(i,j,kk,UMZ) * adv(i,j,kk,UMZ)) / dens_zone;
                   adv(i,j,kk,UTEMP) = temp_zone;
                   for (int n = 0; n < NumSpec; n++) {
                       adv(i,j,kk,UFS+n) = dens_zone * zone_mass_frac[n];
                   }
#if NAUX_NET > 0
                   for (int n = 0; n < NumAux; n++) {
                       adv(i,j,kk,UFX+n) = dens_zone * zone_aux_vars[n];
                   }
#endif


                   density_above = dens_zone;
                   pressure_above = pressure_zone;

                }
            });
        }

    }


    if (bcr[URHO].hi(2) == amrex::BCType::ext_dir && hi[2] > domhi[2]) {

        if (zr_bc_kind == EXTERNAL_HYDROSTATIC) {
#ifndef AMREX_USE_GPU
            amrex::Error("ERROR: HSE boundaries not implemented for +Z");
#endif
        }
    }
#endif

}
