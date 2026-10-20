#include <HydroSim.H>
#include <HydroSim_helpers.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#include <cmath>

#include <parabolic_profile.H>
#include <limiters.H>


#include "Radiation.H"

#include "group_advection.H"

#include "flux_limiter.H"

using namespace amrex;

static AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real cell_centered_limiter (int i, int j, int k, int g,
                            AMREX_D_DECL(Array4<Real const> const& qx,
                                         Array4<Real const> const& qy,
                                         Array4<Real const> const& qz))
{
  Real lamc;
#if AMREX_SPACEDIM == 1
  lamc = 0.5_rt * (qx(i,j,k,GDFLUXLIM+g) + qx(i+1,j,k,GDFLUXLIM+g));
#endif
#if AMREX_SPACEDIM == 2
  lamc = 0.25_rt * (qx(i,j,k,GDFLUXLIM+g) + qx(i+1,j,k,GDFLUXLIM+g) +
                    qy(i,j,k,GDFLUXLIM+g) + qy(i,j+1,k,GDFLUXLIM+g));
#endif
#if AMREX_SPACEDIM == 3
  lamc = (qx(i,j,k,GDFLUXLIM+g) + qx(i+1,j,k,GDFLUXLIM+g) +
          qy(i,j,k,GDFLUXLIM+g) + qy(i,j+1,k,GDFLUXLIM+g) +
          qz(i,j,k,GDFLUXLIM+g) + qz(i,j,k+1,GDFLUXLIM+g)) / 6.0_rt;
#endif
  return lamc;
}

void
HydroSim::rad_cons_update(const Box& bx,
                       Array4<Real> const& cons_new,
                       Array4<Real const> const& RadEnergyIn,
                       Array4<Real> const& RadEnergyOut,
                       Array4<Real const> const& radiative_flux1,
                       Array4<Real const> const& qx,
                       Array4<Real const> const& area1,
#if AMREX_SPACEDIM >= 2
                       Array4<Real const> const& radiative_flux2,
                       Array4<Real const> const& qy,
                       Array4<Real const> const& area2,
#endif
#if AMREX_SPACEDIM == 3
                       Array4<Real const> const& radiative_flux3,
                       Array4<Real const> const& qz,
                       Array4<Real const> const& face_area3,
#endif
                       int& rad_step_count,
                       Array4<Real const> const& vol,
                       const Real dt)
{


  auto dx = geom.CellSizeArray();
  const int coord_type = geom.Coord();

  GpuArray<Real, NGROUPS> RadEnergyScale = {0.0};
  GpuArray<Real, NGROUPS> log_group_width = {0.0};

  if (NGROUPS > 1) {
    for (int g = 0; g < NGROUPS; ++g) {
      log_group_width[g] = radiation->log_group_widths[g];
    }
    if (radiation::frequency_advection_mode == 1) {
      for (int g = 0; g < NGROUPS; g++) {
        RadEnergyScale[g] = log_group_width[g];
      }
    } else {
      for (int g = 0; g < NGROUPS; g++) {
        RadEnergyScale[g] = radiation->freq_center[g] * log_group_width[g];
      }
    }
  }



  amrex::ParallelFor(bx, NGROUPS,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int g) noexcept
  {

    RadEnergyOut(i,j,k,g) = RadEnergyIn(i,j,k,g) + dt *
      (  radiative_flux1(i,j,k,g) * area1(i,j,k) - radiative_flux1(i+1,j,k,g) * area1(i+1,j,k)
#if AMREX_SPACEDIM >= 2
       + radiative_flux2(i,j,k,g) * area2(i,j,k) - radiative_flux2(i,j+1,k,g) * area2(i,j+1,k)
#endif
#if AMREX_SPACEDIM == 3
       + radiative_flux3(i,j,k,g) * face_area3(i,j,k) - radiative_flux3(i,j,k+1,g) * face_area3(i,j,k+1)
#endif
         ) / vol(i,j,k);
  });


  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real radiation_pressure_grad_x = 0.0;
    Real radiation_pressure_grad_y = 0.0;
    Real radiation_pressure_grad_z = 0.0;

    Real lamc = 0.0;
    for (int g = 0; g < NGROUPS; g++) {

      lamc = cell_centered_limiter(i, j, k, g, AMREX_D_DECL(qx, qy, qz));

      radiation_pressure_grad_x = radiation_pressure_grad_x + lamc * (qx(i+1,j,k,GDRADENG+g) - qx(i,j,k,GDRADENG+g))/dx[0];
#if AMREX_SPACEDIM >= 2
      radiation_pressure_grad_y = radiation_pressure_grad_y + lamc * (qy(i,j+1,k,GDRADENG+g) - qy(i,j,k,GDRADENG+g))/dx[1];
#endif
#if AMREX_SPACEDIM == 3
      radiation_pressure_grad_z = radiation_pressure_grad_z + lamc * (qz(i,j,k+1,GDRADENG+g) - qz(i,j,k,GDRADENG+g))/dx[2];
#endif
    }




    Real cell_density_new = cons_new(i,j,k,URHO);

    Real x_momentum_new1 = cons_new(i,j,k,UMX);
    Real y_momentum_new1 = cons_new(i,j,k,UMY);
    Real z_momentum_new1 = cons_new(i,j,k,UMZ);

    Real ek1 = (x_momentum_new1 * x_momentum_new1 +
                y_momentum_new1 * y_momentum_new1 +
                z_momentum_new1 * z_momentum_new1) / (2.0_rt * cell_density_new);


    cons_new(i,j,k,UMX) -= dt * radiation_pressure_grad_x;
    cons_new(i,j,k,UMY) -= dt * radiation_pressure_grad_y;
    cons_new(i,j,k,UMZ) -= dt * radiation_pressure_grad_z;

    Real x_momentum_new2 = cons_new(i,j,k,UMX);
    Real y_momentum_new2 = cons_new(i,j,k,UMY);
    Real z_momentum_new2 = cons_new(i,j,k,UMZ);

    Real ek2 = (x_momentum_new2 * x_momentum_new2 +
                y_momentum_new2 * y_momentum_new2 +
                z_momentum_new2 * z_momentum_new2) / (2.0_rt * cell_density_new);

    Real dek = ek2 - ek1;


    cons_new(i,j,k,UEDEN) = cons_new(i,j,k,UEDEN) + dek;

    if (!radiation::moving_frame) {
      RadEnergyOut(i,j,k,0) = RadEnergyOut(i,j,k,0) - dek;
    }

  });



  if (radiation::moving_frame) {

    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    reduce_op.eval(bx, reduce_data,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
    {

      Real ux = 0.5_rt * (qx(i,j,k,GDU) + qx(i+1,j,k,GDU));
#if AMREX_SPACEDIM >= 2
      Real uy = 0.5_rt * (qy(i,j,k,GDV) + qy(i,j+1,k,GDV));
#endif
#if AMREX_SPACEDIM == 3
      Real uz = 0.5_rt * (qz(i,j,k,GDW) + qz(i,j,k+1,GDW));
#endif

      Real dudx[3] = {0, 0, 0};
      Real dudy[3] = {0, 0, 0};
      Real dudz[3] = {0, 0, 0};

      dudx[0] = (qx(i+1,j,k,GDU) - qx(i,j,k,GDU))/dx[0];
      dudx[1] = (qx(i+1,j,k,GDV) - qx(i,j,k,GDV))/dx[0];
      dudx[2] = (qx(i+1,j,k,GDW) - qx(i,j,k,GDW))/dx[0];

#if AMREX_SPACEDIM >= 2
      dudy[0] = (qy(i,j+1,k,GDU) - qy(i,j,k,GDU))/dx[1];
      dudy[1] = (qy(i,j+1,k,GDV) - qy(i,j,k,GDV))/dx[1];
      dudy[2] = (qy(i,j+1,k,GDW) - qy(i,j,k,GDW))/dx[1];
#endif

#if AMREX_SPACEDIM == 3
      dudz[0] = (qz(i,j,k+1,GDU) - qz(i,j,k,GDU))/dx[2];
      dudz[1] = (qz(i,j,k+1,GDV) - qz(i,j,k,GDV))/dx[2];
      dudz[2] = (qz(i,j,k+1,GDW) - qz(i,j,k,GDW))/dx[2];
#endif

      Real divu = dudx[0] + dudy[1] + dudz[2];

      Real af[NGROUPS];

      for (int g = 0; g < NGROUPS; g++) {

        Real energy_grad[3] = {0., 0., 0.};

        energy_grad[0] = (qx(i+1,j,k,GDRADENG+g) - qx(i,j,k,GDRADENG+g))/dx[0];
#if AMREX_SPACEDIM >= 2
        energy_grad[1] = (qy(i,j+1,k,GDRADENG+g) - qy(i,j,k,GDRADENG+g))/dx[1];
#endif
#if AMREX_SPACEDIM == 3
        energy_grad[2] = (qz(i,j,k+1,GDRADENG+g) - qz(i,j,k,GDRADENG+g))/dx[2];
#endif

        Real NormalGradVel[3];

        NormalGradVel[0] = energy_grad[0] * dudx[0] + energy_grad[1] * dudx[1] + energy_grad[2] * dudx[2];
        NormalGradVel[1] = energy_grad[0] * dudy[0] + energy_grad[1] * dudy[1] + energy_grad[2] * dudy[2];
        NormalGradVel[2] = energy_grad[0] * dudz[0] + energy_grad[1] * dudz[1] + energy_grad[2] * dudz[2];

        Real normalDoubleDot = (energy_grad[0] * NormalGradVel[0] + energy_grad[1] * NormalGradVel[1] + energy_grad[2] * NormalGradVel[2]) /
          (energy_grad[0] * energy_grad[0] + energy_grad[1] * energy_grad[1] + energy_grad[2] * energy_grad[2] + 1.e-50_rt);

        Real lamc;
        lamc = cell_centered_limiter(i, j, k, g, AMREX_D_DECL(qx, qy, qz));

        Real RadClosure = Eddington_factor(lamc);
        Real f1 = (1.0_rt - RadClosure) * 0.5_rt;
        Real f2 = (3.0_rt * RadClosure - 1.0_rt) * 0.5_rt;
        af[g] = -(f1*divu + f2*normalDoubleDot);

        if (radiation::frequency_advection_mode == 1) {
          Real RadClosureXp = Eddington_factor(qx(i+1,j,k,GDFLUXLIM+g));
          Real RadClosureXm = Eddington_factor(qx(i,j,k,GDFLUXLIM+g));
#if AMREX_SPACEDIM >= 2
          Real RadClosureYp = Eddington_factor(qy(i,j+1,k,GDFLUXLIM+g));
          Real RadClosureYm = Eddington_factor(qy(i,j,k,GDFLUXLIM+g));
#endif
#if AMREX_SPACEDIM == 3
          Real RadClosureZp = Eddington_factor(qz(i,j,k+1,GDFLUXLIM+g));
          Real RadClosureZm = Eddington_factor(qz(i,j,k,GDFLUXLIM+g));
#endif

          Real closure_weight_xp = 0.5_rt * (1.0_rt - RadClosureXp);
          Real closure_weight_xm = 0.5_rt * (1.0_rt - RadClosureXm);
#if AMREX_SPACEDIM >= 2
          Real closure_weight_yp = 0.5_rt * (1.0_rt - RadClosureYp);
          Real closure_weight_ym = 0.5_rt * (1.0_rt - RadClosureYm);
#endif
#if AMREX_SPACEDIM == 3
          Real closure_weight_zp = 0.5_rt * (1.0_rt - RadClosureZp);
          Real closure_weight_zm = 0.5_rt * (1.0_rt - RadClosureZm);
#endif

          Real RadEnergyGrad[3];

          RadEnergyGrad[0] = (closure_weight_xp*qx(i+1,j,k,GDRADENG+g) - closure_weight_xm*qx(i,j,k,GDRADENG+g)) / dx[0];
#if AMREX_SPACEDIM >= 2
          RadEnergyGrad[1] = (closure_weight_yp*qy(i,j+1,k,GDRADENG+g) - closure_weight_ym*qy(i,j,k,GDRADENG+g)) / dx[1];
#endif
#if AMREX_SPACEDIM == 3
          RadEnergyGrad[2] = (closure_weight_zp*qz(i,j,k+1,GDRADENG+g) - closure_weight_zm*qz(i,j,k,GDRADENG+g)) / dx[2];
#endif


#if AMREX_SPACEDIM == 1
          Real RadEnergyCenter = 0.5_rt * (qx(i,j,k,GDRADENG+g) + qx(i+1,j,k,GDRADENG+g));
          RadEnergyOut(i,j,k,g) = RadEnergyOut(i,j,k,g) + dt * ux * RadEnergyGrad[0]
            - dt * f2 * RadEnergyCenter * normalDoubleDot;
#endif
#if AMREX_SPACEDIM == 2
          Real RadEnergyCenter = 0.25_rt * (qx(i,j,k,GDRADENG+g) + qx(i+1,j,k,GDRADENG+g) +
                                 qy(i,j,k,GDRADENG+g) + qy(i,j+1,k,GDRADENG+g));
          RadEnergyOut(i,j,k,g) = RadEnergyOut(i,j,k,g) + dt * (ux * RadEnergyGrad[0] + uy * RadEnergyGrad[1])
            - dt * f2 * RadEnergyCenter * normalDoubleDot;
#endif
#if AMREX_SPACEDIM == 3
          Real RadEnergyCenter = (qx(i,j,k,GDRADENG+g) + qx(i+1,j,k,GDRADENG+g) +
                       qy(i,j,k,GDRADENG+g) + qy(i,j+1,k,GDRADENG+g) +
                       qz(i,j,k,GDRADENG+g) + qz(i,j,k+1,GDRADENG+g) ) / 6.0_rt;
          RadEnergyOut(i,j,k,g) = RadEnergyOut(i,j,k,g) + dt * (ux * RadEnergyGrad[0] + uy * RadEnergyGrad[1] + uz * RadEnergyGrad[2])
            - dt * f2 * RadEnergyCenter * normalDoubleDot;
#endif

        }
      }

      int rad_step_count_tmp = 1;

      if (NGROUPS > 1) {
        Real ustar[NGROUPS];
        for (int g = 0; g < NGROUPS; g++) {
          ustar[g] = RadEnergyOut(i,j,k,g) / RadEnergyScale[g];
        }

        integrate_group_energies(NGROUPS, ustar, af, log_group_width.begin(), dt, rad_step_count_tmp);

        for (int g = 0; g < NGROUPS; g++) {
          RadEnergyOut(i,j,k,g) = ustar[g] * RadEnergyScale[g];
        }
      }

      return rad_step_count_tmp;

    });

    ReduceTuple hv = reduce_data.value();
    rad_step_count = amrex::get<0>(hv);

  }
}

#if (AMREX_SPACEDIM < 3)
static AMREX_GPU_HOST_DEVICE AMREX_INLINE
void add_area_source_terms (Array4<Real> const& q, int i, int j, int k,
                            Real source_density, Real source_pressure, Real source,
                            const Real* rad_source_terms, const Real* lam, Real ldensity_floor)
{
  q(i,j,k,QRHO) = q(i,j,k,QRHO) + source_density;
  q(i,j,k,QRHO) = amrex::max(q(i,j,k,QRHO), ldensity_floor);
  q(i,j,k,QPRES) = q(i,j,k,QPRES) + source_pressure;
  q(i,j,k,QEINTDENS) = q(i,j,k,QEINTDENS) + source;
  for (int g = 0; g < NGROUPS; g++) {
    q(i,j,k,QRADENG+g) += rad_source_terms[g];
  }
  q(i,j,k,QPTOTAL) = q(i,j,k,QPTOTAL) + source_pressure;
  q(i,j,k,QETOTDENS) = q(i,j,k,QEINTDENS);
  for (int g = 0; g < NGROUPS; g++) {
    q(i,j,k,QPTOTAL) += lam[g]*rad_source_terms[g];
    q(i,j,k,QETOTDENS) += q(i,j,k,QRADENG+g);
  }
}
#endif

void
HydroSim::trace_parabolic_radiative(const Box& bx,
                      const int idir,
                      Array4<Real const> const& field_arr,
                      Array4<Real const> const& inv_density_arr,
                      Array4<Real const> const& prim_arr,
                      Array4<Real const> const& qthermo_arr,
                      Array4<Real const> const& prim_source,
                      Array4<Real> const& qm,
                      Array4<Real> const& qp,
#if (AMREX_SPACEDIM < 3)
                      Array4<Real const> const& dloga,
#endif
                      const Box& vbx,
                      const Real dt) {



  const auto dx = geom.CellSizeArray();

  Real hdt = 0.5_rt * dt;
  Real dtdx = dt / dx[idir];

  auto lo = bx.loVect3d();
  auto hi = bx.hiVect3d();

  auto vlo = vbx.loVect3d();
  auto vhi = vbx.hiVect3d();

#ifndef AMREX_USE_GPU


  GpuArray<int, NQSOURCE> trace_source_flags;


  for (int n = 0; n < NQSOURCE; n++) {
    trace_source_flags[n] = 0;

    for (int k = lo[2]-2*dg2; k <= hi[2]+2*dg2; k++) {
      for (int j = lo[1]-2*dg1; j <= hi[1]+2*dg1; j++) {
        for (int i = lo[0]-2; i <= hi[0]+2; i++) {
          if (std::abs(prim_source(i,j,k,n)) > 0.0_rt) {
            trace_source_flags[n] = 1;
            break;
          }
        }
        if (trace_source_flags[n] == 1) break;
      }
      if (trace_source_flags[n] == 1) break;
    }
  }
#endif


  const int velocity_order[3][3] = {{QU, QV, QW}, {QV, QW, QU}, {QW, QU, QV}};

  const int QUN = velocity_order[idir][0];
  const int QUT = velocity_order[idir][1];
  const int QTRANSV2 = velocity_order[idir][2];

  Real ldensity_floor = small_dens;
  Real local_pressure_floor = pressure_cutoff;

  AMREX_PARALLEL_FOR_3D(bx, i, j, k,
  {

   Real lam0[NGROUPS];
   Real lamp[NGROUPS];
   Real limiter_minus[NGROUPS];

    for (int g = 0; g < NGROUPS; g++) {
      lam0[g] = qthermo_arr(i,j,k,QFLUXLIM+g);
      lamp[g] = qthermo_arr(i,j,k,QFLUXLIM+g);
      limiter_minus[g] = qthermo_arr(i,j,k,QFLUXLIM+g);
    }

    Real rho = prim_arr(i,j,k,QRHO);

    Real gas_sound_speed_sq = qthermo_arr(i,j,k,QCG)*qthermo_arr(i,j,k,QCG);
    Real cc = qthermo_arr(i,j,k,QC);
    Real csq = cc*cc;

    Real un = prim_arr(i,j,k,QUN);

    Real p = prim_arr(i,j,k,QPRES);
    Real energy_density_gas = prim_arr(i,j,k,QEINTDENS);
    Real h_g = ( (p+energy_density_gas)/rho)/csq;

    Real gas_adiabatic_index = qthermo_arr(i,j,k,QGAMMA1G);

    Real total_pressure = prim_arr(i,j,k,QPTOTAL);

    Real er[NGROUPS];
    Real hr[NGROUPS];
    for (int g = 0; g < NGROUPS; g++) {
      er[g] = prim_arr(i,j,k,QRADENG+g);
      hr[g] = (lam0[g] + 1.0_rt)*er[g]/rho;
    }

    Real s[5];

    Real flat = 1.0;

    if (hydrosim::first_order_scheme) {
        flat = 0.0;
    }
    else if (hydrosim::use_shock_flatten) {
        flat = hydro::flatten(i, j, k, prim_arr, QPRES);

#ifdef RADIATION
        flat *= hydro::flatten(i, j, k, prim_arr, QPTOTAL);

        if (radiation::radiation_flatten_cutoff > 0.0) {
            if ( prim_arr(i-1,j,k,QU) + prim_arr(i,j-1,k,QV) + prim_arr(i,j,k-1,QW) >
                 prim_arr(i+1,j,k,QU) + prim_arr(i,j+1,k,QV) + prim_arr(i,j,k+1,QW) ) {

                if (prim_arr(i,j,k,QPRES) < radiation::radiation_flatten_cutoff * prim_arr(i,j,k,QPTOTAL)) {
                    flat = 0.0;
                }
            }
        }
#endif
    }

    Real sm;
    Real sp;

    Real Ip[NQ][3];
    Real Im[NQ][3];



    for (int n = 0; n < NQPRIMBASE; n++) {
      if (n == QTEMP) continue;

      gather_stencil(prim_arr, idir, i, j, k, n, s);
      parabolic_reconstruct(s, flat, sm, sp);
      integrate_parabolic_profile(sm, sp, s[i0], un, cc, dtdx, Ip[n], Im[n]);

    }


    for (int passive_index = 0; passive_index < npassive; ++passive_index) {

      const int n = passive_prim_index(passive_index);
      const int nc = passive_cons_index(passive_index);

      fill_passive_stencil(field_arr, inv_density_arr, idir, i, j, k, nc, s);
      parabolic_reconstruct(s, flat, sm, sp);
      integrate_parabolic_profile(sm, sp, s[i0], un, cc, dtdx, Ip[n], Im[n]);

    }


    Real TraceP_src[NQSOURCE][3];
    Real TraceM_src[NQSOURCE][3];

    for (int n = 0; n < NQSOURCE; n++) {

#ifndef AMREX_USE_GPU
      int trace_enabled = trace_source_flags[n];
#else
      int trace_enabled = 0;
      if (idir == 0) {
        for (int b = i-2; b <= i+2; b++) {
          if (std::abs(prim_source(b,j,k,n)) > 0.0_rt) {
            trace_enabled = 1;
            break;
          }
        }
      } else if (idir == 1) {
        for (int b = j-2; b <= j+2; b++) {
          if (std::abs(prim_source(i,b,k,n)) > 0.0_rt) {
            trace_enabled = 1;
            break;
          }
        }
      } else {
        for (int b = k-2; b <= k+2; b++) {
          if (std::abs(prim_source(i,j,b,n)) > 0.0_rt) {
            trace_enabled = 1;
            break;
          }
        }
      }
#endif

      if (trace_enabled) {

          gather_stencil(prim_source, idir, i, j, k, n, s);
          parabolic_reconstruct(s, flat, sm, sp);
          integrate_parabolic_profile(sm, sp, s[i0], un, cc, dtdx, TraceP_src[n], TraceM_src[n]);

      } else {
        TraceP_src[n][0] = 0.0_rt;
        TraceP_src[n][1] = 0.0_rt;
        TraceP_src[n][2] = 0.0_rt;

        TraceM_src[n][0] = 0.0_rt;
        TraceM_src[n][1] = 0.0_rt;
        TraceM_src[n][2] = 0.0_rt;
      }

    }




    for (int passive_index = 0; passive_index < npassive; passive_index++) {

      int n = passive_prim_index(passive_index);

      if ((idir == 0 && i >= vlo[0]) ||
          (idir == 1 && j >= vlo[1]) ||
          (idir == 2 && k >= vlo[2])) {


        qp(i,j,k,n) = Im[n][1];
      }

      if (idir == 0 && i <= vhi[0]) {
        qm(i+1,j,k,n) = Ip[n][1];
      } else if (idir == 1 && j <= vhi[1]) {
        qm(i,j+1,k,n) = Ip[n][1];
      } else if (idir == 2 && k <= vhi[2]) {
        qm(i,j,k+1,n) = Ip[n][1];
      }
    }


    if ((idir == 0 && i >= vlo[0]) ||
        (idir == 1 && j >= vlo[1]) ||
        (idir == 2 && k >= vlo[2])) {

      Real ref_density = Im[QRHO][0];
      Real ref_normal_velocity = Im[QUN][0];

      Real ref_pressure = Im[QPRES][0];
      Real ref_energy_density_gas = Im[QEINTDENS][0];

      Real ref_total_pressure = Im[QPTOTAL][0];
      Real ref_rad_energy[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        ref_rad_energy[g] = Im[QRADENG+g][0];
      }

      ref_density = amrex::max(ref_density, ldensity_floor);
      ref_pressure = amrex::max(ref_pressure, local_pressure_floor);



      Real dum = ref_normal_velocity - Im[QUN][0] - hdt*TraceM_src[QUN][0];
      Real pressure_jump_minus = ref_total_pressure - Im[QPTOTAL][0] - hdt*TraceM_src[QPRES][0];

      Real drho = ref_density - Im[QRHO][1] - hdt*TraceM_src[QRHO][1];
      Real pressure_jump = ref_total_pressure - Im[QPTOTAL][1] - hdt*TraceM_src[QPRES][1];
      Real energy_density_jump_gas = ref_energy_density_gas - Im[QEINTDENS][1] - hdt*TraceM_src[QEINTDENS][1];

      Real der[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        der[g] = ref_rad_energy[g] - Im[QRADENG+g][1];
      }

      Real dup = ref_normal_velocity - Im[QUN][2] - hdt*TraceM_src[QUN][2];
      Real pressure_jump_plus = ref_total_pressure - Im[QPTOTAL][2] - hdt*TraceM_src[QPRES][2];



      Real acoustic_amp_minus = 0.5_rt*(pressure_jump_minus/(rho*cc) - dum)*rho/cc;
      Real acoustic_amp_plus = 0.5_rt*(pressure_jump_plus/(rho*cc) + dup)*rho/cc;
      Real entropy_amp_density = drho - pressure_jump/csq;
      Real entropy_amp_energy_gas = energy_density_jump_gas - pressure_jump*h_g;

      Real rad_wave_amplitude[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        rad_wave_amplitude[g] = der[g] - pressure_jump/csq*hr[g];
      }

      acoustic_amp_minus = un-cc > 0.0_rt ? 0.0_rt : -acoustic_amp_minus;
      acoustic_amp_plus = un+cc > 0.0_rt ? 0.0_rt : -acoustic_amp_plus;
      entropy_amp_density = un > 0.0_rt ? 0.0_rt : -entropy_amp_density;
      entropy_amp_energy_gas = un > 0.0_rt ? 0.0_rt : -entropy_amp_energy_gas;

      for (int g = 0; g < NGROUPS; g++) {
        rad_wave_amplitude[g] = un > 0.0_rt ? 0.0_rt : -rad_wave_amplitude[g];
      }

      qp(i,j,k,QRHO) = amrex::max(ldensity_floor, ref_density + acoustic_amp_plus + acoustic_amp_minus + entropy_amp_density);
      qp(i,j,k,QUN) = ref_normal_velocity + (acoustic_amp_plus - acoustic_amp_minus)*cc/rho;
      qp(i,j,k,QEINTDENS) = ref_energy_density_gas + (acoustic_amp_plus + acoustic_amp_minus)*h_g*csq + entropy_amp_energy_gas;

      qp(i,j,k,QPRES) = ref_pressure + (acoustic_amp_plus + acoustic_amp_minus)*gas_sound_speed_sq;
      for (int g = 0; g < NGROUPS; g++) {
        qp(i,j,k,QPRES) += -lamp[g]*rad_wave_amplitude[g];
      }
      qp(i,j,k,QPRES) = amrex::max(local_pressure_floor, qp(i,j,k,QPRES));

      qp(i,j,k,QPTOTAL) = ref_total_pressure + (acoustic_amp_plus + acoustic_amp_minus)*csq;
      qp(i,j,k,QETOTDENS) = qp(i,j,k,QEINTDENS);

      Real rad_prim_value;
      for (int g = 0; g < NGROUPS; g++) {
        rad_prim_value = ref_rad_energy[g] + (acoustic_amp_plus + acoustic_amp_minus)*hr[g] + rad_wave_amplitude[g];
        qp(i,j,k,QRADENG+g) = rad_prim_value;
        qp(i,j,k,QETOTDENS) += rad_prim_value;
      }

      for (int g = 0; g < NGROUPS; g++) {
        if (qp(i,j,k,QRADENG+g) < 0.0_rt) {
          Real clipped_rad_energy = -qp(i,j,k,QRADENG+g);
          qp(i,j,k,QRADENG+g) = 0.0_rt;
          qp(i,j,k,QPTOTAL) += lamp[g] * clipped_rad_energy;
          qp(i,j,k,QETOTDENS) += clipped_rad_energy;
        }
      }


      qp(i,j,k,QUT) = Im[QUT][1] + hdt*TraceM_src[QUT][1];
      qp(i,j,k,QTRANSV2) = Im[QTRANSV2][1] + hdt*TraceM_src[QTRANSV2][1];

    }


    if ((idir == 0 && i <= vhi[0]) ||
        (idir == 1 && j <= vhi[1]) ||
        (idir == 2 && k <= vhi[2])) {

      Real ref_density = Ip[QRHO][2];
      Real ref_normal_velocity = Ip[QUN][2];

      Real ref_pressure = Ip[QPRES][2];
      Real ref_energy_density_gas = Ip[QEINTDENS][2];

      Real ref_total_pressure = Ip[QPTOTAL][2];
      Real ref_rad_energy[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        ref_rad_energy[g] = Ip[QRADENG+g][2];
      }

      ref_density = amrex::max(ref_density, ldensity_floor);
      ref_pressure = amrex::max(ref_pressure, local_pressure_floor);


      Real dum = ref_normal_velocity - Ip[QUN][0] - hdt*TraceP_src[QUN][0];
      Real pressure_jump_minus = ref_total_pressure - Ip[QPTOTAL][0] - hdt*TraceP_src[QPRES][0];

      Real drho = ref_density - Ip[QRHO][1] - hdt*TraceP_src[QRHO][1];
      Real pressure_jump = ref_total_pressure - Ip[QPTOTAL][1] - hdt*TraceP_src[QPRES][1];
      Real energy_density_jump_gas = ref_energy_density_gas - Ip[QEINTDENS][1] - hdt*TraceP_src[QEINTDENS][1];

      Real der[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        der[g]  = ref_rad_energy[g]  - Ip[QRADENG+g][1];
      }

      Real dup = ref_normal_velocity - Ip[QUN][2] - hdt*TraceP_src[QUN][2];
      Real pressure_jump_plus = ref_total_pressure - Ip[QPTOTAL][2] - hdt*TraceP_src[QPRES][2];



      Real acoustic_amp_minus = 0.5_rt*(pressure_jump_minus/(rho*cc) - dum)*rho/cc;
      Real acoustic_amp_plus = 0.5_rt*(pressure_jump_plus/(rho*cc) + dup)*rho/cc;
      Real entropy_amp_density = drho - pressure_jump/csq;
      Real entropy_amp_energy_gas = energy_density_jump_gas - pressure_jump*h_g;

      Real rad_wave_amplitude[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        rad_wave_amplitude[g] = der[g] - pressure_jump/csq*hr[g];
      }

      acoustic_amp_minus = un-cc > 0.0_rt ? -acoustic_amp_minus : 0.0_rt;
      acoustic_amp_plus = un+cc > 0.0_rt ? -acoustic_amp_plus : 0.0_rt;
      entropy_amp_density = un > 0.0_rt ? -entropy_amp_density : 0.0_rt;
      entropy_amp_energy_gas = un > 0.0_rt ? -entropy_amp_energy_gas : 0.0_rt;

      for (int g = 0; g < NGROUPS; g++) {
        rad_wave_amplitude[g] = un > 0.0_rt ? -rad_wave_amplitude[g] : 0.0_rt;
      }


      if (idir == 0) {
        qm(i+1,j,k,QRHO) = amrex::max(ldensity_floor, ref_density + acoustic_amp_plus + acoustic_amp_minus + entropy_amp_density);
        qm(i+1,j,k,QUN) = ref_normal_velocity + (acoustic_amp_plus - acoustic_amp_minus)*cc/rho;
        qm(i+1,j,k,QEINTDENS) = ref_energy_density_gas + (acoustic_amp_plus + acoustic_amp_minus)*h_g*csq + entropy_amp_energy_gas;

        qm(i+1,j,k,QPRES) = ref_pressure + (acoustic_amp_plus + acoustic_amp_minus)*gas_sound_speed_sq;
        for (int g = 0; g < NGROUPS; g++) {
          qm(i+1,j,k,QPRES) += -limiter_minus[g]*rad_wave_amplitude[g];
        }

        qm(i+1,j,k,QPTOTAL) = ref_total_pressure + (acoustic_amp_plus + acoustic_amp_minus)*csq;
        qm(i+1,j,k,QETOTDENS) = qm(i+1,j,k,QEINTDENS);

        Real rad_prim_value;
        for (int g = 0; g < NGROUPS; g++) {
          rad_prim_value = ref_rad_energy[g] + (acoustic_amp_plus + acoustic_amp_minus)*hr[g] + rad_wave_amplitude[g];
          qm(i+1,j,k,QRADENG+g) = rad_prim_value;
          qm(i+1,j,k,QETOTDENS) += rad_prim_value;
        }

        for (int g = 0; g < NGROUPS; g++) {
          if (qm(i+1,j,k,QRADENG+g) < 0.0_rt) {
            Real clipped_rad_energy = -qm(i+1,j,k,QRADENG+g);
            qm(i+1,j,k,QRADENG+g) = 0.0_rt;
            qm(i+1,j,k,QPTOTAL) += limiter_minus[g] * clipped_rad_energy;
            qm(i+1,j,k,QETOTDENS) += clipped_rad_energy;
          }
        }

        qm(i+1,j,k,QUT) = Ip[QUT][1] + hdt*TraceP_src[QUT][1];
        qm(i+1,j,k,QTRANSV2) = Ip[QTRANSV2][1] + hdt*TraceP_src[QTRANSV2][1];

      } else if (idir == 1) {
        qm(i,j+1,k,QRHO) = amrex::max(ldensity_floor, ref_density + acoustic_amp_plus + acoustic_amp_minus + entropy_amp_density);
        qm(i,j+1,k,QUN) = ref_normal_velocity + (acoustic_amp_plus - acoustic_amp_minus)*cc/rho;
        qm(i,j+1,k,QEINTDENS) = ref_energy_density_gas + (acoustic_amp_plus + acoustic_amp_minus)*h_g*csq + entropy_amp_energy_gas;

        qm(i,j+1,k,QPRES) = ref_pressure + (acoustic_amp_plus + acoustic_amp_minus)*gas_sound_speed_sq;
        for (int g = 0; g < NGROUPS; g++) {
          qm(i,j+1,k,QPRES) += -limiter_minus[g]*rad_wave_amplitude[g];
        }

        qm(i,j+1,k,QPTOTAL) = ref_total_pressure + (acoustic_amp_plus + acoustic_amp_minus)*csq;
        qm(i,j+1,k,QETOTDENS) = qm(i,j+1,k,QEINTDENS);

        Real rad_prim_value;
        for (int g = 0; g < NGROUPS; g++) {
          rad_prim_value = ref_rad_energy[g] + (acoustic_amp_plus + acoustic_amp_minus)*hr[g] + rad_wave_amplitude[g];
          qm(i,j+1,k,QRADENG+g) = rad_prim_value;
          qm(i,j+1,k,QETOTDENS) += rad_prim_value;
        }

        for (int g = 0; g < NGROUPS; g++) {
          rad_prim_value = ref_rad_energy[g] + (acoustic_amp_plus + acoustic_amp_minus)*hr[g] + rad_wave_amplitude[g];
          qm(i,j+1,k,QRADENG+g) = rad_prim_value;
          qm(i,j+1,k,QETOTDENS) += rad_prim_value;
        }

        for (int g = 0; g < NGROUPS; g++) {
          if (qm(i,j+1,k,QRADENG+g) < 0.0_rt) {
            Real clipped_rad_energy = -qm(i,j+1,k,QRADENG+g);
            qm(i,j+1,k,QRADENG+g) = 0.0_rt;
            qm(i,j+1,k,QPTOTAL) += limiter_minus[g] * clipped_rad_energy;
            qm(i,j+1,k,QETOTDENS) += clipped_rad_energy;
          }
        }

        qm(i,j+1,k,QUT) = Ip[QUT][1] + hdt*TraceP_src[QUT][1];
        qm(i,j+1,k,QTRANSV2) = Ip[QTRANSV2][1] + hdt*TraceP_src[QTRANSV2][1];


      } else {

        qm(i,j,k+1,QRHO) = amrex::max(ldensity_floor, ref_density + acoustic_amp_plus + acoustic_amp_minus + entropy_amp_density);
        qm(i,j,k+1,QUN) = ref_normal_velocity + (acoustic_amp_plus - acoustic_amp_minus)*cc/rho;
        qm(i,j,k+1,QEINTDENS) = ref_energy_density_gas + (acoustic_amp_plus + acoustic_amp_minus)*h_g*csq + entropy_amp_energy_gas;

        qm(i,j,k+1,QPRES) = ref_pressure + (acoustic_amp_plus + acoustic_amp_minus)*gas_sound_speed_sq;
        for (int g = 0; g < NGROUPS; g++) {
          qm(i,j,k+1,QPRES) += -limiter_minus[g]*rad_wave_amplitude[g];
        }

        qm(i,j,k+1,QPTOTAL) = ref_total_pressure + (acoustic_amp_plus + acoustic_amp_minus)*csq;
        qm(i,j,k+1,QETOTDENS) = qm(i,j,k+1,QEINTDENS);

        Real rad_prim_value;
        for (int g = 0; g < NGROUPS; g++) {
          rad_prim_value = ref_rad_energy[g] + (acoustic_amp_plus + acoustic_amp_minus)*hr[g] + rad_wave_amplitude[g];
          qm(i,j,k+1,QRADENG+g) = rad_prim_value;
          qm(i,j,k+1,QETOTDENS) += rad_prim_value;
        }

        for (int g = 0; g < NGROUPS; g++) {
          rad_prim_value = ref_rad_energy[g] + (acoustic_amp_plus + acoustic_amp_minus)*hr[g] + rad_wave_amplitude[g];
          qm(i,j,k+1,QRADENG+g) = rad_prim_value;
          qm(i,j,k+1,QETOTDENS) += rad_prim_value;
        }

        for (int g = 0; g < NGROUPS; g++) {
          if (qm(i,j,k+1,QRADENG+g) < 0.0_rt) {
            Real clipped_rad_energy = -qm(i,j,k+1,QRADENG+g);
            qm(i,j,k+1,QRADENG+g) = 0.0_rt;
            qm(i,j,k+1,QPTOTAL) += limiter_minus[g] * clipped_rad_energy;
            qm(i,j,k+1,QETOTDENS) += clipped_rad_energy;
          }
        }

        qm(i,j,k+1,QUT) = Ip[QUT][1] + hdt*TraceP_src[QUT][1];
        qm(i,j,k+1,QTRANSV2) = Ip[QTRANSV2][1] + hdt*TraceP_src[QTRANSV2][1];

      }

    }


#if (AMREX_SPACEDIM < 3)
    if (idir == 0 && dloga(i,j,k) != 0.0_rt) {
      Real courant_local = dt/dx[0]*(cc+std::abs(un));
      Real eta = (1.0_rt - courant_local)/(cc*dt*std::abs(dloga(i,j,k)));
      Real limited_dlog_area = amrex::min(eta, 1.0_rt)*dloga(i,j,k);
      Real source_density = -0.5_rt*dt*rho*limited_dlog_area*un;
      Real source_pressure = source_density*gas_sound_speed_sq;
      Real source = source_pressure*h_g;
      Real rad_source_terms[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        rad_source_terms[g] = -0.5_rt*dt*limited_dlog_area*un*(lam0[g] + 1.0_rt)*er[g];
      }

      if (i <= vhi[0]) {
        add_area_source_terms(qm, i+1, j, k, source_density, source_pressure, source,
                              rad_source_terms, limiter_minus, ldensity_floor);
      }

      if (i >= vlo[0]) {
        add_area_source_terms(qp, i, j, k, source_density, source_pressure, source,
                              rad_source_terms, lamp, ldensity_floor);
      }
    }
#endif

  });
}
