#include "HydroSim.H"
#include "HydroSim_helpers.H"

#include "Radiation.H"
#include "RadiativeHydro.H"
#include "flux_limiter.H"

using namespace amrex;

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


  // radiation energy update.

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

  // add the radiation pressure gradient to the momentum for all
  // directions

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    // radiation contribution -- this is sum{lambda E_r}
    Real radiation_pressure_grad_x = 0.0;
    Real radiation_pressure_grad_y = 0.0;
    Real radiation_pressure_grad_z = 0.0;

    Real lamc = 0.0;
    for (int g = 0; g < NGROUPS; g++) {

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

      radiation_pressure_grad_x = radiation_pressure_grad_x + lamc * (qx(i+1,j,k,GDRADENG+g) - qx(i,j,k,GDRADENG+g))/dx[0];
#if AMREX_SPACEDIM >= 2
      radiation_pressure_grad_y = radiation_pressure_grad_y + lamc * (qy(i,j+1,k,GDRADENG+g) - qy(i,j,k,GDRADENG+g))/dx[1];
#endif
#if AMREX_SPACEDIM == 3
      radiation_pressure_grad_z = radiation_pressure_grad_z + lamc * (qz(i,j,k+1,GDRADENG+g) - qz(i,j,k,GDRADENG+g))/dx[2];
#endif
    }


    // we now want to compute the change in the kinetic energy -- we
    // base this off of S_new, since that already is updated with hydro

    // note, we need to include the Z component here too (even
    // for 1- and 2-d), since rotation might be in play

    Real cell_density_new = cons_new(i,j,k,URHO);

    // this update includes the hydro fluxes and grad{p} from hydro
    Real x_momentum_new1 = cons_new(i,j,k,UMX);
    Real y_momentum_new1 = cons_new(i,j,k,UMY);
    Real z_momentum_new1 = cons_new(i,j,k,UMZ);

    Real ek1 = (x_momentum_new1 * x_momentum_new1 +
                y_momentum_new1 * y_momentum_new1 +
                z_momentum_new1 * z_momentum_new1) / (2.0_rt * cell_density_new);


    // now add the radiation pressure gradient
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

    // update the kinetic energy with the radiation contribution

    cons_new(i,j,k,UEDEN) = cons_new(i,j,k,UEDEN) + dek;

    if (!radiation::moving_frame) {
      // ! mixed-frame (single group only)
      RadEnergyOut(i,j,k,0) = RadEnergyOut(i,j,k,0) - dek;
    }

  });


  // Add radiation source terms to rho*u, rhoE, and Er

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

      // Note that for single group, frequency_advection_mode is always 1
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
                qz(i,j,k,GDFLUXLIM+g) + qz(i,j,k+1,GDFLUXLIM+g) ) / 6.0_rt;
#endif

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

