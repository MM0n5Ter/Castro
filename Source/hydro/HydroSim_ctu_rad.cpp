#include "HydroSim.H"
#include "HydroSim_util.H"

#include "Radiation.H"
#include "RadHydro.H"
#include "fluxlimiter.H"

using namespace amrex;

int
HydroSim::ctu_rad_cons_update(const Box& bx,
                       Array4<Real> const& U_new,
                       Array4<Real const> const& Erin,
                       Array4<Real> const& Erout,
                       Array4<Real const> const& radflux1,
                       Array4<Real const> const& qx,
                       Array4<Real const> const& area1,
#if AMREX_SPACEDIM >= 2
                       Array4<Real const> const& radflux2,
                       Array4<Real const> const& qy,
                       Array4<Real const> const& area2,
#endif
#if AMREX_SPACEDIM == 3
                       Array4<Real const> const& radflux3,
                       Array4<Real const> const& qz,
                       Array4<Real const> const& area3,
#endif
                       Array4<Real const> const& vol,
                       const Real dt)
{


  auto dx = geom.CellSizeArray();
  const int coord_type = geom.Coord();

  GpuArray<Real, NGROUPS> Erscale = {0.0};
  GpuArray<Real, NGROUPS> dlognu = {0.0};

  if (NGROUPS > 1) {
    for (int g = 0; g < NGROUPS; ++g) {
      dlognu[g] = radiation->dlognugroup[g];
    }
    if (radiation::fspace_advection_type == 1) {
      for (int g = 0; g < NGROUPS; g++) {
        Erscale[g] = dlognu[g];
      }
    } else {
      for (int g = 0; g < NGROUPS; g++) {
        Erscale[g] = radiation->nugroup[g] * dlognu[g];
      }
    }
  }


  // radiation energy update.

  amrex::ParallelFor(bx, NGROUPS,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int g) noexcept
  {

    Erout(i,j,k,g) = Erin(i,j,k,g) + dt *
      (  radflux1(i,j,k,g) * area1(i,j,k) - radflux1(i+1,j,k,g) * area1(i+1,j,k)
#if AMREX_SPACEDIM >= 2
       + radflux2(i,j,k,g) * area2(i,j,k) - radflux2(i,j+1,k,g) * area2(i,j+1,k)
#endif
#if AMREX_SPACEDIM == 3
       + radflux3(i,j,k,g) * area3(i,j,k) - radflux3(i,j,k+1,g) * area3(i,j,k+1)
#endif
         ) / vol(i,j,k);
  });

  // add the radiation pressure gradient to the momentum for all
  // directions

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    // radiation contribution -- this is sum{lambda E_r}
    Real dprdx = 0.0;
    Real dprdy = 0.0;
    Real dprdz = 0.0;

    Real lamc = 0.0;
    for (int g = 0; g < NGROUPS; g++) {

#if AMREX_SPACEDIM == 1
      lamc = 0.5_rt * (qx(i,j,k,IFCFLXLIM+g) + qx(i+1,j,k,IFCFLXLIM+g));
#endif
#if AMREX_SPACEDIM == 2
      lamc = 0.25_rt * (qx(i,j,k,IFCFLXLIM+g) + qx(i+1,j,k,IFCFLXLIM+g) +
                        qy(i,j,k,IFCFLXLIM+g) + qy(i,j+1,k,IFCFLXLIM+g));
#endif
#if AMREX_SPACEDIM == 3
      lamc = (qx(i,j,k,IFCFLXLIM+g) + qx(i+1,j,k,IFCFLXLIM+g) +
              qy(i,j,k,IFCFLXLIM+g) + qy(i,j+1,k,IFCFLXLIM+g) +
              qz(i,j,k,IFCFLXLIM+g) + qz(i,j,k+1,IFCFLXLIM+g)) / 6.0_rt;
#endif

      dprdx = dprdx + lamc * (qx(i+1,j,k,IFCERAD+g) - qx(i,j,k,IFCERAD+g))/dx[0];
#if AMREX_SPACEDIM >= 2
      dprdy = dprdy + lamc * (qy(i,j+1,k,IFCERAD+g) - qy(i,j,k,IFCERAD+g))/dx[1];
#endif
#if AMREX_SPACEDIM == 3
      dprdz = dprdz + lamc * (qz(i,j,k+1,IFCERAD+g) - qz(i,j,k,IFCERAD+g))/dx[2];
#endif
    }


    // we now want to compute the change in the kinetic energy -- we
    // base this off of S_new, since that already is updated with hydro

    // note, we need to include the Z component here too (even
    // for 1- and 2-d), since rotation might be in play

    Real urho_new = U_new(i,j,k,URHO);

    // this update includes the hydro fluxes and grad{p} from hydro
    Real umx_new1 = U_new(i,j,k,UMX);
    Real umy_new1 = U_new(i,j,k,UMY);
    Real umz_new1 = U_new(i,j,k,UMZ);

    Real ek1 = (umx_new1 * umx_new1 +
                umy_new1 * umy_new1 +
                umz_new1 * umz_new1) / (2.0_rt * urho_new);


    // now add the radiation pressure gradient
    U_new(i,j,k,UMX) -= dt * dprdx;
    U_new(i,j,k,UMY) -= dt * dprdy;
    U_new(i,j,k,UMZ) -= dt * dprdz;

    Real umx_new2 = U_new(i,j,k,UMX);
    Real umy_new2 = U_new(i,j,k,UMY);
    Real umz_new2 = U_new(i,j,k,UMZ);

    Real ek2 = (umx_new2 * umx_new2 +
                umy_new2 * umy_new2 +
                umz_new2 * umz_new2) / (2.0_rt * urho_new);

    Real dek = ek2 - ek1;

    // update the kinetic energy with the radiation contribution

    U_new(i,j,k,UEDEN) = U_new(i,j,k,UEDEN) + dek;

    if (!radiation::comoving) {
      // ! mixed-frame (single group only)
      Erout(i,j,k,0) = Erout(i,j,k,0) - dek;
    }

  });


  // Add radiation source terms to rho*u, rhoE, and Er

  int n_freq_substeps = -1;

  if (radiation::comoving) {

    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    reduce_op.eval(bx, reduce_data,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
    {

      Real ux = 0.5_rt * (qx(i,j,k,IFCU) + qx(i+1,j,k,IFCU));
#if AMREX_SPACEDIM >= 2
      Real uy = 0.5_rt * (qy(i,j,k,IFCV) + qy(i,j+1,k,IFCV));
#endif
#if AMREX_SPACEDIM == 3
      Real uz = 0.5_rt * (qz(i,j,k,IFCW) + qz(i,j,k+1,IFCW));
#endif

      Real dudx[3] = {0, 0, 0};
      Real dudy[3] = {0, 0, 0};
      Real dudz[3] = {0, 0, 0};

      dudx[0] = (qx(i+1,j,k,IFCU) - qx(i,j,k,IFCU))/dx[0];
      dudx[1] = (qx(i+1,j,k,IFCV) - qx(i,j,k,IFCV))/dx[0];
      dudx[2] = (qx(i+1,j,k,IFCW) - qx(i,j,k,IFCW))/dx[0];

#if AMREX_SPACEDIM >= 2
      dudy[0] = (qy(i,j+1,k,IFCU) - qy(i,j,k,IFCU))/dx[1];
      dudy[1] = (qy(i,j+1,k,IFCV) - qy(i,j,k,IFCV))/dx[1];
      dudy[2] = (qy(i,j+1,k,IFCW) - qy(i,j,k,IFCW))/dx[1];
#endif

#if AMREX_SPACEDIM == 3
      dudz[0] = (qz(i,j,k+1,IFCU) - qz(i,j,k,IFCU))/dx[2];
      dudz[1] = (qz(i,j,k+1,IFCV) - qz(i,j,k,IFCV))/dx[2];
      dudz[2] = (qz(i,j,k+1,IFCW) - qz(i,j,k,IFCW))/dx[2];
#endif

      Real divu = dudx[0] + dudy[1] + dudz[2];

      // Note that for single group, fspace_advection_type is always 1
      GpuArray<Real, NGROUPS> af;

      for (int g = 0; g < NGROUPS; g++) {

        Real nhat[3] = {0., 0., 0.};

        nhat[0] = (qx(i+1,j,k,IFCERAD+g) - qx(i,j,k,IFCERAD+g))/dx[0];
#if AMREX_SPACEDIM >= 2
        nhat[1] = (qy(i,j+1,k,IFCERAD+g) - qy(i,j,k,IFCERAD+g))/dx[1];
#endif
#if AMREX_SPACEDIM == 3
        nhat[2] = (qz(i,j,k+1,IFCERAD+g) - qz(i,j,k,IFCERAD+g))/dx[2];
#endif

        Real grad_u_dot_n[3];

        grad_u_dot_n[0] = nhat[0] * dudx[0] + nhat[1] * dudx[1] + nhat[2] * dudx[2];
        grad_u_dot_n[1] = nhat[0] * dudy[0] + nhat[1] * dudy[1] + nhat[2] * dudy[2];
        grad_u_dot_n[2] = nhat[0] * dudz[0] + nhat[1] * dudz[1] + nhat[2] * dudz[2];

        Real nn_colon_grad_u = (nhat[0] * grad_u_dot_n[0] + nhat[1] * grad_u_dot_n[1] + nhat[2] * grad_u_dot_n[2]) /
          (nhat[0] * nhat[0] + nhat[1] * nhat[1] + nhat[2] * nhat[2] + 1.e-50_rt);

        Real lamc;
#if AMREX_SPACEDIM == 1
        lamc = 0.5_rt * (qx(i,j,k,IFCFLXLIM+g) + qx(i+1,j,k,IFCFLXLIM+g));
#endif
#if AMREX_SPACEDIM == 2
        lamc = 0.25_rt * (qx(i,j,k,IFCFLXLIM+g) + qx(i+1,j,k,IFCFLXLIM+g) +
                          qy(i,j,k,IFCFLXLIM+g) + qy(i,j+1,k,IFCFLXLIM+g));
#endif
#if AMREX_SPACEDIM == 3
        lamc = (qx(i,j,k,IFCFLXLIM+g) + qx(i+1,j,k,IFCFLXLIM+g) +
                qy(i,j,k,IFCFLXLIM+g) + qy(i,j+1,k,IFCFLXLIM+g) +
                qz(i,j,k,IFCFLXLIM+g) + qz(i,j,k+1,IFCFLXLIM+g) ) / 6.0_rt;
#endif

        Real Eddf = Edd_factor(lamc);
        Real f1 = (1.0_rt - Eddf) * 0.5_rt;
        Real f2 = (3.0_rt * Eddf - 1.0_rt) * 0.5_rt;
        af[g] = -(f1*divu + f2*nn_colon_grad_u);

        if (radiation::fspace_advection_type == 1) {
          Real Eddfxp = Edd_factor(qx(i+1,j,k,IFCFLXLIM+g));
          Real Eddfxm = Edd_factor(qx(i,j,k,IFCFLXLIM+g));
#if AMREX_SPACEDIM >= 2
          Real Eddfyp = Edd_factor(qy(i,j+1,k,IFCFLXLIM+g));
          Real Eddfym = Edd_factor(qy(i,j,k,IFCFLXLIM+g));
#endif
#if AMREX_SPACEDIM == 3
          Real Eddfzp = Edd_factor(qz(i,j,k+1,IFCFLXLIM+g));
          Real Eddfzm = Edd_factor(qz(i,j,k,IFCFLXLIM+g));
#endif

          Real f1xp = 0.5_rt * (1.0_rt - Eddfxp);
          Real f1xm = 0.5_rt * (1.0_rt - Eddfxm);
#if AMREX_SPACEDIM >= 2
          Real f1yp = 0.5_rt * (1.0_rt - Eddfyp);
          Real f1ym = 0.5_rt * (1.0_rt - Eddfym);
#endif
#if AMREX_SPACEDIM == 3
          Real f1zp = 0.5_rt * (1.0_rt - Eddfzp);
          Real f1zm = 0.5_rt * (1.0_rt - Eddfzm);
#endif

          Real Gf1E[3];

          Gf1E[0] = (f1xp*qx(i+1,j,k,IFCERAD+g) - f1xm*qx(i,j,k,IFCERAD+g)) / dx[0];
#if AMREX_SPACEDIM >= 2
          Gf1E[1] = (f1yp*qy(i,j+1,k,IFCERAD+g) - f1ym*qy(i,j,k,IFCERAD+g)) / dx[1];
#endif
#if AMREX_SPACEDIM == 3
          Gf1E[2] = (f1zp*qz(i,j,k+1,IFCERAD+g) - f1zm*qz(i,j,k,IFCERAD+g)) / dx[2];
#endif


#if AMREX_SPACEDIM == 1
          Real E_face_avg = 0.5_rt * (qx(i,j,k,IFCERAD+g) + qx(i+1,j,k,IFCERAD+g));
          Erout(i,j,k,g) = Erout(i,j,k,g) + dt * ux * Gf1E[0]
            - dt * f2 * E_face_avg * nn_colon_grad_u;
#endif
#if AMREX_SPACEDIM == 2
          Real E_face_avg = 0.25_rt * (qx(i,j,k,IFCERAD+g) + qx(i+1,j,k,IFCERAD+g) +
                                 qy(i,j,k,IFCERAD+g) + qy(i,j+1,k,IFCERAD+g));
          Erout(i,j,k,g) = Erout(i,j,k,g) + dt * (ux * Gf1E[0] + uy * Gf1E[1])
            - dt * f2 * E_face_avg * nn_colon_grad_u;
#endif
#if AMREX_SPACEDIM == 3
          Real E_face_avg = (qx(i,j,k,IFCERAD+g) + qx(i+1,j,k,IFCERAD+g) +
                       qy(i,j,k,IFCERAD+g) + qy(i,j+1,k,IFCERAD+g) +
                       qz(i,j,k,IFCERAD+g) + qz(i,j,k+1,IFCERAD+g) ) / 6.0_rt;
          Erout(i,j,k,g) = Erout(i,j,k,g) + dt * (ux * Gf1E[0] + uy * Gf1E[1] + uz * Gf1E[2])
            - dt * f2 * E_face_avg * nn_colon_grad_u;
#endif

        }
      }

      int n_freq_substeps_tmp = 1;

      if (NGROUPS > 1) {
        GpuArray<Real, NGROUPS> ustar;
        for (int g = 0; g < NGROUPS; g++) {
          ustar[g] = Erout(i,j,k,g) / Erscale[g];
        }

        update_one_species(NGROUPS, ustar, af, dlognu, dt, n_freq_substeps_tmp);

        for (int g = 0; g < NGROUPS; g++) {
          Erout(i,j,k,g) = ustar[g] * Erscale[g];
        }
      }

      return n_freq_substeps_tmp;

    });

    ReduceTuple hv = reduce_data.value();
    n_freq_substeps = amrex::get<0>(hv);

  }

  return n_freq_substeps;
}

