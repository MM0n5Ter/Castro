#include <HydroSim.H>




#include <parabolic_profile.H>
#include <limiters.H>
#include <HydroSim_helpers.H>

#include <primitive_conversion.H>


#include <eos.H>

using namespace amrex;
using namespace reconstruction;

void
HydroSim::flux_from_state(const Box& bx,
                            Array4<Real const> const& interface_state,
                            Array4<Real> const& F,
                            const int idir) {

  int iu, iv1, iv2;
  int im1, im2, im3;

  auto coord = geom.Coord();
  auto has_pressure_flux = momentum_flux_has_pressure(idir, idir, coord);

  if (idir == 0) {
    iu = QU;
    iv1 = QV;
    iv2 = QW;
    im1 = UMX;
    im2 = UMY;
    im3 = UMZ;

  } else if (idir == 1) {
    iu = QV;
    iv1 = QU;
    iv2 = QW;
    im1 = UMY;
    im2 = UMX;
    im3 = UMZ;

  } else {
    iu = QW;
    iv1 = QU;
    iv2 = QV;
    im1 = UMZ;
    im2 = UMX;
    im3 = UMY;
  }


  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real advect_velocity = interface_state(i,j,k,iu);
    Real rho_internal_energy = interface_state(i,j,k,QEINTDENS);

    F(i,j,k,URHO) = interface_state(i,j,k,QRHO)*advect_velocity;

    F(i,j,k,im1) = F(i,j,k,URHO)*interface_state(i,j,k,iu);
    if (has_pressure_flux) {
      F(i,j,k,im1) += interface_state(i,j,k,QPRES);
    }
    F(i,j,k,im2) = F(i,j,k,URHO)*interface_state(i,j,k,iv1);
    F(i,j,k,im3) = F(i,j,k,URHO)*interface_state(i,j,k,iv2);

    Real rho_total_energy = rho_internal_energy + 0.5_rt * interface_state(i,j,k,QRHO)*
      (interface_state(i,j,k,iu)*interface_state(i,j,k,iu) +
       interface_state(i,j,k,iv1)*interface_state(i,j,k,iv1) +
       interface_state(i,j,k,iv2)*interface_state(i,j,k,iv2));

    F(i,j,k,UEDEN) = advect_velocity*(rho_total_energy + interface_state(i,j,k,QPRES));
    F(i,j,k,UEINT) = advect_velocity*rho_internal_energy;

    F(i,j,k,UTEMP) = 0.0;
#ifdef SHOCK_VAR
    F(i,j,k,UFLAG_SHOCK) = 0.0;
#endif

    for (int passive_index = 0; passive_index < npassive; passive_index++) {
      int n  = passive_cons_index(passive_index);
      int nqp = passive_prim_index(passive_index);

      F(i,j,k,n) = F(i,j,k,URHO)*interface_state(i,j,k,nqp);
    }

  });
}

void
HydroSim::save_godunov_state(const Box& bx,
                            Array4<Real const> const& interface_state,
                            Array4<Real> const& godunov_state) {

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    godunov_state(i,j,k,GDU) = interface_state(i,j,k,QU);
    godunov_state(i,j,k,GDV) = interface_state(i,j,k,QV);
    godunov_state(i,j,k,GDW) = interface_state(i,j,k,QW);
    godunov_state(i,j,k,GDPRESSURE) = interface_state(i,j,k,QPRES);
  });
}

void
HydroSim::lines_diffusion_flux(const Box& bx,
                           const int idir,
                           Array4<Real const> const& U,
                           Array4<Real const> const& cond,
                           Array4<Real> const& flux) {

  const auto dx = geom.CellSizeArray();

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real face_conductivity;
    Real diffusion_term;

    if (idir == 0) {
      face_conductivity = 0.5_rt * (cond(i,j,k) + cond(i-1,j,k));
      diffusion_term = -face_conductivity * (U(i,j,k,UTEMP) - U(i-1,j,k,UTEMP)) / dx[0];

    } else if (idir == 1) {
      face_conductivity = 0.5_rt * (cond(i,j,k) + cond(i,j-1,k));
      diffusion_term = -face_conductivity * (U(i,j,k,UTEMP) - U(i,j-1,k,UTEMP)) / dx[1];

    } else {
      face_conductivity = 0.5_rt * (cond(i,j,k) + cond(i,j,k-1));
      diffusion_term = -face_conductivity * (U(i,j,k,UTEMP) - U(i,j,k-1,UTEMP)) / dx[2];
    }

    flux(i,j,k,UEINT) += diffusion_term;
    flux(i,j,k,UEDEN) += diffusion_term;

  });
}

void
HydroSim::lines_cons_update(const Box& bx,
#ifdef SHOCK_VAR
                   Array4<Real const> const& shk,
#endif
                   Array4<Real const> const& sourceTerms,
                   Array4<Real> const& update,
                   const Real dt,
                   Array4<Real const> const& xface_flux,
#if AMREX_SPACEDIM >= 2
                   Array4<Real const> const& flux1,
#endif
#if AMREX_SPACEDIM == 3
                   Array4<Real const> const& flux2,
#endif
                   Array4<Real const> const& face_area0,
#if AMREX_SPACEDIM >= 2
                   Array4<Real const> const& area1,
#endif
#if AMREX_SPACEDIM == 3
                   Array4<Real const> const& area2,
#endif
#if AMREX_SPACEDIM <= 2
                   Array4<Real const> const& q0,
#if AMREX_SPACEDIM == 2
                   Array4<Real const> const& q1,
#endif
#endif
                   Array4<Real const> const& vol) {

  amrex::ignore_unused(dt);

#if AMREX_SPACEDIM <= 2
  const auto dx = geom.CellSizeArray();
  auto coord = geom.Coord();
#if AMREX_SPACEDIM == 2
  auto prob_lo = geom.ProbLoArray();
#endif
#endif

  amrex::ParallelFor(bx, NUM_STATE,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {

#if AMREX_SPACEDIM == 1
    update(i,j,k,n) += (xface_flux(i,j,k,n) * face_area0(i,j,k) - xface_flux(i+1,j,k,n) * face_area0(i+1,j,k) ) / vol(i,j,k);

#elif AMREX_SPACEDIM == 2
    update(i,j,k,n) += (xface_flux(i,j,k,n) * face_area0(i,j,k) - xface_flux(i+1,j,k,n) * face_area0(i+1,j,k) +
                        flux1(i,j,k,n) * area1(i,j,k) - flux1(i,j+1,k,n) * area1(i,j+1,k) ) / vol(i,j,k);

#else
    update(i,j,k,n) += (xface_flux(i,j,k,n) * face_area0(i,j,k) - xface_flux(i+1,j,k,n) * face_area0(i+1,j,k) +
                        flux1(i,j,k,n) * area1(i,j,k) - flux1(i,j+1,k,n) * area1(i,j+1,k) +
                        flux2(i,j,k,n) * area2(i,j,k) - flux2(i,j,k+1,n) * area2(i,j,k+1) ) / vol(i,j,k);
#endif

#if AMREX_SPACEDIM <= 2
    if (do_hydro == 1) {
        if (n == UMX && !momentum_flux_has_pressure(0, 0, coord)) {

            update(i,j,k,UMX) -= (q0(i+1,j,k,GDPRESSURE) - q0(i,j,k,GDPRESSURE)) / dx[0];

#if AMREX_SPACEDIM == 2
        } else if (n == UMY && !momentum_flux_has_pressure(1, 1, coord)) {

            Real r = prob_lo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
            update(i,j,k,UMY) -= (q1(i,j+1,k,GDPRESSURE) - q1(i,j,k,GDPRESSURE)) / (r * dx[1]);
#endif
        }
    }
#endif

    if (n < NUM_SOURCE) {
      update(i,j,k,n) += sourceTerms(i,j,k,n);
    }

  });

#ifdef SHOCK_VAR

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    update(i,j,k,UFLAG_SHOCK) = shk(i,j,k) / dt;
  });
#endif

}

void
HydroSim::cons_update_hydro(const Box& bx,
                     Array4<Real> const& cons_new,
                     Array4<Real> const& xface_flux,
                     Array4<Real const> const& qx,
#if AMREX_SPACEDIM >= 2
                     Array4<Real> const& flux1,
                     Array4<Real const> const& qy,
#endif
#if AMREX_SPACEDIM == 3
                     Array4<Real> const& flux2,
                     Array4<Real const> const& qz,
#endif
                     const Real dt)
{
  auto geomdata = geom.data();

  amrex::ParallelFor(bx, NUM_STATE,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {
    Real volinv = 1.0 / cell_geometry::volume(i, j, k, geomdata);

    cons_new(i,j,k,n) = cons_new(i,j,k,n) + dt *
        ( xface_flux(i,  j,k,n) * cell_geometry::area(i,   j, k, 0, geomdata)
        - xface_flux(i+1,j,k,n) * cell_geometry::area(i+1, j, k, 0, geomdata)
#if AMREX_SPACEDIM >= 2
        + flux1(i,j,  k,n) * cell_geometry::area(i, j,   k, 1, geomdata)
        - flux1(i,j+1,k,n) * cell_geometry::area(i, j+1, k, 1, geomdata)
#endif
#if AMREX_SPACEDIM == 3
        + flux2(i,j,k,  n) * cell_geometry::area(i, j, k,   2, geomdata)
        - flux2(i,j,k+1,n) * cell_geometry::area(i, j, k+1, 2, geomdata)
#endif
        ) * volinv;

    if (n == UEINT) {

      Real pdu = (qx(i+1,j,k,GDPRESSURE) + qx(i,j,k,GDPRESSURE)) *
          (qx(i+1,j,k,GDU) * cell_geometry::area(i+1, j, k, 0, geomdata) -
           qx(i,  j,k,GDU) * cell_geometry::area(i,   j, k, 0, geomdata));

#if AMREX_SPACEDIM >= 2
      pdu += (qy(i,j+1,k,GDPRESSURE) + qy(i,j,k,GDPRESSURE)) *
          (qy(i,j+1,k,GDV) * cell_geometry::area(i, j+1, k, 1, geomdata) -
           qy(i,j,  k,GDV) * cell_geometry::area(i, j,   k, 1, geomdata));
#endif

#if AMREX_SPACEDIM == 3
      pdu += (qz(i,j,k+1,GDPRESSURE) + qz(i,j,k,GDPRESSURE)) *
          (qz(i,j,k+1,GDW) * cell_geometry::area(i, j, k+1, 2, geomdata) -
           qz(i,j,k  ,GDW) * cell_geometry::area(i, j, k,   2, geomdata));
#endif

      pdu = 0.5 * pdu * volinv;

      cons_new(i,j,k,n) = cons_new(i,j,k,n) - dt * pdu;

    } else if (n == UMX && !momentum_flux_has_pressure(0, 0, geomdata.Coord())) {

        cons_new(i,j,k,UMX) += - dt * (qx(i+1,j,k,GDPRESSURE) - qx(i,j,k,GDPRESSURE)) / geomdata.CellSize(0);

#if AMREX_SPACEDIM >= 2
    } else if (n == UMY && !momentum_flux_has_pressure(1, 1, geomdata.Coord())) {

        Real r = geomdata.ProbLo(0) + (static_cast<Real>(i) + 0.5_rt) * geomdata.CellSize(0);
        cons_new(i,j,k,UMY) += - dt * (qy(i,j+1,k,GDPRESSURE) - qy(i,j,k,GDPRESSURE)) / (r * geomdata.CellSize(1));
#endif
    }
  });
}

void
HydroSim::inject_deferred_source(const Box& bx, const int idir, const Real dt,
                                 Array4<Real> const& prim_left,
                                 Array4<Real> const& prim_right,
                                 Array4<Real const> const& deferred_source)
{

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {


        Real pressure_old = prim_left(i,j,k,QPRES);
        Real rhoe_old = prim_left(i,j,k,QEINTDENS);

        if (idir == 0) {
            prim_left(i,j,k,QPRES) += 0.5 * dt * deferred_source(i-1,j,k,QPRES);
            prim_left(i,j,k,QEINTDENS) += 0.5 * dt * deferred_source(i-1,j,k,QEINTDENS);
        } else if (idir == 1) {
            prim_left(i,j,k,QPRES) += 0.5 * dt * deferred_source(i,j-1,k,QPRES);
            prim_left(i,j,k,QEINTDENS) += 0.5 * dt * deferred_source(i,j-1,k,QEINTDENS);
        } else {
            prim_left(i,j,k,QPRES) += 0.5 * dt * deferred_source(i,j,k-1,QPRES);
            prim_left(i,j,k,QEINTDENS) += 0.5 * dt * deferred_source(i,j,k-1,QEINTDENS);
        }

        if (prim_left(i,j,k,QPRES) < pressure_cutoff || prim_left(i,j,k,QEINTDENS) < small_dens * eint_floor) {
            prim_left(i,j,k,QPRES) = pressure_old;
            prim_left(i,j,k,QEINTDENS) = rhoe_old;
        }

        pressure_old = prim_right(i,j,k,QPRES);
        rhoe_old = prim_right(i,j,k,QEINTDENS);

        prim_right(i,j,k,QPRES) += 0.5 * dt * deferred_source(i,j,k,QPRES);
        prim_right(i,j,k,QEINTDENS) += 0.5 * dt * deferred_source(i,j,k,QEINTDENS);

        if (prim_right(i,j,k,QPRES) < pressure_cutoff || prim_right(i,j,k,QEINTDENS) < small_dens * eint_floor) {
            prim_right(i,j,k,QPRES) = pressure_old;
            prim_right(i,j,k,QEINTDENS) = rhoe_old;
        }


        for (int passive_index = 0; passive_index < npassive; passive_index++) {

            int n = passive_prim_index(passive_index);

            if (idir == 0) {
                prim_left(i,j,k,n) += 0.5 * dt * deferred_source(i-1,j,k,n);
            } else if (idir == 1) {
                prim_left(i,j,k,n) += 0.5 * dt * deferred_source(i,j-1,k,n);
            } else {
                prim_left(i,j,k,n) += 0.5 * dt * deferred_source(i,j,k-1,n);
            }
            prim_right(i,j,k,n) += 0.5 * dt * deferred_source(i,j,k,n);

            if (n >= QFS && n <= QFS-1+NumSpec) {
                prim_left(i,j,k,n) = amrex::Clamp(prim_left(i,j,k,n), 0.0_rt, 1.0_rt);
                prim_right(i,j,k,n) = amrex::Clamp(prim_right(i,j,k,n), 0.0_rt, 1.0_rt);
            }
        }

    });

}

void
HydroSim::clip_fluxes_at_density_floor(const Box& bx,
                                         int idir,
                                         Array4<Real const> const& u,
                                         Array4<Real const> const& vol,
                                         Array4<Real> const& flux,
                                         Array4<Real const> const& face_area_arr,
                                         Real dt,
                                         bool scale_by_area_dt)
{

    const Real floor_margin = 1.1_rt;


    Real floor_density = small_dens * floor_margin;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

        Real right_density = u(i,j,k,URHO);
        Real right_volume = vol(i,j,k);

        Real left_density, left_volume;

        if (idir == 0) {
            left_density = u(i-1,j,k,URHO);
            left_volume = vol(i-1,j,k);
        }
        else if (idir == 1) {
            left_density = u(i,j-1,k,URHO);
            left_volume = vol(i,j-1,k);
        }
        else {
            left_density = u(i,j,k-1,URHO);
            left_volume = vol(i,j,k-1);
        }


        Real right_flux_scale = 1.0_rt / right_volume;
        Real left_flux_scale = 1.0_rt / left_volume;

        if (scale_by_area_dt) {
            right_flux_scale *= dt * face_area_arr(i,j,k);
            left_flux_scale *= dt * face_area_arr(i,j,k);
        }


        Real right_density_change = right_flux_scale * flux(i,j,k,URHO);
        Real left_density_change = left_flux_scale * flux(i,j,k,URHO);


        if (right_density >= floor_density && std::abs(right_density_change) > 0.0_rt && right_density + 2 * AMREX_SPACEDIM * right_density_change < floor_density) {
            Real floor_scale = std::abs((floor_density - right_density) / (2 * AMREX_SPACEDIM * right_density_change));
            for (int n = 0; n < NUM_STATE; ++n) {
                flux(i,j,k,n) = flux(i,j,k,n) * floor_scale;
            }
        }
        else if (left_density >= floor_density && std::abs(left_density_change) > 0.0_rt && left_density - 2 * AMREX_SPACEDIM * left_density_change < floor_density) {
            Real floor_scale = std::abs((floor_density - left_density) / (2 * AMREX_SPACEDIM * left_density_change));
            for (int n = 0; n < NUM_STATE; ++n) {
                flux(i,j,k,n) = flux(i,j,k,n) * floor_scale;
            }
        }

    });
}

void
HydroSim::balance_species_flux(const Box& bx,
                                 Array4<Real> const& flux) {


  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real sum = 0.0_rt;

    for (int n = UFS; n < UFS+NumSpec; n++) {
      sum += flux(i,j,k,n);
    }

    Real fac = 1.0_rt;


    if (std::abs(sum) > std::numeric_limits<Real>::epsilon() * std::abs(flux(i,j,k,URHO))) {
      fac = flux(i,j,k,URHO) / sum;
    }

    for (int n = UFS; n < UFS+NumSpec; n++) {
      flux(i,j,k,n) = flux(i,j,k,n) * fac;
    }
  });
}

void
HydroSim::scale_flux_by_area(const Box& bx,
                   Array4<Real> const& flux,
                   Array4<Real const> const& face_area_arr,
                   const Real dt) {

  amrex::ParallelFor(bx, NUM_STATE,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {
    flux(i,j,k,n) = dt * flux(i,j,k,n) * face_area_arr(i,j,k);
  });
}

