#include <HydroSim.H>
#include <HydroSim_helpers.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

using namespace amrex;

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
HydroSim::corner_ppm_states(const Box& bx, const Box& vbx,
                       Array4<Real const> const& field_arr,
                       Array4<Real const> const& inv_density_arr,
                       Array4<Real const> const& prim_arr,
                       Array4<Real const> const& qthermo_arr,
                       Array4<Real const> const& prim_source,
                       Array4<Real> const& qxm,
                       Array4<Real> const& qxp,
#if AMREX_SPACEDIM >= 2
                       Array4<Real> const& qym,
                       Array4<Real> const& qyp,
#endif
#if AMREX_SPACEDIM == 3
                       Array4<Real> const& qzm,
                       Array4<Real> const& qzp,
#endif
#if AMREX_SPACEDIM < 3
                       Array4<Real const> const& logAreaRateX,
#endif
#if AMREX_SPACEDIM == 2
                       Array4<Real const> const& logAreaRateY,
#endif
                       const Real dt) {


  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {

    if (idir == 0) {
      trace_piecewise_parabolic(bx,
                idir,
                field_arr, inv_density_arr, prim_arr, qthermo_arr, prim_source,
                qxm, qxp,
#if AMREX_SPACEDIM <= 2
                logAreaRateX,
#endif
                vbx, dt);

      set_reflecting_boundary_states(bx, 0, qxm, qxp);

#if AMREX_SPACEDIM >= 2
    } else if (idir == 1) {
      trace_piecewise_parabolic(bx,
                idir,
                field_arr, inv_density_arr, prim_arr, qthermo_arr, prim_source,
                qym, qyp,
#if AMREX_SPACEDIM <= 2
                logAreaRateY,
#endif
                vbx, dt);

      set_reflecting_boundary_states(bx, 1, qym, qyp);
#endif


#if AMREX_SPACEDIM == 3
    } else {
      trace_piecewise_parabolic(bx,
                idir,
                field_arr, inv_density_arr, prim_arr, qthermo_arr, prim_source,
                qzm, qzp,
                vbx, dt);

      set_reflecting_boundary_states(bx, 2, qzm, qzp);
#endif
    }
  }
}


#ifdef RADIATION
void
HydroSim::corner_ppm_rad_states(const Box& bx, const Box& vbx,
                           Array4<Real const> const& field_arr,
                           Array4<Real const> const& inv_density_arr,
                           Array4<Real const> const& prim_arr,
                           Array4<Real const> const& qthermo_arr,
                           Array4<Real const> const& prim_source,
                           Array4<Real> const& qxm,
                           Array4<Real> const& qxp,
#if AMREX_SPACEDIM >= 2
                           Array4<Real> const& qym,
                           Array4<Real> const& qyp,
#endif
#if AMREX_SPACEDIM == 3
                           Array4<Real> const& qzm,
                           Array4<Real> const& qzp,
#endif
#if AMREX_SPACEDIM < 3
                           Array4<Real const> const& logAreaRateX,
#endif
#if AMREX_SPACEDIM == 2
                           Array4<Real const> const& logAreaRateY,
#endif
                           const Real dt) {


  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {

    if (idir == 0) {

      trace_parabolic_radiative(bx,
                    idir,
                    field_arr, inv_density_arr, prim_arr, qthermo_arr, prim_source,
                    qxm, qxp,
#if AMREX_SPACEDIM <= 2
                    logAreaRateX,
#endif
                    vbx, dt);

      set_reflecting_boundary_states(bx, 0, qxm, qxp);

#if AMREX_SPACEDIM >= 2
    } else if (idir == 1) {
      trace_parabolic_radiative(bx,
                    idir,
                    field_arr, inv_density_arr, prim_arr, qthermo_arr, prim_source,
                    qym, qyp,
#if AMREX_SPACEDIM <= 2
                    logAreaRateY,
#endif
                    vbx, dt);

      set_reflecting_boundary_states(bx, 1, qym, qyp);
#endif


#if AMREX_SPACEDIM == 3
    } else {
      trace_parabolic_radiative(bx,
                    idir,
                    field_arr, inv_density_arr, prim_arr, qthermo_arr, prim_source,
                    qzm, qzp,
                    vbx, dt);

      set_reflecting_boundary_states(bx, 2, qzm, qzp);

#endif
    }
  }
}
#endif


void
HydroSim::corner_plm_states(const Box& bx, const Box& vbx,
                       Array4<Real const> const& field_arr,
                       Array4<Real const> const& inv_density_arr,
                       Array4<Real const> const& prim_arr,
                       Array4<Real const> const& qthermo_arr,
                       Array4<Real const> const& prim_source,
                       Array4<Real> const& qxm,
                       Array4<Real> const& qxp,
#if AMREX_SPACEDIM >= 2
                       Array4<Real> const& qym,
                       Array4<Real> const& qyp,
#endif
#if AMREX_SPACEDIM == 3
                       Array4<Real> const& qzm,
                       Array4<Real> const& qzp,
#endif
#if AMREX_SPACEDIM < 3
                       Array4<Real const> const& logAreaRateX,
#endif
#if AMREX_SPACEDIM == 2
                       Array4<Real const> const& logAreaRateY,
#endif
                       const Real dt) {



#ifdef RADIATION
#ifndef AMREX_USE_GPU
  amrex::Error("ppm_recon_kind <=0 is not supported in with radiation");
#endif
#endif

  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {


    if (idir == 0) {
      trace_piecewise_linear(bx, 0,
                field_arr, inv_density_arr, prim_arr, qthermo_arr,
                qxm, qxp,
#if AMREX_SPACEDIM < 3
                logAreaRateX,
#endif
                prim_source, vbx, dt);

      set_reflecting_boundary_states(bx, 0, qxm, qxp);

#if AMREX_SPACEDIM >= 2
    } else if (idir == 1) {
      trace_piecewise_linear(bx, 1,
                field_arr, inv_density_arr, prim_arr, qthermo_arr,
                qym, qyp,
#if AMREX_SPACEDIM < 3
                logAreaRateY,
#endif
                prim_source, vbx, dt);

      set_reflecting_boundary_states(bx, 1, qym, qyp);
#endif

#if AMREX_SPACEDIM == 3
    } else {
      trace_piecewise_linear(bx, 2,
                field_arr, inv_density_arr, prim_arr, qthermo_arr,
                qzm, qzp,
                prim_source, vbx, dt);

      set_reflecting_boundary_states(bx, 2, qzm, qzp);
#endif
    }
  }
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
