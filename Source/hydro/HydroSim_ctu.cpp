#include <HydroSim.H>
#include <HydroSim_util.H>
#include <advection_util.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

using namespace amrex;

void
HydroSim::cons_update_hydro(const Box& bx,
                     Array4<Real> const& U_new,
                     GpuArray<Array4<Real>, AMREX_SPACEDIM> const& flux,
                     GpuArray<Array4<Real const>, AMREX_SPACEDIM> const& qe,
                     const Real dt)
{
  auto geomdata = geom.data();

  amrex::ParallelFor(bx, NUM_STATE,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {
    Real volinv = 1.0 / geometry_util::volume({i, j, k}, geomdata);

    U_new(i,j,k,n) = U_new(i,j,k,n) + dt *
        ( flux[0](i,  j,k,n) * geometry_util::area({i,   j, k}, 0, geomdata)
        - flux[0](i+1,j,k,n) * geometry_util::area({i+1, j, k}, 0, geomdata)
#if AMREX_SPACEDIM >= 2
        + flux[1](i,j,  k,n) * geometry_util::area({i, j,   k}, 1, geomdata)
        - flux[1](i,j+1,k,n) * geometry_util::area({i, j+1, k}, 1, geomdata)
#endif
#if AMREX_SPACEDIM == 3
        + flux[2](i,j,k,  n) * geometry_util::area({i, j, k},   2, geomdata)
        - flux[2](i,j,k+1,n) * geometry_util::area({i, j, k+1}, 2, geomdata)
#endif
        ) * volinv;

    // Add the p div(u) source term to (rho e).
    if (n == UEINT) {

      Real pdu = (qe[0](i+1,j,k,IFCPRES) + qe[0](i,j,k,IFCPRES)) *
          (qe[0](i+1,j,k,IFCU) * geometry_util::area({i+1, j, k}, 0, geomdata) -
           qe[0](i,  j,k,IFCU) * geometry_util::area({i,   j, k}, 0, geomdata));

#if AMREX_SPACEDIM >= 2
      pdu += (qe[1](i,j+1,k,IFCPRES) + qe[1](i,j,k,IFCPRES)) *
          (qe[1](i,j+1,k,IFCV) * geometry_util::area({i, j+1, k}, 1, geomdata) -
           qe[1](i,j,  k,IFCV) * geometry_util::area({i, j,   k}, 1, geomdata));
#endif

#if AMREX_SPACEDIM == 3
      pdu += (qe[2](i,j,k+1,IFCPRES) + qe[2](i,j,k,IFCPRES)) *
          (qe[2](i,j,k+1,IFCW) * geometry_util::area({i, j, k+1}, 2, geomdata) -
           qe[2](i,j,k  ,IFCW) * geometry_util::area({i, j, k},   2, geomdata));
#endif

      pdu = 0.5 * pdu * volinv;

      U_new(i,j,k,n) = U_new(i,j,k,n) - dt * pdu;

    } else if (n == UMX && !momentum_flux_has_pressure(0, 0, geomdata.Coord())) {
        // Add gradp term to momentum equation -- only for axisymmetric
        // coords (and only for the radial flux).

        U_new(i,j,k,UMX) += - dt * (qe[0](i+1,j,k,IFCPRES) - qe[0](i,j,k,IFCPRES)) / geomdata.CellSize(0);

#if AMREX_SPACEDIM >= 2
    } else if (n == UMY && !momentum_flux_has_pressure(1, 1, geomdata.Coord())) {
        // Add gradp term to polar(theta) momentum equation for Spherical 2D geometry

        Real r = geomdata.ProbLo(0) + (static_cast<Real>(i) + 0.5_rt) * geomdata.CellSize(0);
        U_new(i,j,k,UMY) += - dt * (qe[1](i,j+1,k,IFCPRES) - qe[1](i,j,k,IFCPRES)) / (r * geomdata.CellSize(1));
#endif
    }
  });
}


void
HydroSim::ppm_interface_states(const Box& bx, const Box& vbx,
                       Array4<Real const> const& U_arr,
                       Array4<Real const> const& rho_inv_arr,
                       Array4<Real const> const& q_arr,
                       Array4<Real const> const& qaux_arr,
                       Array4<Real const> const& srcQ,
                       Array<Array4<Real>, AMREX_SPACEDIM> const& qm,
                       Array<Array4<Real>, AMREX_SPACEDIM> const& qp,
                       Array<Array4<Real const>, AMREX_SPACEDIM> const& dloga,
                       const Real dt) {

  // Compute the normal interface states by reconstructing
  // the primitive variables using the piecewise parabolic method
  // and doing characteristic tracing.  We do not apply the
  // transverse terms here.

  TraceFields fields{U_arr, rho_inv_arr, q_arr, qaux_arr, srcQ, {}};

  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {

    fields.dloga = dloga[idir];
    trace_ppm(bx, idir, fields, qm[idir], qp[idir], vbx, dt);

    mirror_states_at_walls(bx, idir, qm[idir], qp[idir]);
  }
}


#ifdef RADIATION
void
HydroSim::ppm_rad_interface_states(const Box& bx, const Box& vbx,
                           Array4<Real const> const& U_arr,
                           Array4<Real const> const& rho_inv_arr,
                           Array4<Real const> const& q_arr,
                           Array4<Real const> const& qaux_arr,
                           Array4<Real const> const& srcQ,
                           Array<Array4<Real>, AMREX_SPACEDIM> const& qm,
                           Array<Array4<Real>, AMREX_SPACEDIM> const& qp,
                           Array<Array4<Real const>, AMREX_SPACEDIM> const& dloga,
                           const Real dt) {


  TraceFields fields{U_arr, rho_inv_arr, q_arr, qaux_arr, srcQ, {}};

  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {

    fields.dloga = dloga[idir];
    trace_ppm_rad(bx, idir, fields, qm[idir], qp[idir], vbx, dt);

    mirror_states_at_walls(bx, idir, qm[idir], qp[idir]);
  }
}
#endif


void
HydroSim::plm_interface_states(const Box& bx, const Box& vbx,
                       Array4<Real const> const& U_arr,
                       Array4<Real const> const& rho_inv_arr,
                       Array4<Real const> const& q_arr,
                       Array4<Real const> const& qaux_arr,
                       Array4<Real const> const& srcQ,
                       Array<Array4<Real>, AMREX_SPACEDIM> const& qm,
                       Array<Array4<Real>, AMREX_SPACEDIM> const& qp,
                       Array<Array4<Real const>, AMREX_SPACEDIM> const& dloga,
                       const Real dt) {

  // Compute the normal interface states by reconstructing
  // the primitive variables using piecewise linear slopes and doing
  // characteristic tracing.  We do not apply the transverse terms here.
  //
  // .. todo::
  //    we can get rid of the the different temporary q Godunov
  //    state arrays
  //


#ifdef RADIATION
#ifndef AMREX_USE_GPU
  amrex::Error("ppm_type <=0 is not supported in with radiation");
#endif
#endif

  TraceFields fields{U_arr, rho_inv_arr, q_arr, qaux_arr, srcQ, {}};

  // Compute all slopes
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {

    // compute the interface states

    fields.dloga = dloga[idir];
    trace_plm(bx, idir, fields, qm[idir], qp[idir], vbx, dt);

    mirror_states_at_walls(bx, idir, qm[idir], qp[idir]);
  }
}


void
hydro::add_sdc_source_to_states(const Box& bx, const int idir, const Real dt,
                                Array4<Real> const& qleft,
                                Array4<Real> const& qright,
                                Array4<Real const> const& sdc_src)
{

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

        // of the state variables, only pressure, rhoe, and
        // composition have source terms

        Real p_old = qleft(i,j,k,QPRES);
        Real rhoe_old = qleft(i,j,k,QRHOEI);

        if (idir == 0) {
            qleft(i,j,k,QPRES) += 0.5 * dt * sdc_src(i-1,j,k,QPRES);
            qleft(i,j,k,QRHOEI) += 0.5 * dt * sdc_src(i-1,j,k,QRHOEI);
        } else if (idir == 1) {
            qleft(i,j,k,QPRES) += 0.5 * dt * sdc_src(i,j-1,k,QPRES);
            qleft(i,j,k,QRHOEI) += 0.5 * dt * sdc_src(i,j-1,k,QRHOEI);
        } else {
            qleft(i,j,k,QPRES) += 0.5 * dt * sdc_src(i,j,k-1,QPRES);
            qleft(i,j,k,QRHOEI) += 0.5 * dt * sdc_src(i,j,k-1,QRHOEI);
        }

        if (qleft(i,j,k,QPRES) < small_pres || qleft(i,j,k,QRHOEI) < small_dens * small_ener) {
            qleft(i,j,k,QPRES) = p_old;
            qleft(i,j,k,QRHOEI) = rhoe_old;
        }

        p_old = qright(i,j,k,QPRES);
        rhoe_old = qright(i,j,k,QRHOEI);

        qright(i,j,k,QPRES) += 0.5 * dt * sdc_src(i,j,k,QPRES);
        qright(i,j,k,QRHOEI) += 0.5 * dt * sdc_src(i,j,k,QRHOEI);

        if (qright(i,j,k,QPRES) < small_pres || qright(i,j,k,QRHOEI) < small_dens * small_ener) {
            qright(i,j,k,QPRES) = p_old;
            qright(i,j,k,QRHOEI) = rhoe_old;
        }


        for (int ipassive = 0; ipassive < npassive; ipassive++) {

            int n = prim_passive_index(ipassive);

            if (idir == 0) {
                qleft(i,j,k,n) += 0.5 * dt * sdc_src(i-1,j,k,n);
            } else if (idir == 1) {
                qleft(i,j,k,n) += 0.5 * dt * sdc_src(i,j-1,k,n);
            } else {
                qleft(i,j,k,n) += 0.5 * dt * sdc_src(i,j,k-1,n);
            }
            qright(i,j,k,n) += 0.5 * dt * sdc_src(i,j,k,n);

            if (n >= QSPEC && n <= QSPEC-1+NumSpec) {
                // mass fractions should be in [0, 1]
                qleft(i,j,k,n) = amrex::Clamp(qleft(i,j,k,n), 0.0_rt, 1.0_rt);
                qright(i,j,k,n) = amrex::Clamp(qright(i,j,k,n), 0.0_rt, 1.0_rt);
            }
        }

    });

}
