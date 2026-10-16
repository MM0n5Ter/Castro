#include <HydroSim.H>
#include <HydroSim_util.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#include <cmath>

#include <slope.H>
#include <reconstruction.H>
#include <flatten.H>

using namespace amrex;
using namespace reconstruction;

void
HydroSim::trace_plm(const Box& bx, const int idir,
                  TraceFields const& fields,
                  Array4<Real> const& qm,
                  Array4<Real> const& qp,
                  const Box& vbx,
                  const Real dt) {

  // here, bx is the box we loop over -- this can include ghost cells
  // vbx is the valid box (no ghost cells)

  auto const& U_arr = fields.U;
  auto const& rho_inv_arr = fields.rho_inv;
  auto const& q_arr = fields.q;
  auto const& qaux_arr = fields.qaux;
  auto const& srcQ = fields.srcQ;
  [[maybe_unused]] auto const& dloga = fields.dloga;

  const auto dx = geom.CellSizeArray();
  const auto problo = geom.ProbLoArray();
  const int coord = geom.Coord();

  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  bool lo_symm = lo_bc[idir] == amrex::PhysBCType::symmetry;
  bool hi_symm = hi_bc[idir] == amrex::PhysBCType::symmetry;

  const auto domlo = geom.Domain().loVect3d();
  const auto domhi = geom.Domain().hiVect3d();

  auto vlo = vbx.loVect3d();
  auto vhi = vbx.hiVect3d();

#ifndef AMREX_USE_GPU
  if (ppm_type != 0) {
    std::cout << "Oops -- shouldn't be in tracexy with ppm_type != 0" << std::endl;
    amrex::Error("Error:: trace_3d.f90 :: tracexy");
  }
#endif

  int QUN;
  int QUT;
  int QUTT;

  if (idir == 0) {
    QUN = QU;
    QUT = QV;
    QUTT = QW;

  } else if (idir == 1) {
    QUN = QV;
    QUT = QW;
    QUTT = QU;

  } else {
    QUN = QW;
    QUT = QU;
    QUTT = QV;
  }

  Real lsmall_dens = small_dens;
  Real lsmall_pres = small_pres;

  constexpr int NEVEC = 6;
  constexpr int IEVEC_RHO = 0;
  constexpr int IEVEC_UN = 1;
  constexpr int IEVEC_UT = 2;
  constexpr int IEVEC_UTT = 3;
  constexpr int IEVEC_P = 4;
  constexpr int IEVEC_RE = 5;

  int cvars[NEVEC];
  cvars[IEVEC_RHO] = QRHO;
  cvars[IEVEC_UN] = QUN;
  cvars[IEVEC_UT] = QUT;
  cvars[IEVEC_UTT] = QUTT;
  cvars[IEVEC_P] = QPRES;
  cvars[IEVEC_RE] = QRHOEI;

  // Compute left and right traced states

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real dtdL = dt / dx[idir];
    Real dL = dx[idir];

    // Want dt/(rdtheta) instead of dt/dtheta for 2d Spherical
    if (coord == 2 && idir == 1) {
        Real r = problo[0] + static_cast<Real>(i + 0.5_rt) * dx[0];
        dL = r * dx[1];
        dtdL = dt / dL;
    }

    bool lo_bc_test = lo_symm && ((idir == 0 && i == domlo[0]) ||
                                  (idir == 1 && j == domlo[1]) ||
                                  (idir == 2 && k == domlo[2]));

    bool hi_bc_test = hi_symm && ((idir == 0 && i == domhi[0]) ||
                                  (idir == 1 && j == domhi[1]) ||
                                  (idir == 2 && k == domhi[2]));

    const ReflectEdge zone_edge = reflect_edge(lo_bc_test, hi_bc_test);

    Real cc = qaux_arr(i,j,k,QC);
    Real csq = cc*cc;
    Real rho = q_arr(i,j,k,QRHO);

    Real un = q_arr(i,j,k,QUN);
    Real ut = q_arr(i,j,k,QUT);
    Real utt = q_arr(i,j,k,QUTT);

    Real p = q_arr(i,j,k,QPRES);
    Real rhoe = q_arr(i,j,k,QRHOEI);
    Real enth = (rhoe+p)/(rho*csq);

    Real dq[NEVEC];
    ZoneStencil s;

    Real flat = 1.0;

    if (hydrosim::first_order_hydro) {
        flat = 0.0;
    }
    else if (hydrosim::use_flattening) {
        flat = hydro::flatten(i, j, k, q_arr, QPRES);

#ifdef RADIATION
        flat *= hydro::flatten(i, j, k, q_arr, QPTOTAL);

        if (radiation::flatten_pp_threshold > 0.0) {
            Real vel_sum_m = q_arr(i-1,j,k,QU);
            Real vel_sum_p = q_arr(i+1,j,k,QU);
#if AMREX_SPACEDIM >= 2
            vel_sum_m += q_arr(i,j-1,k,QV);
            vel_sum_p += q_arr(i,j+1,k,QV);
#endif
#if AMREX_SPACEDIM == 3
            vel_sum_m += q_arr(i,j,k-1,QW);
            vel_sum_p += q_arr(i,j,k+1,QW);
#endif
            if (vel_sum_m > vel_sum_p) {
                if (q_arr(i,j,k,QPRES) < radiation::flatten_pp_threshold * q_arr(i,j,k,QPTOTAL)) {
                    flat = 0.0;
                }
            }
        }
#endif
    }

    for (int n = 0; n < NEVEC; n++) {
      int v = cvars[n];

      s = load_stencil(q_arr, idir, i, j, k, v);

      bool vtest = v == QUN;
      dq[n] = uslope(s, flat, vtest ? zone_edge : ReflectEdge::None);
    }

    // are we doing well-balanced?
    if (use_pslope == 1) {

      HydrostaticStencils hse;
      hse.rho = load_stencil(q_arr, idir, i, j, k, QRHO);
      hse.p = load_stencil(q_arr, idir, i, j, k, QPRES);
      hse.src = load_stencil(srcQ, idir, i, j, k, QUN);

      Real dp = dq[IEVEC_P];
      pslope(hse, flat, zone_edge, dL, dp);
      dq[IEVEC_P] = dp;

    }

    Real alpham = 0.5_rt*(dq[IEVEC_P]/(rho*cc) - dq[IEVEC_UN])*(rho/cc);
    Real alphap = 0.5_rt*(dq[IEVEC_P]/(rho*cc) + dq[IEVEC_UN])*(rho/cc);
    Real alpha0r = dq[IEVEC_RHO] - dq[IEVEC_P]/csq;
    Real alpha0e = dq[IEVEC_RE] - dq[IEVEC_P]*enth;
    Real alpha0ut = dq[IEVEC_UT];
    Real alpha0utt = dq[IEVEC_UTT];

    GpuArray<Real, 3> e;
    e[0] = un - cc;
    e[1] = un;
    e[2] = un + cc;

    // construct the right state on the i interface

    Real ref_fac = 0.5_rt*(1.0_rt + dtdL*amrex::min(e[0], 0.0_rt));
    Real rho_ref = rho - ref_fac*dq[IEVEC_RHO];
    Real un_ref = un - ref_fac*dq[IEVEC_UN];
    Real ut_ref = ut - ref_fac*dq[IEVEC_UT];
    Real utt_ref = utt - ref_fac*dq[IEVEC_UTT];
    Real p_ref = p - ref_fac*dq[IEVEC_P];
    Real rhoe_ref = rhoe - ref_fac*dq[IEVEC_RE];

    // this is -(1/2) ( 1 + dt/dx lambda) (l . dq) r
    Real trace_fac0 = 0.0_rt; //  FOURTH*dtdx*(e(1) - e(1))*(1.0_rt - sign(1.0_rt, e(1)))
    Real trace_fac1 = 0.25_rt*dtdL*(e[0] - e[1])*(1.0_rt - std::copysign(1.0_rt, e[1]));
    Real trace_fac2 = 0.25_rt*dtdL*(e[0] - e[2])*(1.0_rt - std::copysign(1.0_rt, e[2]));

    Real ap_right = trace_fac2*alphap;
    Real am_right = trace_fac0*alpham;

    Real az_rho_right = trace_fac1*alpha0r;
    Real az_e_right = trace_fac1*alpha0e;
    Real az_ut_right = trace_fac1*alpha0ut;
    Real az_utt_right = trace_fac1*alpha0utt;

    if ((idir == 0 && i >= vlo[0]) ||
        (idir == 1 && j >= vlo[1]) ||
        (idir == 2 && k >= vlo[2])) {

      qp(i,j,k,QRHO) = amrex::max(lsmall_dens, rho_ref + ap_right + am_right + az_rho_right);
      qp(i,j,k,QUN) = un_ref + (ap_right - am_right)*cc/rho;
      qp(i,j,k,QUT) = ut_ref + az_ut_right;
      qp(i,j,k,QUTT) = utt_ref + az_utt_right;
      qp(i,j,k,QPRES) = amrex::max(lsmall_pres, p_ref + (ap_right + am_right)*csq);
      qp(i,j,k,QRHOEI) = rhoe_ref + (ap_right + am_right)*enth*csq + az_e_right;

      // add the source terms
      qp(i,j,k,QRHO  ) += 0.5_rt*dt*srcQ(i,j,k,QRHO);
      qp(i,j,k,QRHO  ) = amrex::max(lsmall_dens, qp(i,j,k,QRHO));
      qp(i,j,k,QUN   ) += 0.5_rt*dt*srcQ(i,j,k,QUN);
      qp(i,j,k,QUT   ) += 0.5_rt*dt*srcQ(i,j,k,QUT);
      qp(i,j,k,QUTT  ) += 0.5_rt*dt*srcQ(i,j,k,QUTT);
      qp(i,j,k,QRHOEI) += 0.5_rt*dt*srcQ(i,j,k,QRHOEI);
      qp(i,j,k,QPRES ) += 0.5_rt*dt*srcQ(i,j,k,QPRES);

    }

    // now construct the left state on the i+1 interface

    ref_fac = 0.5_rt*(1.0_rt - dtdL*amrex::max(e[2], 0.0_rt));
    rho_ref = rho + ref_fac*dq[IEVEC_RHO];
    un_ref = un + ref_fac*dq[IEVEC_UN];
    ut_ref = ut + ref_fac*dq[IEVEC_UT];
    utt_ref = utt + ref_fac*dq[IEVEC_UTT];
    p_ref = p + ref_fac*dq[IEVEC_P];
    rhoe_ref = rhoe + ref_fac*dq[IEVEC_RE];

    trace_fac0 = 0.25_rt*dtdL*(e[2] - e[0])*(1.0_rt + std::copysign(1.0_rt, e[0]));
    trace_fac1 = 0.25_rt*dtdL*(e[2] - e[1])*(1.0_rt + std::copysign(1.0_rt, e[1]));
    trace_fac2 = 0.0_rt; //  FOURTH*dtdx*(e(3) - e(3))*(1.0_rt + sign(1.0_rt, e(3)))

    Real ap_left = trace_fac2*alphap;
    Real am_left = trace_fac0*alpham;

    Real az_rho_left = trace_fac1*alpha0r;
    Real az_e_left = trace_fac1*alpha0e;
    Real az_ut_left = trace_fac1*alpha0ut;
    Real az_utt_left = trace_fac1*alpha0utt;

    if (idir == 0 && i <= vhi[0]) {
      qm(i+1,j,k,QRHO) = amrex::max(lsmall_dens, rho_ref + ap_left + am_left + az_rho_left);
      qm(i+1,j,k,QUN) = un_ref + (ap_left - am_left)*cc/rho;
      qm(i+1,j,k,QUT) = ut_ref + az_ut_left;
      qm(i+1,j,k,QUTT) = utt_ref + az_utt_left;
      qm(i+1,j,k,QPRES) = amrex::max(lsmall_pres, p_ref + (ap_left + am_left)*csq);
      qm(i+1,j,k,QRHOEI) = rhoe_ref + (ap_left + am_left)*enth*csq + az_e_left;

      // add the source terms
      qm(i+1,j,k,QRHO) = amrex::max(lsmall_dens, qm(i+1,j,k,QRHO) + 0.5_rt*dt*srcQ(i,j,k,QRHO));
      qm(i+1,j,k,QUN) += 0.5_rt*dt*srcQ(i,j,k,QUN);
      qm(i+1,j,k,QUT) += 0.5_rt*dt*srcQ(i,j,k,QUT);
      qm(i+1,j,k,QUTT) += 0.5_rt*dt*srcQ(i,j,k,QUTT);
      qm(i+1,j,k,QRHOEI) += 0.5_rt*dt*srcQ(i,j,k,QRHOEI);
      qm(i+1,j,k,QPRES) += 0.5_rt*dt*srcQ(i,j,k,QPRES);

    } else if (idir == 1 && j <= vhi[1]) {
      qm(i,j+1,k,QRHO) = amrex::max(lsmall_dens, rho_ref + ap_left + am_left + az_rho_left);
      qm(i,j+1,k,QUN) = un_ref + (ap_left - am_left)*cc/rho;
      qm(i,j+1,k,QUT) = ut_ref + az_ut_left;
      qm(i,j+1,k,QUTT) = utt_ref + az_utt_left;
      qm(i,j+1,k,QPRES) = max(lsmall_pres, p_ref + (ap_left + am_left)*csq);
      qm(i,j+1,k,QRHOEI) = rhoe_ref + (ap_left + am_left)*enth*csq + az_e_left;

      // add the source terms
      qm(i,j+1,k,QRHO) = amrex::max(lsmall_dens, qm(i,j+1,k,QRHO) + 0.5_rt*dt*srcQ(i,j,k,QRHO));
      qm(i,j+1,k,QUN) += 0.5_rt*dt*srcQ(i,j,k,QUN);
      qm(i,j+1,k,QUT) += 0.5_rt*dt*srcQ(i,j,k,QUT);
      qm(i,j+1,k,QUTT) += 0.5_rt*dt*srcQ(i,j,k,QUTT);
      qm(i,j+1,k,QRHOEI) += 0.5_rt*dt*srcQ(i,j,k,QRHOEI);
      qm(i,j+1,k,QPRES) += 0.5_rt*dt*srcQ(i,j,k,QPRES);

    } else if (idir == 2 && k <= vhi[2]) {
      qm(i,j,k+1,QRHO) = amrex::max(lsmall_dens, rho_ref + ap_left + am_left + az_rho_left);
      qm(i,j,k+1,QUN) = un_ref + (ap_left - am_left)*cc/rho;
      qm(i,j,k+1,QUT) = ut_ref + az_ut_left;
      qm(i,j,k+1,QUTT) = utt_ref + az_utt_left;
      qm(i,j,k+1,QPRES) = amrex::max(lsmall_pres, p_ref + (ap_left + am_left)*csq);
      qm(i,j,k+1,QRHOEI) = rhoe_ref + (ap_left + am_left)*enth*csq + az_e_left;

      // add the source terms
      qm(i,j,k+1,QRHO) = amrex::max(lsmall_dens, qm(i,j,k+1,QRHO) + 0.5_rt*dt*srcQ(i,j,k,QRHO));
      qm(i,j,k+1,QUN) += 0.5_rt*dt*srcQ(i,j,k,QUN);
      qm(i,j,k+1,QUT) += 0.5_rt*dt*srcQ(i,j,k,QUT);
      qm(i,j,k+1,QUTT) += 0.5_rt*dt*srcQ(i,j,k,QUTT);
      qm(i,j,k+1,QRHOEI) += 0.5_rt*dt*srcQ(i,j,k,QRHOEI);
      qm(i,j,k+1,QPRES) += 0.5_rt*dt*srcQ(i,j,k,QPRES);

    }

#if (AMREX_SPACEDIM < 3)
    // geometry source terms
    // these only apply to the x-states for cylindrical and spherical
    // or y-states for spherical

    if (dloga(i,j,k) != 0.0_rt) {
      Real courn = dtdL*(cc + std::abs(un));
      Real eta = (1.0_rt-courn)/(cc*dt*std::abs(dloga(i,j,k)));
      Real dlogatmp = amrex::min(eta, 1.0_rt)*dloga(i,j,k);
      Real geo_src_rho = -0.5_rt*dt*rho*dlogatmp*un;
      Real geo_src_p = geo_src_rho*csq;
      Real source = geo_src_p*enth;

      if (idir == 0 && i <= vhi[0]) {
        qm(i+1,j,k,QRHO) += geo_src_rho;
        qm(i+1,j,k,QRHO) = amrex::max(qm(i+1,j,k,QRHO), lsmall_dens);
        qm(i+1,j,k,QPRES) += geo_src_p;
        qm(i+1,j,k,QRHOEI) += source;
      }

      if (idir == 1 && j <= vhi[1]) {
        qm(i,j+1,k,QRHO) += geo_src_rho;
        qm(i,j+1,k,QRHO) = amrex::max(qm(i,j+1,k,QRHO), lsmall_dens);
        qm(i,j+1,k,QPRES) += geo_src_p;
        qm(i,j+1,k,QRHOEI) += source;
      }

      if ((idir == 0 && i >= vlo[0]) ||
          (idir == 1 && j >= vlo[1])) {
        qp(i,j,k,QRHO) += geo_src_rho;
        qp(i,j,k,QRHO) = amrex::max(qp(i,j,k,QRHO), lsmall_dens);
        qp(i,j,k,QPRES) += geo_src_p;
        qp(i,j,k,QRHOEI) += source;
      }

    }

#endif

    for (int ipassive = 0; ipassive < npassive; ipassive++) {
      const int nc = cons_passive_index(ipassive);
      const int n = prim_passive_index(ipassive);

      // get the slope

      s = load_passive_stencil(U_arr, rho_inv_arr, idir, i, j, k, nc);
      Real dX = uslope(s, flat, ReflectEdge::None);

      // Right state
      if ((idir == 0 && i >= vlo[0]) ||
          (idir == 1 && j >= vlo[1]) ||
          (idir == 2 && k >= vlo[2])) {

        Real spzero = un >= 0.0_rt ? -1.0_rt : un*dtdL;
        qp(i,j,k,n) = s[i0] + 0.5_rt*(-1.0_rt - spzero)*dX;
      }

      // Left state
      Real spzero = un >= 0.0_rt ? un*dtdL : 1.0_rt;
      Real adv_corr_left = 0.5_rt*(1.0_rt - spzero )*dX;

      if (idir == 0 && i <= vhi[0]) {
        qm(i+1,j,k,n) = s[i0] + adv_corr_left;
      } else if (idir == 1 && j <= vhi[1]) {
        qm(i,j+1,k,n) = s[i0] + adv_corr_left;
      } else if (idir == 2 && k <= vhi[2]) {
        qm(i,j,k+1,n) = s[i0] + adv_corr_left;
      }

    }

  });
}
