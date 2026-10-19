#include <HydroSim.H>
#include <HydroSim_helpers.H>

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
HydroSim::trace_piecewise_linear(const Box& bx, const int idir,
                  Array4<Real const> const& field_arr,
                  Array4<Real const> const& inv_density_arr,
                  Array4<Real const> const& prim_arr,
                  Array4<Real const> const& qthermo_arr,
                  Array4<Real> const& qm,
                  Array4<Real> const& qp,
#if AMREX_SPACEDIM < 3
                  Array4<Real const> const& dloga,
#endif
                  Array4<Real const> const& prim_source,
                  const Box& vbx,
                  const Real dt) {


  const auto dx = geom.CellSizeArray();
  const auto problo = geom.ProbLoArray();
  const int coord = geom.Coord();

  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  bool lo_symmetric = lo_bc[idir] == amrex::PhysBCType::symmetry;
  bool hi_symmetric = hi_bc[idir] == amrex::PhysBCType::symmetry;

  const auto domlo = geom.Domain().loVect3d();
  const auto domhi = geom.Domain().hiVect3d();

  auto vlo = vbx.loVect3d();
  auto vhi = vbx.hiVect3d();

#ifndef AMREX_USE_GPU
  if (ppm_recon_kind != 0) {
    std::cout << "Oops -- shouldn't be in tracexy with ppm_recon_kind != 0" << std::endl;
    amrex::Error("Error:: trace_3d.f90 :: tracexy");
  }
#endif

  int QUN;
  int QUT;
  int QTRANSV2;

  if (idir == 0) {
    QUN = QU;
    QUT = QV;
    QTRANSV2 = QW;

  } else if (idir == 1) {
    QUN = QV;
    QUT = QW;
    QTRANSV2 = QU;

  } else {
    QUN = QW;
    QUT = QU;
    QTRANSV2 = QV;
  }

  Real ldensity_floor = small_dens;
  Real local_pressure_floor = pressure_cutoff;

  constexpr int NUM_EIGVEC = 6;
  constexpr int EIGVEC_RHO = 0;
  constexpr int EIGVEC_NORMAL = 1;
  constexpr int EIGVEC_TRANSV1 = 2;
  constexpr int EIGVEC_TRANSV2 = 3;
  constexpr int EIGVEC_PRESSURE = 4;
  constexpr int EIGVEC_ENERGY = 5;

  int char_var_map[NUM_EIGVEC];
  char_var_map[EIGVEC_RHO] = QRHO;
  char_var_map[EIGVEC_NORMAL] = QUN;
  char_var_map[EIGVEC_TRANSV1] = QUT;
  char_var_map[EIGVEC_TRANSV2] = QTRANSV2;
  char_var_map[EIGVEC_PRESSURE] = QPRES;
  char_var_map[EIGVEC_ENERGY] = QEINTDENS;


  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real dt_over_dx = dt / dx[idir];
    Real dL = dx[idir];

    if (coord == 2 && idir == 1) {
        Real r = problo[0] + static_cast<Real>(i + 0.5_rt) * dx[0];
        dL = r * dx[1];
        dt_over_dx = dt / dL;
    }

    bool at_lo_symmetry = lo_symmetric && ((idir == 0 && i == domlo[0]) ||
                                  (idir == 1 && j == domlo[1]) ||
                                  (idir == 2 && k == domlo[2]));

    bool at_hi_symmetry = hi_symmetric && ((idir == 0 && i == domhi[0]) ||
                                  (idir == 1 && j == domhi[1]) ||
                                  (idir == 2 && k == domhi[2]));

    Real cc = qthermo_arr(i,j,k,QC);
    Real csq = cc*cc;
    Real rho = prim_arr(i,j,k,QRHO);

    Real un = prim_arr(i,j,k,QUN);
    Real ut = prim_arr(i,j,k,QUT);
    Real utt = prim_arr(i,j,k,QTRANSV2);

    Real p = prim_arr(i,j,k,QPRES);
    Real rhoe = prim_arr(i,j,k,QEINTDENS);
    Real enth = (rhoe+p)/(rho*csq);

    Real dq[NUM_EIGVEC];
    Real s[nslp];

    Real flat = 1.0;

    if (hydrosim::first_order_scheme) {
        flat = 0.0;
    }
    else if (hydrosim::use_shock_flatten) {
        flat = hydro::flatten(i, j, k, prim_arr, QPRES);

#ifdef RADIATION
        flat *= hydro::flatten(i, j, k, prim_arr, QPTOTAL);

        if (radiation::radiation_flatten_cutoff > 0.0) {
            Real speed_sum_minus = prim_arr(i-1,j,k,QU);
            Real speed_sum_plus = prim_arr(i+1,j,k,QU);
#if AMREX_SPACEDIM >= 2
            speed_sum_minus += prim_arr(i,j-1,k,QV);
            speed_sum_plus += prim_arr(i,j+1,k,QV);
#endif
#if AMREX_SPACEDIM == 3
            speed_sum_minus += prim_arr(i,j,k-1,QW);
            speed_sum_plus += prim_arr(i,j,k+1,QW);
#endif
            if (speed_sum_minus > speed_sum_plus) {
                if (prim_arr(i,j,k,QPRES) < radiation::radiation_flatten_cutoff * prim_arr(i,j,k,QPTOTAL)) {
                    flat = 0.0;
                }
            }
        }
#endif
    }

    for (int n = 0; n < NUM_EIGVEC; n++) {
      int v = char_var_map[n];

      gather_stencil(prim_arr, idir, i, j, k, v, s);

      bool is_normal_velocity = v == QUN;
      dq[n] = limited_slope(s, flat, at_lo_symmetry && is_normal_velocity, at_hi_symmetry && is_normal_velocity);
    }

    if (use_pressure_slope == 1) {

      Real rho_stencil[nslp];
      Real src[nslp];

      gather_stencil(prim_arr, idir, i, j, k, QPRES, s);
      gather_stencil(prim_arr, idir, i, j, k, QRHO, rho_stencil);
      gather_stencil(prim_source, idir, i, j, k, QUN, src);

      Real dp = dq[EIGVEC_PRESSURE];
      pressure_slope(rho_stencil, s, src, flat, at_lo_symmetry, at_hi_symmetry, dL, dp);
      dq[EIGVEC_PRESSURE] = dp;

    }

    Real acoustic_amp_minus = 0.5_rt*(dq[EIGVEC_PRESSURE]/(rho*cc) - dq[EIGVEC_NORMAL])*(rho/cc);
    Real acoustic_amp_plus = 0.5_rt*(dq[EIGVEC_PRESSURE]/(rho*cc) + dq[EIGVEC_NORMAL])*(rho/cc);
    Real entropy_amp_density = dq[EIGVEC_RHO] - dq[EIGVEC_PRESSURE]/csq;
    Real entropy_amp_energy = dq[EIGVEC_ENERGY] - dq[EIGVEC_PRESSURE]*enth;
    Real entropy_amp_transv1 = dq[EIGVEC_TRANSV1];
    Real entropy_amp_transv2 = dq[EIGVEC_TRANSV2];

    Real e[3];
    e[0] = un - cc;
    e[1] = un;
    e[2] = un + cc;


    Real ref_fac = 0.5_rt*(1.0_rt + dt_over_dx*amrex::min(e[0], 0.0_rt));
    Real ref_density = rho - ref_fac*dq[EIGVEC_RHO];
    Real ref_normal_velocity = un - ref_fac*dq[EIGVEC_NORMAL];
    Real ref_transv1_velocity = ut - ref_fac*dq[EIGVEC_TRANSV1];
    Real ref_transv2_velocity = utt - ref_fac*dq[EIGVEC_TRANSV2];
    Real ref_pressure = p - ref_fac*dq[EIGVEC_PRESSURE];
    Real ref_energy_density = rhoe - ref_fac*dq[EIGVEC_ENERGY];

    Real trace_coef_minus = 0.0_rt;
    Real trace_coef_entropy = 0.25_rt*dt_over_dx*(e[0] - e[1])*(1.0_rt - std::copysign(1.0_rt, e[1]));
    Real trace_coef_plus = 0.25_rt*dt_over_dx*(e[0] - e[2])*(1.0_rt - std::copysign(1.0_rt, e[2]));

    Real acoustic_plus_right = trace_coef_plus*acoustic_amp_plus;
    Real acoustic_minus_right = trace_coef_minus*acoustic_amp_minus;

    Real entropy_density_right = trace_coef_entropy*entropy_amp_density;
    Real entropy_energy_right = trace_coef_entropy*entropy_amp_energy;
    Real entropy_transv1_right = trace_coef_entropy*entropy_amp_transv1;
    Real entropy_transv2_right = trace_coef_entropy*entropy_amp_transv2;

    if ((idir == 0 && i >= vlo[0]) ||
        (idir == 1 && j >= vlo[1]) ||
        (idir == 2 && k >= vlo[2])) {

      qp(i,j,k,QRHO) = amrex::max(ldensity_floor, ref_density + acoustic_plus_right + acoustic_minus_right + entropy_density_right);
      qp(i,j,k,QUN) = ref_normal_velocity + (acoustic_plus_right - acoustic_minus_right)*cc/rho;
      qp(i,j,k,QUT) = ref_transv1_velocity + entropy_transv1_right;
      qp(i,j,k,QTRANSV2) = ref_transv2_velocity + entropy_transv2_right;
      qp(i,j,k,QPRES) = amrex::max(local_pressure_floor, ref_pressure + (acoustic_plus_right + acoustic_minus_right)*csq);
      qp(i,j,k,QEINTDENS) = ref_energy_density + (acoustic_plus_right + acoustic_minus_right)*enth*csq + entropy_energy_right;

      qp(i,j,k,QRHO  ) += 0.5_rt*dt*prim_source(i,j,k,QRHO);
      qp(i,j,k,QRHO  ) = amrex::max(ldensity_floor, qp(i,j,k,QRHO));
      qp(i,j,k,QUN   ) += 0.5_rt*dt*prim_source(i,j,k,QUN);
      qp(i,j,k,QUT   ) += 0.5_rt*dt*prim_source(i,j,k,QUT);
      qp(i,j,k,QTRANSV2  ) += 0.5_rt*dt*prim_source(i,j,k,QTRANSV2);
      qp(i,j,k,QEINTDENS) += 0.5_rt*dt*prim_source(i,j,k,QEINTDENS);
      qp(i,j,k,QPRES ) += 0.5_rt*dt*prim_source(i,j,k,QPRES);

    }


    ref_fac = 0.5_rt*(1.0_rt - dt_over_dx*amrex::max(e[2], 0.0_rt));
    ref_density = rho + ref_fac*dq[EIGVEC_RHO];
    ref_normal_velocity = un + ref_fac*dq[EIGVEC_NORMAL];
    ref_transv1_velocity = ut + ref_fac*dq[EIGVEC_TRANSV1];
    ref_transv2_velocity = utt + ref_fac*dq[EIGVEC_TRANSV2];
    ref_pressure = p + ref_fac*dq[EIGVEC_PRESSURE];
    ref_energy_density = rhoe + ref_fac*dq[EIGVEC_ENERGY];

    trace_coef_minus = 0.25_rt*dt_over_dx*(e[2] - e[0])*(1.0_rt + std::copysign(1.0_rt, e[0]));
    trace_coef_entropy = 0.25_rt*dt_over_dx*(e[2] - e[1])*(1.0_rt + std::copysign(1.0_rt, e[1]));
    trace_coef_plus = 0.0_rt;

    Real acoustic_plus_left = trace_coef_plus*acoustic_amp_plus;
    Real acoustic_minus_left = trace_coef_minus*acoustic_amp_minus;

    Real entropy_density_left = trace_coef_entropy*entropy_amp_density;
    Real entropy_energy_left = trace_coef_entropy*entropy_amp_energy;
    Real entropy_transv1_left = trace_coef_entropy*entropy_amp_transv1;
    Real entropy_transv2_left = trace_coef_entropy*entropy_amp_transv2;

    if (idir == 0 && i <= vhi[0]) {
      qm(i+1,j,k,QRHO) = amrex::max(ldensity_floor, ref_density + acoustic_plus_left + acoustic_minus_left + entropy_density_left);
      qm(i+1,j,k,QUN) = ref_normal_velocity + (acoustic_plus_left - acoustic_minus_left)*cc/rho;
      qm(i+1,j,k,QUT) = ref_transv1_velocity + entropy_transv1_left;
      qm(i+1,j,k,QTRANSV2) = ref_transv2_velocity + entropy_transv2_left;
      qm(i+1,j,k,QPRES) = amrex::max(local_pressure_floor, ref_pressure + (acoustic_plus_left + acoustic_minus_left)*csq);
      qm(i+1,j,k,QEINTDENS) = ref_energy_density + (acoustic_plus_left + acoustic_minus_left)*enth*csq + entropy_energy_left;

      qm(i+1,j,k,QRHO) = amrex::max(ldensity_floor, qm(i+1,j,k,QRHO) + 0.5_rt*dt*prim_source(i,j,k,QRHO));
      qm(i+1,j,k,QUN) += 0.5_rt*dt*prim_source(i,j,k,QUN);
      qm(i+1,j,k,QUT) += 0.5_rt*dt*prim_source(i,j,k,QUT);
      qm(i+1,j,k,QTRANSV2) += 0.5_rt*dt*prim_source(i,j,k,QTRANSV2);
      qm(i+1,j,k,QEINTDENS) += 0.5_rt*dt*prim_source(i,j,k,QEINTDENS);
      qm(i+1,j,k,QPRES) += 0.5_rt*dt*prim_source(i,j,k,QPRES);

    } else if (idir == 1 && j <= vhi[1]) {
      qm(i,j+1,k,QRHO) = amrex::max(ldensity_floor, ref_density + acoustic_plus_left + acoustic_minus_left + entropy_density_left);
      qm(i,j+1,k,QUN) = ref_normal_velocity + (acoustic_plus_left - acoustic_minus_left)*cc/rho;
      qm(i,j+1,k,QUT) = ref_transv1_velocity + entropy_transv1_left;
      qm(i,j+1,k,QTRANSV2) = ref_transv2_velocity + entropy_transv2_left;
      qm(i,j+1,k,QPRES) = max(local_pressure_floor, ref_pressure + (acoustic_plus_left + acoustic_minus_left)*csq);
      qm(i,j+1,k,QEINTDENS) = ref_energy_density + (acoustic_plus_left + acoustic_minus_left)*enth*csq + entropy_energy_left;

      qm(i,j+1,k,QRHO) = amrex::max(ldensity_floor, qm(i,j+1,k,QRHO) + 0.5_rt*dt*prim_source(i,j,k,QRHO));
      qm(i,j+1,k,QUN) += 0.5_rt*dt*prim_source(i,j,k,QUN);
      qm(i,j+1,k,QUT) += 0.5_rt*dt*prim_source(i,j,k,QUT);
      qm(i,j+1,k,QTRANSV2) += 0.5_rt*dt*prim_source(i,j,k,QTRANSV2);
      qm(i,j+1,k,QEINTDENS) += 0.5_rt*dt*prim_source(i,j,k,QEINTDENS);
      qm(i,j+1,k,QPRES) += 0.5_rt*dt*prim_source(i,j,k,QPRES);

    } else if (idir == 2 && k <= vhi[2]) {
      qm(i,j,k+1,QRHO) = amrex::max(ldensity_floor, ref_density + acoustic_plus_left + acoustic_minus_left + entropy_density_left);
      qm(i,j,k+1,QUN) = ref_normal_velocity + (acoustic_plus_left - acoustic_minus_left)*cc/rho;
      qm(i,j,k+1,QUT) = ref_transv1_velocity + entropy_transv1_left;
      qm(i,j,k+1,QTRANSV2) = ref_transv2_velocity + entropy_transv2_left;
      qm(i,j,k+1,QPRES) = amrex::max(local_pressure_floor, ref_pressure + (acoustic_plus_left + acoustic_minus_left)*csq);
      qm(i,j,k+1,QEINTDENS) = ref_energy_density + (acoustic_plus_left + acoustic_minus_left)*enth*csq + entropy_energy_left;

      qm(i,j,k+1,QRHO) = amrex::max(ldensity_floor, qm(i,j,k+1,QRHO) + 0.5_rt*dt*prim_source(i,j,k,QRHO));
      qm(i,j,k+1,QUN) += 0.5_rt*dt*prim_source(i,j,k,QUN);
      qm(i,j,k+1,QUT) += 0.5_rt*dt*prim_source(i,j,k,QUT);
      qm(i,j,k+1,QTRANSV2) += 0.5_rt*dt*prim_source(i,j,k,QTRANSV2);
      qm(i,j,k+1,QEINTDENS) += 0.5_rt*dt*prim_source(i,j,k,QEINTDENS);
      qm(i,j,k+1,QPRES) += 0.5_rt*dt*prim_source(i,j,k,QPRES);

    }

#if (AMREX_SPACEDIM < 3)

    if (dloga(i,j,k) != 0.0_rt) {
      Real courant_local = dt_over_dx*(cc + std::abs(un));
      Real eta = (1.0_rt-courant_local)/(cc*dt*std::abs(dloga(i,j,k)));
      Real limited_dlog_area = amrex::min(eta, 1.0_rt)*dloga(i,j,k);
      Real source_density = -0.5_rt*dt*rho*limited_dlog_area*un;
      Real source_pressure = source_density*csq;
      Real source = source_pressure*enth;

      if (idir == 0 && i <= vhi[0]) {
        qm(i+1,j,k,QRHO) += source_density;
        qm(i+1,j,k,QRHO) = amrex::max(qm(i+1,j,k,QRHO), ldensity_floor);
        qm(i+1,j,k,QPRES) += source_pressure;
        qm(i+1,j,k,QEINTDENS) += source;
      }

      if (idir == 1 && j <= vhi[1]) {
        qm(i,j+1,k,QRHO) += source_density;
        qm(i,j+1,k,QRHO) = amrex::max(qm(i,j+1,k,QRHO), ldensity_floor);
        qm(i,j+1,k,QPRES) += source_pressure;
        qm(i,j+1,k,QEINTDENS) += source;
      }

      if ((idir == 0 && i >= vlo[0]) ||
          (idir == 1 && j >= vlo[1])) {
        qp(i,j,k,QRHO) += source_density;
        qp(i,j,k,QRHO) = amrex::max(qp(i,j,k,QRHO), ldensity_floor);
        qp(i,j,k,QPRES) += source_pressure;
        qp(i,j,k,QEINTDENS) += source;
      }

    }

#endif

    for (int passive_index = 0; passive_index < npassive; passive_index++) {
      const int nc = passive_cons_index(passive_index);
      const int n = passive_prim_index(passive_index);


      fill_passive_stencil(field_arr, inv_density_arr, idir, i, j, k, nc, s);
      Real dX = limited_slope(s, flat, false, false);

      if ((idir == 0 && i >= vlo[0]) ||
          (idir == 1 && j >= vlo[1]) ||
          (idir == 2 && k >= vlo[2])) {

        Real zero_speed_factor = un >= 0.0_rt ? -1.0_rt : un*dt_over_dx;
        qp(i,j,k,n) = s[i0] + 0.5_rt*(-1.0_rt - zero_speed_factor)*dX;
      }

      Real zero_speed_factor = un >= 0.0_rt ? un*dt_over_dx : 1.0_rt;
      Real advect_correction_left = 0.5_rt*(1.0_rt - zero_speed_factor )*dX;

      if (idir == 0 && i <= vhi[0]) {
        qm(i+1,j,k,n) = s[i0] + advect_correction_left;
      } else if (idir == 1 && j <= vhi[1]) {
        qm(i,j+1,k,n) = s[i0] + advect_correction_left;
      } else if (idir == 2 && k <= vhi[2]) {
        qm(i,j,k+1,n) = s[i0] + advect_correction_left;
      }

    }

  });
}
