#include <HydroSim.H>
#include <HydroSim_helpers.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#include <cmath>

#include <parabolic_profile.H>
#include <stencil_tools.H>
#include <limiters.H>

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

  constexpr int velocity_order[3][3] = {{QU, QV, QW}, {QV, QW, QU}, {QW, QU, QV}};

  const int QUN = velocity_order[idir][0];
  const int QUT = velocity_order[idir][1];
  const int QTRANSV2 = velocity_order[idir][2];

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

    auto qm_upper = [&] (int n) -> Real& {
      return qm(i+(idir==0), j+(idir==1), k+(idir==2), n);
    };

    const bool has_upper_face = (idir == 0 && i <= vhi[0]) ||
                                (idir == 1 && j <= vhi[1]) ||
                                (idir == 2 && k <= vhi[2]);

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

    if (has_upper_face) {
      qm_upper(QRHO) = amrex::max(ldensity_floor, ref_density + acoustic_plus_left + acoustic_minus_left + entropy_density_left);
      qm_upper(QUN) = ref_normal_velocity + (acoustic_plus_left - acoustic_minus_left)*cc/rho;
      qm_upper(QUT) = ref_transv1_velocity + entropy_transv1_left;
      qm_upper(QTRANSV2) = ref_transv2_velocity + entropy_transv2_left;
      qm_upper(QPRES) = amrex::max(local_pressure_floor, ref_pressure + (acoustic_plus_left + acoustic_minus_left)*csq);
      qm_upper(QEINTDENS) = ref_energy_density + (acoustic_plus_left + acoustic_minus_left)*enth*csq + entropy_energy_left;

      qm_upper(QRHO) = amrex::max(ldensity_floor, qm_upper(QRHO) + 0.5_rt*dt*prim_source(i,j,k,QRHO));
      qm_upper(QUN) += 0.5_rt*dt*prim_source(i,j,k,QUN);
      qm_upper(QUT) += 0.5_rt*dt*prim_source(i,j,k,QUT);
      qm_upper(QTRANSV2) += 0.5_rt*dt*prim_source(i,j,k,QTRANSV2);
      qm_upper(QEINTDENS) += 0.5_rt*dt*prim_source(i,j,k,QEINTDENS);
      qm_upper(QPRES) += 0.5_rt*dt*prim_source(i,j,k,QPRES);

    }

#if (AMREX_SPACEDIM < 3)

    if (dloga(i,j,k) != 0.0_rt) {
      Real courant_local = dt_over_dx*(cc + std::abs(un));
      Real eta = (1.0_rt-courant_local)/(cc*dt*std::abs(dloga(i,j,k)));
      Real limited_dlog_area = amrex::min(eta, 1.0_rt)*dloga(i,j,k);
      Real source_density = -0.5_rt*dt*rho*limited_dlog_area*un;
      Real source_pressure = source_density*csq;
      Real source = source_pressure*enth;

      if (has_upper_face) {
        qm_upper(QRHO) += source_density;
        qm_upper(QRHO) = amrex::max(qm_upper(QRHO), ldensity_floor);
        qm_upper(QPRES) += source_pressure;
        qm_upper(QEINTDENS) += source;
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

      if (has_upper_face) {
        qm_upper(n) = s[i0] + advect_correction_left;
      }

    }

  });
}

void
HydroSim::trace_piecewise_parabolic(const Box& bx,
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
  const auto problo = geom.ProbLoArray();
  const int coord = geom.Coord();

  Real hdt = 0.5_rt * dt;

#ifndef AMREX_USE_GPU
  auto lo = bx.loVect3d();
  auto hi = bx.hiVect3d();
#endif

  auto vlo = vbx.loVect3d();
  auto vhi = vbx.hiVect3d();


#ifndef AMREX_USE_GPU


  GpuArray<int, NQSOURCE> trace_source_flags{};


  for (int n = 0; n < NQSOURCE; n++) {


    if ((coord == 2 || (idir == 0 && coord == 1))
        && (n == QRHO || n == QPRES || n == QEINTDENS)) {
        trace_source_flags[n] = 1;
        continue;
    }

    for (int k = lo[2]-2*dg2; k <= hi[2]+2*dg2; k++) {
      for (int j = lo[1]-2*dg1; j <= hi[1]+2*dg1; j++) {
        for (int i = lo[0]-2; i <= hi[0]+2; i++) {
          if (std::abs(prim_source(i,j,k,n)) > 0.0_rt) {
            trace_source_flags[n] = 1;
            break;
          }
        }
        if (trace_source_flags[n] == 1) {
            break;
        }
      }
      if (trace_source_flags[n] == 1) {
          break;
      }
    }
  }
#endif


  constexpr int velocity_comp[3] = {QU, QV, QW};

  const int QUN = velocity_comp[idir];
  const int QUT = velocity_comp[(idir + 1) % 3];
  const int QTRANSV2 = velocity_comp[(idir + 2) % 3];

  Real ldensity_floor = small_dens;
  Real local_pressure_floor = pressure_cutoff;

  const Real face_dt_over_dx = dt / dx[idir];
  const Real face_width = dx[idir];

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real dt_over_dx = face_dt_over_dx;
    Real dL = face_width;

    if (coord == 2 && idir == 1) {
        Real r = problo[0] + static_cast<Real>(i + 0.5_rt) * dx[0];
        dL = r * dx[1];
        dt_over_dx = dt / dL;
    }

    Real cc = qthermo_arr(i,j,k,QC);
    Real un = prim_arr(i,j,k,QUN);

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

    Real sm;
    Real sp;



    Real trace_p_rho[3];
    Real trace_m_rho[3];

    gather_stencil(prim_arr, idir, i, j, k, QRHO, s);
    parabolic_reconstruct(s, flat, sm, sp);
    integrate_parabolic_profile(sm, sp, s[i0], un, cc, dt_over_dx, trace_p_rho, trace_m_rho);


    Real trace_p_normal_vel_0;
    Real trace_m_normal_vel_0;
    Real trace_p_normal_vel_2;
    Real trace_m_normal_vel_2;

    gather_stencil(prim_arr, idir, i, j, k, QUN, s);
    parabolic_reconstruct(s, flat, sm, sp);
    integrate_parabolic_profile_single(sm, sp, s[i0], un-cc, dt_over_dx, trace_p_normal_vel_0, trace_m_normal_vel_0);
    integrate_parabolic_profile_single(sm, sp, s[i0], un+cc, dt_over_dx, trace_p_normal_vel_2, trace_m_normal_vel_2);


    Real trace_p_pressure[3];
    Real trace_m_pressure[3];

    gather_stencil(prim_arr, idir, i, j, k, QPRES, s);

    bool in_hydrostatic{};


    amrex::Real p_minus_hydrostatic{};
    amrex::Real p_plus_hydrostatic{};

    if (hydrosim::use_pressure_slope || hydrosim::ppm_equil_recon) {
        Real rho_stencil[nslp];
        Real src[nslp];

        gather_stencil(prim_arr, idir, i, j, k, QRHO, rho_stencil);
        gather_stencil(prim_source, idir, i, j, k, QUN, src);

        in_hydrostatic = reconstruct_parabolic_hydrostatic(rho_stencil, s, src, flat, dL, sm, sp);

        if (in_hydrostatic && hydrosim::ppm_equil_recon) {
            integrate_parabolic_profile(sm, sp, 0.0_rt, un, cc, dt_over_dx, trace_p_pressure, trace_m_pressure);
            p_minus_hydrostatic = s[i0] - 0.5_rt * dL * rho_stencil[i0] * src[i0];
            p_plus_hydrostatic = s[i0] + 0.5_rt * dL * rho_stencil[i0] * src[i0];

        } else {
            integrate_parabolic_profile(sm, sp, s[i0], un, cc, dt_over_dx, trace_p_pressure, trace_m_pressure);
        }

    } else {
        parabolic_reconstruct(s, flat, sm, sp);
        integrate_parabolic_profile(sm, sp, s[i0], un, cc, dt_over_dx, trace_p_pressure, trace_m_pressure);
    }


    Real trace_p_energy_density[3];
    Real trace_m_energy_density[3];

    gather_stencil(prim_arr, idir, i, j, k, QEINTDENS, s);
    parabolic_reconstruct(s, flat, sm, sp);
    integrate_parabolic_profile(sm, sp, s[i0], un, cc, dt_over_dx, trace_p_energy_density, trace_m_energy_density);


    Real trace_p_transv1_vel_1;
    Real trace_m_transv1_vel_1;
    Real trace_p_transv2_vel_1;
    Real trace_m_transv2_vel_1;

    gather_stencil(prim_arr, idir, i, j, k, QUT, s);
    parabolic_reconstruct(s, flat, sm, sp);
    integrate_parabolic_profile_single(sm, sp, s[i0], un, dt_over_dx, trace_p_transv1_vel_1, trace_m_transv1_vel_1);

    gather_stencil(prim_arr, idir, i, j, k, QTRANSV2, s);
    parabolic_reconstruct(s, flat, sm, sp);
    integrate_parabolic_profile_single(sm, sp, s[i0], un, dt_over_dx, trace_p_transv2_vel_1, trace_m_transv2_vel_1);


    Real trace_p_gamma1_0;
    Real trace_m_gamma1_0;
    Real trace_p_gamma1_2;
    Real trace_m_gamma1_2;

    gather_stencil(qthermo_arr, idir, i, j, k, QGAMMA1, s);
    parabolic_reconstruct(s, flat, sm, sp);
    integrate_parabolic_profile_single(sm, sp, s[i0], un-cc, dt_over_dx, trace_p_gamma1_0, trace_m_gamma1_0);
    integrate_parabolic_profile_single(sm, sp, s[i0], un+cc, dt_over_dx, trace_p_gamma1_2, trace_m_gamma1_2);


    int trace_enabled;


    Real trace_p_src_rho[3] = {0.0_rt};
    Real trace_m_src_rho[3] = {0.0_rt};

#ifndef AMREX_USE_GPU
    trace_enabled = trace_source_flags[QRHO];
#else
    if (coord == 2 || (idir == 0 && coord == 1)) {
        trace_enabled = 1;
    } else {
        trace_enabled = source_trace_needed(prim_source, idir, i, j, k, QRHO);
    }
#endif

    if (trace_enabled) {
        gather_stencil(prim_source, idir, i, j, k, QRHO, s);
#if AMREX_SPACEDIM <= 2
        if (coord == 2 || (idir == 0 && coord == 1)) {
            add_geometric_density_source(prim_arr, dloga, i, j, k, QUN, s);
        }
#endif
        parabolic_reconstruct(s, flat, sm, sp);
        integrate_parabolic_profile(sm, sp, s[i0], un, cc, dt_over_dx, trace_p_src_rho, trace_m_src_rho);
    }


    Real trace_p_src_normal_vel_0 = 0.0_rt;
    Real trace_m_src_normal_vel_0 = 0.0_rt;
    Real trace_p_src_normal_vel_2 = 0.0_rt;
    Real trace_m_src_normal_vel_2 = 0.0_rt;

#ifndef AMREX_USE_GPU
    trace_enabled = trace_source_flags[QUN];
#else
    trace_enabled = source_trace_needed(prim_source, idir, i, j, k, QUN);
#endif

    if (trace_enabled) {
        gather_stencil(prim_source, idir, i, j, k, QUN, s);
        parabolic_reconstruct(s, flat, sm, sp);
        integrate_parabolic_profile_single(sm, sp, s[i0], un-cc, dt_over_dx, trace_p_src_normal_vel_0, trace_m_src_normal_vel_0);
        integrate_parabolic_profile_single(sm, sp, s[i0], un+cc, dt_over_dx, trace_p_src_normal_vel_2, trace_m_src_normal_vel_2);
    }


    Real trace_p_src_pressure[3] = {0.0_rt};
    Real trace_m_src_pressure[3] = {0.0_rt};

#ifndef AMREX_USE_GPU
    trace_enabled = trace_source_flags[QPRES];
#else
    if (coord == 2 || (idir == 0 && coord == 1)) {
        trace_enabled = 1;
    } else {
        trace_enabled = source_trace_needed(prim_source, idir, i, j, k, QPRES);
    }
#endif

    if (trace_enabled) {
        gather_stencil(prim_source, idir, i, j, k, QPRES, s);
#if AMREX_SPACEDIM <= 2
        if (coord == 2 || (idir == 0 && coord == 1)) {
            add_geometric_pressure_source(prim_arr, qthermo_arr, dloga, i, j, k, QUN, s);
        }
#endif
        parabolic_reconstruct(s, flat, sm, sp);
        integrate_parabolic_profile(sm, sp, s[i0], un, cc, dt_over_dx, trace_p_src_pressure, trace_m_src_pressure);
    }


    Real trace_p_src_energy_density[3] = {0.0_rt};
    Real trace_m_src_energy_density[3] = {0.0_rt};

#ifndef AMREX_USE_GPU
    trace_enabled = trace_source_flags[QEINTDENS];
#else
    if (coord == 2 || (idir == 0 && coord == 1)) {
        trace_enabled = 1;
    } else {
        trace_enabled = source_trace_needed(prim_source, idir, i, j, k, QEINTDENS);
    }
#endif

    if (trace_enabled) {
        gather_stencil(prim_source, idir, i, j, k, QEINTDENS, s);
#if AMREX_SPACEDIM <= 2
        if (coord == 2 || (idir == 0 && coord == 1)) {
            add_geometric_energy_source(prim_arr, dloga, i, j, k, QUN, s);
        }
#endif
        parabolic_reconstruct(s, flat, sm, sp);
        integrate_parabolic_profile(sm, sp, s[i0], un, cc, dt_over_dx, trace_p_src_energy_density, trace_m_src_energy_density);
    }


    Real trace_p_src_transv1_vel_1 = 0.0_rt;
    Real trace_m_src_transv1_vel_1 = 0.0_rt;

#ifndef AMREX_USE_GPU
    trace_enabled = trace_source_flags[QUT];
#else
    trace_enabled = source_trace_needed(prim_source, idir, i, j, k, QUT);
#endif

    if (trace_enabled) {
        gather_stencil(prim_source, idir, i, j, k, QUT, s);
        parabolic_reconstruct(s, flat, sm, sp);
        integrate_parabolic_profile_single(sm, sp, s[i0], un, dt_over_dx, trace_p_src_transv1_vel_1, trace_m_src_transv1_vel_1);
    }

    Real trace_p_src_transv2_vel_1 = 0.0_rt;
    Real trace_m_src_transv2_vel_1 = 0.0_rt;

#ifndef AMREX_USE_GPU
    trace_enabled = trace_source_flags[QTRANSV2];
#else
    trace_enabled = source_trace_needed(prim_source, idir, i, j, k, QTRANSV2);
#endif

    if (trace_enabled) {
        gather_stencil(prim_source, idir, i, j, k, QTRANSV2, s);
        parabolic_reconstruct(s, flat, sm, sp);
        integrate_parabolic_profile_single(sm, sp, s[i0], un, dt_over_dx, trace_p_src_transv2_vel_1, trace_m_src_transv2_vel_1);
    }




    Real trace_p_passive;
    Real trace_m_passive;

    const int ip = i + (idir == 0);
    const int jp = j + (idir == 1);
    const int kp = k + (idir == 2);

    for (int passive_index = 0; passive_index < npassive; passive_index++) {

        const int nc = passive_cons_index(passive_index);
        const int n = passive_prim_index(passive_index);

        fill_passive_stencil(field_arr, inv_density_arr, idir, i, j, k, nc, s);
        parabolic_reconstruct(s, flat, sm, sp);
        integrate_parabolic_profile_single(sm, sp, s[i0], un, dt_over_dx, trace_p_passive, trace_m_passive);


        if ((idir == 0 && i >= vlo[0]) ||
            (idir == 1 && j >= vlo[1]) ||
            (idir == 2 && k >= vlo[2])) {


            qp(i,j,k,n) = trace_m_passive;
        }

        if ((idir == 0 && i <= vhi[0]) ||
            (idir == 1 && j <= vhi[1]) ||
            (idir == 2 && k <= vhi[2])) {
            qm(ip,jp,kp,n) = trace_p_passive;
        }
    }



    amrex::Real fac = (hydrosim::ppm_equil_recon && in_hydrostatic) ? 0.0_rt : 1.0_rt;


    if ((idir == 0 && i >= vlo[0]) ||
        (idir == 1 && j >= vlo[1]) ||
        (idir == 2 && k >= vlo[2])) {

      Real ref_density = trace_m_rho[0] + hdt * trace_m_src_rho[0];
      Real ref_normal_velocity = trace_m_normal_vel_0 + fac * hdt * trace_m_src_normal_vel_0;

      Real ref_pressure = trace_m_pressure[0] + hdt * trace_m_src_pressure[0];
      Real ref_energy_density_gas = trace_m_energy_density[0] + hdt * trace_m_src_energy_density[0];

      Real ref_gamma1_gas = trace_m_gamma1_0;

      ref_density = std::max(ref_density, ldensity_floor);

      Real ref_density_inv = 1.0_rt/ref_density;
      ref_pressure = std::max(ref_pressure, local_pressure_floor);

      Real ref_sound_speed_sq = ref_gamma1_gas * (p_minus_hydrostatic + ref_pressure) * ref_density_inv;
      Real ref_sound_speed = std::sqrt(ref_sound_speed_sq);
      Real ref_sound_speed_inv = 1.0_rt/ref_sound_speed;
      Real ref_enthalpy_gas = ((p_minus_hydrostatic + ref_pressure) + ref_energy_density_gas) * ref_density_inv;





      Real dum = ref_normal_velocity - trace_m_normal_vel_0 - fac * hdt * trace_m_src_normal_vel_0;
      Real pressure_jump_minus = ref_pressure - trace_m_pressure[0] - hdt * trace_m_src_pressure[0];

      Real drho = ref_density - trace_m_rho[1] - hdt * trace_m_src_rho[1];
      Real pressure_jump = ref_pressure - trace_m_pressure[1] - hdt * trace_m_src_pressure[1];
      Real energy_density_jump_gas = ref_energy_density_gas - trace_m_energy_density[1] - hdt * trace_m_src_energy_density[1];

      Real dup = ref_normal_velocity - trace_m_normal_vel_2 - fac * hdt * trace_m_src_normal_vel_2;
      Real pressure_jump_plus = ref_pressure - trace_m_pressure[2] - hdt * trace_m_src_pressure[2];



      Real acoustic_amp_minus = 0.5_rt*(pressure_jump_minus*ref_density_inv*ref_sound_speed_inv - dum)*ref_density*ref_sound_speed_inv;
      Real acoustic_amp_plus = 0.5_rt*(pressure_jump_plus*ref_density_inv*ref_sound_speed_inv + dup)*ref_density*ref_sound_speed_inv;
      Real entropy_amp_density = drho - pressure_jump/ref_sound_speed_sq;
      Real entropy_amp_energy_gas = energy_density_jump_gas - pressure_jump*ref_enthalpy_gas/ref_sound_speed_sq;

      acoustic_amp_minus = un-cc > 0.0_rt ? 0.0_rt : -acoustic_amp_minus;
      acoustic_amp_plus = un+cc > 0.0_rt ? 0.0_rt : -acoustic_amp_plus;
      entropy_amp_density = un > 0.0_rt ? 0.0_rt : -entropy_amp_density;
      entropy_amp_energy_gas = un > 0.0_rt ? 0.0_rt : -entropy_amp_energy_gas;

      qp(i,j,k,QRHO) = std::max(ldensity_floor, ref_density +  acoustic_amp_plus + acoustic_amp_minus + entropy_amp_density);
      qp(i,j,k,QUN) = ref_normal_velocity + (acoustic_amp_plus - acoustic_amp_minus)*ref_sound_speed*ref_density_inv;
      qp(i,j,k,QEINTDENS) = std::max(hydrosim::small_dens * hydrosim::eint_floor,
                                    ref_energy_density_gas + (acoustic_amp_plus + acoustic_amp_minus)*ref_enthalpy_gas + entropy_amp_energy_gas);
      qp(i,j,k,QPRES) = std::max(local_pressure_floor, p_minus_hydrostatic + ref_pressure + (acoustic_amp_plus + acoustic_amp_minus)*ref_sound_speed_sq);


      qp(i,j,k,QUT) = trace_m_transv1_vel_1 + hdt * trace_m_src_transv1_vel_1;
      qp(i,j,k,QTRANSV2) = trace_m_transv2_vel_1 + hdt * trace_m_src_transv2_vel_1;

    }


    if ((idir == 0 && i <= vhi[0]) ||
        (idir == 1 && j <= vhi[1]) ||
        (idir == 2 && k <= vhi[2])) {

      Real ref_density = trace_p_rho[2] + hdt * trace_p_src_rho[2];
      Real ref_normal_velocity = trace_p_normal_vel_2 + fac * hdt * trace_p_src_normal_vel_2;

      Real ref_pressure = trace_p_pressure[2] + hdt * trace_p_src_pressure[2];
      Real ref_energy_density_gas = trace_p_energy_density[2] + hdt * trace_p_src_energy_density[2];

      Real ref_gamma1_gas = trace_p_gamma1_2;

      ref_density = std::max(ref_density, ldensity_floor);
      Real ref_density_inv = 1.0_rt/ref_density;
      ref_pressure = std::max(ref_pressure, local_pressure_floor);

      Real ref_sound_speed_sq = ref_gamma1_gas * (p_plus_hydrostatic + ref_pressure) * ref_density_inv;
      Real ref_sound_speed = std::sqrt(ref_sound_speed_sq);
      Real ref_sound_speed_inv = 1.0_rt/ref_sound_speed;
      Real ref_enthalpy_gas = ((p_plus_hydrostatic + ref_pressure) + ref_energy_density_gas) * ref_density_inv;


      Real dum = ref_normal_velocity - trace_p_normal_vel_0 - fac * hdt * trace_p_src_normal_vel_0;
      Real pressure_jump_minus  = ref_pressure - trace_p_pressure[0] - hdt * trace_p_src_pressure[0];

      Real drho = ref_density - trace_p_rho[1] - hdt * trace_p_src_rho[1];
      Real pressure_jump = ref_pressure - trace_p_pressure[1] - hdt * trace_p_src_pressure[1];
      Real energy_density_jump_gas = ref_energy_density_gas - trace_p_energy_density[1] - hdt * trace_p_src_energy_density[1];

      Real dup = ref_normal_velocity - trace_p_normal_vel_2 - fac * hdt * trace_p_src_normal_vel_2;
      Real pressure_jump_plus = ref_pressure - trace_p_pressure[2] - hdt * trace_p_src_pressure[2];



      Real acoustic_amp_minus = 0.5_rt*(pressure_jump_minus*ref_density_inv*ref_sound_speed_inv - dum)*ref_density*ref_sound_speed_inv;
      Real acoustic_amp_plus = 0.5_rt*(pressure_jump_plus*ref_density_inv*ref_sound_speed_inv + dup)*ref_density*ref_sound_speed_inv;
      Real entropy_amp_density = drho - pressure_jump/ref_sound_speed_sq;
      Real entropy_amp_energy_gas = energy_density_jump_gas - pressure_jump*ref_enthalpy_gas/ref_sound_speed_sq;

      acoustic_amp_minus = un-cc > 0.0_rt ? -acoustic_amp_minus : 0.0_rt;
      acoustic_amp_plus = un+cc > 0.0_rt ? -acoustic_amp_plus : 0.0_rt;
      entropy_amp_density = un > 0.0_rt ? -entropy_amp_density : 0.0_rt;
      entropy_amp_energy_gas = un > 0.0_rt ? -entropy_amp_energy_gas : 0.0_rt;

      qm(ip,jp,kp,QRHO) = std::max(ldensity_floor, ref_density +  acoustic_amp_plus + acoustic_amp_minus + entropy_amp_density);
      qm(ip,jp,kp,QUN) = ref_normal_velocity + (acoustic_amp_plus - acoustic_amp_minus)*ref_sound_speed*ref_density_inv;
      qm(ip,jp,kp,QEINTDENS) = std::max(hydrosim::small_dens * hydrosim::eint_floor,
                                      ref_energy_density_gas + (acoustic_amp_plus + acoustic_amp_minus)*ref_enthalpy_gas + entropy_amp_energy_gas);
      qm(ip,jp,kp,QPRES) = std::max(local_pressure_floor, p_plus_hydrostatic + ref_pressure + (acoustic_amp_plus + acoustic_amp_minus)*ref_sound_speed_sq);

      qm(ip,jp,kp,QUT) = trace_p_transv1_vel_1 + hdt*trace_p_src_transv1_vel_1;
      qm(ip,jp,kp,QTRANSV2) = trace_p_transv2_vel_1 + hdt*trace_p_src_transv2_vel_1;

    }

  });
}
