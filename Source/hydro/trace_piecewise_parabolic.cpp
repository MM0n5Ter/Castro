#include <HydroSim.H>
#include <HydroSim_helpers.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#include <cmath>

#include <ppm.H>
#include <reconstruction.H>
#include <flatten.H>

using namespace amrex;
using namespace reconstruction;

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


  int QUN = -1;
  int QUT = -1;
  int QTRANSV2 = -1;

  if (idir == 0) {
    QUN = QU;
    QUT = QV;
    QTRANSV2 = QW;
  } else if (idir == 1) {
    QUN = QV;
    QUT = QW;
    QTRANSV2 = QU;
  } else if (idir == 2) {
    QUN = QW;
    QUT = QU;
    QTRANSV2 = QV;
  }

  Real ldensity_floor = small_dens;
  Real local_pressure_floor = pressure_cutoff;

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

        if (idir == 0 && i <= vhi[0]) {
            qm(i+1,j,k,n) = trace_p_passive;
        } else if (idir == 1 && j <= vhi[1]) {
            qm(i,j+1,k,n) = trace_p_passive;
        } else if (idir == 2 && k <= vhi[2]) {
            qm(i,j,k+1,n) = trace_p_passive;
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

      if (idir == 0) {
        qm(i+1,j,k,QRHO) = std::max(ldensity_floor, ref_density +  acoustic_amp_plus + acoustic_amp_minus + entropy_amp_density);
        qm(i+1,j,k,QUN) = ref_normal_velocity + (acoustic_amp_plus - acoustic_amp_minus)*ref_sound_speed*ref_density_inv;
        qm(i+1,j,k,QEINTDENS) = std::max(hydrosim::small_dens * hydrosim::eint_floor,
                                        ref_energy_density_gas + (acoustic_amp_plus + acoustic_amp_minus)*ref_enthalpy_gas + entropy_amp_energy_gas);
        qm(i+1,j,k,QPRES) = std::max(local_pressure_floor, p_plus_hydrostatic + ref_pressure + (acoustic_amp_plus + acoustic_amp_minus)*ref_sound_speed_sq);

        qm(i+1,j,k,QUT) = trace_p_transv1_vel_1 + hdt*trace_p_src_transv1_vel_1;
        qm(i+1,j,k,QTRANSV2) = trace_p_transv2_vel_1 + hdt*trace_p_src_transv2_vel_1;

      } else if (idir == 1) {
        qm(i,j+1,k,QRHO) = std::max(ldensity_floor, ref_density +  acoustic_amp_plus + acoustic_amp_minus + entropy_amp_density);
        qm(i,j+1,k,QUN) = ref_normal_velocity + (acoustic_amp_plus - acoustic_amp_minus)*ref_sound_speed*ref_density_inv;
        qm(i,j+1,k,QEINTDENS) = std::max(hydrosim::small_dens * hydrosim::eint_floor,
                                      ref_energy_density_gas + (acoustic_amp_plus + acoustic_amp_minus)*ref_enthalpy_gas + entropy_amp_energy_gas);
        qm(i,j+1,k,QPRES) = std::max(local_pressure_floor, p_plus_hydrostatic + ref_pressure + (acoustic_amp_plus + acoustic_amp_minus)*ref_sound_speed_sq);

        qm(i,j+1,k,QUT) = trace_p_transv1_vel_1 + hdt*trace_p_src_transv1_vel_1;
        qm(i,j+1,k,QTRANSV2) = trace_p_transv2_vel_1 + hdt*trace_p_src_transv2_vel_1;

      } else if (idir == 2) {
        qm(i,j,k+1,QRHO) = std::max(ldensity_floor, ref_density +  acoustic_amp_plus + acoustic_amp_minus + entropy_amp_density);
        qm(i,j,k+1,QUN) = ref_normal_velocity + (acoustic_amp_plus - acoustic_amp_minus)*ref_sound_speed*ref_density_inv;
        qm(i,j,k+1,QEINTDENS) = std::max(hydrosim::small_dens * hydrosim::eint_floor,
                                        ref_energy_density_gas + (acoustic_amp_plus + acoustic_amp_minus)*ref_enthalpy_gas + entropy_amp_energy_gas);
        qm(i,j,k+1,QPRES) = std::max(local_pressure_floor, p_plus_hydrostatic + ref_pressure + (acoustic_amp_plus + acoustic_amp_minus)*ref_sound_speed_sq);

        qm(i,j,k+1,QUT) = trace_p_transv1_vel_1 + hdt*trace_p_src_transv1_vel_1;
        qm(i,j,k+1,QTRANSV2) = trace_p_transv2_vel_1 + hdt*trace_p_src_transv2_vel_1;
      }

    }

  });
}
