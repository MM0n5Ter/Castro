#ifndef ITERATIVE_RIEMANN_H
#define ITERATIVE_RIEMANN_H

#include <AMReX_REAL.H>

using namespace amrex::literals;

#include <solver_constants.H>
#include <riemann_states.H>
#ifdef RADIATION
#include <Radiation.H>
#include <radiation_helpers.H>
#endif

namespace ShockPairSolver {
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    compute_wave_speed_sq(const amrex::Real p, const amrex::Real v,
          const amrex::Real gam, const amrex::Real gamma_drift_rate, amrex::Real& gamma_star_value,
          const amrex::Real gamma_lower_bound, const amrex::Real gmax, const amrex::Real csq,
          const amrex::Real pstar, amrex::Real& wsq) {




      gamma_star_value = (pstar-p)*gamma_drift_rate/(pstar+p) + gam;
      gamma_star_value = amrex::Clamp(gamma_star_value, gamma_lower_bound, gmax);


      amrex::Real alpha = pstar - (gamma_star_value - 1.0_rt)*p/(gam - 1.0_rt);
      if (alpha == 0.0_rt) {
        alpha = solver_constants::small_rel_pressure*(pstar + p);
      }

      amrex::Real beta = pstar + 0.5_rt*(gamma_star_value - 1.0_rt)*(pstar+p);

      wsq = (pstar-p)*beta/(v*alpha);

      if (std::abs(pstar - p) < solver_constants::small_rel_pressure*(pstar + p)) {
        wsq = csq;
      }
      wsq = amrex::max(wsq, (0.5_rt * (gam - 1.0_rt)/gam)*csq);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    shock_pair_direct(const InterfaceState& ql, const InterfaceState& qr, const InterfaceAux& interface_aux,
              InterfaceState& interface_state) {




      amrex::Real impedance_floor = small_dens * interface_aux.sound_floor;

      amrex::Real wl = amrex::max(impedance_floor, std::sqrt(std::abs(ql.gamma1_cell * ql.p * ql.rho)));
      amrex::Real wr = amrex::max(impedance_floor, std::sqrt(std::abs(qr.gamma1_cell * qr.p * qr.rho)));

      amrex::Real inv_impedance_sum = 1.0_rt/(wl + wr);
      amrex::Real pstar = ((wr * ql.p + wl * qr.p) + wl * wr * (ql.un - qr.un)) * inv_impedance_sum;
      amrex::Real ustar = ((wl * ql.un + wr * qr.un) + (ql.p - qr.p)) * inv_impedance_sum;

      pstar = amrex::max(pstar, pressure_cutoff);


      if (std::abs(ustar) < solver_constants::small_velocity_ratio * 0.5_rt * (std::abs(ql.un) + std::abs(qr.un))) {
          ustar = 0.0_rt;
      }


      amrex::Real sgnm = std::copysign(1.0_rt, ustar);
      if (ustar == 0.0_rt) {
          sgnm = 0.0_rt;
      }

      amrex::Real fp = 0.5_rt*(1.0_rt + sgnm);
      amrex::Real fm = 0.5_rt*(1.0_rt - sgnm);

      auto upwind_mix = [=] (amrex::Real l, amrex::Real r) {
          return fp * l + fm * r;
      };

      amrex::Real ro = upwind_mix(ql.rho, qr.rho);
      amrex::Real uo = upwind_mix(ql.un, qr.un);
      amrex::Real po = upwind_mix(ql.p, qr.p);
      amrex::Real reo = upwind_mix(ql.rhoe, qr.rhoe);
      amrex::Real upwind_gamma1 = upwind_mix(ql.gamma1_cell, qr.gamma1_cell);
    #ifdef RADIATION
      for (int g = 0; g < NGROUPS; g++) {
          interface_state.lam[g] = upwind_mix(ql.lam[g], qr.lam[g]);
      }

      if (ustar == 0) {
          for (int g = 0; g < NGROUPS; g++) {
              interface_state.lam[g] = 2.0_rt * (ql.lam[g] * qr.lam[g]) / (ql.lam[g] + qr.lam[g] + 1.e-50_rt);
          }
      }

      amrex::Real upwind_pressure_gas = upwind_mix(ql.p_g, qr.p_g);
      amrex::Real upwind_energy_density_group[NGROUPS];
      amrex::Real upwind_pressure_group[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
          upwind_energy_density_group[g] = upwind_mix(ql.er[g], qr.er[g]);
          upwind_pressure_group[g] = interface_state.lam[g] * upwind_energy_density_group[g];
      }
      amrex::Real upwind_energy_density_gas = upwind_mix(ql.energy_density_gas, qr.energy_density_gas);
      amrex::Real gamma1_upwind_gas = upwind_mix(ql.gamma1_gas, qr.gamma1_gas);
    #endif

      ro = amrex::max(small_dens, ro);

      amrex::Real inv_upwind_density = 1.0_rt / ro;

      amrex::Real co = std::sqrt(std::abs(upwind_gamma1 * po * inv_upwind_density));
      co = amrex::max(interface_aux.sound_floor, co);
      amrex::Real inv_sound_squared = 1.0_rt / (co*co);


      interface_state.ut = upwind_mix(ql.ut, qr.ut);
      interface_state.utt = upwind_mix(ql.utt, qr.utt);


      amrex::Real drho = (pstar - po)*inv_sound_squared;
      amrex::Real rstar = ro + drho;
      rstar = amrex::max(small_dens, rstar);

    #ifdef RADIATION
      amrex::Real star_energy_gas = upwind_energy_density_gas + drho*(upwind_energy_density_gas + upwind_pressure_gas)*inv_upwind_density;

      amrex::Real sound_speed_gas = std::sqrt(std::abs(gamma1_upwind_gas*upwind_pressure_gas*inv_upwind_density));
      sound_speed_gas = amrex::max(interface_aux.sound_floor, sound_speed_gas);

      amrex::Real star_pressure_gas = upwind_pressure_gas + drho*sound_speed_gas*sound_speed_gas;
      star_pressure_gas = amrex::max(star_pressure_gas, pressure_cutoff);

      amrex::Real star_energy_group[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
          star_energy_group[g] = upwind_energy_density_group[g] + drho*(upwind_energy_density_group[g] + upwind_pressure_group[g])*inv_upwind_density;
      }
    #else
      amrex::Real enthalpy_jump_coeff = (reo + po)*inv_upwind_density*inv_sound_squared;
      amrex::Real star_energy = reo + (pstar - po)*enthalpy_jump_coeff;
    #endif

      amrex::Real cstar = std::sqrt(std::abs(upwind_gamma1*pstar/rstar));
      cstar = amrex::max(cstar, interface_aux.sound_floor);



      amrex::Real spout = co - sgnm*uo;
      amrex::Real spin = cstar - sgnm*ustar;

      amrex::Real shock_vel = 0.5_rt*(spin + spout);

      if (pstar-po > 0.0_rt) {
          spin = shock_vel;
          spout = shock_vel;
      }

      amrex::Real scr = spout - spin;
      if (spout-spin == 0.0_rt) {
          scr = solver_constants::small * interface_aux.mean_sound_speed;
      }

      amrex::Real frac = (1.0_rt + (spout + spin)/scr)*0.5_rt;
      frac = amrex::Clamp(frac, 0.0_rt, 1.0_rt);

      interface_state.rho = frac*rstar + (1.0_rt - frac)*ro;
      interface_state.un = frac*ustar + (1.0_rt - frac)*uo;

    #ifdef RADIATION
      amrex::Real godunov_pressure_total = frac*pstar + (1.0_rt - frac)*po;
      amrex::Real godunov_pressure_gas = frac*star_pressure_gas + (1.0_rt - frac)*upwind_pressure_gas;
      amrex::Real godunov_energy_density_gas = frac*star_energy_gas + (1.0_rt - frac)*upwind_energy_density_gas;
      amrex::Real godunov_energy_density_group[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
          godunov_energy_density_group[g] = frac*star_energy_group[g] + (1.0_rt - frac)*upwind_energy_density_group[g];
      }
    #else
      interface_state.p = frac*pstar + (1.0_rt - frac)*po;
      amrex::Real godunov_energy_density = frac*star_energy + (1.0_rt - frac)*reo;
    #endif


      if (spout < 0.0_rt) {
          interface_state.rho = ro;
          interface_state.un = uo;
    #ifdef RADIATION
          godunov_pressure_total = po;
          godunov_pressure_gas = upwind_pressure_gas;
          godunov_energy_density_gas = upwind_energy_density_gas;
          for (int g = 0; g < NGROUPS; g++) {
              godunov_energy_density_group[g] = upwind_energy_density_group[g];
          }
    #else
          interface_state.p = po;
          godunov_energy_density = reo;
    #endif
      }

      if (spin >= 0.0_rt) {
          interface_state.rho = rstar;
          interface_state.un = ustar;
    #ifdef RADIATION
          godunov_pressure_total = pstar;
          godunov_pressure_gas = star_pressure_gas;
          godunov_energy_density_gas = star_energy_gas;
          for (int g = 0; g < NGROUPS; g++) {
              godunov_energy_density_group[g] = star_energy_group[g];
          }
    #else
          interface_state.p = pstar;
          godunov_energy_density = star_energy;
    #endif
        }

    #ifdef RADIATION
      for (int g = 0; g < NGROUPS; g++) {
          interface_state.er[g] = amrex::max(godunov_energy_density_group[g], 0.0_rt);
      }

      interface_state.p_g = godunov_pressure_gas;
      interface_state.p = godunov_pressure_total;
      interface_state.energy_density_gas = godunov_energy_density_gas;

      interface_state.rhoe = godunov_energy_density_gas;
      for (int g = 0; g < NGROUPS; g++) {
          interface_state.rhoe += godunov_energy_density_group[g];
      }

    #else
      interface_state.p = amrex::max(interface_state.p, pressure_cutoff);
      interface_state.rhoe = godunov_energy_density;
    #endif

      interface_state.un = interface_state.un * interface_aux.boundary_factor;


    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    bisect_star_pressure(amrex::Real& star_pressure_lower, amrex::Real& star_pressure_upper,
                    const amrex::Real ul, const amrex::Real pl, const amrex::Real specific_volume_left,
                    const amrex::Real gamma_energy_left, const amrex::Real clsql,
                    const amrex::Real ur, const amrex::Real pr, const amrex::Real specific_volume_right,
                    const amrex::Real gamma_energy_right, const amrex::Real clsqr,
                    const amrex::Real gamma_drift_rate, const amrex::Real gamma_lower_bound, const amrex::Real gmax,
                    const int max_shock_iters, const amrex::Real pstar_tolerance,
                    amrex::Real& pstar, amrex::Real& gamma_energy_star,
                    bool& converged, amrex::GpuArray<amrex::Real, solver_constants::STAR_BISECT_SCALE * solver_constants::STAR_HISTORY_DEPTH>& star_pressure_bisect_history) {



      amrex::Real wlsq = 0.0;
      compute_wave_speed_sq(pl, specific_volume_left, gamma_energy_left, gamma_drift_rate,
             gamma_energy_star, gamma_lower_bound, gmax, clsql, star_pressure_lower, wlsq);

      amrex::Real wrsq = 0.0;
      compute_wave_speed_sq(pr, specific_volume_right, gamma_energy_right, gamma_drift_rate,
             gamma_energy_star, gamma_lower_bound, gmax, clsqr, star_pressure_lower, wrsq);

      amrex::Real wl = 1.0_rt / std::sqrt(wlsq);
      amrex::Real wr = 1.0_rt / std::sqrt(wrsq);

      amrex::Real star_velocity_left = ul - (star_pressure_lower - pstar)*wl;
      amrex::Real star_velocity_right = ur + (star_pressure_lower - pstar)*wr;

      amrex::Real f_lo = star_velocity_left - star_velocity_right;

      compute_wave_speed_sq(pl, specific_volume_left, gamma_energy_left, gamma_drift_rate,
            gamma_energy_star, gamma_lower_bound, gmax, clsql, star_pressure_upper, wlsq);

      compute_wave_speed_sq(pr, specific_volume_right, gamma_energy_right, gamma_drift_rate,
            gamma_energy_star, gamma_lower_bound, gmax, clsqr, star_pressure_upper, wrsq);




      converged = false;
      amrex::Real star_pressure_mid = 0.0;

      for (int iter = 0; iter < solver_constants::STAR_BISECT_SCALE * max_shock_iters; iter++) {

        star_pressure_mid = 0.5_rt * (star_pressure_lower + star_pressure_upper);
        star_pressure_bisect_history[iter] = star_pressure_mid;

        compute_wave_speed_sq(pl, specific_volume_left, gamma_energy_left, gamma_drift_rate,
              gamma_energy_star, gamma_lower_bound, gmax, clsql, star_pressure_mid, wlsq);

        compute_wave_speed_sq(pr, specific_volume_right, gamma_energy_right, gamma_drift_rate,
              gamma_energy_star, gamma_lower_bound, gmax, clsqr, star_pressure_mid, wrsq);

        wl = 1.0_rt / std::sqrt(wlsq);
        wr = 1.0_rt / std::sqrt(wrsq);

        star_velocity_left = ul - (star_pressure_mid - pl)*wl;
        star_velocity_right = ur - (star_pressure_mid - pr)*wr;

        amrex::Real f_c = star_velocity_left - star_velocity_right;

        if ( 0.5_rt * std::abs(star_pressure_lower - star_pressure_upper) < pstar_tolerance * star_pressure_mid ) {
          converged = true;
          break;
        }

        if (f_lo * f_c < 0.0_rt) {
          star_pressure_upper = star_pressure_mid;
        } else {
          star_pressure_lower = star_pressure_mid;
          f_lo = f_c;
        }
      }

      pstar = star_pressure_mid;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    shock_pair_iterate(const InterfaceState& ql, const InterfaceState& qr, const InterfaceAux& interface_aux,
              InterfaceState& interface_state) {


      constexpr amrex::Real weakwv = 1.e-3_rt;

    #ifndef AMREX_USE_GPU
      amrex::GpuArray<amrex::Real, solver_constants::STAR_HISTORY_DEPTH> star_pressure_history;
    #endif


      amrex::Real specific_volume_left = 1.0_rt / ql.rho;
      amrex::Real specific_volume_right = 1.0_rt / qr.rho;

      amrex::Real clsql = ql.gamma1_cell * ql.p * ql.rho;
      amrex::Real clsqr = qr.gamma1_cell * qr.p * qr.rho;

      amrex::Real gamma_energy_left = ql.p / ql.rhoe + 1.0_rt;
      amrex::Real gamma_energy_right = qr.p / qr.rhoe + 1.0_rt;

      amrex::Real gamma_lower_bound = amrex::min(gamma_energy_left, gamma_energy_right, 1.0_rt);
      amrex::Real gmax = amrex::max(gamma_energy_left, gamma_energy_right, 2.0_rt);

      amrex::Real gamma_energy_avg = 0.5_rt*(gamma_energy_left + gamma_energy_right);
      amrex::Real gamma1_avg = 0.5_rt*(ql.gamma1_cell + qr.gamma1_cell);

      amrex::Real gamma_drift_rate = 2.0_rt*(1.0_rt - gamma_energy_avg/gamma1_avg)*(gamma_energy_avg - 1.0_rt);

      amrex::Real impedance_floor = small_dens * interface_aux.sound_floor;
      amrex::Real wl = amrex::max(impedance_floor, std::sqrt(std::abs(clsql)));
      amrex::Real wr = amrex::max(impedance_floor, std::sqrt(std::abs(clsqr)));

      amrex::Real pstar = ql.p + ( (qr.p - ql.p) - wr*(qr.un - ql.un) ) * wl / (wl + wr);
      pstar = amrex::max(pstar, pressure_cutoff);

      amrex::Real gamma_energy_star = 0.0;

      amrex::Real wlsq = 0.0;
      compute_wave_speed_sq(ql.p, specific_volume_left, gamma_energy_left, gamma_drift_rate, gamma_energy_star,
            gamma_lower_bound, gmax, clsql, pstar, wlsq);

      amrex::Real wrsq = 0.0;
      compute_wave_speed_sq(qr.p, specific_volume_right, gamma_energy_right, gamma_drift_rate, gamma_energy_star,
            gamma_lower_bound, gmax, clsqr, pstar, wrsq);

      amrex::Real star_pressure_prev = pstar;

      wl = std::sqrt(wlsq);
      wr = std::sqrt(wrsq);

      amrex::Real star_velocity_left = ql.un - (pstar - ql.p) / wl;
      amrex::Real star_velocity_right = qr.un + (pstar - qr.p) / wr;

      pstar = ql.p + ( (qr.p - ql.p) - wr * (qr.un - ql.un) ) * wl / (wl + wr);
      pstar = amrex::max(pstar, pressure_cutoff);

      auto converged = false;

      int iter = 0;
      while ((iter < shock_star_max_iters && !converged) || iter < 2) {

          compute_wave_speed_sq(ql.p, specific_volume_left, gamma_energy_left, gamma_drift_rate, gamma_energy_star,
                gamma_lower_bound, gmax, clsql, pstar, wlsq);

          compute_wave_speed_sq(qr.p, specific_volume_right, gamma_energy_right, gamma_drift_rate, gamma_energy_star,
                gamma_lower_bound, gmax, clsqr, pstar, wrsq);


          wl = 1.0_rt / std::sqrt(wlsq);
          wr = 1.0_rt / std::sqrt(wrsq);

          amrex::Real star_velocity_right_prev = star_velocity_right;
          amrex::Real star_velocity_left_prev = star_velocity_left;

          star_velocity_right = qr.un - (qr.p - pstar) * wr;
          star_velocity_left = ql.un + (ql.p - pstar) * wl;

          amrex::Real dpditer = std::abs(star_pressure_prev - pstar);

          amrex::Real zp = std::abs(star_velocity_left - star_velocity_left_prev);
          if (zp - weakwv * interface_aux.mean_sound_speed <= 0.0_rt) {
              zp = dpditer * wl;
          }

          amrex::Real zm = std::abs(star_velocity_right - star_velocity_right_prev);
          if (zm - weakwv * interface_aux.mean_sound_speed <= 0.0_rt) {
              zm = dpditer * wr;
          }

          amrex::Real denom = dpditer / amrex::max(zp + zm, solver_constants::small * interface_aux.mean_sound_speed);
          star_pressure_prev = pstar;
          pstar = pstar - denom*(star_velocity_right - star_velocity_left);
          pstar = amrex::max(pstar, pressure_cutoff);

          amrex::Real err = std::abs(pstar - star_pressure_prev);
          if (err < riemann_star_tol * pstar) {
              converged = true;
          }

    #ifndef AMREX_USE_GPU
          star_pressure_history[iter] = pstar;
    #endif

          iter++;
      }


      if (!converged) {

          if (riemann_search_mode == 0) {

    #ifndef AMREX_USE_GPU
              std::cout <<  "pstar history: " << std::endl;
              for (int history_index=0; history_index < shock_star_max_iters; history_index++) {
                  std::cout << history_index << " " << star_pressure_history[history_index] << std::endl;
              }

              std::cout << std::endl;
              std::cout << "left state: " << std::endl << ql << std::endl;
              std::cout << "right state: " << std::endl << qr << std::endl;
              std::cout << "aux information: " << std::endl << interface_aux << std::endl;

              amrex::Error("ERROR: non-convergence in the Riemann solver");
    #endif

          } else if (riemann_search_mode == 1) {

              pstar = ql.p + ( (qr.p - ql.p) - wr * (qr.un - ql.un) ) * wl / (wl + wr);

          } else if (riemann_search_mode == 2) {

    #ifndef AMREX_USE_GPU
              amrex::Real star_pressure_min = 1.e200;
              amrex::Real star_pressure_max = -1.e200;
              for (int n = shock_star_max_iters-6; n < shock_star_max_iters; n++) {
                  star_pressure_min = std::min(star_pressure_min, star_pressure_history[n]);
                  star_pressure_max = std::max(star_pressure_max, star_pressure_history[n]);
              }

              star_pressure_min = amrex::max(star_pressure_min, pressure_cutoff);
              star_pressure_max = amrex::max(star_pressure_max, pressure_cutoff);

              amrex::GpuArray<amrex::Real, solver_constants::STAR_BISECT_SCALE * solver_constants::STAR_HISTORY_DEPTH> star_pressure_bisect_history;

              bisect_star_pressure(star_pressure_min, star_pressure_max,
                              ql.un, ql.p, specific_volume_left, gamma_energy_left, clsql,
                              qr.un, qr.p, specific_volume_right, gamma_energy_right, clsqr,
                              gamma_drift_rate, gamma_lower_bound, gmax,
                              shock_star_max_iters, riemann_star_tol,
                              pstar, gamma_energy_star, converged, star_pressure_bisect_history);

              if (!converged) {

                  std::cout << "pstar history: " << std::endl;
                  for (int history_index = 0; history_index < shock_star_max_iters; history_index++) {
                      std::cout << history_index << " " << star_pressure_history[history_index] << std::endl;
                  }
                  std::cout << "pstar extra history: " << std::endl;
                  for (int history_index = 0; history_index < solver_constants::STAR_BISECT_SCALE * shock_star_max_iters; history_index++) {
                      std::cout << history_index << " " << star_pressure_bisect_history[history_index] << std::endl;
                  }

                  std::cout << std::endl;
                  std::cout << "left state: " << std::endl << ql << std::endl;
                  std::cout << "right state: " << std::endl << qr << std::endl;
                  std::cout << "aux information: " << std::endl << interface_aux << std::endl;

                  amrex::Error("ERROR: non-convergence in the Riemann solver");
              }

    #endif
          } else {

    #ifndef AMREX_USE_GPU
              amrex::Error("ERROR: unrecognized riemann_search_mode option.");
    #endif
          }

      }

      star_velocity_right = qr.un - (qr.p - pstar) * wr;
      star_velocity_left = ql.un + (ql.p - pstar) * wl;

      amrex::Real ustar = 0.5_rt * (star_velocity_left + star_velocity_right);

      if (std::abs(ustar) < solver_constants::small_velocity_ratio * 0.5_rt * (std::abs(ql.un) + std::abs(qr.un))) {
          ustar = 0.0_rt;
      }

      amrex::Real ro;
      amrex::Real uo;
      amrex::Real po;
      amrex::Real specific_volume_upwind;
      amrex::Real upwind_gamma1;
      amrex::Real gamma_energy_upwind;

      if (ustar > 0.0_rt) {
          uo = ql.un;
          po = ql.p;
          specific_volume_upwind = specific_volume_left;
          upwind_gamma1 = ql.gamma1_cell;
          gamma_energy_upwind = gamma_energy_left;

      } else if (ustar < 0.0_rt) {
          uo = qr.un;
          po = qr.p;
          specific_volume_upwind = specific_volume_right;
          upwind_gamma1 = qr.gamma1_cell;
          gamma_energy_upwind = gamma_energy_right;

      } else {
          uo = 0.5_rt * (ql.un + qr.un);
          po = 0.5_rt * (ql.p + qr.p);
          specific_volume_upwind = 0.5_rt * (specific_volume_left + specific_volume_right);
          upwind_gamma1 = 0.5_rt * (ql.gamma1_cell + qr.gamma1_cell);
          gamma_energy_upwind = 0.5_rt * (gamma_energy_left + gamma_energy_right);
      }

      ro = amrex::max(small_dens, 1.0_rt/specific_volume_upwind);
      specific_volume_upwind = 1.0_rt/ro;

      amrex::Real co = std::sqrt(std::abs(upwind_gamma1*po*specific_volume_upwind));
      co = amrex::max(interface_aux.sound_floor, co);
      amrex::Real lagrangian_sound_sq = amrex::Math::powi<2>(co*ro);

      amrex::Real wosq = 0.0;
      compute_wave_speed_sq(po, specific_volume_upwind, gamma_energy_upwind, gamma_drift_rate, gamma_energy_star,
            gamma_lower_bound, gmax, lagrangian_sound_sq, pstar, wosq);

      amrex::Real sgnm = std::copysign(1.0_rt, ustar);

      amrex::Real wo = std::sqrt(wosq);
      amrex::Real dpjmp = pstar - po;

      amrex::Real rstar = 1.0_rt - ro*dpjmp/wosq;
      rstar = ro/rstar;
      rstar = amrex::max(small_dens, rstar);

      amrex::Real cstar = std::sqrt(std::abs(upwind_gamma1 * pstar / rstar));
      cstar = amrex::max(cstar, interface_aux.sound_floor);

      amrex::Real spout = co - sgnm*uo;
      amrex::Real spin = cstar - sgnm*ustar;

      amrex::Real shock_vel = wo*specific_volume_upwind - sgnm*uo;

      if (pstar - po >= 0.0_rt) {
          spin = shock_vel;
          spout = shock_vel;
      }

      amrex::Real frac = 0.5_rt*(1.0_rt + (spin + spout)/amrex::max(amrex::max(spout-spin, spin+spout),
                                                             solver_constants::small * interface_aux.mean_sound_speed));

      if (ustar > 0.0_rt) {
          interface_state.ut = ql.ut;
          interface_state.utt = ql.utt;
      } else if (ustar < 0.0_rt) {
          interface_state.ut = qr.ut;
          interface_state.utt = qr.utt;
      } else {
          interface_state.ut = 0.5_rt * (ql.ut + qr.ut);
          interface_state.utt = 0.5_rt * (ql.utt + qr.utt);
      }

      interface_state.rho = frac*rstar + (1.0_rt - frac)*ro;
      interface_state.un = frac*ustar + (1.0_rt - frac)*uo;
      interface_state.p = frac*pstar + (1.0_rt - frac)*po;
      amrex::Real gamma_energy_interface = frac*gamma_energy_star + (1.0_rt-frac)*gamma_energy_upwind;

      if (spout < 0.0_rt) {
          interface_state.rho = ro;
          interface_state.un = uo;
          interface_state.p = po;
          gamma_energy_interface = gamma_energy_upwind;
      }

      if (spin >= 0.0_rt) {
          interface_state.rho = rstar;
          interface_state.un = ustar;
          interface_state.p = pstar;
          gamma_energy_interface = gamma_energy_star;
      }

      interface_state.p = amrex::max(interface_state.p, pressure_cutoff);

      interface_state.un = interface_state.un * interface_aux.boundary_factor;

      interface_state.rhoe = interface_state.p / (gamma_energy_interface - 1.0_rt);


    }
}

#endif
