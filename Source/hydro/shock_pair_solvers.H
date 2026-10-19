#ifndef SHOCK_PAIR_SOLVERS_H
#define SHOCK_PAIR_SOLVERS_H

#include <AMReX_REAL.H>

using namespace amrex::literals;

#include <solver_constants.H>
#include <solver_types.H>
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

      // compute the lagrangian wave speeds -- this is the approximate
      // version for the Colella & Glaz algorithm


      // First predict a value of game across the shock

      gamma_star_value = (pstar-p)*gamma_drift_rate/(pstar+p) + gam;
      gamma_star_value = amrex::Clamp(gamma_star_value, gamma_lower_bound, gmax);

      // Now use that predicted value of game with the R-H jump conditions
      // to compute the wave speed.

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

      // we want to zero
      // f(p*) = u*_l(p*) - u*_r(p*)
      // we'll do bisection
      //
      // this version is for the approximate Colella & Glaz
      // version


      // lo bounds
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

      // hi bounds
      compute_wave_speed_sq(pl, specific_volume_left, gamma_energy_left, gamma_drift_rate,
            gamma_energy_star, gamma_lower_bound, gmax, clsql, star_pressure_upper, wlsq);

      compute_wave_speed_sq(pr, specific_volume_right, gamma_energy_right, gamma_drift_rate,
            gamma_energy_star, gamma_lower_bound, gmax, clsqr, star_pressure_upper, wrsq);

      // wl = 1.0_rt / std::sqrt(wlsq);
      // wr = 1.0_rt / std::sqrt(wrsq);

      //star_velocity_left = ul - (star_pressure_upper - pstar)*wl;
      //star_velocity_right = ur + (star_pressure_upper - pstar)*wr;

      //amrex::Real f_hi = star_velocity_left - star_velocity_right;

      // bisection
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
          // root is in the left half
          star_pressure_upper = star_pressure_mid;
          //f_hi = f_c;
        } else {
          star_pressure_lower = star_pressure_mid;
          f_lo = f_c;
        }
      }

      pstar = star_pressure_mid;
    }


    ///
    /// The Colella-Glaz Riemann solver for pure hydrodynamics.  This is a
    /// two shock approximate state Riemann solver.
    ///
    /// @param bx         the box to operate over
    /// @param ql         the left interface state
    /// @param qr         the right interface state
    /// @param qthermo_arr   the auxiliary state
    /// @param interface_state       the full Godunov state on the interface
    /// @param idir       coordinate direction for the solve (0 = x, 1 = y, 2 = z)
    ///
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    shock_pair_iterate(const InterfaceState& ql, const InterfaceState& qr, const InterfaceAux& interface_aux,
              InterfaceState& interface_state) {

      // this implements the approximate Riemann solver of Colella & Glaz
      //

      constexpr amrex::Real weakwv = 1.e-3_rt;

    #ifndef AMREX_USE_GPU
      amrex::GpuArray<amrex::Real, solver_constants::STAR_HISTORY_DEPTH> star_pressure_history;
    #endif


      // common quantities
      amrex::Real specific_volume_left = 1.0_rt / ql.rho;
      amrex::Real specific_volume_right = 1.0_rt / qr.rho;

      // lagrangian sound speeds
      amrex::Real clsql = ql.gamma1_cell * ql.p * ql.rho;
      amrex::Real clsqr = qr.gamma1_cell * qr.p * qr.rho;

      // Note: in the original Colella & Glaz, they predicted
      // gamma_e to the interfaces using a special (non-hyperbolic)
      // evolution equation.  In HydroSim, we instead bring (rho e)
      // to the edges, so we construct the necessary gamma_e here from
      // what we have on the interfaces.
      amrex::Real gamma_energy_left = ql.p / ql.rhoe + 1.0_rt;
      amrex::Real gamma_energy_right = qr.p / qr.rhoe + 1.0_rt;

      // these should consider a wider average of the cell-centered
      // gammas
      amrex::Real gamma_lower_bound = amrex::min(gamma_energy_left, gamma_energy_right, 1.0_rt);
      amrex::Real gmax = amrex::max(gamma_energy_left, gamma_energy_right, 2.0_rt);

      amrex::Real gamma_energy_avg = 0.5_rt*(gamma_energy_left + gamma_energy_right);
      amrex::Real gamma1_avg = 0.5_rt*(ql.gamma1_cell + qr.gamma1_cell);

      amrex::Real gamma_drift_rate = 2.0_rt*(1.0_rt - gamma_energy_avg/gamma1_avg)*(gamma_energy_avg - 1.0_rt);

      amrex::Real impedance_floor = small_dens * interface_aux.sound_floor;
      amrex::Real wl = amrex::max(impedance_floor, std::sqrt(std::abs(clsql)));
      amrex::Real wr = amrex::max(impedance_floor, std::sqrt(std::abs(clsqr)));

      // make an initial guess for pstar -- this is a two-shock
      // approximation
      //pstar = ((wr*pl + wl*pr) + wl*wr*(ul - ur))/(wl + wr)
      amrex::Real pstar = ql.p + ( (qr.p - ql.p) - wr*(qr.un - ql.un) ) * wl / (wl + wr);
      pstar = amrex::max(pstar, pressure_cutoff);

      // get the shock speeds -- this computes W_s
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

      // R-H jump conditions give ustar across each wave -- these
      // should be equal when we are done iterating.  Our notation
      // here is a little funny, comparing to CG, star_velocity_left = u*_L and
      // star_velocity_right = u*_R.
      amrex::Real star_velocity_left = ql.un - (pstar - ql.p) / wl;
      amrex::Real star_velocity_right = qr.un + (pstar - qr.p) / wr;

      // revise our pstar guess
      // pstar = ((wr*pl + wl*pr) + wl*wr*(ul - ur))/(wl + wr)
      pstar = ql.p + ( (qr.p - ql.p) - wr * (qr.un - ql.un) ) * wl / (wl + wr);
      pstar = amrex::max(pstar, pressure_cutoff);

      // secant iteration
      bool converged = false;

      int iter = 0;
      while ((iter < shock_star_max_iters && !converged) || iter < 2) {

          compute_wave_speed_sq(ql.p, specific_volume_left, gamma_energy_left, gamma_drift_rate, gamma_energy_star,
                gamma_lower_bound, gmax, clsql, pstar, wlsq);

          compute_wave_speed_sq(qr.p, specific_volume_right, gamma_energy_right, gamma_drift_rate, gamma_energy_star,
                gamma_lower_bound, gmax, clsqr, pstar, wrsq);


          // NOTE: these are really the inverses of the wave speeds!
          wl = 1.0_rt / std::sqrt(wlsq);
          wr = 1.0_rt / std::sqrt(wrsq);

          amrex::Real star_velocity_right_prev = star_velocity_right;
          amrex::Real star_velocity_left_prev = star_velocity_left;

          star_velocity_right = qr.un - (qr.p - pstar) * wr;
          star_velocity_left = ql.un + (ql.p - pstar) * wl;

          amrex::Real dpditer = std::abs(star_pressure_prev - pstar);

          // Here we are going to do the Secant iteration version in
          // CG.  Note that what we call zp and zm here are not
          // actually the Z_p = |dp*/du*_p| defined in CG, by rather
          // simply |du*_p| (or something that looks like dp/Z!).
          amrex::Real zp = std::abs(star_velocity_left - star_velocity_left_prev);
          if (zp - weakwv * interface_aux.mean_sound_speed <= 0.0_rt) {
              zp = dpditer * wl;
          }

          amrex::Real zm = std::abs(star_velocity_right - star_velocity_right_prev);
          if (zm - weakwv * interface_aux.mean_sound_speed <= 0.0_rt) {
              zm = dpditer * wr;
          }

          // the new pstar is found
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

      // If we failed to converge using the secant iteration, we
      // can either stop here; or, revert to the original
      // two-shock estimate for pstar; or do a bisection root
      // find using the bounds established by the most recent
      // iterations.

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

              // we don't store the history if we are in CUDA, so
              // we can't do this
    #ifndef AMREX_USE_GPU
              // first try to find a reasonable bounds
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

      // we converged!  construct the single ustar for the region
      // between the left and right waves, using the updated wave speeds
      star_velocity_right = qr.un - (qr.p - pstar) * wr;  // careful -- here wl, wr are 1/W
      star_velocity_left = ql.un + (ql.p - pstar) * wl;

      amrex::Real ustar = 0.5_rt * (star_velocity_left + star_velocity_right);

      // for symmetry preservation, if ustar is really small, then we
      // set it to zero
      if (std::abs(ustar) < solver_constants::small_velocity_ratio * 0.5_rt * (std::abs(ql.un) + std::abs(qr.un))) {
          ustar = 0.0_rt;
      }

      // sample the solution -- here we look first at the direction
      // that the contact is moving.  This tells us if we need to
      // worry about the L/L* states or the R*/R states.
      amrex::Real ro;
      amrex::Real uo;
      amrex::Real po;
      amrex::Real specific_volume_upwind;
      amrex::Real upwind_gamma1;
      amrex::Real gamma_energy_upwind;

      if (ustar > 0.0_rt) {
          //ro = ql.rho;
          uo = ql.un;
          po = ql.p;
          specific_volume_upwind = specific_volume_left;
          upwind_gamma1 = ql.gamma1_cell;
          gamma_energy_upwind = gamma_energy_left;

      } else if (ustar < 0.0_rt) {
          //ro = qr.rho;
          uo = qr.un;
          po = qr.p;
          specific_volume_upwind = specific_volume_right;
          upwind_gamma1 = qr.gamma1_cell;
          gamma_energy_upwind = gamma_energy_right;

      } else {
          //ro = 0.5_rt * (ql.rho + qr.rho);
          uo = 0.5_rt * (ql.un + qr.un);
          po = 0.5_rt * (ql.p + qr.p);
          specific_volume_upwind = 0.5_rt * (specific_volume_left + specific_volume_right);
          upwind_gamma1 = 0.5_rt * (ql.gamma1_cell + qr.gamma1_cell);
          gamma_energy_upwind = 0.5_rt * (gamma_energy_left + gamma_energy_right);
      }

      // use tau = 1/rho as the independent variable here
      ro = amrex::max(small_dens, 1.0_rt/specific_volume_upwind);
      specific_volume_upwind = 1.0_rt/ro;

      amrex::Real co = std::sqrt(std::abs(upwind_gamma1*po*specific_volume_upwind));
      co = amrex::max(interface_aux.sound_floor, co);
      amrex::Real lagrangian_sound_sq = amrex::Math::powi<2>(co*ro);

      // now that we know which state (left or right) we need to worry
      // about, get the value of gamma_energy_star and wosq across the wave we
      // are dealing with.
      amrex::Real wosq = 0.0;
      compute_wave_speed_sq(po, specific_volume_upwind, gamma_energy_upwind, gamma_drift_rate, gamma_energy_star,
            gamma_lower_bound, gmax, lagrangian_sound_sq, pstar, wosq);

      amrex::Real sgnm = std::copysign(1.0_rt, ustar);

      amrex::Real wo = std::sqrt(wosq);
      amrex::Real dpjmp = pstar - po;

      // is this max really necessary?
      //rstar=max(ONE-ro*dpjmp/wosq, (gamma_energy_upwind-ONE)/(gamma_energy_upwind+ONE))
      amrex::Real rstar = 1.0_rt - ro*dpjmp/wosq;
      rstar = ro/rstar;
      rstar = amrex::max(small_dens, rstar);

      amrex::Real cstar = std::sqrt(std::abs(upwind_gamma1 * pstar / rstar));
      cstar = amrex::max(cstar, interface_aux.sound_floor);

      amrex::Real spout = co - sgnm*uo;
      amrex::Real spin = cstar - sgnm*ustar;

      //shock_vel = 0.5_rt*(spin + spout)
      amrex::Real shock_vel = wo*specific_volume_upwind - sgnm*uo;

      if (pstar - po >= 0.0_rt) {
          spin = shock_vel;
          spout = shock_vel;
      }

      amrex::Real frac = 0.5_rt*(1.0_rt + (spin + spout)/amrex::max(amrex::max(spout-spin, spin+spout),
                                                             solver_constants::small * interface_aux.mean_sound_speed));

      // the transverse velocity states only depend on the
      // direction that the contact moves
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

      // linearly interpolate between the star and normal state -- this covers the
      // case where we are inside the rarefaction fan.
      interface_state.rho = frac*rstar + (1.0_rt - frac)*ro;
      interface_state.un = frac*ustar + (1.0_rt - frac)*uo;
      interface_state.p = frac*pstar + (1.0_rt - frac)*po;
      amrex::Real gamma_energy_interface = frac*gamma_energy_star + (1.0_rt-frac)*gamma_energy_upwind;

      // now handle the cases where instead we are fully in the
      // star or fully in the original (l/r) state
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

      // compute the total energy from the internal, p/(gamma - 1), and the kinetic
      interface_state.rhoe = interface_state.p / (gamma_energy_interface - 1.0_rt);

      // we'll do the passive scalars separately

    }


    ///
    /// The Colella-Glaz-Ferguson Riemann solver for hydrodynamics and
    /// radiation hydrodynamics.  This is a two shock approximate state
    /// Riemann solver.
    ///
    /// @param bx          the box to operate over
    /// @param ql          the left interface state
    /// @param qr          the right interface state
    /// @param qthermo_arr    the auxiliary state
    /// @param interface_state        the full Godunov state on the interface
    /// @param lambda_int  the radiation flux limiter on the interface
    ///
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    shock_pair_direct(const InterfaceState& ql, const InterfaceState& qr, const InterfaceAux& interface_aux,
              InterfaceState& interface_state) {

      // Colella, Glaz, and Ferguson solver
      //
      // this is a 2-shock solver that uses a very simple approximation for the
      // star state, and carries an auxiliary jump condition for (rho e) to
      // deal with a real gas


      // estimate the star state: pstar, ustar

      amrex::Real impedance_floor = small_dens * interface_aux.sound_floor;

      amrex::Real wl = amrex::max(impedance_floor, std::sqrt(std::abs(ql.gamma1_cell * ql.p * ql.rho)));
      amrex::Real wr = amrex::max(impedance_floor, std::sqrt(std::abs(qr.gamma1_cell * qr.p * qr.rho)));

      amrex::Real inv_impedance_sum = 1.0_rt/(wl + wr);
      amrex::Real pstar = ((wr * ql.p + wl * qr.p) + wl * wr * (ql.un - qr.un)) * inv_impedance_sum;
      amrex::Real ustar = ((wl * ql.un + wr * qr.un) + (ql.p - qr.p)) * inv_impedance_sum;

      pstar = amrex::max(pstar, pressure_cutoff);

      // for symmetry preservation, if ustar is really small, then we
      // set it to zero

      if (std::abs(ustar) < solver_constants::small_velocity_ratio * 0.5_rt * (std::abs(ql.un) + std::abs(qr.un))) {
          ustar = 0.0_rt;
      }

      // look at the contact to determine which region we are in

      // this just determines which of the left or right states is still
      // in play.  We still need to look at the other wave to determine
      // if the star state or this state is on the interface.
      amrex::Real sgnm = std::copysign(1.0_rt, ustar);
      if (ustar == 0.0_rt) {
          sgnm = 0.0_rt;
      }

      amrex::Real fp = 0.5_rt*(1.0_rt + sgnm);
      amrex::Real fm = 0.5_rt*(1.0_rt - sgnm);

      amrex::Real ro = fp * ql.rho + fm * qr.rho;
      amrex::Real uo = fp * ql.un + fm * qr.un;
      amrex::Real po = fp * ql.p + fm * qr.p;
      amrex::Real reo = fp * ql.rhoe + fm * qr.rhoe;
      amrex::Real upwind_gamma1 = fp * ql.gamma1_cell + fm * qr.gamma1_cell;
    #ifdef RADIATION
      for (int g = 0; g < NGROUPS; g++) {
          interface_state.lam[g] = fp * ql.lam[g] + fm * qr.lam[g];
      }

      if (ustar == 0) {
          // harmonic average
          for (int g = 0; g < NGROUPS; g++) {
              interface_state.lam[g] = 2.0_rt * (ql.lam[g] * qr.lam[g]) / (ql.lam[g] + qr.lam[g] + 1.e-50_rt);
          }
      }

      amrex::Real upwind_pressure_gas = fp * ql.p_g + fm * qr.p_g;
      amrex::Real upwind_energy_density_group[NGROUPS];
      amrex::Real upwind_pressure_group[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
          upwind_energy_density_group[g] = fp * ql.er[g] + fm * qr.er[g];
          upwind_pressure_group[g] = interface_state.lam[g] * upwind_energy_density_group[g];
      }
      amrex::Real upwind_energy_density_gas = fp * ql.energy_density_gas + fm * qr.energy_density_gas;
      amrex::Real gamma1_upwind_gas = fp * ql.gamma1_gas + fm * qr.gamma1_gas;
    #endif

      ro = amrex::max(small_dens, ro);

      amrex::Real inv_upwind_density = 1.0_rt / ro;

      amrex::Real co = std::sqrt(std::abs(upwind_gamma1 * po * inv_upwind_density));
      co = amrex::max(interface_aux.sound_floor, co);
      amrex::Real inv_sound_squared = 1.0_rt / (co*co);

      // we can already deal with the transverse velocities -- they
      // only jump across the contact

      interface_state.ut = fp * ql.ut + fm * qr.ut;
      interface_state.utt = fp * ql.utt + fm * qr.utt;

      // compute the rest of the star state

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

      // finish sampling the solution

      // look at the remaining wave to determine if the star state or the
      // 'o' state above is on the interface

      // the values of u +/- c on either side of the non-contact wave
      amrex::Real spout = co - sgnm*uo;
      amrex::Real spin = cstar - sgnm*ustar;

      // a simple estimate of the shock speed
      amrex::Real shock_vel = 0.5_rt*(spin + spout);

      if (pstar-po > 0.0_rt) {
          spin = shock_vel;
          spout = shock_vel;
      }

      amrex::Real scr = spout - spin;
      if (spout-spin == 0.0_rt) {
          scr = solver_constants::small * interface_aux.mean_sound_speed;
      }

      // interpolate for the case that we are in a rarefaction
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

      // as it stands now, we set things assuming that the rarefaction
      // spans the interface.  We overwrite that here depending on the
      // wave speeds

      // look at the speeds on either side of the remaining wave
      // to determine which region we are in
      if (spout < 0.0_rt) {
          // the l or r state is on the interface
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
          // the star state is on the interface
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

      // Enforce that fluxes through a symmetry plane or wall are hard zero.
      interface_state.un = interface_state.un * interface_aux.boundary_factor;

      // we'll do the passive scalars separately

    }
}
#endif
