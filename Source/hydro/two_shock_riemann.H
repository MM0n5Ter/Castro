#ifndef TWO_SHOCK_RIEMANN_H
#define TWO_SHOCK_RIEMANN_H

#include <AMReX_REAL.H>
#include <AMReX_Tuple.H>

using namespace amrex::literals;

#include <riemann_constants.H>
#include <riemann_type.H>
#ifdef RADIATION
#include <Radiation.H>
#include <rad_util.H>
#endif

namespace TwoShock {

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::GpuTuple<amrex::Real, amrex::Real>
    lagrangian_wave_sq(const amrex::Real p, const amrex::Real v,
          const amrex::Real gam, const amrex::Real gamma_slope,
          const amrex::Real gmin, const amrex::Real gmax, const amrex::Real csq,
          const amrex::Real pstar) {

      // compute the lagrangian wave speeds -- this is the approximate
      // version.  Returns the
      // predicted game across the shock and the squared wave speed.


      // First predict a value of game across the shock

      amrex::Real gstar = (pstar-p)*gamma_slope/(pstar+p) + gam;
      gstar = amrex::Clamp(gstar, gmin, gmax);

      // Now use that predicted value of game with the R-H jump conditions
      // to compute the wave speed.

      amrex::Real alpha = pstar - (gstar - 1.0_rt)*p/(gam - 1.0_rt);
      if (alpha == 0.0_rt) {
        alpha = riemann_constants::small_p_frac*(pstar + p);
      }

      amrex::Real beta = pstar + 0.5_rt*(gstar - 1.0_rt)*(pstar+p);

      amrex::Real wsq = (pstar-p)*beta/(v*alpha);

      if (std::abs(pstar - p) < riemann_constants::small_p_frac*(pstar + p)) {
        wsq = csq;
      }
      wsq = amrex::max(wsq, (0.5_rt * (gam - 1.0_rt)/gam)*csq);

      return {gstar, wsq};
    }


    ///
    /// one side of the interface in the Lagrangian variables of the
    /// two-shock iteration: velocity, pressure, specific volume,
    /// game and the Lagrangian sound speed squared
    ///
    struct LagrangianState
    {
        amrex::Real u;
        amrex::Real p;
        amrex::Real tau;
        amrex::Real gam;
        amrex::Real csq;
    };


    ///
    /// the game bounds and the iteration limits of the pstar bisection
    ///
    struct BisectionControls
    {
        amrex::Real gamma_slope;
        amrex::Real gmin;
        amrex::Real gmax;
        int maxiter;
        amrex::Real tol;
    };


    ///
    /// the outcome of the pstar bisection
    ///
    struct BisectionResult
    {
        amrex::Real pstar;
        amrex::Real gamstar;
        bool converged;
    };


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    BisectionResult
    pstar_bisection(amrex::Real& pstar_lo, amrex::Real& pstar_hi,
                    LagrangianState const& left, LagrangianState const& right,
                    BisectionControls const& ctrl, const amrex::Real pstar,
                    amrex::GpuArray<amrex::Real, riemann_constants::PSTAR_BISECT_FACTOR * riemann_constants::HISTORY_SIZE>& pstar_hist_extra) {

      // we want to zero
      // f(p*) = u*_l(p*) - u*_r(p*)
      // we'll do bisection


      // lo bounds
      amrex::Real gamstar = 0.0;

      amrex::Real wlsq = 0.0;
      amrex::Tie(gamstar, wlsq) = lagrangian_wave_sq(left.p, left.tau, left.gam, ctrl.gamma_slope,
                                        ctrl.gmin, ctrl.gmax, left.csq, pstar_lo);

      amrex::Real wrsq = 0.0;
      amrex::Tie(gamstar, wrsq) = lagrangian_wave_sq(right.p, right.tau, right.gam, ctrl.gamma_slope,
                                        ctrl.gmin, ctrl.gmax, right.csq, pstar_lo);

      amrex::Real wl = 1.0_rt / std::sqrt(wlsq);
      amrex::Real wr = 1.0_rt / std::sqrt(wrsq);

      amrex::Real ustar_l = left.u - (pstar_lo - pstar)*wl;
      amrex::Real ustar_r = right.u + (pstar_lo - pstar)*wr;

      amrex::Real f_lo = ustar_l - ustar_r;

      // hi bounds
      amrex::Tie(gamstar, wlsq) = lagrangian_wave_sq(left.p, left.tau, left.gam, ctrl.gamma_slope,
                                        ctrl.gmin, ctrl.gmax, left.csq, pstar_hi);

      amrex::Tie(gamstar, wrsq) = lagrangian_wave_sq(right.p, right.tau, right.gam, ctrl.gamma_slope,
                                        ctrl.gmin, ctrl.gmax, right.csq, pstar_hi);

      // wl = 1.0_rt / std::sqrt(wlsq);
      // wr = 1.0_rt / std::sqrt(wrsq);

      //ustar_l = left.u - (pstar_hi - pstar)*wl;
      //ustar_r = right.u + (pstar_hi - pstar)*wr;

      //amrex::Real f_hi = ustar_l - ustar_r;

      // bisection
      bool converged = false;
      amrex::Real pstar_c = 0.0;

      for (int iter = 0; iter < riemann_constants::PSTAR_BISECT_FACTOR * ctrl.maxiter; iter++) {

        pstar_c = 0.5_rt * (pstar_lo + pstar_hi);
        pstar_hist_extra[iter] = pstar_c;

        amrex::Tie(gamstar, wlsq) = lagrangian_wave_sq(left.p, left.tau, left.gam, ctrl.gamma_slope,
                                          ctrl.gmin, ctrl.gmax, left.csq, pstar_c);

        amrex::Tie(gamstar, wrsq) = lagrangian_wave_sq(right.p, right.tau, right.gam, ctrl.gamma_slope,
                                          ctrl.gmin, ctrl.gmax, right.csq, pstar_c);

        wl = 1.0_rt / std::sqrt(wlsq);
        wr = 1.0_rt / std::sqrt(wrsq);

        ustar_l = left.u - (pstar_c - left.p)*wl;
        ustar_r = right.u - (pstar_c - right.p)*wr;

        amrex::Real f_c = ustar_l - ustar_r;

        if ( 0.5_rt * std::abs(pstar_lo - pstar_hi) < ctrl.tol * pstar_c ) {
          converged = true;
          break;
        }

        if (f_lo * f_c < 0.0_rt) {
          // root is in the left half
          pstar_hi = pstar_c;
          //f_hi = f_c;
        } else {
          pstar_lo = pstar_c;
          f_lo = f_c;
        }
      }

      return {pstar_c, gamstar, converged};
    }


    ///
    /// The Riemann solver for pure hydrodynamics.  This is a
    /// two shock approximate state Riemann solver.
    ///
    /// @param bx         the box to operate over
    /// @param ql         the left interface state
    /// @param qr         the right interface state
    /// @param qaux_arr   the auxiliary state
    /// @param q_riemann       the full Godunov state on the interface
    /// @param idir       coordinate direction for the solve (0 = x, 1 = y, 2 = z)
    ///
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    riemann_two_shock_iterative(const RiemannVars& ql, const RiemannVars& qr, const RiemannParams& riemann_aux,
              RiemannVars& q_riemann) {

      constexpr amrex::Real weakwv = 1.e-3_rt;

    #ifndef AMREX_USE_GPU
      amrex::GpuArray<amrex::Real, riemann_constants::HISTORY_SIZE> pstar_hist;
    #endif


      // common quantities
      amrex::Real taul = 1.0_rt / ql.rho;
      amrex::Real taur = 1.0_rt / qr.rho;

      // lagrangian sound speeds
      amrex::Real clsql = ql.gamma1 * ql.p * ql.rho;
      amrex::Real clsqr = qr.gamma1 * qr.p * qr.rho;

      // Note: in the original, they predicted
      // gamma_e to the interfaces using a special (non-hyperbolic)
      // evolution equation.  In HydroSim, we instead bring (rho e)
      // to the edges, so we construct the necessary gamma_e here from
      // what we have on the interfaces.
      amrex::Real gamma_e_l = ql.p / ql.rhoe + 1.0_rt;
      amrex::Real gamma_e_r = qr.p / qr.rhoe + 1.0_rt;

      // these should consider a wider average of the cell-centered
      // gammas
      amrex::Real gmin = amrex::min(gamma_e_l, gamma_e_r, 1.0_rt);
      amrex::Real gmax = amrex::max(gamma_e_l, gamma_e_r, 2.0_rt);

      amrex::Real gamma_e_avg = 0.5_rt*(gamma_e_l + gamma_e_r);
      amrex::Real gamma1_avg = 0.5_rt*(ql.gamma1 + qr.gamma1);

      amrex::Real gamma_slope = 2.0_rt*(1.0_rt - gamma_e_avg/gamma1_avg)*(gamma_e_avg - 1.0_rt);

      amrex::Real wsmall = small_dens * riemann_aux.cs_floor;
      amrex::Real wl = amrex::max(wsmall, std::sqrt(std::abs(clsql)));
      amrex::Real wr = amrex::max(wsmall, std::sqrt(std::abs(clsqr)));

      // make an initial guess for pstar -- this is a two-shock
      // approximation
      //pstar = ((wr*pl + wl*pr) + wl*wr*(ul - ur))/(wl + wr)
      amrex::Real pstar = ql.p + ( (qr.p - ql.p) - wr*(qr.vel.un - ql.vel.un) ) * wl / (wl + wr);
      pstar = amrex::max(pstar, small_pres);

      // get the shock speeds -- this computes W_s
      amrex::Real gamstar = 0.0;

      amrex::Real wlsq = 0.0;
      amrex::Tie(gamstar, wlsq) = lagrangian_wave_sq(ql.p, taul, gamma_e_l, gamma_slope,
                                        gmin, gmax, clsql, pstar);

      amrex::Real wrsq = 0.0;
      amrex::Tie(gamstar, wrsq) = lagrangian_wave_sq(qr.p, taur, gamma_e_r, gamma_slope,
                                        gmin, gmax, clsqr, pstar);

      amrex::Real pstar_old = pstar;

      wl = std::sqrt(wlsq);
      wr = std::sqrt(wrsq);

      // R-H jump conditions give ustar across each wave -- these
      // should be equal when we are done iterating.  Our notation
      // here is a little funny, ustar_l = u*_L and
      // ustar_r = u*_R.
      amrex::Real ustar_l = ql.vel.un - (pstar - ql.p) / wl;
      amrex::Real ustar_r = qr.vel.un + (pstar - qr.p) / wr;

      // revise our pstar guess
      // pstar = ((wr*pl + wl*pr) + wl*wr*(ul - ur))/(wl + wr)
      pstar = ql.p + ( (qr.p - ql.p) - wr * (qr.vel.un - ql.vel.un) ) * wl / (wl + wr);
      pstar = amrex::max(pstar, small_pres);

      // secant iteration
      bool converged = false;

      int iter = 0;
      while ((iter < riemann_shock_maxiter && !converged) || iter < 2) {

          amrex::Tie(gamstar, wlsq) = lagrangian_wave_sq(ql.p, taul, gamma_e_l, gamma_slope,
                                            gmin, gmax, clsql, pstar);

          amrex::Tie(gamstar, wrsq) = lagrangian_wave_sq(qr.p, taur, gamma_e_r, gamma_slope,
                                            gmin, gmax, clsqr, pstar);


          // NOTE: these are really the inverses of the wave speeds!
          wl = 1.0_rt / std::sqrt(wlsq);
          wr = 1.0_rt / std::sqrt(wrsq);

          amrex::Real ustar_r_old = ustar_r;
          amrex::Real ustar_l_old = ustar_l;

          ustar_r = qr.vel.un - (qr.p - pstar) * wr;
          ustar_l = ql.vel.un + (ql.p - pstar) * wl;

          amrex::Real dpditer = std::abs(pstar_old - pstar);

          // Here we are going to do the Secant iteration version.
          // Note that what we call zp and zm here are not
          // actually the Z_p = |dp*/du*_p|, by rather
          // simply |du*_p| (or something that looks like dp/Z!).
          amrex::Real zp = std::abs(ustar_l - ustar_l_old);
          if (zp - weakwv * riemann_aux.cs_mean <= 0.0_rt) {
              zp = dpditer * wl;
          }

          amrex::Real zm = std::abs(ustar_r - ustar_r_old);
          if (zm - weakwv * riemann_aux.cs_mean <= 0.0_rt) {
              zm = dpditer * wr;
          }

          // the new pstar is found
          amrex::Real denom = dpditer / amrex::max(zp + zm, riemann_constants::small * riemann_aux.cs_mean);
          pstar_old = pstar;
          pstar = pstar - denom*(ustar_r - ustar_l);
          pstar = amrex::max(pstar, small_pres);

          amrex::Real err = std::abs(pstar - pstar_old);
          if (err < riemann_pstar_tol * pstar) {
              converged = true;
          }

    #ifndef AMREX_USE_GPU
          pstar_hist[iter] = pstar;
    #endif

          iter++;
      }

      // If we failed to converge using the secant iteration, we
      // can either stop here; or, revert to the original
      // two-shock estimate for pstar; or do a bisection root
      // find using the bounds established by the most recent
      // iterations.

      if (!converged) {

          if (riemann_cg_blend == 0) {

    #ifndef AMREX_USE_GPU
              std::cout <<  "pstar history: " << std::endl;
              for (int iter_l=0; iter_l < riemann_shock_maxiter; iter_l++) {
                  std::cout << iter_l << " " << pstar_hist[iter_l] << std::endl;
              }

              std::cout << std::endl;
              std::cout << "left state: " << std::endl << ql << std::endl;
              std::cout << "right state: " << std::endl << qr << std::endl;
              std::cout << "aux information: " << std::endl << riemann_aux << std::endl;

              amrex::Error("ERROR: non-convergence in the Riemann solver");
    #endif

          } else if (riemann_cg_blend == 1) {

              pstar = ql.p + ( (qr.p - ql.p) - wr * (qr.vel.un - ql.vel.un) ) * wl / (wl + wr);

          } else if (riemann_cg_blend == 2) {

              // we don't store the history if we are in CUDA, so
              // we can't do this
    #ifndef AMREX_USE_GPU
              // first try to find a reasonable bounds
              amrex::Real pstarl = 1.e200;
              amrex::Real pstaru = -1.e200;
              for (int n = riemann_shock_maxiter-6; n < riemann_shock_maxiter; n++) {
                  pstarl = std::min(pstarl, pstar_hist[n]);
                  pstaru = std::max(pstaru, pstar_hist[n]);
              }

              pstarl = amrex::max(pstarl, small_pres);
              pstaru = amrex::max(pstaru, small_pres);

              amrex::GpuArray<amrex::Real, riemann_constants::PSTAR_BISECT_FACTOR * riemann_constants::HISTORY_SIZE> pstar_hist_extra;

              const LagrangianState left{ql.vel.un, ql.p, taul, gamma_e_l, clsql};
              const LagrangianState right{qr.vel.un, qr.p, taur, gamma_e_r, clsqr};
              const BisectionControls ctrl{gamma_slope, gmin, gmax,
                                           riemann_shock_maxiter, riemann_pstar_tol};

              const auto bisection = pstar_bisection(pstarl, pstaru,
                                                     left, right, ctrl, pstar,
                                                     pstar_hist_extra);
              pstar = bisection.pstar;
              gamstar = bisection.gamstar;
              converged = bisection.converged;

              if (!converged) {

                  std::cout << "pstar history: " << std::endl;
                  for (int iter_l = 0; iter_l < riemann_shock_maxiter; iter_l++) {
                      std::cout << iter_l << " " << pstar_hist[iter_l] << std::endl;
                  }
                  std::cout << "pstar extra history: " << std::endl;
                  for (int iter_l = 0; iter_l < riemann_constants::PSTAR_BISECT_FACTOR * riemann_shock_maxiter; iter_l++) {
                      std::cout << iter_l << " " << pstar_hist_extra[iter_l] << std::endl;
                  }

                  std::cout << std::endl;
                  std::cout << "left state: " << std::endl << ql << std::endl;
                  std::cout << "right state: " << std::endl << qr << std::endl;
                  std::cout << "aux information: " << std::endl << riemann_aux << std::endl;

                  amrex::Error("ERROR: non-convergence in the Riemann solver");
              }

    #endif
          } else {

    #ifndef AMREX_USE_GPU
              amrex::Error("ERROR: unrecognized riemann_cg_blend option.");
    #endif
          }

      }

      // we converged!  construct the single ustar for the region
      // between the left and right waves, using the updated wave speeds
      ustar_r = qr.vel.un - (qr.p - pstar) * wr;  // careful -- here wl, wr are 1/W
      ustar_l = ql.vel.un + (ql.p - pstar) * wl;

      amrex::Real ustar = 0.5_rt * (ustar_l + ustar_r);

      // for symmetry preservation, if ustar is really small, then we
      // set it to zero
      if (std::abs(ustar) < riemann_constants::smallu * 0.5_rt * (std::abs(ql.vel.un) + std::abs(qr.vel.un))) {
          ustar = 0.0_rt;
      }

      // sample the solution -- here we look first at the direction
      // that the contact is moving.  This tells us if we need to
      // worry about the L/L* states or the R*/R states.
      amrex::Real ro;
      amrex::Real uo;
      amrex::Real po;
      amrex::Real tauo;
      amrex::Real gamma1_o;
      amrex::Real gamma_e_o;

      if (ustar > 0.0_rt) {
          //ro = ql.rho;
          uo = ql.vel.un;
          po = ql.p;
          tauo = taul;
          gamma1_o = ql.gamma1;
          gamma_e_o = gamma_e_l;

      } else if (ustar < 0.0_rt) {
          //ro = qr.rho;
          uo = qr.vel.un;
          po = qr.p;
          tauo = taur;
          gamma1_o = qr.gamma1;
          gamma_e_o = gamma_e_r;

      } else {
          //ro = 0.5_rt * (ql.rho + qr.rho);
          uo = 0.5_rt * (ql.vel.un + qr.vel.un);
          po = 0.5_rt * (ql.p + qr.p);
          tauo = 0.5_rt * (taul + taur);
          gamma1_o = 0.5_rt * (ql.gamma1 + qr.gamma1);
          gamma_e_o = 0.5_rt * (gamma_e_l + gamma_e_r);
      }

      // use tau = 1/rho as the independent variable here
      ro = amrex::max(small_dens, 1.0_rt/tauo);
      tauo = 1.0_rt/ro;

      amrex::Real co = std::sqrt(std::abs(gamma1_o*po*tauo));
      co = amrex::max(riemann_aux.cs_floor, co);
      amrex::Real clsq = amrex::Math::powi<2>(co*ro);

      // now that we know which state (left or right) we need to worry
      // about, get the value of gamstar and wosq across the wave we
      // are dealing with.
      amrex::Real wosq = 0.0;
      amrex::Tie(gamstar, wosq) = lagrangian_wave_sq(po, tauo, gamma_e_o, gamma_slope,
                                        gmin, gmax, clsq, pstar);

      amrex::Real sgnm = std::copysign(1.0_rt, ustar);

      amrex::Real wo = std::sqrt(wosq);
      amrex::Real dpjmp = pstar - po;

      // is this max really necessary?
      //rstar=max(ONE-ro*dpjmp/wosq, (gamma_e_o-ONE)/(gamma_e_o+ONE))
      amrex::Real rstar = 1.0_rt - ro*dpjmp/wosq;
      rstar = ro/rstar;
      rstar = amrex::max(small_dens, rstar);

      amrex::Real cstar = std::sqrt(std::abs(gamma1_o * pstar / rstar));
      cstar = amrex::max(cstar, riemann_aux.cs_floor);

      amrex::Real spout = co - sgnm*uo;
      amrex::Real spin = cstar - sgnm*ustar;

      //ushock = 0.5_rt*(spin + spout)
      amrex::Real ushock = wo*tauo - sgnm*uo;

      if (pstar - po >= 0.0_rt) {
          spin = ushock;
          spout = ushock;
      }

      amrex::Real frac = 0.5_rt*(1.0_rt + (spin + spout)/amrex::max(amrex::max(spout-spin, spin+spout),
                                                             riemann_constants::small * riemann_aux.cs_mean));

      // the transverse velocity states only depend on the
      // direction that the contact moves
      if (ustar > 0.0_rt) {
          q_riemann.vel.ut = ql.vel.ut;
          q_riemann.vel.utt = ql.vel.utt;
      } else if (ustar < 0.0_rt) {
          q_riemann.vel.ut = qr.vel.ut;
          q_riemann.vel.utt = qr.vel.utt;
      } else {
          q_riemann.vel.ut = 0.5_rt * (ql.vel.ut + qr.vel.ut);
          q_riemann.vel.utt = 0.5_rt * (ql.vel.utt + qr.vel.utt);
      }

      // linearly interpolate between the star and normal state -- this covers the
      // case where we are inside the rarefaction fan.
      q_riemann.rho = frac*rstar + (1.0_rt - frac)*ro;
      q_riemann.vel.un = frac*ustar + (1.0_rt - frac)*uo;
      q_riemann.p = frac*pstar + (1.0_rt - frac)*po;
      amrex::Real gamma_e_face = frac*gamstar + (1.0_rt-frac)*gamma_e_o;

      // now handle the cases where instead we are fully in the
      // star or fully in the original (l/r) state
      if (spout < 0.0_rt) {
          q_riemann.rho = ro;
          q_riemann.vel.un = uo;
          q_riemann.p = po;
          gamma_e_face = gamma_e_o;
      }

      if (spin >= 0.0_rt) {
          q_riemann.rho = rstar;
          q_riemann.vel.un = ustar;
          q_riemann.p = pstar;
          gamma_e_face = gamstar;
      }

      q_riemann.p = amrex::max(q_riemann.p, small_pres);

      q_riemann.vel.un = q_riemann.vel.un * riemann_aux.boundary_scale;

      // compute the total energy from the internal, p/(gamma - 1), and the kinetic
      q_riemann.rhoe = q_riemann.p / (gamma_e_face - 1.0_rt);

      // we'll do the passive scalars separately

    }


    ///
    /// The Riemann solver for hydrodynamics and
    /// radiation hydrodynamics.  This is a two shock approximate state
    /// Riemann solver.
    ///
    /// @param bx          the box to operate over
    /// @param ql          the left interface state
    /// @param qr          the right interface state
    /// @param qaux_arr    the auxiliary state
    /// @param q_riemann        the full Godunov state on the interface
    /// @param lambda_int  the radiation flux limiter on the interface
    ///
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    riemann_two_shock_approx(const RiemannVars& ql, const RiemannVars& qr, const RiemannParams& riemann_aux,
              RiemannVars& q_riemann) {

      // this is a 2-shock solver that uses a very simple approximation for the
      // star state, and carries an auxiliary jump condition for (rho e) to
      // deal with a real gas


      // estimate the star state: pstar, ustar

      amrex::Real wsmall = small_dens * riemann_aux.cs_floor;

      amrex::Real wl = amrex::max(wsmall, std::sqrt(std::abs(ql.gamma1 * ql.p * ql.rho)));
      amrex::Real wr = amrex::max(wsmall, std::sqrt(std::abs(qr.gamma1 * qr.p * qr.rho)));

      amrex::Real wwinv = 1.0_rt/(wl + wr);
      amrex::Real pstar = ((wr * ql.p + wl * qr.p) + wl * wr * (ql.vel.un - qr.vel.un)) * wwinv;
      amrex::Real ustar = ((wl * ql.vel.un + wr * qr.vel.un) + (ql.p - qr.p)) * wwinv;

      pstar = amrex::max(pstar, small_pres);

      // for symmetry preservation, if ustar is really small, then we
      // set it to zero

      if (std::abs(ustar) < riemann_constants::smallu * 0.5_rt * (std::abs(ql.vel.un) + std::abs(qr.vel.un))) {
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
      amrex::Real uo = fp * ql.vel.un + fm * qr.vel.un;
      amrex::Real po = fp * ql.p + fm * qr.p;
      amrex::Real rhoe_o = fp * ql.rhoe + fm * qr.rhoe;
      amrex::Real gamma1_o = fp * ql.gamma1 + fm * qr.gamma1;
    #ifdef RADIATION
      for (int g = 0; g < NGROUPS; g++) {
          q_riemann.rad.lam[g] = fp * ql.rad.lam[g] + fm * qr.rad.lam[g];
      }

      if (ustar == 0) {
          // harmonic average
          for (int g = 0; g < NGROUPS; g++) {
              q_riemann.rad.lam[g] = 2.0_rt * (ql.rad.lam[g] * qr.rad.lam[g]) / (ql.rad.lam[g] + qr.rad.lam[g] + 1.e-50_rt);
          }
      }

      amrex::Real po_g = fp * ql.rad.p_g + fm * qr.rad.p_g;
      amrex::Real rhoe_o_r[NGROUPS];
      amrex::Real po_r[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
          rhoe_o_r[g] = fp * ql.rad.er[g] + fm * qr.rad.er[g];
          po_r[g] = q_riemann.rad.lam[g] * rhoe_o_r[g];
      }
      amrex::Real rhoe_o_g = fp * ql.rad.rhoe_g + fm * qr.rad.rhoe_g;
      amrex::Real gamma1_o_gas = fp * ql.rad.gamma1_gas + fm * qr.rad.gamma1_gas;
    #endif

      ro = amrex::max(small_dens, ro);

      amrex::Real roinv = 1.0_rt / ro;

      amrex::Real co = std::sqrt(std::abs(gamma1_o * po * roinv));
      co = amrex::max(riemann_aux.cs_floor, co);
      amrex::Real co2inv = 1.0_rt / (co*co);

      // we can already deal with the transverse velocities -- they
      // only jump across the contact

      q_riemann.vel.ut = fp * ql.vel.ut + fm * qr.vel.ut;
      q_riemann.vel.utt = fp * ql.vel.utt + fm * qr.vel.utt;

      // compute the rest of the star state

      amrex::Real drho = (pstar - po)*co2inv;
      amrex::Real rstar = ro + drho;
      rstar = amrex::max(small_dens, rstar);

    #ifdef RADIATION
      amrex::Real estar_g = rhoe_o_g + drho*(rhoe_o_g + po_g)*roinv;

      amrex::Real co_g = std::sqrt(std::abs(gamma1_o_gas*po_g*roinv));
      co_g = amrex::max(riemann_aux.cs_floor, co_g);

      amrex::Real pstar_g = po_g + drho*co_g*co_g;
      pstar_g = amrex::max(pstar_g, small_pres);

      amrex::Real estar_r[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
          estar_r[g] = rhoe_o_r[g] + drho*(rhoe_o_r[g] + po_r[g])*roinv;
      }
    #else
      amrex::Real enth_o = (rhoe_o + po)*roinv*co2inv;
      amrex::Real estar = rhoe_o + (pstar - po)*enth_o;
    #endif

      amrex::Real cstar = std::sqrt(std::abs(gamma1_o*pstar/rstar));
      cstar = amrex::max(cstar, riemann_aux.cs_floor);

      // finish sampling the solution

      // look at the remaining wave to determine if the star state or the
      // 'o' state above is on the interface

      // the values of u +/- c on either side of the non-contact wave
      amrex::Real spout = co - sgnm*uo;
      amrex::Real spin = cstar - sgnm*ustar;

      // a simple estimate of the shock speed
      amrex::Real ushock = 0.5_rt*(spin + spout);

      if (pstar-po > 0.0_rt) {
          spin = ushock;
          spout = ushock;
      }

      amrex::Real scr = spout - spin;
      if (spout-spin == 0.0_rt) {
          scr = riemann_constants::small * riemann_aux.cs_mean;
      }

      // interpolate for the case that we are in a rarefaction
      amrex::Real frac = (1.0_rt + (spout + spin)/scr)*0.5_rt;
      frac = amrex::Clamp(frac, 0.0_rt, 1.0_rt);

      q_riemann.rho = frac*rstar + (1.0_rt - frac)*ro;
      q_riemann.vel.un = frac*ustar + (1.0_rt - frac)*uo;

    #ifdef RADIATION
      amrex::Real pface_t = frac*pstar + (1.0_rt - frac)*po;
      amrex::Real pface_g = frac*pstar_g + (1.0_rt - frac)*po_g;
      amrex::Real reface_g = frac*estar_g + (1.0_rt - frac)*rhoe_o_g;
      amrex::Real reface_r[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
          reface_r[g] = frac*estar_r[g] + (1.0_rt - frac)*rhoe_o_r[g];
      }
    #else
      q_riemann.p = frac*pstar + (1.0_rt - frac)*po;
      amrex::Real reface = frac*estar + (1.0_rt - frac)*rhoe_o;
    #endif

      // as it stands now, we set things assuming that the rarefaction
      // spans the interface.  We overwrite that here depending on the
      // wave speeds

      // look at the speeds on either side of the remaining wave
      // to determine which region we are in
      if (spout < 0.0_rt) {
          // the l or r state is on the interface
          q_riemann.rho = ro;
          q_riemann.vel.un = uo;
    #ifdef RADIATION
          pface_t = po;
          pface_g = po_g;
          reface_g = rhoe_o_g;
          for (int g = 0; g < NGROUPS; g++) {
              reface_r[g] = rhoe_o_r[g];
          }
    #else
          q_riemann.p = po;
          reface = rhoe_o;
    #endif
      }

      if (spin >= 0.0_rt) {
          // the star state is on the interface
          q_riemann.rho = rstar;
          q_riemann.vel.un = ustar;
    #ifdef RADIATION
          pface_t = pstar;
          pface_g = pstar_g;
          reface_g = estar_g;
          for (int g = 0; g < NGROUPS; g++) {
              reface_r[g] = estar_r[g];
          }
    #else
          q_riemann.p = pstar;
          reface = estar;
    #endif
        }

    #ifdef RADIATION
      for (int g = 0; g < NGROUPS; g++) {
          q_riemann.rad.er[g] = amrex::max(reface_r[g], 0.0_rt);
      }

      q_riemann.rad.p_g = pface_g;
      q_riemann.p = pface_t;
      q_riemann.rad.rhoe_g = reface_g;

      q_riemann.rhoe = reface_g;
      for (int g = 0; g < NGROUPS; g++) {
          q_riemann.rhoe += reface_r[g];
      }

    #else
      q_riemann.p = amrex::max(q_riemann.p, small_pres);
      q_riemann.rhoe = reface;
    #endif

      // Enforce that fluxes through a symmetry plane or wall are hard zero.
      q_riemann.vel.un = q_riemann.vel.un * riemann_aux.boundary_scale;

      // we'll do the passive scalars separately

    }
}
#endif
