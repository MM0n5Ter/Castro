#ifndef HLL_SOLVERS_H
#define HLL_SOLVERS_H

#include <AMReX_REAL.H>

#include <riemann_type.H>

namespace HLL {

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    hllc_star_state(const int idir, const Real S_k, const Real S_c,
               GpuArray<Real, NQ> const& qstate, GpuArray<Real, NUM_STATE>& U) {

      Real u_k = 0.0;
      if (idir == 0) {
        u_k = qstate[QU];
      } else if (idir == 1) {
        u_k = qstate[QV];
      } else if (idir == 2) {
        u_k = qstate[QW];
      }

      Real star_factor = qstate[QRHO]*(S_k - u_k)/(S_k - S_c);
      U[URHO] = star_factor;

      if (idir == 0) {
        U[UMX]  = star_factor*S_c;
        U[UMY]  = star_factor*qstate[QV];
        U[UMZ]  = star_factor*qstate[QW];

      } else if (idir == 1) {
        U[UMX]  = star_factor*qstate[QU];
        U[UMY]  = star_factor*S_c;
        U[UMZ]  = star_factor*qstate[QW];

      } else {
        U[UMX]  = star_factor*qstate[QU];
        U[UMY]  = star_factor*qstate[QV];
        U[UMZ]  = star_factor*S_c;
      }

      U[UEDEN] = star_factor*(qstate[QRHOEI]/qstate[QRHO] +
                              0.5_rt*(qstate[QU]*qstate[QU] + qstate[QV]*qstate[QV] + qstate[QW]*qstate[QW]) +
                              (S_c - u_k)*(S_c + qstate[QPRES]/(qstate[QRHO]*(S_k - u_k))));
      U[UEINT] = star_factor*qstate[QRHOEI]/qstate[QRHO];

      U[UTEMP] = 0.0; // we don't evolve T

    #ifdef SHOCK_VAR
      U[USHOCK] = 0.0;
    #endif

    #ifdef NSE_NET
      U[UMU_P] = 0.0;
      U[UMU_N] = 0.0;
    #endif

      for (int ipassive = 0; ipassive < npassive; ipassive++) {
          const int n  = cons_passive_index(ipassive);
          const int nqs = prim_passive_index(ipassive);
          U[n] = star_factor*qstate[nqs];
      }
    }


    ///
    /// given a conserved state, compute the flux in direction idir
    ///
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    compute_flux(const int idir, const Real boundary_scale, const int coord,
                 GpuArray<Real, NUM_STATE> const& U, const Real p,
                 GpuArray<Real, NUM_STATE>& F) {

      Real u_flx = U[UMX+idir]/U[URHO];

      if (boundary_scale == 0) {
        u_flx = 0.0;
      }

      F[URHO] = U[URHO]*u_flx;

      F[UMX] = U[UMX]*u_flx;
      F[UMY] = U[UMY]*u_flx;
      F[UMZ] = U[UMZ]*u_flx;

      auto mom_check = momentum_flux_has_pressure(idir, idir, coord);

      if (mom_check) {
        // we do not include the pressure term in any non-Cartesian
        // coordinate directions
        F[UMX+idir] = F[UMX+idir] + p;
      }

      F[UEINT] = U[UEINT]*u_flx;
      F[UEDEN] = (U[UEDEN] + p)*u_flx;

      F[UTEMP] = 0.0;

    #ifdef SHOCK_VAR
      F[USHOCK] = 0.0;
    #endif

    #ifdef NSE_NET
      F[UMU_P] = 0.0;
      F[UMU_N] = 0.0;
    #endif

      for (int ipassive=0; ipassive < npassive; ipassive++) {
          const int n = cons_passive_index(ipassive);
          F[n] = U[n]*u_flx;
      }
    }

    ///
    /// convert the primitive variable state to the conserved state
    ///
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    cons_state(GpuArray<Real, NQ> const& qstate, GpuArray<Real, NUM_STATE>& U) {

      U[URHO] = qstate[QRHO];

      // since we advect all 3 velocity components regardless of dimension, this
      // will be general
      U[UMX]  = qstate[QRHO]*qstate[QU];
      U[UMY]  = qstate[QRHO]*qstate[QV];
      U[UMZ]  = qstate[QRHO]*qstate[QW];

      U[UEDEN] = qstate[QRHOEI] + 0.5_rt*qstate[QRHO]*(qstate[QU]*qstate[QU] + qstate[QV]*qstate[QV] + qstate[QW]*qstate[QW]);
      U[UEINT] = qstate[QRHOEI];

      // we don't care about T here, but initialize it to make NaN
      // checking happy
      U[UTEMP] = 0.0;

    #ifdef SHOCK_VAR
      U[USHOCK] = 0.0;
    #endif

    #ifdef NSE_NET
      U[UMU_P] = 0.0;
      U[UMU_N] = 0.0;
    #endif

      for (int ipassive = 0; ipassive < npassive; ipassive++) {
          const int n  = cons_passive_index(ipassive);
          const int nqs = prim_passive_index(ipassive);
          U[n] = qstate[QRHO]*qstate[nqs];
      }
    }


    ///
    /// A simple HLL Riemann solver for pure hydrodynamics.  This takes just a
    /// single interface's data and returns the HLL flux
    ///
    /// @param ql     the left interface state
    /// @param qr     the right interface state
    /// @param cl     sound speed on the left interface
    /// @param cr     sound speed on the right interface
    /// @param idir   coordinate direction for the solve (0 = x, 1 = y, 2 = z)
    /// @param coord  geometry type (0 = Cartesian, 1 = axisymmetric, 2 = spherical)
    /// @param f      the HLL fluxes
    ///
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    HLL(GpuArray<Real, NQ> const& ql, GpuArray<Real, NQ> const& qr,
        const Real cl, const Real cr,
        const int idir, const int coord,
        GpuArray<Real, NUM_STATE>& flux_hll) {

      // This is the HLLE solver.  We should apply it to zone averages
      // (not reconstructed states) at an interface in the presence of
      // shocks to avoid the odd-even decoupling / carbuncle phenomenon.


      constexpr Real small_hll = 1.e-10_rt;

      int ivel, ivel_t1, ivel_t2;
      int imom, imom_t1, imom_t2;

      if (idir == 0) {
        ivel = QU;
        ivel_t1 = QV;
        ivel_t2 = QW;

        imom = UMX;
        imom_t1 = UMY;
        imom_t2 = UMZ;

      } else if (idir == 1) {
        ivel = QV;
        ivel_t1 = QU;
        ivel_t2 = QW;

        imom = UMY;
        imom_t1 = UMX;
        imom_t2 = UMZ;

      } else {
        ivel = QW;
        ivel_t1 = QU;
        ivel_t2 = QV;

        imom = UMZ;
        imom_t1 = UMX;
        imom_t2 = UMY;
      }

      Real rhol_sqrt = std::sqrt(ql[QRHO]);
      Real rhor_sqrt = std::sqrt(qr[QRHO]);

      Real rhod = 1.0_rt/(rhol_sqrt + rhor_sqrt);


      // compute the average sound speed. This uses an approximation
      // that assumes gamma falls between 1
      // and 5/3
      Real cs_mean = std::sqrt( (rhol_sqrt*cl*cl + rhor_sqrt*cr*cr)*rhod +
                             0.5_rt*rhol_sqrt*rhor_sqrt*rhod*rhod*std::pow(qr[ivel] - ql[ivel], 2));


      // Roe eigenvalues
      Real uavg = (rhol_sqrt*ql[ivel] + rhor_sqrt*qr[ivel])*rhod;

      Real a1 = uavg - cs_mean;
      Real a4 = uavg + cs_mean;


      // signal speeds
      Real bl = std::min(a1, ql[ivel] - cl);
      Real br = std::max(a4, qr[ivel] + cr);

      Real bm = std::min(0.0_rt, bl);
      Real bp = std::max(0.0_rt, br);

      Real bd = bp - bm;

      if (std::abs(bd) < small_hll * std::max(std::abs(bm), std::abs(bp))) {
          return;
      }

      // we'll overwrite the passed in flux with the HLL flux

      bd = 1.0_rt/bd;

      // compute the fluxes -- note that the
      // min/max above picks the correct flux if we are not in the star
      // region

      // density flux
      Real fl_tmp = ql[QRHO]*ql[ivel];
      Real fr_tmp = qr[QRHO]*qr[ivel];

      flux_hll[URHO] = (bp*fl_tmp - bm*fr_tmp)*bd + bp*bm*bd*(qr[QRHO] - ql[QRHO]);

      // normal momentum flux.  Note for 1-d and 2-d non cartesian
      // r-coordinate, we leave off the pressure term and handle that
      // separately in the update, to accommodate different geometries
      fl_tmp = ql[QRHO]*ql[ivel]*ql[ivel];
      fr_tmp = qr[QRHO]*qr[ivel]*qr[ivel];
      if (momentum_flux_has_pressure(idir, idir, coord)) {
        fl_tmp = fl_tmp + ql[QPRES];
        fr_tmp = fr_tmp + qr[QPRES];
      }

      flux_hll[imom] = (bp*fl_tmp - bm*fr_tmp)*bd + bp*bm*bd*(qr[QRHO]*qr[ivel] - ql[QRHO]*ql[ivel]);

      // transverse momentum flux
      fl_tmp = ql[QRHO]*ql[ivel]*ql[ivel_t1];
      fr_tmp = qr[QRHO]*qr[ivel]*qr[ivel_t1];

      flux_hll[imom_t1] = (bp*fl_tmp - bm*fr_tmp)*bd + bp*bm*bd*(qr[QRHO]*qr[ivel_t1] - ql[QRHO]*ql[ivel_t1]);


      fl_tmp = ql[QRHO]*ql[ivel]*ql[ivel_t2];
      fr_tmp = qr[QRHO]*qr[ivel]*qr[ivel_t2];

      flux_hll[imom_t2] = (bp*fl_tmp - bm*fr_tmp)*bd + bp*bm*bd*(qr[QRHO]*qr[ivel_t2] - ql[QRHO]*ql[ivel_t2]);

      // total energy flux
      Real rhoEl = ql[QRHOEI] + 0.5_rt*ql[QRHO]*(ql[ivel]*ql[ivel] + ql[ivel_t1]*ql[ivel_t1] + ql[ivel_t2]*ql[ivel_t2]);
      fl_tmp = ql[ivel]*(rhoEl + ql[QPRES]);

      Real rhoEr = qr[QRHOEI] + 0.5_rt*qr[QRHO]*(qr[ivel]*qr[ivel] + qr[ivel_t1]*qr[ivel_t1] + qr[ivel_t2]*qr[ivel_t2]);
      fr_tmp = qr[ivel]*(rhoEr + qr[QPRES]);

      flux_hll[UEDEN] = (bp*fl_tmp - bm*fr_tmp)*bd + bp*bm*bd*(rhoEr - rhoEl);


      // eint flux
      fl_tmp = ql[QRHOEI]*ql[ivel];
      fr_tmp = qr[QRHOEI]*qr[ivel];

      flux_hll[UEINT] = (bp*fl_tmp - bm*fr_tmp)*bd + bp*bm*bd*(qr[QRHOEI] - ql[QRHOEI]);


      // passively-advected scalar fluxes
      for (int ipassive = 0; ipassive < npassive; ipassive++) {
          const int n  = cons_passive_index(ipassive);
          const int nqs = prim_passive_index(ipassive);

          fl_tmp = ql[QRHO]*ql[nqs]*ql[ivel];
          fr_tmp = qr[QRHO]*qr[nqs]*qr[ivel];

          flux_hll[n] = (bp*fl_tmp - bm*fr_tmp)*bd + bp*bm*bd*(qr[QRHO]*qr[nqs] - ql[QRHO]*ql[nqs]);
      }
    }


    ///
    /// A HLLC Riemann solver for pure hydrodynamics
    ///
    /// @param ql          the left interface state
    /// @param qr          the right interface state
    /// @param qaux_arr    the auxiliary state
    /// @param uflx        the flux through the interface
    /// @param q_riemann        an approximate Godunov state on the interface
    /// @param idir        coordinate direction for the solve (0 = x, 1 = y, 2 = z)
    ///
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    HLLC(const int i, const int j, const int k, const int idir,
         Array4<Real const> const& ql,
         Array4<Real const> const& qr,
         Array4<Real const> const& qaux_arr,
         Array4<Real> const& uflx,
         Array4<Real> const& qface, const FaceComponents qface_comps,
         const GeometryData& geom,
         DomainEdges const& edges) {

        // this is an implementation of the HLLC solver.
        // it uses the simplest estimate of the wave speeds, since
        // those should work for a general EOS.  We also initially do the
        // Riemann construction to get pstar and ustar, since we'll
        // need to know the pressure and velocity on the interface for the
        // pdV term in the internal energy update.

        const Real small = 1.e-8_rt;

        int iu;
        int sx, sy, sz;

        if (idir == 0) {
            iu = QU;
            sx = 1;
            sy = 0;
            sz = 0;

        } else if (idir == 1) {
            iu = QV;
            sx = 0;
            sy = 1;
            sz = 0;

        } else {
            iu = QW;
            sx = 0;
            sy = 0;
            sz = 1;

        }


        int coord = geom.Coord();

        // deal with hard walls
        Real boundary_scale = 1.0_rt;

        if (idir == 0) {
            if ((i == edges.lo[0] && edges.wall_lo) ||
                (i == edges.hi[0]+1 && edges.wall_hi)) {
                boundary_scale = 0.0_rt;
            }

        } else if (idir == 1) {
            if ((j == edges.lo[1] && edges.wall_lo) ||
                (j == edges.hi[1]+1 && edges.wall_hi)) {
                boundary_scale = 0.0_rt;
            }

        } else {
            if ((k == edges.lo[2] && edges.wall_lo) ||
                (k == edges.hi[2]+1 && edges.wall_hi)) {
                boundary_scale = 0.0_rt;
            }
        }


        Real rl = std::max(ql(i,j,k,QRHO), small_dens);

        // pick left velocities based on direction
        Real ul  = ql(i,j,k,iu);

        Real pl = std::max(ql(i,j,k,QPRES), small_pres);

        Real rr = std::max(qr(i,j,k,QRHO), small_dens);

        // pick right velocities based on direction
        Real ur  = qr(i,j,k,iu);

        Real pr = std::max(qr(i,j,k,QPRES), small_pres);

        // now we essentially get p and u on the
        // interface, but we won't use these in any flux construction.
        Real cs_floor = amrex::max(small, small * qaux_arr(i,j,k,QC), small * qaux_arr(i-sx,j-sy,k-sz,QC));
        Real cs_mean = 0.5_rt*(qaux_arr(i,j,k,QC) + qaux_arr(i-sx,j-sy,k-sz,QC));

        Real gamma1_l = qaux_arr(i-sx,j-sy,k-sz,QGAM1);
        Real gamma1_r = qaux_arr(i,j,k,QGAM1);

    #ifdef TRUE_SDC
        if (use_reconstructed_gamma1 == 1) {
            gamma1_l = ql(i,j,k,QGAM1P);
            gamma1_r = qr(i,j,k,QGAM1P);
        }
    #endif

        Real wsmall = small_dens*cs_floor;
        Real wl = std::max(wsmall, std::sqrt(std::abs(gamma1_l*pl*rl)));
        Real wr = std::max(wsmall, std::sqrt(std::abs(gamma1_r*pr*rr)));

        Real wwinv = 1.0_rt/(wl + wr);
        Real pstar = ((wr*pl + wl*pr) + wl*wr*(ul - ur))*wwinv;
        Real ustar = ((wl*ul + wr*ur) + (pl - pr))*wwinv;

        pstar = std::max(pstar, small_pres);
        // for symmetry preservation, if ustar is really small, then we
        // set it to zero
        if (std::abs(ustar) < riemann_constants::smallu*0.5_rt*(std::abs(ul) + std::abs(ur))){
            ustar = 0.0_rt;
        }

        Real ro;
        Real uo;
        Real po;
        Real gamma1_o;

        if (ustar > 0.0_rt) {
            ro = rl;
            uo = ul;
            po = pl;
            gamma1_o = gamma1_l;

        } else if (!(ustar < 0.0_rt)) {
            ro = 0.5_rt*(rl + rr);
            uo = 0.5_rt*(ul + ur);
            po = 0.5_rt*(pl + pr);
            gamma1_o = 0.5_rt*(gamma1_l + gamma1_r);
        } else {
            ro = rr;
            uo = ur;
            po = pr;
            gamma1_o = gamma1_r;

        }

        ro = std::max(small_dens, ro);

        Real roinv = 1.0_rt/ro;
        Real co = std::sqrt(std::abs(gamma1_o*po*roinv));
        co = std::max(cs_floor, co);
        Real co2inv = 1.0_rt/(co*co);

        Real rstar = ro + (pstar - po)*co2inv;
        rstar = std::max(small_dens, rstar);

        Real cstar = std::sqrt(std::abs(gamma1_o*pstar/rstar));
        cstar = std::max(cstar, cs_floor);

        Real sgnm = std::copysign(1.0_rt, ustar);
        Real spout = co - sgnm*uo;
        Real spin = cstar - sgnm*ustar;
        Real ushock = 0.5_rt*(spin + spout);

        if (pstar-po > 0.0_rt) {
            spin = ushock;
            spout = ushock;
        }

        Real scr = spout-spin;
        if (spout-spin == 0.0_rt) {
            scr = small*cs_mean;
        }

        Real frac = (1.0_rt + (spout + spin)/scr)*0.5_rt;
        frac = amrex::Clamp(frac, 0.0_rt, 1.0_rt);

        Real q_riemann[NQ] = {0.0};

        q_riemann[QRHO] = frac*rstar + (1.0_rt - frac)*ro;
        q_riemann[iu] = frac*ustar + (1.0_rt - frac)*uo;
        q_riemann[QPRES] = frac*pstar + (1.0_rt - frac)*po;


        // now we do the HLLC construction

        // use the simplest estimates of the wave speeds
        Real S_l = std::min(ul - std::sqrt(gamma1_l*pl/rl), ur - std::sqrt(gamma1_r*pr/rr));
        Real S_r = std::max(ul + std::sqrt(gamma1_l*pl/rl), ur + std::sqrt(gamma1_r*pr/rr));

        // estimate of the contact speed
        Real S_c = (pr - pl + rl*ul*(S_l - ul) - rr*ur*(S_r - ur))/
            (rl*(S_l - ul) - rr*(S_r - ur));

        GpuArray<Real, NQ> q_zone;
        GpuArray<Real, NUM_STATE> U_state;
        GpuArray<Real, NUM_STATE> U_star_state;
        GpuArray<Real, NUM_STATE> F_state;

        if (S_r <= 0.0_rt) {
            // R region
            for (int n = 0; n < NQ; n++) {
                q_zone[n] = qr(i,j,k,n);
            }
            cons_state(q_zone, U_state);
            compute_flux(idir, boundary_scale, coord,
                         U_state, pr, F_state);

        } else if (S_r > 0.0_rt && S_c <= 0.0_rt) {
            // R* region
            for (int n = 0; n < NQ; n++) {
                q_zone[n] = qr(i,j,k,n);
            }
            cons_state(q_zone, U_state);
            compute_flux(idir, boundary_scale, coord,
                         U_state, pr, F_state);
            hllc_star_state(idir, S_r, S_c, q_zone, U_star_state);

            // correct the flux
            for (int n = 0; n < NUM_STATE; n++) {
                F_state[n] = F_state[n] + S_r*(U_star_state[n] - U_state[n]);
            }

        } else if (S_c > 0.0_rt && S_l < 0.0_rt) {
            // L* region
            for (int n = 0; n < NQ; n++) {
                q_zone[n] = ql(i,j,k,n);
            }
            cons_state(q_zone, U_state);
            compute_flux(idir, boundary_scale, coord,
                         U_state, pl, F_state);
            hllc_star_state(idir, S_l, S_c, q_zone, U_star_state);

            // correct the flux
            for (int n = 0; n < NUM_STATE; n++) {
                F_state[n] = F_state[n] + S_l*(U_star_state[n] - U_state[n]);
            }

        } else {
            // L region
            for (int n = 0; n < NQ; n++) {
                q_zone[n] = ql(i,j,k,n);
            }
            cons_state(q_zone, U_state);
            compute_flux(idir, boundary_scale, coord,
                         U_state, pl, F_state);
        }

        for (int n = 0; n < NUM_STATE; n++) {
            uflx(i,j,k,n) = F_state[n];
        }


        // now store the state

        if (qface_comps == FaceComponents::Full) {

            for (int n = 0; n < NQ; n++) {
                qface(i,j,k,n) = q_riemann[n];
            }

        } else {

    #ifdef HYBRID_MOMENTUM
            qface(i,j,k,IFCRHO) = q_riemann[QRHO];
    #endif
            qface(i,j,k,IFCU) = q_riemann[QU];
            qface(i,j,k,IFCV) = q_riemann[QV];
            qface(i,j,k,IFCW) = q_riemann[QW];
            qface(i,j,k,IFCPRES) = q_riemann[QPRES];

        }


    }
}

#endif
