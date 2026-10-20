#ifndef CONTACT_SOLVERS_H
#define CONTACT_SOLVERS_H

#include <AMReX_REAL.H>

namespace HLL {
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    conserved_state(const Real* prim_vals, Real* U) {

      U[URHO] = prim_vals[QRHO];

      U[UMX]  = prim_vals[QRHO]*prim_vals[QU];
      U[UMY]  = prim_vals[QRHO]*prim_vals[QV];
      U[UMZ]  = prim_vals[QRHO]*prim_vals[QW];

      U[UEDEN] = prim_vals[QEINTDENS] + 0.5_rt*prim_vals[QRHO]*(prim_vals[QU]*prim_vals[QU] + prim_vals[QV]*prim_vals[QV] + prim_vals[QW]*prim_vals[QW]);
      U[UEINT] = prim_vals[QEINTDENS];

      U[UTEMP] = 0.0;

    #ifdef SHOCK_VAR
      U[UFLAG_SHOCK] = 0.0;
    #endif

    #ifdef NSE_NET
      U[UPROTON_MU] = 0.0;
      U[UNEUTRON_MU] = 0.0;
    #endif

      for (int passive_index = 0; passive_index < npassive; passive_index++) {
          const int n  = passive_cons_index(passive_index);
          const int nqs = passive_prim_index(passive_index);
          U[n] = prim_vals[QRHO]*prim_vals[nqs];
      }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    compute_flux(const int idir, const Real boundary_factor, const int coord,
                 const Real* U, const Real p,
                 Real* F) {

      Real normal_velocity = U[UMX+idir]/U[URHO];

      if (boundary_factor == 0) {
        normal_velocity = 0.0;
      }

      F[URHO] = U[URHO]*normal_velocity;

      F[UMX] = U[UMX]*normal_velocity;
      F[UMY] = U[UMY]*normal_velocity;
      F[UMZ] = U[UMZ]*normal_velocity;

      auto has_pressure_flux = momentum_flux_has_pressure(idir, idir, coord);

      if (has_pressure_flux) {
        F[UMX+idir] = F[UMX+idir] + p;
      }

      F[UEINT] = U[UEINT]*normal_velocity;
      F[UEDEN] = (U[UEDEN] + p)*normal_velocity;

      F[UTEMP] = 0.0;

    #ifdef SHOCK_VAR
      F[UFLAG_SHOCK] = 0.0;
    #endif

    #ifdef NSE_NET
      F[UPROTON_MU] = 0.0;
      F[UNEUTRON_MU] = 0.0;
    #endif

      for (int passive_index=0; passive_index < npassive; passive_index++) {
          const int n = passive_cons_index(passive_index);
          F[n] = U[n]*normal_velocity;
      }
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void
    CONTACT_state(const int idir, const Real S_k, const Real S_c,
               const Real* prim_vals, Real* U) {

      Real u_k = 0.0;
      if (idir == 0) {
        u_k = prim_vals[QU];
      } else if (idir == 1) {
        u_k = prim_vals[QV];
      } else if (idir == 2) {
        u_k = prim_vals[QW];
      }

      Real contact_factor = prim_vals[QRHO]*(S_k - u_k)/(S_k - S_c);
      U[URHO] = contact_factor;

      if (idir == 0) {
        U[UMX]  = contact_factor*S_c;
        U[UMY]  = contact_factor*prim_vals[QV];
        U[UMZ]  = contact_factor*prim_vals[QW];

      } else if (idir == 1) {
        U[UMX]  = contact_factor*prim_vals[QU];
        U[UMY]  = contact_factor*S_c;
        U[UMZ]  = contact_factor*prim_vals[QW];

      } else {
        U[UMX]  = contact_factor*prim_vals[QU];
        U[UMY]  = contact_factor*prim_vals[QV];
        U[UMZ]  = contact_factor*S_c;
      }

      U[UEDEN] = contact_factor*(prim_vals[QEINTDENS]/prim_vals[QRHO] +
                              0.5_rt*(prim_vals[QU]*prim_vals[QU] + prim_vals[QV]*prim_vals[QV] + prim_vals[QW]*prim_vals[QW]) +
                              (S_c - u_k)*(S_c + prim_vals[QPRES]/(prim_vals[QRHO]*(S_k - u_k))));
      U[UEINT] = contact_factor*prim_vals[QEINTDENS]/prim_vals[QRHO];

      U[UTEMP] = 0.0;

    #ifdef SHOCK_VAR
      U[UFLAG_SHOCK] = 0.0;
    #endif

    #ifdef NSE_NET
      U[UPROTON_MU] = 0.0;
      U[UNEUTRON_MU] = 0.0;
    #endif

      for (int passive_index = 0; passive_index < npassive; passive_index++) {
          const int n  = passive_cons_index(passive_index);
          const int nqs = passive_prim_index(passive_index);
          U[n] = contact_factor*prim_vals[nqs];
      }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    CONTACT_FLUX(const int i, const int j, const int k, const int idir,
         Array4<Real const> const& ql,
         Array4<Real const> const& qr,
         Array4<Real const> const& qthermo_arr,
         Array4<Real> const& solver_flux_arr,
         Array4<Real> const& godunov_state, const bool keep_full_state,
         const GeometryData& geom,
         const bool lo_no_flux_wall, const bool hi_no_flux_wall,
         GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {


        const Real small = 1.e-8_rt;

        constexpr int normal_vel[3] = {QU, QV, QW};
        const int iu = normal_vel[idir];

        const int sx = (idir == 0);
        const int sy = (idir == 1);
        const int sz = (idir == 2);


        int coord = geom.Coord();

        Real boundary_factor = 1.0_rt;

        const GpuArray<int, 3> zone{i, j, k};
        if ((zone[idir] == domlo[idir] && lo_no_flux_wall) ||
            (zone[idir] == domhi[idir]+1 && hi_no_flux_wall)) {
            boundary_factor = 0.0_rt;
        }


        Real rl = std::max(ql(i,j,k,QRHO), small_dens);

        Real ul  = ql(i,j,k,iu);

        Real pl = std::max(ql(i,j,k,QPRES), pressure_cutoff);

        Real rr = std::max(qr(i,j,k,QRHO), small_dens);

        Real ur  = qr(i,j,k,iu);

        Real pr = std::max(qr(i,j,k,QPRES), pressure_cutoff);

        Real sound_floor = amrex::max(small, small * qthermo_arr(i,j,k,QC), small * qthermo_arr(i-sx,j-sy,k-sz,QC));
        Real mean_sound_speed = 0.5_rt*(qthermo_arr(i,j,k,QC) + qthermo_arr(i-sx,j-sy,k-sz,QC));

        Real gamma1_left = qthermo_arr(i-sx,j-sy,k-sz,QGAMMA1);
        Real gamma1_right = qthermo_arr(i,j,k,QGAMMA1);

    #ifdef TRUE_SDC
        if (use_recon_gamma1 == 1) {
            gamma1_left = ql(i,j,k,QGC);
            gamma1_right = qr(i,j,k,QGC);
        }
    #endif

        Real impedance_floor = small_dens*sound_floor;
        Real wl = std::max(impedance_floor, std::sqrt(std::abs(gamma1_left*pl*rl)));
        Real wr = std::max(impedance_floor, std::sqrt(std::abs(gamma1_right*pr*rr)));

        Real inv_impedance_sum = 1.0_rt/(wl + wr);
        Real pstar = ((wr*pl + wl*pr) + wl*wr*(ul - ur))*inv_impedance_sum;
        Real ustar = ((wl*ul + wr*ur) + (pl - pr))*inv_impedance_sum;

        pstar = std::max(pstar, pressure_cutoff);
        if (std::abs(ustar) < solver_constants::small_velocity_ratio*0.5_rt*(std::abs(ul) + std::abs(ur))){
            ustar = 0.0_rt;
        }

        Real ro;
        Real uo;
        Real po;
        Real upwind_gamma1;

        if (ustar > 0.0_rt) {
            ro = rl;
            uo = ul;
            po = pl;
            upwind_gamma1 = gamma1_left;

        } else if (!(ustar < 0.0_rt)) {
            ro = 0.5_rt*(rl + rr);
            uo = 0.5_rt*(ul + ur);
            po = 0.5_rt*(pl + pr);
            upwind_gamma1 = 0.5_rt*(gamma1_left + gamma1_right);
        } else {
            ro = rr;
            uo = ur;
            po = pr;
            upwind_gamma1 = gamma1_right;

        }

        ro = std::max(small_dens, ro);

        Real inv_upwind_density = 1.0_rt/ro;
        Real co = std::sqrt(std::abs(upwind_gamma1*po*inv_upwind_density));
        co = std::max(sound_floor, co);
        Real inv_sound_squared = 1.0_rt/(co*co);

        Real rstar = ro + (pstar - po)*inv_sound_squared;
        rstar = std::max(small_dens, rstar);

        Real cstar = std::sqrt(std::abs(upwind_gamma1*pstar/rstar));
        cstar = std::max(cstar, sound_floor);

        Real sgnm = std::copysign(1.0_rt, ustar);
        Real spout = co - sgnm*uo;
        Real spin = cstar - sgnm*ustar;
        Real shock_vel = 0.5_rt*(spin + spout);

        if (pstar-po > 0.0_rt) {
            spin = shock_vel;
            spout = shock_vel;
        }

        Real scr = spout-spin;
        if (spout-spin == 0.0_rt) {
            scr = small*mean_sound_speed;
        }

        Real frac = (1.0_rt + (spout + spin)/scr)*0.5_rt;
        frac = amrex::Clamp(frac, 0.0_rt, 1.0_rt);

        Real interface_state[NQ] = {0.0};

        interface_state[QRHO] = frac*rstar + (1.0_rt - frac)*ro;
        interface_state[iu] = frac*ustar + (1.0_rt - frac)*uo;
        interface_state[QPRES] = frac*pstar + (1.0_rt - frac)*po;



        Real S_l = std::min(ul - std::sqrt(gamma1_left*pl/rl), ur - std::sqrt(gamma1_right*pr/rr));
        Real S_r = std::max(ul + std::sqrt(gamma1_left*pl/rl), ur + std::sqrt(gamma1_right*pr/rr));

        Real S_c = (pr - pl + rl*ul*(S_l - ul) - rr*ur*(S_r - ur))/
            (rl*(S_l - ul) - rr*(S_r - ur));

        Real zone_prim[NQ];
        Real cell_cons[NUM_STATE];
        Real U_contact_state[NUM_STATE];
        Real state_flux[NUM_STATE];

        int region;

        if (S_r <= 0.0_rt) {
            region = 0;
        } else if (S_r > 0.0_rt && S_c <= 0.0_rt) {
            region = 1;
        } else if (!(S_c > 0.0_rt && S_l < 0.0_rt)) {
            region = 3;
        } else {
            region = 2;
        }

        const bool from_right = region < 2;

        for (int n = 0; n < NQ; n++) {
            zone_prim[n] = from_right ? qr(i,j,k,n) : ql(i,j,k,n);
        }
        conserved_state(zone_prim, cell_cons);
        compute_flux(idir, boundary_factor, coord,
                     cell_cons, from_right ? pr : pl, state_flux);

        if (region == 1 || region == 2) {
            const Real S_k = (region == 1) ? S_r : S_l;

            CONTACT_state(idir, S_k, S_c, zone_prim, U_contact_state);

            for (int n = 0; n < NUM_STATE; n++) {
                state_flux[n] = state_flux[n] + S_k*(U_contact_state[n] - cell_cons[n]);
            }
        }

        for (int n = 0; n < NUM_STATE; n++) {
            solver_flux_arr(i,j,k,n) = state_flux[n];
        }



        if (keep_full_state) {

            for (int n = 0; n < NQ; n++) {
                godunov_state(i,j,k,n) = interface_state[n];
            }

        } else {

    #ifdef HYBRID_MOMENTUM
            godunov_state(i,j,k,GDDENSITY) = interface_state[QRHO];
    #endif
            godunov_state(i,j,k,GDU) = interface_state[QU];
            godunov_state(i,j,k,GDV) = interface_state[QV];
            godunov_state(i,j,k,GDW) = interface_state[QW];
            godunov_state(i,j,k,GDPRESSURE) = interface_state[QPRES];

        }


    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void
    HLL(const Real* ql, const Real* qr,
        const Real cl, const Real cr,
        const int idir, const int coord,
        Real* two_wave_flux) {



      constexpr Real small_speed_gap = 1.e-10_rt;

      constexpr int vel_index[3][3] = {{QU, QV, QW}, {QV, QU, QW}, {QW, QU, QV}};
      constexpr int mom_index[3][3] = {{UMX, UMY, UMZ}, {UMY, UMX, UMZ}, {UMZ, UMX, UMY}};

      const int normal_vel_index = vel_index[idir][0];
      const int tan_vel_index = vel_index[idir][1];
      const int tan2_vel_index = vel_index[idir][2];

      const int momentum_index = mom_index[idir][0];
      const int momentum_tan_index = mom_index[idir][1];
      const int momentum_tan2_index = mom_index[idir][2];

      Real left_sqrt_rho = std::sqrt(ql[QRHO]);
      Real right_sqrt_rho = std::sqrt(qr[QRHO]);

      Real inv_sqrt_rho_sum = 1.0_rt/(left_sqrt_rho + right_sqrt_rho);


      Real mean_sound_speed = std::sqrt( (left_sqrt_rho*cl*cl + right_sqrt_rho*cr*cr)*inv_sqrt_rho_sum +
                             0.5_rt*left_sqrt_rho*right_sqrt_rho*inv_sqrt_rho_sum*inv_sqrt_rho_sum*std::pow(qr[normal_vel_index] - ql[normal_vel_index], 2));


      Real roe_velocity = (left_sqrt_rho*ql[normal_vel_index] + right_sqrt_rho*qr[normal_vel_index])*inv_sqrt_rho_sum;

      Real a1 = roe_velocity - mean_sound_speed;
      Real a4 = roe_velocity + mean_sound_speed;


      Real bl = std::min(a1, ql[normal_vel_index] - cl);
      Real br = std::max(a4, qr[normal_vel_index] + cr);

      Real bm = std::min(0.0_rt, bl);
      Real bp = std::max(0.0_rt, br);

      Real bd = bp - bm;

      if (std::abs(bd) < small_speed_gap * std::max(std::abs(bm), std::abs(bp))) {
          return;
      }


      bd = 1.0_rt/bd;

      auto hll_combine = [=] (Real lf, Real rf, Real jump) {
          return (bp*lf - bm*rf)*bd + bp*bm*bd*jump;
      };


      Real left_flux_tmp = ql[QRHO]*ql[normal_vel_index];
      Real right_flux_tmp = qr[QRHO]*qr[normal_vel_index];

      two_wave_flux[URHO] = hll_combine(left_flux_tmp, right_flux_tmp, qr[QRHO] - ql[QRHO]);

      left_flux_tmp = ql[QRHO]*ql[normal_vel_index]*ql[normal_vel_index];
      right_flux_tmp = qr[QRHO]*qr[normal_vel_index]*qr[normal_vel_index];
      if (momentum_flux_has_pressure(idir, idir, coord)) {
        left_flux_tmp = left_flux_tmp + ql[QPRES];
        right_flux_tmp = right_flux_tmp + qr[QPRES];
      }

      two_wave_flux[momentum_index] = hll_combine(left_flux_tmp, right_flux_tmp, qr[QRHO]*qr[normal_vel_index] - ql[QRHO]*ql[normal_vel_index]);

      left_flux_tmp = ql[QRHO]*ql[normal_vel_index]*ql[tan_vel_index];
      right_flux_tmp = qr[QRHO]*qr[normal_vel_index]*qr[tan_vel_index];

      two_wave_flux[momentum_tan_index] = hll_combine(left_flux_tmp, right_flux_tmp, qr[QRHO]*qr[tan_vel_index] - ql[QRHO]*ql[tan_vel_index]);


      left_flux_tmp = ql[QRHO]*ql[normal_vel_index]*ql[tan2_vel_index];
      right_flux_tmp = qr[QRHO]*qr[normal_vel_index]*qr[tan2_vel_index];

      two_wave_flux[momentum_tan2_index] = hll_combine(left_flux_tmp, right_flux_tmp, qr[QRHO]*qr[tan2_vel_index] - ql[QRHO]*ql[tan2_vel_index]);

      Real leftTotalEnergy = ql[QEINTDENS] + 0.5_rt*ql[QRHO]*(ql[normal_vel_index]*ql[normal_vel_index] + ql[tan_vel_index]*ql[tan_vel_index] + ql[tan2_vel_index]*ql[tan2_vel_index]);
      left_flux_tmp = ql[normal_vel_index]*(leftTotalEnergy + ql[QPRES]);

      Real rightTotalEnergy = qr[QEINTDENS] + 0.5_rt*qr[QRHO]*(qr[normal_vel_index]*qr[normal_vel_index] + qr[tan_vel_index]*qr[tan_vel_index] + qr[tan2_vel_index]*qr[tan2_vel_index]);
      right_flux_tmp = qr[normal_vel_index]*(rightTotalEnergy + qr[QPRES]);

      two_wave_flux[UEDEN] = hll_combine(left_flux_tmp, right_flux_tmp, rightTotalEnergy - leftTotalEnergy);


      left_flux_tmp = ql[QEINTDENS]*ql[normal_vel_index];
      right_flux_tmp = qr[QEINTDENS]*qr[normal_vel_index];

      two_wave_flux[UEINT] = hll_combine(left_flux_tmp, right_flux_tmp, qr[QEINTDENS] - ql[QEINTDENS]);


      for (int passive_index = 0; passive_index < npassive; passive_index++) {
          const int n  = passive_cons_index(passive_index);
          const int nqs = passive_prim_index(passive_index);

          left_flux_tmp = ql[QRHO]*ql[nqs]*ql[normal_vel_index];
          right_flux_tmp = qr[QRHO]*qr[nqs]*qr[normal_vel_index];

          two_wave_flux[n] = hll_combine(left_flux_tmp, right_flux_tmp, qr[QRHO]*qr[nqs] - ql[QRHO]*ql[nqs]);
      }
    }
}

#endif
