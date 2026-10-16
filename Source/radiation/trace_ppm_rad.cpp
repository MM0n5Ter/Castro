#include <HydroSim.H>
#include <HydroSim_util.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#include <cmath>

#include <ppm.H>
#include <flatten.H>

using namespace amrex;

void
HydroSim::trace_ppm_rad(const Box& bx,
                      const int idir,
                      TraceFields const& fields,
                      Array4<Real> const& qm,
                      Array4<Real> const& qp,
                      const Box& vbx,
                      const Real dt) {

  // These routines do the characteristic tracing under the parabolic
  // profiles in each zone to the edge / half-time.

  auto const& U_arr = fields.U;
  auto const& rho_inv_arr = fields.rho_inv;
  auto const& q_arr = fields.q;
  auto const& qaux_arr = fields.qaux;
  auto const& srcQ = fields.srcQ;
  [[maybe_unused]] auto const& dloga = fields.dloga;

  // To allow for easy integration of radiation, we adopt the
  // following conventions:
  //
  // rho : mass density
  // u, v, w : velocities
  // p : gas (hydro) pressure
  // ptot : total pressure (note for pure hydro, this is
  //        just the gas pressure)
  // rhoe_g : gas specific internal energy
  // cgas : sound speed for just the gas contribution
  // cc : total sound speed (including radiation)
  // h_g : gas specific enthalpy / cc**2
  // gam_g : the gas Gamma_1
  //
  // for pure hydro, we will only consider:
  //   rho, u, v, w, ptot, rhoe_g, cc, h_g

  const auto dx = geom.CellSizeArray();

  Real hdt = 0.5_rt * dt;
  Real dtdx = dt / dx[idir];

  auto lo = bx.loVect3d();
  auto hi = bx.hiVect3d();

  auto vlo = vbx.loVect3d();
  auto vhi = vbx.hiVect3d();

#ifndef AMREX_USE_GPU

  // if we're on the CPU, we preprocess the sources over the whole
  // tile up front -- we don't want to trace under a source that is
  // empty. This check only needs to be done over the tile we're
  // working on, since the PPM reconstruction and integration done
  // here is only local to this tile.

  GpuArray<int, NQSRC> do_source_trace;


  for (int n = 0; n < NQSRC; n++) {
    do_source_trace[n] = 0;

    for (int k = lo[2]-2*dim2_on; k <= hi[2]+2*dim2_on; k++) {
      for (int j = lo[1]-2*dim1_on; j <= hi[1]+2*dim1_on; j++) {
        for (int i = lo[0]-2; i <= hi[0]+2; i++) {
          if (std::abs(srcQ(i,j,k,n)) > 0.0_rt) {
            do_source_trace[n] = 1;
            break;
          }
        }
        if (do_source_trace[n] == 1) break;
      }
      if (do_source_trace[n] == 1) break;
    }
  }
#endif

  // This does the characteristic tracing to build the interface
  // states using the normal predictor only (no transverse terms).
  //
  // For each zone, we construct Im and Ip arrays -- these are the averages
  // of the various primitive state variables under the parabolic
  // interpolant over the region swept out by one of the 3 different
  // characteristic waves.
  //
  // Im is integrating to the left interface of the current zone
  // (which will be used to build the right ("p") state at that interface)
  // and Ip is integrating to the right interface of the current zone
  // (which will be used to build the left ("m") state at that interface).
  //
  //
  // The choice of reference state is designed to minimize the
  // effects of the characteristic projection.  We subtract the I's
  // off of the reference state, project the quantity such that it is
  // in terms of the characteristic variables, and then add all the
  // jumps that are moving toward the interface to the reference
  // state to get the full state on that interface.

  int QUN, QUT, QUTT;

  if (idir == 0) {
    QUN = QU;
    QUT = QV;
    QUTT = QW;
  } else if (idir == 1) {
    QUN = QV;
    QUT = QW;
    QUTT = QU;
  } else if (idir == 2) {
    QUN = QW;
    QUT = QU;
    QUTT = QV;
  }

  Real lsmall_dens = small_dens;
  Real lsmall_pres = small_pres;

  // Trace to left and right edges using upwind PPM
  AMREX_PARALLEL_FOR_3D(bx, i, j, k,
  {

   Real lam0[NGROUPS];
   Real lamp[NGROUPS];
   Real lamm[NGROUPS];

    for (int g = 0; g < NGROUPS; g++) {
      lam0[g] = qaux_arr(i,j,k,QFLXLIM+g);
      lamp[g] = qaux_arr(i,j,k,QFLXLIM+g);
      lamm[g] = qaux_arr(i,j,k,QFLXLIM+g);
    }

    Real rho = q_arr(i,j,k,QRHO);

    // cgassq is the gas soundspeed **2
    // cc is the total soundspeed **2 (gas + radiation)
    Real cgassq = qaux_arr(i,j,k,QCSNDGAS)*qaux_arr(i,j,k,QCSNDGAS);
    Real cc = qaux_arr(i,j,k,QC);
    Real csq = cc*cc;

    Real un = q_arr(i,j,k,QUN);

    Real p = q_arr(i,j,k,QPRES);
    Real rhoe_g = q_arr(i,j,k,QRHOEI);
    Real h_g = ( (p+rhoe_g)/rho)/csq;

    Real gam_g = qaux_arr(i,j,k,QGAM1GAS);

    Real ptot = q_arr(i,j,k,QPTOTAL);

    Real er[NGROUPS];
    Real hr[NGROUPS];
    for (int g = 0; g < NGROUPS; g++) {
      er[g] = q_arr(i,j,k,QERAD+g);
      hr[g] = (lam0[g] + 1.0_rt)*er[g]/rho;
    }

    // do the parabolic reconstruction and compute the
    // integrals under the characteristic waves
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
            if ( q_arr(i-1,j,k,QU) + q_arr(i,j-1,k,QV) + q_arr(i,j,k-1,QW) >
                 q_arr(i+1,j,k,QU) + q_arr(i,j+1,k,QV) + q_arr(i,j,k+1,QW) ) {

                if (q_arr(i,j,k,QPRES) < radiation::flatten_pp_threshold * q_arr(i,j,k,QPTOTAL)) {
                    flat = 0.0;
                }
            }
        }
#endif
    }

    ParabolaEdges edge;

    ParabolaIntegrals traced[NQ];


    // do the non-passives first

    for (int n = 0; n < NQTHERM; n++) {
      if (n == QTEMP) continue;

      s = load_stencil(q_arr, idir, i, j, k, n);
      edge = ppm_reconstruct(s, flat);
      traced[n] = ppm_profile_integrals(edge.lo, edge.hi, s[i0], un, cc, dtdx);

    }

    // now the passives

    for (int ipassive = 0; ipassive < npassive; ++ipassive) {

      const int n = prim_passive_index(ipassive);
      const int nc = cons_passive_index(ipassive);

      s = load_passive_stencil(U_arr, rho_inv_arr, idir, i, j, k, nc);
      edge = ppm_reconstruct(s, flat);
      traced[n] = ppm_profile_integrals(edge.lo, edge.hi, s[i0], un, cc, dtdx);

    }


    // source terms
    ParabolaIntegrals src_traced[NQSRC];

    for (int n = 0; n < NQSRC; n++) {

      // do we even need to trace (non-zero source?)
#ifndef AMREX_USE_GPU
      int do_trace = do_source_trace[n];
#else
      int do_trace = 0;
      if (idir == 0) {
        for (int b = i-2; b <= i+2; b++) {
          if (std::abs(srcQ(b,j,k,n)) > 0.0_rt) {
            do_trace = 1;
            break;
          }
        }
      } else if (idir == 1) {
        for (int b = j-2; b <= j+2; b++) {
          if (std::abs(srcQ(i,b,k,n)) > 0.0_rt) {
            do_trace = 1;
            break;
          }
        }
      } else {
        for (int b = k-2; b <= k+2; b++) {
          if (std::abs(srcQ(i,j,b,n)) > 0.0_rt) {
            do_trace = 1;
            break;
          }
        }
      }
#endif

      if (do_trace) {

          s = load_stencil(srcQ, idir, i, j, k, n);
          edge = ppm_reconstruct(s, flat);
          src_traced[n] = ppm_profile_integrals(edge.lo, edge.hi, s[i0], un, cc, dtdx);

      } else {
        src_traced[n] = {};
      }

    }


    // do the passives separately

    // the passive stuff is the same regardless of the tracing

    for (int ipassive = 0; ipassive < npassive; ipassive++) {

      int n = prim_passive_index(ipassive);

      // Plus state on face i
      if ((idir == 0 && i >= vlo[0]) ||
          (idir == 1 && j >= vlo[1]) ||
          (idir == 2 && k >= vlo[2])) {

        // We have
        //
        // q_l = q_ref - Proj{(q_ref - I)}
        //
        // and Proj{} represents the characteristic projection.
        // But for these, there is only 1-wave that matters, the u
        // wave, so no projection is needed.  Since we are not
        // projecting, the reference state doesn't matter

        qp(i,j,k,n) = traced[n].lo[1];
      }

      // Minus state on face i+1
      if (idir == 0 && i <= vhi[0]) {
        qm(i+1,j,k,n) = traced[n].hi[1];
      } else if (idir == 1 && j <= vhi[1]) {
        qm(i,j+1,k,n) = traced[n].hi[1];
      } else if (idir == 2 && k <= vhi[2]) {
        qm(i,j,k+1,n) = traced[n].hi[1];
      }
    }

    // plus state on face i

    if ((idir == 0 && i >= vlo[0]) ||
        (idir == 1 && j >= vlo[1]) ||
        (idir == 2 && k >= vlo[2])) {

      // Set the reference state
      // This will be the fastest moving state to the left
      Real rho_ref = traced[QRHO].lo[0];
      Real un_ref = traced[QUN].lo[0];

      Real p_ref = traced[QPRES].lo[0];
      Real rhoe_g_ref = traced[QRHOEI].lo[0];

      Real ptot_ref = traced[QPTOTAL].lo[0];
      Real er_ref[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        er_ref[g] = traced[QERAD+g].lo[0];
      }

      rho_ref = amrex::max(rho_ref, lsmall_dens);
      p_ref = amrex::max(p_ref, lsmall_pres);

      // *m are the jumps carried by u-c
      // *p are the jumps carried by u+c

      // Note: for the transverse velocities, the jump is carried
      //       only by the u wave (the contact)

      // we also add the sources here so they participate in the tracing
      Real dum = un_ref - traced[QUN].lo[0] - hdt*src_traced[QUN].lo[0];
      Real dptotm = ptot_ref - traced[QPTOTAL].lo[0] - hdt*src_traced[QPRES].lo[0];

      Real drho = rho_ref - traced[QRHO].lo[1] - hdt*src_traced[QRHO].lo[1];
      Real dptot = ptot_ref - traced[QPTOTAL].lo[1] - hdt*src_traced[QPRES].lo[1];
      Real drhoe_g = rhoe_g_ref - traced[QRHOEI].lo[1] - hdt*src_traced[QRHOEI].lo[1];

      Real der[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        der[g] = er_ref[g] - traced[QERAD+g].lo[1];
      }

      Real dup = un_ref - traced[QUN].lo[2] - hdt*src_traced[QUN].lo[2];
      Real dptotp = ptot_ref - traced[QPTOTAL].lo[2] - hdt*src_traced[QPRES].lo[2];

      // {rho, u, p, (rho e)} eigensystem

      // These are analogous to the beta's from the original PPM
      // (except we work with rho instead of tau).  This is
      // simply (l . dq), where dq = qref - I(q)

      Real alpham = 0.5_rt*(dptotm/(rho*cc) - dum)*rho/cc;
      Real alphap = 0.5_rt*(dptotp/(rho*cc) + dup)*rho/cc;
      Real alpha0r = drho - dptot/csq;
      Real alpha0e_g = drhoe_g - dptot*h_g;

      Real alphar[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        alphar[g] = der[g] - dptot/csq*hr[g];
      }

      alpham = un-cc > 0.0_rt ? 0.0_rt : -alpham;
      alphap = un+cc > 0.0_rt ? 0.0_rt : -alphap;
      alpha0r = un > 0.0_rt ? 0.0_rt : -alpha0r;
      alpha0e_g = un > 0.0_rt ? 0.0_rt : -alpha0e_g;

      for (int g = 0; g < NGROUPS; g++) {
        alphar[g] = un > 0.0_rt ? 0.0_rt : -alphar[g];
      }

      // The final interface states are just
      // q_s = q_ref - sum(l . dq) r
      // note that the a{mpz}right as defined above have the minus already
      qp(i,j,k,QRHO) = amrex::max(lsmall_dens, rho_ref + alphap + alpham + alpha0r);
      qp(i,j,k,QUN) = un_ref + (alphap - alpham)*cc/rho;
      qp(i,j,k,QRHOEI) = rhoe_g_ref + (alphap + alpham)*h_g*csq + alpha0e_g;

      qp(i,j,k,QPRES) = p_ref + (alphap + alpham)*cgassq;
      for (int g = 0; g < NGROUPS; g++) {
        qp(i,j,k,QPRES) += -lamp[g]*alphar[g];
      }
      qp(i,j,k,QPRES) = amrex::max(lsmall_pres, qp(i,j,k,QPRES));

      qp(i,j,k,QPTOTAL) = ptot_ref + (alphap + alpham)*csq;
      qp(i,j,k,QRHOEITOT) = qp(i,j,k,QRHOEI);

      Real qrtmp;
      for (int g = 0; g < NGROUPS; g++) {
        qrtmp = er_ref[g] + (alphap + alpham)*hr[g] + alphar[g];
        qp(i,j,k,QERAD+g) = qrtmp;
        qp(i,j,k,QRHOEITOT) += qrtmp;
      }

      for (int g = 0; g < NGROUPS; g++) {
        if (qp(i,j,k,QERAD+g) < 0.0_rt) {
          Real er_foo = -qp(i,j,k,QERAD+g);
          qp(i,j,k,QERAD+g) = 0.0_rt;
          qp(i,j,k,QPTOTAL) += lamp[g] * er_foo;
          qp(i,j,k,QRHOEITOT) += er_foo;
        }
      }

      // Transverse velocities -- there's no projection here, so
      // we don't need a reference state.  We only care about
      // the state traced under the middle wave

      // Recall that I already takes the limit of the parabola
      // in the event that the wave is not moving toward the
      // interface
      qp(i,j,k,QUT) = traced[QUT].lo[1] + hdt*src_traced[QUT].lo[1];
      qp(i,j,k,QUTT) = traced[QUTT].lo[1] + hdt*src_traced[QUTT].lo[1];

    }

    // minus state on face i + 1

    if ((idir == 0 && i <= vhi[0]) ||
        (idir == 1 && j <= vhi[1]) ||
        (idir == 2 && k <= vhi[2])) {

      // Set the reference state
      // This will be the fastest moving state to the right
      Real rho_ref = traced[QRHO].hi[2];
      Real un_ref = traced[QUN].hi[2];

      Real p_ref = traced[QPRES].hi[2];
      Real rhoe_g_ref = traced[QRHOEI].hi[2];

      Real ptot_ref = traced[QPTOTAL].hi[2];
      Real er_ref[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        er_ref[g] = traced[QERAD+g].hi[2];
      }

      rho_ref = amrex::max(rho_ref, lsmall_dens);
      p_ref = amrex::max(p_ref, lsmall_pres);

      // *m are the jumps carried by u-c
      // *p are the jumps carried by u+c

      Real dum = un_ref - traced[QUN].hi[0] - hdt*src_traced[QUN].hi[0];
      Real dptotm = ptot_ref - traced[QPTOTAL].hi[0] - hdt*src_traced[QPRES].hi[0];

      Real drho = rho_ref - traced[QRHO].hi[1] - hdt*src_traced[QRHO].hi[1];
      Real dptot = ptot_ref - traced[QPTOTAL].hi[1] - hdt*src_traced[QPRES].hi[1];
      Real drhoe_g = rhoe_g_ref - traced[QRHOEI].hi[1] - hdt*src_traced[QRHOEI].hi[1];

      Real der[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        der[g]  = er_ref[g]  - traced[QERAD+g].hi[1];
      }

      Real dup = un_ref - traced[QUN].hi[2] - hdt*src_traced[QUN].hi[2];
      Real dptotp = ptot_ref - traced[QPTOTAL].hi[2] - hdt*src_traced[QPRES].hi[2];

      // {rho, u, p, (rho e)} eigensystem

      // These are analogous to the beta's from the original PPM
      // (except we work with rho instead of tau).  This is
      // simply (l . dq), where dq = qref - I(q)

      Real alpham = 0.5_rt*(dptotm/(rho*cc) - dum)*rho/cc;
      Real alphap = 0.5_rt*(dptotp/(rho*cc) + dup)*rho/cc;
      Real alpha0r = drho - dptot/csq;
      Real alpha0e_g = drhoe_g - dptot*h_g;

      Real alphar[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        alphar[g] = der[g] - dptot/csq*hr[g];
      }

      alpham = un-cc > 0.0_rt ? -alpham : 0.0_rt;
      alphap = un+cc > 0.0_rt ? -alphap : 0.0_rt;
      alpha0r = un > 0.0_rt ? -alpha0r : 0.0_rt;
      alpha0e_g = un > 0.0_rt ? -alpha0e_g : 0.0_rt;

      for (int g = 0; g < NGROUPS; g++) {
        alphar[g] = un > 0.0_rt ? -alphar[g] : 0.0_rt;
      }

      // The final interface states are just
      // q_s = q_ref - sum (l . dq) r
      // note that the a{mpz}left as defined above have the minus already

      if (idir == 0) {
        qm(i+1,j,k,QRHO) = amrex::max(lsmall_dens, rho_ref + alphap + alpham + alpha0r);
        qm(i+1,j,k,QUN) = un_ref + (alphap - alpham)*cc/rho;
        qm(i+1,j,k,QRHOEI) = rhoe_g_ref + (alphap + alpham)*h_g*csq + alpha0e_g;

        qm(i+1,j,k,QPRES) = p_ref + (alphap + alpham)*cgassq;
        for (int g = 0; g < NGROUPS; g++) {
          qm(i+1,j,k,QPRES) += -lamm[g]*alphar[g];
        }

        qm(i+1,j,k,QPTOTAL) = ptot_ref + (alphap + alpham)*csq;
        qm(i+1,j,k,QRHOEITOT) = qm(i+1,j,k,QRHOEI);

        Real qrtmp;
        for (int g = 0; g < NGROUPS; g++) {
          qrtmp = er_ref[g] + (alphap + alpham)*hr[g] + alphar[g];
          qm(i+1,j,k,QERAD+g) = qrtmp;
          qm(i+1,j,k,QRHOEITOT) += qrtmp;
        }

        for (int g = 0; g < NGROUPS; g++) {
          if (qm(i+1,j,k,QERAD+g) < 0.0_rt) {
            Real er_foo = -qm(i+1,j,k,QERAD+g);
            qm(i+1,j,k,QERAD+g) = 0.0_rt;
            qm(i+1,j,k,QPTOTAL) += lamm[g] * er_foo;
            qm(i+1,j,k,QRHOEITOT) += er_foo;
          }
        }

        // transverse velocities
        qm(i+1,j,k,QUT) = traced[QUT].hi[1] + hdt*src_traced[QUT].hi[1];
        qm(i+1,j,k,QUTT) = traced[QUTT].hi[1] + hdt*src_traced[QUTT].hi[1];

      } else if (idir == 1) {
        qm(i,j+1,k,QRHO) = amrex::max(lsmall_dens, rho_ref + alphap + alpham + alpha0r);
        qm(i,j+1,k,QUN) = un_ref + (alphap - alpham)*cc/rho;
        qm(i,j+1,k,QRHOEI) = rhoe_g_ref + (alphap + alpham)*h_g*csq + alpha0e_g;

        qm(i,j+1,k,QPRES) = p_ref + (alphap + alpham)*cgassq;
        for (int g = 0; g < NGROUPS; g++) {
          qm(i,j+1,k,QPRES) += -lamm[g]*alphar[g];
        }

        qm(i,j+1,k,QPTOTAL) = ptot_ref + (alphap + alpham)*csq;
        qm(i,j+1,k,QRHOEITOT) = qm(i,j+1,k,QRHOEI);

        Real qrtmp;
        for (int g = 0; g < NGROUPS; g++) {
          qrtmp = er_ref[g] + (alphap + alpham)*hr[g] + alphar[g];
          qm(i,j+1,k,QERAD+g) = qrtmp;
          qm(i,j+1,k,QRHOEITOT) += qrtmp;
        }

        for (int g = 0; g < NGROUPS; g++) {
          qrtmp = er_ref[g] + (alphap + alpham)*hr[g] + alphar[g];
          qm(i,j+1,k,QERAD+g) = qrtmp;
          qm(i,j+1,k,QRHOEITOT) += qrtmp;
        }

        for (int g = 0; g < NGROUPS; g++) {
          if (qm(i,j+1,k,QERAD+g) < 0.0_rt) {
            Real er_foo = -qm(i,j+1,k,QERAD+g);
            qm(i,j+1,k,QERAD+g) = 0.0_rt;
            qm(i,j+1,k,QPTOTAL) += lamm[g] * er_foo;
            qm(i,j+1,k,QRHOEITOT) += er_foo;
          }
        }

        // transverse velocities
        qm(i,j+1,k,QUT) = traced[QUT].hi[1] + hdt*src_traced[QUT].hi[1];
        qm(i,j+1,k,QUTT) = traced[QUTT].hi[1] + hdt*src_traced[QUTT].hi[1];


      } else {

        qm(i,j,k+1,QRHO) = amrex::max(lsmall_dens, rho_ref + alphap + alpham + alpha0r);
        qm(i,j,k+1,QUN) = un_ref + (alphap - alpham)*cc/rho;
        qm(i,j,k+1,QRHOEI) = rhoe_g_ref + (alphap + alpham)*h_g*csq + alpha0e_g;

        qm(i,j,k+1,QPRES) = p_ref + (alphap + alpham)*cgassq;
        for (int g = 0; g < NGROUPS; g++) {
          qm(i,j,k+1,QPRES) += -lamm[g]*alphar[g];
        }

        qm(i,j,k+1,QPTOTAL) = ptot_ref + (alphap + alpham)*csq;
        qm(i,j,k+1,QRHOEITOT) = qm(i,j,k+1,QRHOEI);

        Real qrtmp;
        for (int g = 0; g < NGROUPS; g++) {
          qrtmp = er_ref[g] + (alphap + alpham)*hr[g] + alphar[g];
          qm(i,j,k+1,QERAD+g) = qrtmp;
          qm(i,j,k+1,QRHOEITOT) += qrtmp;
        }

        for (int g = 0; g < NGROUPS; g++) {
          qrtmp = er_ref[g] + (alphap + alpham)*hr[g] + alphar[g];
          qm(i,j,k+1,QERAD+g) = qrtmp;
          qm(i,j,k+1,QRHOEITOT) += qrtmp;
        }

        for (int g = 0; g < NGROUPS; g++) {
          if (qm(i,j,k+1,QERAD+g) < 0.0_rt) {
            Real er_foo = -qm(i,j,k+1,QERAD+g);
            qm(i,j,k+1,QERAD+g) = 0.0_rt;
            qm(i,j,k+1,QPTOTAL) += lamm[g] * er_foo;
            qm(i,j,k+1,QRHOEITOT) += er_foo;
          }
        }

        // transverse velocities
        qm(i,j,k+1,QUT) = traced[QUT].hi[1] + hdt*src_traced[QUT].hi[1];
        qm(i,j,k+1,QUTT) = traced[QUTT].hi[1] + hdt*src_traced[QUTT].hi[1];

      }

    }

    // geometry source terms

#if (AMREX_SPACEDIM < 3)
    // these only apply for x states (idir = 0)
    if (idir == 0 && dloga(i,j,k) != 0.0_rt) {
      Real courn = dt/dx[0]*(cc+std::abs(un));
      Real eta = (1.0_rt - courn)/(cc*dt*std::abs(dloga(i,j,k)));
      Real dlogatmp = amrex::min(eta, 1.0_rt)*dloga(i,j,k);
      Real geo_src_rho = -0.5_rt*dt*rho*dlogatmp*un;
      Real geo_src_p = geo_src_rho*cgassq;
      Real source = geo_src_p*h_g;
      Real geo_src_er[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        geo_src_er[g] = -0.5_rt*dt*dlogatmp*un*(lam0[g] + 1.0_rt)*er[g];
      }

      if (i <= vhi[0]) {
        qm(i+1,j,k,QRHO) = qm(i+1,j,k,QRHO) + geo_src_rho;
        qm(i+1,j,k,QRHO) = amrex::max(qm(i+1,j,k,QRHO), lsmall_dens);
        qm(i+1,j,k,QPRES ) = qm(i+1,j,k,QPRES ) + geo_src_p;
        qm(i+1,j,k,QRHOEI) = qm(i+1,j,k,QRHOEI) + source;
        for (int g = 0; g < NGROUPS; g++) {
          qm(i+1,j,k,QERAD+g) += geo_src_er[g];
        }
        qm(i+1,j,k,QPTOTAL) = qm(i+1,j,k,QPTOTAL) + geo_src_p;
        qm(i+1,j,k,QRHOEITOT) = qm(i+1,j,k,QRHOEI);
        for (int g = 0; g < NGROUPS; g++) {
          qm(i+1,j,k,QPTOTAL) += lamm[g]*geo_src_er[g];
          qm(i+1,j,k,QRHOEITOT) += qm(i+1,j,k,QERAD+g);
          }
      }

      if (i >= vlo[0]) {
        qp(i,j,k,QRHO) = qp(i,j,k,QRHO) + geo_src_rho;
        qp(i,j,k,QRHO) = amrex::max(qp(i,j,k,QRHO), lsmall_dens);
        qp(i,j,k,QPRES) = qp(i,j,k,QPRES) + geo_src_p;
        qp(i,j,k,QRHOEI) = qp(i,j,k,QRHOEI) + source;
        for (int g = 0; g < NGROUPS; g++) {
          qp(i,j,k,QERAD+g) += geo_src_er[g];
        }
        qp(i,j,k,QPTOTAL) = qp(i,j,k,QPTOTAL) + geo_src_p;
        qp(i,j,k,QRHOEITOT) = qp(i,j,k,QRHOEI);
        for (int g = 0; g < NGROUPS; g++) {
          qp(i,j,k,QPTOTAL) += lamp[g]*geo_src_er[g];
          qp(i,j,k,QRHOEITOT) += qp(i,j,k,QERAD+g);
        }
      }
    }
#endif

  });
}
