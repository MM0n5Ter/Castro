#include <HydroSim.H>

#include <ppm.H>

using namespace amrex;

#include <mhd_eigen.H>

void
HydroSim::ppm_mhd(const Box& bx,
                Array4<Real const> const& q_arr,
                Array4<Real const> const& qaux,
                Array4<Real const> const& flatcoef,
                Array4<Real const> const& Bx,
                Array4<Real const> const& By,
                Array4<Real const> const& Bz,
                Array4<Real const> const& srcQ,
                const int idir,
                const Real dt,
                Array4<Real> const& qleft,
                Array4<Real> const& qright) {

  // these loops are over cell-centers and for each cell-center, we find the left and
  // right interface states

  const auto dx = geom.CellSizeArray();

  Real dtdx = dt/dx[idir];

  // these are the characteristic variables for this direction
  int cvars[NEVEC];

  if (idir == 0) {

    // component (Bx) is omitted

    cvars[IEVEC_RHO] = QRHO;
    cvars[IEVEC_U] = QU;
    cvars[IEVEC_V] = QV;
    cvars[IEVEC_W] = QW;
    cvars[IEVEC_P] = QPRES;
    cvars[IEVEC_BT] = QMAGY;
    cvars[IEVEC_BTT] = QMAGZ;

  } else if (idir == 1) {

    // component (By) is omitted

    cvars[IEVEC_RHO] = QRHO;
    cvars[IEVEC_U] = QU;
    cvars[IEVEC_V] = QV;
    cvars[IEVEC_W] = QW;
    cvars[IEVEC_P] = QPRES;
    cvars[IEVEC_BT] = QMAGX;
    cvars[IEVEC_BTT] = QMAGZ;

  } else {

    // component (Bz) is omitted

    cvars[IEVEC_RHO] = QRHO;
    cvars[IEVEC_U] = QU;
    cvars[IEVEC_V] = QV;
    cvars[IEVEC_W] = QW;
    cvars[IEVEC_P] = QPRES;
    cvars[IEVEC_BT] = QMAGX;
    cvars[IEVEC_BTT] = QMAGY;

  }


  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    // compute the eigenvectors and eigenvalues for this coordinate direction

    Array1D<Real, 0, NQ-1> q_zone;
    for (int n = 0; n < NQ; n++) {
      q_zone(n) = q_arr(i,j,k,n);
    }

    Real as = qaux(i,j,k,QC);

    const Array1D<Real, 0, NEVEC-1> lam = evals(as, q_zone, idir);

    Eigenvectors evec;

    if (idir == 0) {
      evec = evecx(as, q_zone);

    } else if (idir == 1) {
      evec = evecy(as, q_zone);

    } else {
      evec = evecz(as, q_zone);
    }

    // do the parabolic reconstruction and compute the integrals under
    // the characteristic waves
    ZoneStencil s;
    Real flat = flatcoef(i,j,k);
    ParabolaEdges edge;

    Real Ip[NEVEC][NEVEC];  // first component is var, second is wave
    Real Im[NEVEC][NEVEC];

    Real q_ref_left[NEVEC];
    Real q_ref_right[NEVEC];


    for (int n = 0; n < NEVEC; n++) {

      int v = cvars[n];

      s = load_stencil(q_arr, idir, i, j, k, v);
      edge = ppm_reconstruct(s, flat);

      Real Ipt = 0.0;
      Real Imt = 0.0;

      // loop over waves and integrate under the parabola
      for (int ii = 0; ii < NEVEC; ii++) {
        Real speed = lam(ii);

        amrex::Tie(Ipt, Imt) = ppm_int_profile_single(edge.lo, edge.hi, s[i0], speed, dtdx);

        // we do a special correction here for waves that are not
        // moving toward the interface.  It doesn't make sense to
        // integrate under the parabola in this case, so we simply
        // create a piecewise linear slope here.
        if (lam(ii) >= 0.0_rt) {
          Ip[n][ii] = Ipt;
        } else {
          Ip[n][ii] = 0.5_rt * dtdx * (edge.hi - edge.lo);
        }

        if (lam(ii) <= 0.0_rt) {
          Im[n][ii] = Imt;
        } else {
          Im[n][ii] = 0.5_rt * dtdx * (edge.hi - edge.lo);
        }

      }

      // and now the reference states

      // q_{i+1/2, L} uses the fastest wave moving to the right
      Real lambda_max = amrex::max(lam(NEVEC-1), 0.0_rt);
      amrex::Tie(Ipt, Imt) = ppm_int_profile_single(edge.lo, edge.hi, s[i0], lambda_max, dtdx);
      q_ref_left[n] = Ipt;

      // q_{i+1/2, L} uses the fastest wave moving to the left
      Real lambda_min = amrex::min(lam(0), 0.0_rt);
      amrex::Tie(Ipt, Imt) = ppm_int_profile_single(edge.lo, edge.hi, s[i0], lambda_min, dtdx);
      q_ref_right[n] = Imt;

    }

    // MHD Source Terms
    Real smhd[NEVEC];

    smhd[IEVEC_RHO] = 0.0_rt;
    smhd[IEVEC_U] = q_zone(QMAGX) / q_zone(QRHO);
    smhd[IEVEC_V] = q_zone(QMAGY) / q_zone(QRHO);
    smhd[IEVEC_W] = q_zone(QMAGZ) / q_zone(QRHO);
    smhd[IEVEC_P] = q_zone(QMAGX) * q_zone(QU) +
                    q_zone(QMAGY) * q_zone(QV) +
                    q_zone(QMAGZ) * q_zone(QW);

    if (idir == 0) {
      smhd[IEVEC_BT] = q_zone(QV);
      smhd[IEVEC_BTT] = q_zone(QW);

      // cross-talk of normal magnetic field direction
      for (auto & source : smhd) {
          source *= (Bx(i+1,j,k) - Bx(i,j,k)) / dx[idir];
      }

    } else if (idir == 1) {
      smhd[IEVEC_BT] = q_zone(QU);
      smhd[IEVEC_BTT] = q_zone(QW);

      // cross-talk of normal magnetic field direction
      for (auto & source : smhd) {
          source *= (By(i,j+1,k) - By(i,j,k)) / dx[idir];
      }

    } else {
      smhd[IEVEC_BT] = q_zone(QU);
      smhd[IEVEC_BTT] = q_zone(QV);

      // cross-talk of normal magnetic field direction
      for (auto & source : smhd) {
          source *= (Bz(i,j,k+1) - Bz(i,j,k)) / dx[idir];
      }
    }

    // Perform the characteristic projection.  Since we are using
    // Using HLLD, we sum over all eigenvalues

    // right state at i-1/2

    // Im is the integral from the left edge, so we take as the
    // reference state the fastest wave moving to the left

    Real summ_m[NEVEC] = {0.0_rt};

    // loop over the waves
    for (int ii = 0; ii < NEVEC; ii++) {

      Real LdQ = 0.0_rt;

      // loop over variables in Im[n][ii]
      for (int n = 0; n < NEVEC; n++) {
        if (lam(ii) <= 0.0_rt) {
          LdQ += evec.left(ii,n) * (q_ref_right[n] - Im[n][ii]);
        } else {
          // in this case, the integral Im is a slope
          LdQ += (lam(0) - lam(ii)) * evec.left(ii,n) * Im[n][ii];
        }
      }

      // add the contribution of this wave to each variable
      for (int n = 0; n < NEVEC; n++) {
        summ_m[n] += LdQ * evec.right(n,ii);
      }
    }

    qright(i,j,k,QRHO) = amrex::max(small_dens,
                                    q_ref_right[IEVEC_RHO] - summ_m[IEVEC_RHO] + 0.5_rt*dt*smhd[IEVEC_RHO]);
    qright(i,j,k,QU) = q_ref_right[IEVEC_U] - summ_m[IEVEC_U] + 0.5_rt*dt*smhd[IEVEC_U];
    qright(i,j,k,QV) = q_ref_right[IEVEC_V] - summ_m[IEVEC_V] + 0.5_rt*dt*smhd[IEVEC_V];
    qright(i,j,k,QW) = q_ref_right[IEVEC_W] - summ_m[IEVEC_W] + 0.5_rt*dt*smhd[IEVEC_W];
    qright(i,j,k,QPRES) = amrex::max(small_pres,
                                     q_ref_right[IEVEC_P] - summ_m[IEVEC_P] + 0.5_rt*dt*smhd[IEVEC_P]);

    if (idir == 0) {
      qright(i,j,k,QMAGX) = Bx(i,j,k); // Bx stuff
      qright(i,j,k,QMAGY) = q_ref_right[IEVEC_BT] -  summ_m[IEVEC_BT] + 0.5_rt*dt*smhd[IEVEC_BT];
      qright(i,j,k,QMAGZ) = q_ref_right[IEVEC_BTT] - summ_m[IEVEC_BTT] + 0.5_rt*dt*smhd[IEVEC_BTT];

    } else if (idir == 1) {
      qright(i,j,k,QMAGX) = q_ref_right[IEVEC_BT] - summ_m[IEVEC_BT] + 0.5_rt*dt*smhd[IEVEC_BT];
      qright(i,j,k,QMAGY) = By(i,j,k); // By stuff
      qright(i,j,k,QMAGZ) = q_ref_right[IEVEC_BTT] - summ_m[IEVEC_BTT] + 0.5_rt*dt*smhd[IEVEC_BTT];

    } else {
      qright(i,j,k,QMAGX) = q_ref_right[IEVEC_BT] - summ_m[IEVEC_BT] + 0.5_rt*dt*smhd[IEVEC_BT];
      qright(i,j,k,QMAGY) = q_ref_right[IEVEC_BTT] - summ_m[IEVEC_BTT] + 0.5_rt*dt*smhd[IEVEC_BTT];
      qright(i,j,k,QMAGZ) = Bz(i,j,k); // Bz stuff
    }


    // left state at i+1/2

    // Ip is the integral from the right edge, so we take as the
    // reference state the fastest wave moving to the right

    Real summ_p[NEVEC] = {0.0_rt};

    // loop over the waves
    for (int ii = 0; ii < NEVEC; ii++) {

      Real LdQ = 0.0_rt;

      // loop over variables in Im[n][ii]
      for (int n = 0; n < NEVEC; n++) {
        if (lam(ii) >= 0.0_rt) {
          LdQ += evec.left(ii,n) * (q_ref_left[n] - Ip[n][ii]);
        } else {
          // in this case, the integral Ip is a slope
          LdQ += (lam(NEVEC-1) - lam(ii)) * evec.left(ii,n) * Ip[n][ii];
        }
      }

      // add the contribution of this wave to each variable
      for (int n = 0; n < NEVEC; n++) {
        summ_p[n] += LdQ * evec.right(n,ii);
      }
    }

    if (idir == 0) {
      qleft(i+1,j,k,QRHO) = amrex::max(small_dens,
                                       q_ref_left[IEVEC_RHO] - summ_p[IEVEC_RHO] + 0.5_rt*dt*smhd[IEVEC_RHO]);
      qleft(i+1,j,k,QU) = q_ref_left[IEVEC_U] - summ_p[IEVEC_U] + 0.5_rt*dt*smhd[IEVEC_U];
      qleft(i+1,j,k,QV) = q_ref_left[IEVEC_V] - summ_p[IEVEC_V] + 0.5_rt*dt*smhd[IEVEC_V];
      qleft(i+1,j,k,QW) = q_ref_left[IEVEC_W] - summ_p[IEVEC_W] + 0.5_rt*dt*smhd[IEVEC_W];
      qleft(i+1,j,k,QPRES) = amrex::max(small_pres,
                                        q_ref_left[IEVEC_P] - summ_p[IEVEC_P] + 0.5_rt*dt*smhd[IEVEC_P]);

      qleft(i+1,j,k,QMAGX) = Bx(i+1,j,k); // Bx stuff
      qleft(i+1,j,k,QMAGY) = q_ref_left[IEVEC_BT] - summ_p[IEVEC_BT] + 0.5_rt*dt*smhd[IEVEC_BT];
      qleft(i+1,j,k,QMAGZ) = q_ref_left[IEVEC_BTT] - summ_p[IEVEC_BTT] + 0.5_rt*dt*smhd[IEVEC_BTT];

    } else if (idir == 1) {
      qleft(i,j+1,k,QRHO) = amrex::max(small_dens,
                                       q_ref_left[IEVEC_RHO] - summ_p[IEVEC_RHO] + 0.5_rt*dt*smhd[IEVEC_RHO]);
      qleft(i,j+1,k,QU) = q_ref_left[IEVEC_U] - summ_p[IEVEC_U] + 0.5_rt*dt*smhd[IEVEC_U];
      qleft(i,j+1,k,QV) = q_ref_left[IEVEC_V] - summ_p[IEVEC_V] + 0.5_rt*dt*smhd[IEVEC_V];
      qleft(i,j+1,k,QW) = q_ref_left[IEVEC_W] - summ_p[IEVEC_W] + 0.5_rt*dt*smhd[IEVEC_W];
      qleft(i,j+1,k,QPRES) = amrex::max(small_pres,
                                        q_ref_left[IEVEC_P] - summ_p[IEVEC_P] + 0.5_rt*dt*smhd[IEVEC_P]);

      qleft(i,j+1,k,QMAGX) = q_ref_left[IEVEC_BT] - summ_p[IEVEC_BT] + 0.5_rt*dt*smhd[IEVEC_BT];
      qleft(i,j+1,k,QMAGY) = By(i,j+1,k); // By stuff
      qleft(i,j+1,k,QMAGZ) = q_ref_left[IEVEC_BTT] - summ_p[IEVEC_BTT] + 0.5_rt*dt*smhd[IEVEC_BTT];

    } else {
      qleft(i,j,k+1,QRHO) = amrex::max(small_dens,
                                       q_ref_left[IEVEC_RHO] - summ_p[IEVEC_RHO] + 0.5_rt*dt*smhd[IEVEC_RHO]);
      qleft(i,j,k+1,QU) = q_ref_left[IEVEC_U] - summ_p[IEVEC_U] + 0.5_rt*dt*smhd[IEVEC_U];
      qleft(i,j,k+1,QV) = q_ref_left[IEVEC_V] - summ_p[IEVEC_V] + 0.5_rt*dt*smhd[IEVEC_V];
      qleft(i,j,k+1,QW) = q_ref_left[IEVEC_W] - summ_p[IEVEC_W] + 0.5_rt*dt*smhd[IEVEC_W];
      qleft(i,j,k+1,QPRES) = amrex::max(small_pres,
                                        q_ref_left[IEVEC_P] - summ_p[IEVEC_P] + 0.5_rt*dt*smhd[IEVEC_P]);

      qleft(i,j,k+1,QMAGX) = q_ref_left[IEVEC_BT] - summ_p[IEVEC_BT] + 0.5_rt*dt*smhd[IEVEC_BT];
      qleft(i,j,k+1,QMAGY) = q_ref_left[IEVEC_BTT] - summ_p[IEVEC_BTT] + 0.5_rt*dt*smhd[IEVEC_BTT];
      qleft(i,j,k+1,QMAGZ) = Bz(i,j,k+1); // Bz stuff
    }


    // species
    for (int n = 0; n < NumSpec; n++) {

      int v = QSPEC+n;

      Real Ips;
      Real Ims;

      Real un;

      if (idir == 0) {
        un = q_arr(i,j,k,QU);

      } else if (idir == 1) {
        un = q_arr(i,j,k,QV);

      } else {
        un = q_arr(i,j,k,QW);
      }

      s = load_stencil(q_arr, idir, i, j, k, v);
      edge = ppm_reconstruct(s, flat);
      amrex::Tie(Ips, Ims) = ppm_int_profile_single(edge.lo, edge.hi, s[i0], un, dtdx);

      if (idir == 0) {
        qleft(i+1,j,k,QSPEC+n) = Ips;
      } else if (idir == 1) {
        qleft(i,j+1,k,QSPEC+n) = Ips;
      } else {
        qleft(i,j,k+1,QSPEC+n) = Ips;
      }
      qright(i,j,k,QSPEC+n) = Ims;
    }

    // rho e
    eos_t eos_state;

    if (idir == 0) {
      eos_state.rho = qleft(i+1,j,k,QRHO);
      eos_state.p = qleft(i+1,j,k,QPRES);
      eos_state.T = q_arr(i,j,k,QTEMP); // some initial guess?
      for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = qleft(i+1,j,k,QSPEC+n);
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
        eos_state.aux[n] = qleft(i+1,j,k,QAUXV+n);
      }
#endif
      eos(eos_input_rp, eos_state);
      qleft(i+1,j,k,QRHOEI) = eos_state.e * eos_state.rho;

    } else if (idir == 1) {
      eos_state.rho = qleft(i,j+1,k,QRHO);
      eos_state.p = qleft(i,j+1,k,QPRES);
      eos_state.T = q_arr(i,j,k,QTEMP); // some initial guess?
      for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = qleft(i,j+1,k,QSPEC+n);
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
        eos_state.aux[n] = qleft(i,j+1,k,QAUXV+n);
      }
#endif
      eos(eos_input_rp, eos_state);
      qleft(i,j+1,k,QRHOEI) = eos_state.e * eos_state.rho;

    } else {
      eos_state.rho = qleft(i,j,k+1,QRHO);
      eos_state.p = qleft(i,j,k+1,QPRES);
      eos_state.T = q_arr(i,j,k,QTEMP); // some initial guess?
      for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = qleft(i,j,k+1,QSPEC+n);
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
        eos_state.aux[n] = qleft(i,j,k+1,QAUXV+n);
      }
#endif
      eos(eos_input_rp, eos_state);
      qleft(i,j,k+1,QRHOEI) = eos_state. e * eos_state.rho;
    }

    eos_state.rho = qright(i,j,k,QRHO);
    eos_state.p = qright(i,j,k,QPRES);
    for (int n = 0; n < NumSpec; n++) {
      eos_state.xn[n] = qright(i,j,k,QSPEC+n);
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
      eos_state.aux[n] = qright(i,j,k,QAUXV+n);
    }
#endif

    eos(eos_input_rp, eos_state);
    qright(i,j,k,QRHOEI) = eos_state.e * eos_state.rho;

    // add source terms
    if (idir == 0) {
      qleft(i+1,j,k,QRHO) = amrex::max(small_dens,
                                       qleft(i+1,j,k,QRHO) + 0.5_rt*dt*srcQ(i,j,k,QRHO));
      qleft(i+1,j,k,QU) = qleft(i+1,j,k,QU) + 0.5_rt*dt*srcQ(i,j,k,QU);
      qleft(i+1,j,k,QV) = qleft(i+1,j,k,QV) + 0.5_rt*dt*srcQ(i,j,k,QV);
      qleft(i+1,j,k,QW) = qleft(i+1,j,k,QW) + 0.5_rt*dt*srcQ(i,j,k,QW);
      qleft(i+1,j,k,QPRES) = qleft(i+1,j,k,QPRES) + 0.5_rt*dt*srcQ(i,j,k,QPRES);
      qleft(i+1,j,k,QRHOEI) = qleft(i+1,j,k,QRHOEI) + 0.5_rt*dt*srcQ(i,j,k,QRHOEI);

    } else if (idir == 1) {
      qleft(i,j+1,k,QRHO) = amrex::max(small_dens,
                                       qleft(i,j+1,k,QRHO) + 0.5_rt*dt*srcQ(i,j,k,QRHO));
      qleft(i,j+1,k,QU) = qleft(i,j+1,k,QU) + 0.5_rt*dt*srcQ(i,j,k,QU);
      qleft(i,j+1,k,QV) = qleft(i,j+1,k,QV) + 0.5_rt*dt*srcQ(i,j,k,QV);
      qleft(i,j+1,k,QW) = qleft(i,j+1,k,QW) + 0.5_rt*dt*srcQ(i,j,k,QW);
      qleft(i,j+1,k,QPRES) = qleft(i,j+1,k,QPRES) + 0.5_rt*dt*srcQ(i,j,k,QPRES);
      qleft(i,j+1,k,QRHOEI) = qleft(i,j+1,k,QRHOEI) + 0.5_rt*dt*srcQ(i,j,k,QRHOEI);

    } else {
      qleft(i,j,k+1,QRHO) = amrex::max(small_dens, qleft(i,j,k+1,QRHO) + 0.5_rt*dt*srcQ(i,j,k,QRHO));
      qleft(i,j,k+1,QU) = qleft(i,j,k+1,QU) + 0.5_rt*dt*srcQ(i,j,k,QU);
      qleft(i,j,k+1,QV) = qleft(i,j,k+1,QV) + 0.5_rt*dt*srcQ(i,j,k,QV);
      qleft(i,j,k+1,QW) = qleft(i,j,k+1,QW) + 0.5_rt*dt*srcQ(i,j,k,QW);
      qleft(i,j,k+1,QPRES) = qleft(i,j,k+1,QPRES) + 0.5_rt*dt*srcQ(i,j,k,QPRES);
      qleft(i,j,k+1,QRHOEI) = qleft(i,j,k+1,QRHOEI) + 0.5_rt*dt*srcQ(i,j,k,QRHOEI);
    }

    qright(i,j,k,QRHO) = amrex::max(small_dens, qright(i,j,k,QRHO) + 0.5_rt*dt*srcQ(i,j,k,QRHO));
    qright(i,j,k,QU) = qright(i,j,k,QU) + 0.5_rt*dt*srcQ(i,j,k,QU);
    qright(i,j,k,QV) = qright(i,j,k,QV) + 0.5_rt*dt*srcQ(i,j,k,QV);
    qright(i,j,k,QW) = qright(i,j,k,QW) + 0.5_rt*dt*srcQ(i,j,k,QW);
    qright(i,j,k,QPRES) = qright(i,j,k,QPRES) + 0.5_rt*dt*srcQ(i,j,k,QPRES);
    qright(i,j,k,QRHOEI) = qright(i,j,k,QRHOEI) + 0.5_rt*dt*srcQ(i,j,k,QRHOEI);

  });
}
