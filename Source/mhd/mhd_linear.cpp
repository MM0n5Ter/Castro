#include <HydroSim.H>

using namespace amrex;

#include <mhd_wave_basis.H>
#include <slope.H>
void
HydroSim::plm(const Box& bx,
            const int idir,
            Array4<Real const> const& s,
            Array4<Real const> const& qthermo,
            Array4<Real const> const& flatten_factor,
            Array4<Real const> const& Bx,
            Array4<Real const> const& By,
            Array4<Real const> const& Bz,
            Array4<Real> const& prim_left,
            Array4<Real> const& prim_right,
            Array4<Real const> const& prim_source,
            const Real dt) {

  // these loops are over cell-centers and for each cell-center, we find the left and
  // right interface states

  const auto dx = geom.CellSizeArray();

  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  bool lo_symmetric = lo_bc[idir] == amrex::PhysBCType::symmetry;
  bool hi_symmetric = hi_bc[idir] == amrex::PhysBCType::symmetry;

  const auto domlo = geom.Domain().loVect3d();
  const auto domhi = geom.Domain().hiVect3d();

  Real dtdx = dt/dx[idir];

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    bool at_lo_symmetry = lo_symmetric && ((idir == 0 && i == domlo[0]) ||
                                  (idir == 1 && j == domlo[1]) ||
                                  (idir == 2 && k == domlo[2]));

    bool at_hi_symmetry = hi_symmetric && ((idir == 0 && i == domhi[0]) ||
                                  (idir == 1 && j == domhi[1]) ||
                                  (idir == 2 && k == domhi[2]));

    // compute the 1-sided differences used for the slopes

    Real Q[NUM_EIGVEC][nslp];

    // we use a reduced eigensystem, the normal B field component is
    // omitted

    // for reflect BCs, we need to know what the normal component of
    // the velocity is
    int QUN;

    gather_stencil(s, idir, i, j, k, QRHO, Q[EIGVEC_RHO]);
    gather_stencil(s, idir, i, j, k, QU, Q[EIGVEC_U]);
    gather_stencil(s, idir, i, j, k, QV, Q[EIGVEC_V]);
    gather_stencil(s, idir, i, j, k, QW, Q[EIGVEC_W]);
    gather_stencil(s, idir, i, j, k, QPRES, Q[EIGVEC_PRESSURE]);

    if (idir == 0) {
        QUN = EIGVEC_U;

        // component (Bx) is omitted
        gather_stencil(s, idir, i, j, k, QBFIELDY, Q[EIGVEC_BPERP1]);
        gather_stencil(s, idir, i, j, k, QBFIELDZ, Q[EIGVEC_BPERP2]);

    } else if (idir == 1) {

        QUN = EIGVEC_V;

        // component (By) is omitted
        gather_stencil(s, idir, i, j, k, QBFIELDX, Q[EIGVEC_BPERP1]);
        gather_stencil(s, idir, i, j, k, QBFIELDZ, Q[EIGVEC_BPERP2]);

    } else {

        QUN = EIGVEC_W;

        // component (Bz) is omitted
        gather_stencil(s, idir, i, j, k, QBFIELDX, Q[EIGVEC_BPERP1]);
        gather_stencil(s, idir, i, j, k, QBFIELDY, Q[EIGVEC_BPERP2]);

    }

    // compute the eigenvectors and eigenvalues for this coordinate direction

    Array1D<Real, 0, NQ-1> zone_prim;
    for (int n = 0; n < NQ; n++) {
      zone_prim(n) = s(i,j,k,n);
    }

    Real as = qthermo(i,j,k,QC);

    Array1D<Real, 0, NUM_EIGVEC-1> lam;

    evals(lam, as, zone_prim, idir);

    Array2D<Real, 0, NUM_EIGVEC-1, 0, NUM_EIGVEC-1> left_basis;
    Array2D<Real, 0, NUM_EIGVEC-1, 0, NUM_EIGVEC-1> right_basis;

    if (idir == 0) {
      eigenbasis_x(left_basis, right_basis, as, zone_prim);

    } else if (idir == 1) {
      eigenbasis_y(left_basis, right_basis, as, zone_prim);

    } else {
      eigenbasis_z(left_basis, right_basis, as, zone_prim);
    }

    // MHD Source Terms
    Real mhd_source[NUM_EIGVEC];

    mhd_source[EIGVEC_RHO] = 0.0_rt;
    mhd_source[EIGVEC_U] = zone_prim(QBFIELDX) / zone_prim(QRHO);
    mhd_source[EIGVEC_V] = zone_prim(QBFIELDY) / zone_prim(QRHO);
    mhd_source[EIGVEC_W] = zone_prim(QBFIELDZ) / zone_prim(QRHO);
    mhd_source[EIGVEC_PRESSURE] = zone_prim(QBFIELDX) * zone_prim(QU) +
                    zone_prim(QBFIELDY) * zone_prim(QV) +
                    zone_prim(QBFIELDZ) * zone_prim(QW);

    if (idir == 0) {
      mhd_source[EIGVEC_BPERP1] = zone_prim(QV);
      mhd_source[EIGVEC_BPERP2] = zone_prim(QW);

      // cross-talk of normal magnetic field direction
      for (auto & source : mhd_source) {
          source *= (Bx(i+1,j,k) - Bx(i,j,k)) / dx[idir];
      }

    } else if (idir == 1) {
      mhd_source[EIGVEC_BPERP1] = zone_prim(QU);
      mhd_source[EIGVEC_BPERP2] = zone_prim(QW);

      // cross-talk of normal magnetic field direction
      for (auto & source : mhd_source) {
          source *= (By(i,j+1,k) - By(i,j,k)) / dx[idir];
      }

    } else {
      mhd_source[EIGVEC_BPERP1] = zone_prim(QU);
      mhd_source[EIGVEC_BPERP2] = zone_prim(QV);

      // cross-talk of normal magnetic field direction
      for (auto & source : mhd_source) {
          source *= (Bz(i,j,k+1) - Bz(i,j,k)) / dx[idir];
      }
    }

    // compute the slopes
    Real dq[NUM_EIGVEC] = {};

    if (mhd_char_limit == 1) {

      // we are limiting on characteristic variables
      for (int ii = 0; ii < NUM_EIGVEC; ii++) {

        // construct the ii-th primitive variables
        Real W[nslp] = {};
        for (int n = 0; n < NUM_EIGVEC; n++) {
            for (int idx = 0; idx < nslp; ++idx) {
                W[idx] += left_basis(ii, n) * Q[n][idx];
            }
        }

        // now limit
        Real dW = limited_slope(W, flatten_factor(i,j,k), false, false);

        // now add this charactistic variable's contribution to the
        // primitive variable slope
        for (int n = 0; n < NUM_EIGVEC; n++) {
          dq[n] += dW * right_basis(n, ii);
        }
      }

    } else {

      // we are limiting on primitive variables

      for (int ii = 0; ii < NUM_EIGVEC; ii++) {
        bool is_normal_velocity = ii == QUN;
        dq[ii] = limited_slope(Q[ii], flatten_factor(i,j,k), at_lo_symmetry && is_normal_velocity, at_hi_symmetry && is_normal_velocity);
      }

    }


    // Perform the characteristic projection.  Since we are using
    // Using HLLD, we sum over all eigenvalues
    Real plus_sum[NUM_EIGVEC] = {0.0_rt};
    Real minus_sum[NUM_EIGVEC] = {0.0_rt};

    for (int ii = 0; ii < NUM_EIGVEC; ii++) {
      Real Ldq = 0.0;

      for (int n = 0; n < NUM_EIGVEC; n++) {
        Ldq += left_basis(ii,n) * dq[n];
      }

      for (int n = 0; n < NUM_EIGVEC; n++) {
        plus_sum[n] += (1.0_rt - dtdx * lam(ii)) * Ldq * right_basis(n,ii);
        minus_sum[n] -= (1.0_rt + dtdx * lam(ii)) * Ldq * right_basis(n,ii);
      }
    }

    // left state at i+1/2

    if (idir == 0) {
      prim_left(i+1,j,k,QRHO) = amrex::max(small_dens,
                                       zone_prim(QRHO) + 0.5_rt*plus_sum[EIGVEC_RHO] + 0.5_rt*dt*mhd_source[EIGVEC_RHO]);
      prim_left(i+1,j,k,QU) = zone_prim(QU) + 0.5_rt*plus_sum[EIGVEC_U] + 0.5_rt*dt*mhd_source[EIGVEC_U];
      prim_left(i+1,j,k,QV) = zone_prim(QV) + 0.5_rt*plus_sum[EIGVEC_V] + 0.5_rt*dt*mhd_source[EIGVEC_V];
      prim_left(i+1,j,k,QW) = zone_prim(QW) + 0.5_rt*plus_sum[EIGVEC_W] + 0.5_rt*dt*mhd_source[EIGVEC_W];
      prim_left(i+1,j,k,QPRES) = amrex::max(pressure_cutoff,
                                        zone_prim(QPRES) + 0.5_rt*plus_sum[EIGVEC_PRESSURE] + 0.5_rt*dt*mhd_source[EIGVEC_PRESSURE]);

      prim_left(i+1,j,k,QBFIELDX) = Bx(i+1,j,k); // Bx stuff
      prim_left(i+1,j,k,QBFIELDY) = zone_prim(QBFIELDY) + 0.5_rt*plus_sum[EIGVEC_BPERP1] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP1];
      prim_left(i+1,j,k,QBFIELDZ) = zone_prim(QBFIELDZ) + 0.5_rt*plus_sum[EIGVEC_BPERP2] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP2];

    } else if (idir == 1) {
      prim_left(i,j+1,k,QRHO) = amrex::max(small_dens,
                                       zone_prim(QRHO) + 0.5_rt*plus_sum[EIGVEC_RHO] + 0.5_rt*dt*mhd_source[EIGVEC_RHO]);
      prim_left(i,j+1,k,QU) = zone_prim(QU) + 0.5_rt*plus_sum[EIGVEC_U] + 0.5_rt*dt*mhd_source[EIGVEC_U];
      prim_left(i,j+1,k,QV) = zone_prim(QV) + 0.5_rt*plus_sum[EIGVEC_V] + 0.5_rt*dt*mhd_source[EIGVEC_V];
      prim_left(i,j+1,k,QW) = zone_prim(QW) + 0.5_rt*plus_sum[EIGVEC_W] + 0.5_rt*dt*mhd_source[EIGVEC_W];
      prim_left(i,j+1,k,QPRES) = amrex::max(pressure_cutoff,
                                        zone_prim(QPRES) + 0.5_rt*plus_sum[EIGVEC_PRESSURE] + 0.5_rt*dt*mhd_source[EIGVEC_PRESSURE]);

      prim_left(i,j+1,k,QBFIELDX) = zone_prim(QBFIELDX) + 0.5_rt*plus_sum[EIGVEC_BPERP1] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP1];
      prim_left(i,j+1,k,QBFIELDY) = By(i,j+1,k); // By stuff
      prim_left(i,j+1,k,QBFIELDZ) = zone_prim(QBFIELDZ) + 0.5_rt*plus_sum[EIGVEC_BPERP2] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP2];

    } else {
      prim_left(i,j,k+1,QRHO) = amrex::max(small_dens,
                                       zone_prim(QRHO) + 0.5_rt*plus_sum[EIGVEC_RHO] + 0.5_rt*dt*mhd_source[EIGVEC_RHO]);
      prim_left(i,j,k+1,QU) = zone_prim(QU) + 0.5_rt*plus_sum[EIGVEC_U] + 0.5_rt*dt*mhd_source[EIGVEC_U];
      prim_left(i,j,k+1,QV) = zone_prim(QV) + 0.5_rt*plus_sum[EIGVEC_V] + 0.5_rt*dt*mhd_source[EIGVEC_V];
      prim_left(i,j,k+1,QW) = zone_prim(QW) + 0.5_rt*plus_sum[EIGVEC_W] + 0.5_rt*dt*mhd_source[EIGVEC_W];
      prim_left(i,j,k+1,QPRES) = amrex::max(pressure_cutoff,
                                        zone_prim(QPRES) + 0.5_rt*plus_sum[EIGVEC_PRESSURE] + 0.5_rt*dt*mhd_source[EIGVEC_PRESSURE]);

      prim_left(i,j,k+1,QBFIELDX) = zone_prim(QBFIELDX) + 0.5_rt*plus_sum[EIGVEC_BPERP1] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP1];
      prim_left(i,j,k+1,QBFIELDY) = zone_prim(QBFIELDY) + 0.5_rt*plus_sum[EIGVEC_BPERP2] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP2];
      prim_left(i,j,k+1,QBFIELDZ) = Bz(i,j,k+1); // Bz stuff
    }

    // right state at i-1/2
    prim_right(i,j,k,QRHO) = amrex::max(small_dens,
                                    zone_prim(QRHO) + 0.5_rt*minus_sum[EIGVEC_RHO] + 0.5_rt*dt*mhd_source[EIGVEC_RHO]);
    prim_right(i,j,k,QU) = zone_prim(QU) + 0.5_rt*minus_sum[EIGVEC_U] + 0.5_rt*dt*mhd_source[EIGVEC_U];
    prim_right(i,j,k,QV) = zone_prim(QV) + 0.5_rt*minus_sum[EIGVEC_V] + 0.5_rt*dt*mhd_source[EIGVEC_V];
    prim_right(i,j,k,QW) = zone_prim(QW) + 0.5_rt*minus_sum[EIGVEC_W] + 0.5_rt*dt*mhd_source[EIGVEC_W];
    prim_right(i,j,k,QPRES) = amrex::max(pressure_cutoff,
                                     zone_prim(QPRES) + 0.5_rt*minus_sum[EIGVEC_PRESSURE] + 0.5_rt*dt*mhd_source[EIGVEC_PRESSURE]);

    if (idir == 0) {
      prim_right(i,j,k,QBFIELDX) = Bx(i,j,k); // Bx stuff
      prim_right(i,j,k,QBFIELDY) = zone_prim(QBFIELDY) + 0.5_rt*minus_sum[EIGVEC_BPERP1] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP1];
      prim_right(i,j,k,QBFIELDZ) = zone_prim(QBFIELDZ) + 0.5_rt*minus_sum[EIGVEC_BPERP2] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP2];

    } else if (idir == 1) {
      prim_right(i,j,k,QBFIELDX) = zone_prim(QBFIELDX) + 0.5_rt*minus_sum[EIGVEC_BPERP1] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP1];
      prim_right(i,j,k,QBFIELDY) = By(i,j,k); // By stuff
      prim_right(i,j,k,QBFIELDZ) = zone_prim(QBFIELDZ) + 0.5_rt*minus_sum[EIGVEC_BPERP2] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP2];

    } else {
      prim_right(i,j,k,QBFIELDX) = zone_prim(QBFIELDX) + 0.5_rt*minus_sum[EIGVEC_BPERP1] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP1];
      prim_right(i,j,k,QBFIELDY) = zone_prim(QBFIELDY) + 0.5_rt*minus_sum[EIGVEC_BPERP2] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP2];
      prim_right(i,j,k,QBFIELDZ) = Bz(i,j,k); // Bz stuff
    }

    // species
    for (int n = 0; n < NumSpec; n++) {
      Real un{};
      Real X[nslp];

      gather_stencil(s, idir, i, j, k, QFS+n, X);

      if (idir == 0) {
        un = s(i,j,k,QU);
      } else if (idir == 1) {
        un = s(i,j,k,QV);
      } else {
        un = s(i,j,k,QW);
      }

      Real dX = limited_slope(X, flatten_factor(i,j,k), false, false);

      if (idir == 0) {
        prim_left(i+1,j,k,QFS+n) = zone_prim(QFS+n) + 0.5_rt*(1.0_rt - dtdx*un) * dX;
      } else if (idir == 1) {
        prim_left(i,j+1,k,QFS+n) = zone_prim(QFS+n) + 0.5_rt*(1.0_rt - dtdx*un) * dX;
      } else {
        prim_left(i,j,k+1,QFS+n) = zone_prim(QFS+n) + 0.5_rt*(1.0_rt - dtdx*un) * dX;
      }
      prim_right(i,j,k,QFS+n) = zone_prim(QFS+n) - 0.5_rt*(1.0_rt + dtdx*un) * dX;
    }

    // rho e
    eos_t eos_state;

    if (idir == 0) {
      eos_state.rho = prim_left(i+1,j,k,QRHO);
      eos_state.p = prim_left(i+1,j,k,QPRES);
      eos_state.T = s(i,j,k,QTEMP); // some initial guess?
      for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = prim_left(i+1,j,k,QFS+n);
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
        eos_state.aux[n] = prim_left(i+1,j,k,QFX+n);
      }
#endif
      eos(eos_input_rp, eos_state);
      prim_left(i+1,j,k,QEINTDENS) = eos_state.e * eos_state.rho;

    } else if (idir == 1) {
      eos_state.rho = prim_left(i,j+1,k,QRHO);
      eos_state.p = prim_left(i,j+1,k,QPRES);
      eos_state.T = s(i,j,k,QTEMP); // some initial guess?
      for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = prim_left(i,j+1,k,QFS+n);
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
        eos_state.aux[n] = prim_left(i,j+1,k,QFX+n);
      }
#endif
      eos(eos_input_rp, eos_state);
      prim_left(i,j+1,k,QEINTDENS) = eos_state.e * eos_state.rho;

    } else {
      eos_state.rho = prim_left(i,j,k+1,QRHO);
      eos_state.p = prim_left(i,j,k+1,QPRES);
      eos_state.T = s(i,j,k,QTEMP); // some initial guess?
      for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = prim_left(i,j,k+1,QFS+n);
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
        eos_state.aux[n] = prim_left(i,j,k+1,QFX+n);
      }
#endif
      eos(eos_input_rp, eos_state);
      prim_left(i,j,k+1,QEINTDENS) = eos_state. e * eos_state.rho;
    }

    eos_state.rho = prim_right(i,j,k,QRHO);
    eos_state.p = prim_right(i,j,k,QPRES);
    for (int n = 0; n < NumSpec; n++) {
      eos_state.xn[n] = prim_right(i,j,k,QFS+n);
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
      eos_state.aux[n] = prim_right(i,j,k,QFX+n);
    }
#endif

    eos(eos_input_rp, eos_state);
    prim_right(i,j,k,QEINTDENS) = eos_state.e * eos_state.rho;

    // add source terms
    if (idir == 0) {
      prim_left(i+1,j,k,QRHO) = amrex::max(small_dens,
                                       prim_left(i+1,j,k,QRHO) + 0.5_rt*dt*prim_source(i,j,k,QRHO));
      prim_left(i+1,j,k,QU) = prim_left(i+1,j,k,QU) + 0.5_rt*dt*prim_source(i,j,k,QU);
      prim_left(i+1,j,k,QV) = prim_left(i+1,j,k,QV) + 0.5_rt*dt*prim_source(i,j,k,QV);
      prim_left(i+1,j,k,QW) = prim_left(i+1,j,k,QW) + 0.5_rt*dt*prim_source(i,j,k,QW);
      prim_left(i+1,j,k,QPRES) = prim_left(i+1,j,k,QPRES) + 0.5_rt*dt*prim_source(i,j,k,QPRES);
      prim_left(i+1,j,k,QEINTDENS) = prim_left(i+1,j,k,QEINTDENS) + 0.5_rt*dt*prim_source(i,j,k,QEINTDENS);

    } else if (idir == 1) {
      prim_left(i,j+1,k,QRHO) = amrex::max(small_dens,
                                       prim_left(i,j+1,k,QRHO) + 0.5_rt*dt*prim_source(i,j,k,QRHO));
      prim_left(i,j+1,k,QU) = prim_left(i,j+1,k,QU) + 0.5_rt*dt*prim_source(i,j,k,QU);
      prim_left(i,j+1,k,QV) = prim_left(i,j+1,k,QV) + 0.5_rt*dt*prim_source(i,j,k,QV);
      prim_left(i,j+1,k,QW) = prim_left(i,j+1,k,QW) + 0.5_rt*dt*prim_source(i,j,k,QW);
      prim_left(i,j+1,k,QPRES) = prim_left(i,j+1,k,QPRES) + 0.5_rt*dt*prim_source(i,j,k,QPRES);
      prim_left(i,j+1,k,QEINTDENS) = prim_left(i,j+1,k,QEINTDENS) + 0.5_rt*dt*prim_source(i,j,k,QEINTDENS);

    } else {
      prim_left(i,j,k+1,QRHO) = amrex::max(small_dens, prim_left(i,j,k+1,QRHO) + 0.5_rt*dt*prim_source(i,j,k,QRHO));
      prim_left(i,j,k+1,QU) = prim_left(i,j,k+1,QU) + 0.5_rt*dt*prim_source(i,j,k,QU);
      prim_left(i,j,k+1,QV) = prim_left(i,j,k+1,QV) + 0.5_rt*dt*prim_source(i,j,k,QV);
      prim_left(i,j,k+1,QW) = prim_left(i,j,k+1,QW) + 0.5_rt*dt*prim_source(i,j,k,QW);
      prim_left(i,j,k+1,QPRES) = prim_left(i,j,k+1,QPRES) + 0.5_rt*dt*prim_source(i,j,k,QPRES);
      prim_left(i,j,k+1,QEINTDENS) = prim_left(i,j,k+1,QEINTDENS) + 0.5_rt*dt*prim_source(i,j,k,QEINTDENS);
    }

    prim_right(i,j,k,QRHO) = amrex::max(small_dens, prim_right(i,j,k,QRHO) + 0.5_rt*dt*prim_source(i,j,k,QRHO));
    prim_right(i,j,k,QU) = prim_right(i,j,k,QU) + 0.5_rt*dt*prim_source(i,j,k,QU);
    prim_right(i,j,k,QV) = prim_right(i,j,k,QV) + 0.5_rt*dt*prim_source(i,j,k,QV);
    prim_right(i,j,k,QW) = prim_right(i,j,k,QW) + 0.5_rt*dt*prim_source(i,j,k,QW);
    prim_right(i,j,k,QPRES) = prim_right(i,j,k,QPRES) + 0.5_rt*dt*prim_source(i,j,k,QPRES);
    prim_right(i,j,k,QEINTDENS) = prim_right(i,j,k,QEINTDENS) + 0.5_rt*dt*prim_source(i,j,k,QEINTDENS);

  });
}
