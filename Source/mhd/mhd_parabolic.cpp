#include <HydroSim.H>

#include <ppm.H>

using namespace amrex;

#include <mhd_wave_basis.H>

void
HydroSim::parabolic_reconstruct_mhd(const Box& bx,
                const int idir,
                Array4<Real const> const& prim_arr,
                Array4<Real const> const& qthermo,
                Array4<Real const> const& flatten_factor,
                Array4<Real const> const& Bx,
                Array4<Real const> const& By,
                Array4<Real const> const& Bz,
                Array4<Real> const& prim_left,
                Array4<Real> const& prim_right,
                Array4<Real const> const& prim_source,
                const Real dt) {


  const auto dx = geom.CellSizeArray();

  Real dtdx = dt/dx[idir];

  int char_var_map[NUM_EIGVEC];

  if (idir == 0) {


    char_var_map[EIGVEC_RHO] = QRHO;
    char_var_map[EIGVEC_U] = QU;
    char_var_map[EIGVEC_V] = QV;
    char_var_map[EIGVEC_W] = QW;
    char_var_map[EIGVEC_PRESSURE] = QPRES;
    char_var_map[EIGVEC_BPERP1] = QBFIELDY;
    char_var_map[EIGVEC_BPERP2] = QBFIELDZ;

  } else if (idir == 1) {


    char_var_map[EIGVEC_RHO] = QRHO;
    char_var_map[EIGVEC_U] = QU;
    char_var_map[EIGVEC_V] = QV;
    char_var_map[EIGVEC_W] = QW;
    char_var_map[EIGVEC_PRESSURE] = QPRES;
    char_var_map[EIGVEC_BPERP1] = QBFIELDX;
    char_var_map[EIGVEC_BPERP2] = QBFIELDZ;

  } else {


    char_var_map[EIGVEC_RHO] = QRHO;
    char_var_map[EIGVEC_U] = QU;
    char_var_map[EIGVEC_V] = QV;
    char_var_map[EIGVEC_W] = QW;
    char_var_map[EIGVEC_PRESSURE] = QPRES;
    char_var_map[EIGVEC_BPERP1] = QBFIELDX;
    char_var_map[EIGVEC_BPERP2] = QBFIELDY;

  }


  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {


    Array1D<Real, 0, NQ-1> zone_prim;
    for (int n = 0; n < NQ; n++) {
      zone_prim(n) = prim_arr(i,j,k,n);
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

    Real s[nslp];
    Real flat = flatten_factor(i,j,k);
    Real sm;
    Real sp;

    Real Ip[NUM_EIGVEC][NUM_EIGVEC];
    Real Im[NUM_EIGVEC][NUM_EIGVEC];

    Real interface_prim_left[NUM_EIGVEC];
    Real interface_prim_right[NUM_EIGVEC];


    for (int n = 0; n < NUM_EIGVEC; n++) {

      int v = char_var_map[n];

      gather_stencil(prim_arr, idir, i, j, k, v, s);
      parabolic_reconstruct(s, flat, sm, sp);

      Real Ipt = 0.0;
      Real Imt = 0.0;

      for (int ii = 0; ii < NUM_EIGVEC; ii++) {
        Real speed = lam(ii);

        integrate_parabolic_profile_single(sm, sp, s[i0], speed, dtdx, Ipt, Imt);

        if (lam(ii) >= 0.0_rt) {
          Ip[n][ii] = Ipt;
        } else {
          Ip[n][ii] = 0.5_rt * dtdx * (sp - sm);
        }

        if (lam(ii) <= 0.0_rt) {
          Im[n][ii] = Imt;
        } else {
          Im[n][ii] = 0.5_rt * dtdx * (sp - sm);
        }

      }


      Real lambda_max = amrex::max(lam(NUM_EIGVEC-1), 0.0_rt);
      integrate_parabolic_profile_single(sm, sp, s[i0], lambda_max, dtdx, Ipt, Imt);
      interface_prim_left[n] = Ipt;

      Real lambda_min = amrex::min(lam(0), 0.0_rt);
      integrate_parabolic_profile_single(sm, sp, s[i0], lambda_min, dtdx, Ipt, Imt);
      interface_prim_right[n] = Imt;

    }

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

      for (auto & source : mhd_source) {
          source *= (Bx(i+1,j,k) - Bx(i,j,k)) / dx[idir];
      }

    } else if (idir == 1) {
      mhd_source[EIGVEC_BPERP1] = zone_prim(QU);
      mhd_source[EIGVEC_BPERP2] = zone_prim(QW);

      for (auto & source : mhd_source) {
          source *= (By(i,j+1,k) - By(i,j,k)) / dx[idir];
      }

    } else {
      mhd_source[EIGVEC_BPERP1] = zone_prim(QU);
      mhd_source[EIGVEC_BPERP2] = zone_prim(QV);

      for (auto & source : mhd_source) {
          source *= (Bz(i,j,k+1) - Bz(i,j,k)) / dx[idir];
      }
    }




    Real minus_sum[NUM_EIGVEC] = {0.0_rt};

    for (int ii = 0; ii < NUM_EIGVEC; ii++) {

      Real LdQ = 0.0_rt;

      for (int n = 0; n < NUM_EIGVEC; n++) {
        if (lam(ii) <= 0.0_rt) {
          LdQ += left_basis(ii,n) * (interface_prim_right[n] - Im[n][ii]);
        } else {
          LdQ += (lam(0) - lam(ii)) * left_basis(ii,n) * Im[n][ii];
        }
      }

      for (int n = 0; n < NUM_EIGVEC; n++) {
        minus_sum[n] += LdQ * right_basis(n,ii);
      }
    }

    prim_right(i,j,k,QRHO) = amrex::max(small_dens,
                                    interface_prim_right[EIGVEC_RHO] - minus_sum[EIGVEC_RHO] + 0.5_rt*dt*mhd_source[EIGVEC_RHO]);
    prim_right(i,j,k,QU) = interface_prim_right[EIGVEC_U] - minus_sum[EIGVEC_U] + 0.5_rt*dt*mhd_source[EIGVEC_U];
    prim_right(i,j,k,QV) = interface_prim_right[EIGVEC_V] - minus_sum[EIGVEC_V] + 0.5_rt*dt*mhd_source[EIGVEC_V];
    prim_right(i,j,k,QW) = interface_prim_right[EIGVEC_W] - minus_sum[EIGVEC_W] + 0.5_rt*dt*mhd_source[EIGVEC_W];
    prim_right(i,j,k,QPRES) = amrex::max(pressure_cutoff,
                                     interface_prim_right[EIGVEC_PRESSURE] - minus_sum[EIGVEC_PRESSURE] + 0.5_rt*dt*mhd_source[EIGVEC_PRESSURE]);

    if (idir == 0) {
      prim_right(i,j,k,QBFIELDX) = Bx(i,j,k);
      prim_right(i,j,k,QBFIELDY) = interface_prim_right[EIGVEC_BPERP1] -  minus_sum[EIGVEC_BPERP1] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP1];
      prim_right(i,j,k,QBFIELDZ) = interface_prim_right[EIGVEC_BPERP2] - minus_sum[EIGVEC_BPERP2] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP2];

    } else if (idir == 1) {
      prim_right(i,j,k,QBFIELDX) = interface_prim_right[EIGVEC_BPERP1] - minus_sum[EIGVEC_BPERP1] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP1];
      prim_right(i,j,k,QBFIELDY) = By(i,j,k);
      prim_right(i,j,k,QBFIELDZ) = interface_prim_right[EIGVEC_BPERP2] - minus_sum[EIGVEC_BPERP2] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP2];

    } else {
      prim_right(i,j,k,QBFIELDX) = interface_prim_right[EIGVEC_BPERP1] - minus_sum[EIGVEC_BPERP1] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP1];
      prim_right(i,j,k,QBFIELDY) = interface_prim_right[EIGVEC_BPERP2] - minus_sum[EIGVEC_BPERP2] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP2];
      prim_right(i,j,k,QBFIELDZ) = Bz(i,j,k);
    }




    Real plus_sum[NUM_EIGVEC] = {0.0_rt};

    for (int ii = 0; ii < NUM_EIGVEC; ii++) {

      Real LdQ = 0.0_rt;

      for (int n = 0; n < NUM_EIGVEC; n++) {
        if (lam(ii) >= 0.0_rt) {
          LdQ += left_basis(ii,n) * (interface_prim_left[n] - Ip[n][ii]);
        } else {
          LdQ += (lam(NUM_EIGVEC-1) - lam(ii)) * left_basis(ii,n) * Ip[n][ii];
        }
      }

      for (int n = 0; n < NUM_EIGVEC; n++) {
        plus_sum[n] += LdQ * right_basis(n,ii);
      }
    }

    if (idir == 0) {
      prim_left(i+1,j,k,QRHO) = amrex::max(small_dens,
                                       interface_prim_left[EIGVEC_RHO] - plus_sum[EIGVEC_RHO] + 0.5_rt*dt*mhd_source[EIGVEC_RHO]);
      prim_left(i+1,j,k,QU) = interface_prim_left[EIGVEC_U] - plus_sum[EIGVEC_U] + 0.5_rt*dt*mhd_source[EIGVEC_U];
      prim_left(i+1,j,k,QV) = interface_prim_left[EIGVEC_V] - plus_sum[EIGVEC_V] + 0.5_rt*dt*mhd_source[EIGVEC_V];
      prim_left(i+1,j,k,QW) = interface_prim_left[EIGVEC_W] - plus_sum[EIGVEC_W] + 0.5_rt*dt*mhd_source[EIGVEC_W];
      prim_left(i+1,j,k,QPRES) = amrex::max(pressure_cutoff,
                                        interface_prim_left[EIGVEC_PRESSURE] - plus_sum[EIGVEC_PRESSURE] + 0.5_rt*dt*mhd_source[EIGVEC_PRESSURE]);

      prim_left(i+1,j,k,QBFIELDX) = Bx(i+1,j,k);
      prim_left(i+1,j,k,QBFIELDY) = interface_prim_left[EIGVEC_BPERP1] - plus_sum[EIGVEC_BPERP1] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP1];
      prim_left(i+1,j,k,QBFIELDZ) = interface_prim_left[EIGVEC_BPERP2] - plus_sum[EIGVEC_BPERP2] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP2];

    } else if (idir == 1) {
      prim_left(i,j+1,k,QRHO) = amrex::max(small_dens,
                                       interface_prim_left[EIGVEC_RHO] - plus_sum[EIGVEC_RHO] + 0.5_rt*dt*mhd_source[EIGVEC_RHO]);
      prim_left(i,j+1,k,QU) = interface_prim_left[EIGVEC_U] - plus_sum[EIGVEC_U] + 0.5_rt*dt*mhd_source[EIGVEC_U];
      prim_left(i,j+1,k,QV) = interface_prim_left[EIGVEC_V] - plus_sum[EIGVEC_V] + 0.5_rt*dt*mhd_source[EIGVEC_V];
      prim_left(i,j+1,k,QW) = interface_prim_left[EIGVEC_W] - plus_sum[EIGVEC_W] + 0.5_rt*dt*mhd_source[EIGVEC_W];
      prim_left(i,j+1,k,QPRES) = amrex::max(pressure_cutoff,
                                        interface_prim_left[EIGVEC_PRESSURE] - plus_sum[EIGVEC_PRESSURE] + 0.5_rt*dt*mhd_source[EIGVEC_PRESSURE]);

      prim_left(i,j+1,k,QBFIELDX) = interface_prim_left[EIGVEC_BPERP1] - plus_sum[EIGVEC_BPERP1] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP1];
      prim_left(i,j+1,k,QBFIELDY) = By(i,j+1,k);
      prim_left(i,j+1,k,QBFIELDZ) = interface_prim_left[EIGVEC_BPERP2] - plus_sum[EIGVEC_BPERP2] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP2];

    } else {
      prim_left(i,j,k+1,QRHO) = amrex::max(small_dens,
                                       interface_prim_left[EIGVEC_RHO] - plus_sum[EIGVEC_RHO] + 0.5_rt*dt*mhd_source[EIGVEC_RHO]);
      prim_left(i,j,k+1,QU) = interface_prim_left[EIGVEC_U] - plus_sum[EIGVEC_U] + 0.5_rt*dt*mhd_source[EIGVEC_U];
      prim_left(i,j,k+1,QV) = interface_prim_left[EIGVEC_V] - plus_sum[EIGVEC_V] + 0.5_rt*dt*mhd_source[EIGVEC_V];
      prim_left(i,j,k+1,QW) = interface_prim_left[EIGVEC_W] - plus_sum[EIGVEC_W] + 0.5_rt*dt*mhd_source[EIGVEC_W];
      prim_left(i,j,k+1,QPRES) = amrex::max(pressure_cutoff,
                                        interface_prim_left[EIGVEC_PRESSURE] - plus_sum[EIGVEC_PRESSURE] + 0.5_rt*dt*mhd_source[EIGVEC_PRESSURE]);

      prim_left(i,j,k+1,QBFIELDX) = interface_prim_left[EIGVEC_BPERP1] - plus_sum[EIGVEC_BPERP1] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP1];
      prim_left(i,j,k+1,QBFIELDY) = interface_prim_left[EIGVEC_BPERP2] - plus_sum[EIGVEC_BPERP2] + 0.5_rt*dt*mhd_source[EIGVEC_BPERP2];
      prim_left(i,j,k+1,QBFIELDZ) = Bz(i,j,k+1);
    }


    for (int n = 0; n < NumSpec; n++) {

      int v = QFS+n;

      Real Ips;
      Real Ims;

      Real un;

      if (idir == 0) {
        un = prim_arr(i,j,k,QU);

      } else if (idir == 1) {
        un = prim_arr(i,j,k,QV);

      } else {
        un = prim_arr(i,j,k,QW);
      }

      gather_stencil(prim_arr, idir, i, j, k, v, s);
      parabolic_reconstruct(s, flat, sm, sp);
      integrate_parabolic_profile_single(sm, sp, s[i0], un, dtdx, Ips, Ims);

      if (idir == 0) {
        prim_left(i+1,j,k,QFS+n) = Ips;
      } else if (idir == 1) {
        prim_left(i,j+1,k,QFS+n) = Ips;
      } else {
        prim_left(i,j,k+1,QFS+n) = Ips;
      }
      prim_right(i,j,k,QFS+n) = Ims;
    }

    eos_t eos_state;

    if (idir == 0) {
      eos_state.rho = prim_left(i+1,j,k,QRHO);
      eos_state.p = prim_left(i+1,j,k,QPRES);
      eos_state.T = prim_arr(i,j,k,QTEMP);
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
      eos_state.T = prim_arr(i,j,k,QTEMP);
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
      eos_state.T = prim_arr(i,j,k,QTEMP);
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
