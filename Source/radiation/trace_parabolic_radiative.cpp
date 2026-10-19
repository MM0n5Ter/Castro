#include <HydroSim.H>
#include <HydroSim_helpers.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#include <cmath>

#include <ppm.H>
#include <flatten.H>

using namespace amrex;

void
HydroSim::trace_parabolic_radiative(const Box& bx,
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

  // These routines do the characteristic tracing under the parabolic
  // profiles in each zone to the edge / half-time.

  // To allow for easy integration of radiation, we adopt the
  // following conventions:
  //
  // rho : mass density
  // u, v, w : velocities
  // p : gas (hydro) pressure
  // total_pressure : total pressure (note for pure hydro, this is
  //        just the gas pressure)
  // energy_density_gas : gas specific internal energy
  // cgas : sound speed for just the gas contribution
  // cc : total sound speed (including radiation)
  // h_g : gas specific enthalpy / cc**2
  // gas_adiabatic_index : the gas Gamma_1
  //
  // for pure hydro, we will only consider:
  //   rho, u, v, w, total_pressure, energy_density_gas, cc, h_g

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

  GpuArray<int, NQSOURCE> trace_source_flags;


  for (int n = 0; n < NQSOURCE; n++) {
    trace_source_flags[n] = 0;

    for (int k = lo[2]-2*dg2; k <= hi[2]+2*dg2; k++) {
      for (int j = lo[1]-2*dg1; j <= hi[1]+2*dg1; j++) {
        for (int i = lo[0]-2; i <= hi[0]+2; i++) {
          if (std::abs(prim_source(i,j,k,n)) > 0.0_rt) {
            trace_source_flags[n] = 1;
            break;
          }
        }
        if (trace_source_flags[n] == 1) break;
      }
      if (trace_source_flags[n] == 1) break;
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

  int QUN, QUT, QTRANSV2;

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

  // Trace to left and right edges using upwind PPM
  AMREX_PARALLEL_FOR_3D(bx, i, j, k,
  {

   Real lam0[NGROUPS];
   Real lamp[NGROUPS];
   Real limiter_minus[NGROUPS];

    for (int g = 0; g < NGROUPS; g++) {
      lam0[g] = qthermo_arr(i,j,k,QFLUXLIM+g);
      lamp[g] = qthermo_arr(i,j,k,QFLUXLIM+g);
      limiter_minus[g] = qthermo_arr(i,j,k,QFLUXLIM+g);
    }

    Real rho = prim_arr(i,j,k,QRHO);

    // gas_sound_speed_sq is the gas soundspeed **2
    // cc is the total soundspeed **2 (gas + radiation)
    Real gas_sound_speed_sq = qthermo_arr(i,j,k,QCG)*qthermo_arr(i,j,k,QCG);
    Real cc = qthermo_arr(i,j,k,QC);
    Real csq = cc*cc;

    Real un = prim_arr(i,j,k,QUN);

    Real p = prim_arr(i,j,k,QPRES);
    Real energy_density_gas = prim_arr(i,j,k,QEINTDENS);
    Real h_g = ( (p+energy_density_gas)/rho)/csq;

    Real gas_adiabatic_index = qthermo_arr(i,j,k,QGAMMA1G);

    Real total_pressure = prim_arr(i,j,k,QPTOTAL);

    Real er[NGROUPS];
    Real hr[NGROUPS];
    for (int g = 0; g < NGROUPS; g++) {
      er[g] = prim_arr(i,j,k,QRADENG+g);
      hr[g] = (lam0[g] + 1.0_rt)*er[g]/rho;
    }

    // do the parabolic reconstruction and compute the
    // integrals under the characteristic waves
    Real s[5];

    Real flat = 1.0;

    if (hydrosim::first_order_scheme) {
        flat = 0.0;
    }
    else if (hydrosim::use_shock_flatten) {
        flat = hydro::flatten(i, j, k, prim_arr, QPRES);

#ifdef RADIATION
        flat *= hydro::flatten(i, j, k, prim_arr, QPTOTAL);

        if (radiation::radiation_flatten_cutoff > 0.0) {
            if ( prim_arr(i-1,j,k,QU) + prim_arr(i,j-1,k,QV) + prim_arr(i,j,k-1,QW) >
                 prim_arr(i+1,j,k,QU) + prim_arr(i,j+1,k,QV) + prim_arr(i,j,k+1,QW) ) {

                if (prim_arr(i,j,k,QPRES) < radiation::radiation_flatten_cutoff * prim_arr(i,j,k,QPTOTAL)) {
                    flat = 0.0;
                }
            }
        }
#endif
    }

    Real sm;
    Real sp;

    Real Ip[NQ][3];
    Real Im[NQ][3];


    // do the non-passives first

    for (int n = 0; n < NQPRIMBASE; n++) {
      if (n == QTEMP) continue;

      gather_stencil(prim_arr, idir, i, j, k, n, s);
      parabolic_reconstruct(s, flat, sm, sp);
      integrate_parabolic_profile(sm, sp, s[i0], un, cc, dtdx, Ip[n], Im[n]);

    }

    // now the passives

    for (int passive_index = 0; passive_index < npassive; ++passive_index) {

      const int n = passive_prim_index(passive_index);
      const int nc = passive_cons_index(passive_index);

      fill_passive_stencil(field_arr, inv_density_arr, idir, i, j, k, nc, s);
      parabolic_reconstruct(s, flat, sm, sp);
      integrate_parabolic_profile(sm, sp, s[i0], un, cc, dtdx, Ip[n], Im[n]);

    }


    // source terms
    Real TraceP_src[NQSOURCE][3];
    Real TraceM_src[NQSOURCE][3];

    for (int n = 0; n < NQSOURCE; n++) {

      // do we even need to trace (non-zero source?)
#ifndef AMREX_USE_GPU
      int trace_enabled = trace_source_flags[n];
#else
      int trace_enabled = 0;
      if (idir == 0) {
        for (int b = i-2; b <= i+2; b++) {
          if (std::abs(prim_source(b,j,k,n)) > 0.0_rt) {
            trace_enabled = 1;
            break;
          }
        }
      } else if (idir == 1) {
        for (int b = j-2; b <= j+2; b++) {
          if (std::abs(prim_source(i,b,k,n)) > 0.0_rt) {
            trace_enabled = 1;
            break;
          }
        }
      } else {
        for (int b = k-2; b <= k+2; b++) {
          if (std::abs(prim_source(i,j,b,n)) > 0.0_rt) {
            trace_enabled = 1;
            break;
          }
        }
      }
#endif

      if (trace_enabled) {

          gather_stencil(prim_source, idir, i, j, k, n, s);
          parabolic_reconstruct(s, flat, sm, sp);
          integrate_parabolic_profile(sm, sp, s[i0], un, cc, dtdx, TraceP_src[n], TraceM_src[n]);

      } else {
        TraceP_src[n][0] = 0.0_rt;
        TraceP_src[n][1] = 0.0_rt;
        TraceP_src[n][2] = 0.0_rt;

        TraceM_src[n][0] = 0.0_rt;
        TraceM_src[n][1] = 0.0_rt;
        TraceM_src[n][2] = 0.0_rt;
      }

    }


    // do the passives separately

    // the passive stuff is the same regardless of the tracing

    for (int passive_index = 0; passive_index < npassive; passive_index++) {

      int n = passive_prim_index(passive_index);

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

        qp(i,j,k,n) = Im[n][1];
      }

      // Minus state on face i+1
      if (idir == 0 && i <= vhi[0]) {
        qm(i+1,j,k,n) = Ip[n][1];
      } else if (idir == 1 && j <= vhi[1]) {
        qm(i,j+1,k,n) = Ip[n][1];
      } else if (idir == 2 && k <= vhi[2]) {
        qm(i,j,k+1,n) = Ip[n][1];
      }
    }

    // plus state on face i

    if ((idir == 0 && i >= vlo[0]) ||
        (idir == 1 && j >= vlo[1]) ||
        (idir == 2 && k >= vlo[2])) {

      // Set the reference state
      // This will be the fastest moving state to the left
      Real ref_density = Im[QRHO][0];
      Real ref_normal_velocity = Im[QUN][0];

      Real ref_pressure = Im[QPRES][0];
      Real ref_energy_density_gas = Im[QEINTDENS][0];

      Real ref_total_pressure = Im[QPTOTAL][0];
      Real ref_rad_energy[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        ref_rad_energy[g] = Im[QRADENG+g][0];
      }

      ref_density = amrex::max(ref_density, ldensity_floor);
      ref_pressure = amrex::max(ref_pressure, local_pressure_floor);

      // *m are the jumps carried by u-c
      // *p are the jumps carried by u+c

      // Note: for the transverse velocities, the jump is carried
      //       only by the u wave (the contact)

      // we also add the sources here so they participate in the tracing
      Real dum = ref_normal_velocity - Im[QUN][0] - hdt*TraceM_src[QUN][0];
      Real pressure_jump_minus = ref_total_pressure - Im[QPTOTAL][0] - hdt*TraceM_src[QPRES][0];

      Real drho = ref_density - Im[QRHO][1] - hdt*TraceM_src[QRHO][1];
      Real pressure_jump = ref_total_pressure - Im[QPTOTAL][1] - hdt*TraceM_src[QPRES][1];
      Real energy_density_jump_gas = ref_energy_density_gas - Im[QEINTDENS][1] - hdt*TraceM_src[QEINTDENS][1];

      Real der[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        der[g] = ref_rad_energy[g] - Im[QRADENG+g][1];
      }

      Real dup = ref_normal_velocity - Im[QUN][2] - hdt*TraceM_src[QUN][2];
      Real pressure_jump_plus = ref_total_pressure - Im[QPTOTAL][2] - hdt*TraceM_src[QPRES][2];

      // {rho, u, p, (rho e)} eigensystem

      // These are analogous to the beta's from the original PPM
      // (except we work with rho instead of tau).  This is
      // simply (l . dq), where dq = qref - I(q)

      Real acoustic_amp_minus = 0.5_rt*(pressure_jump_minus/(rho*cc) - dum)*rho/cc;
      Real acoustic_amp_plus = 0.5_rt*(pressure_jump_plus/(rho*cc) + dup)*rho/cc;
      Real entropy_amp_density = drho - pressure_jump/csq;
      Real entropy_amp_energy_gas = energy_density_jump_gas - pressure_jump*h_g;

      Real rad_wave_amplitude[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        rad_wave_amplitude[g] = der[g] - pressure_jump/csq*hr[g];
      }

      acoustic_amp_minus = un-cc > 0.0_rt ? 0.0_rt : -acoustic_amp_minus;
      acoustic_amp_plus = un+cc > 0.0_rt ? 0.0_rt : -acoustic_amp_plus;
      entropy_amp_density = un > 0.0_rt ? 0.0_rt : -entropy_amp_density;
      entropy_amp_energy_gas = un > 0.0_rt ? 0.0_rt : -entropy_amp_energy_gas;

      for (int g = 0; g < NGROUPS; g++) {
        rad_wave_amplitude[g] = un > 0.0_rt ? 0.0_rt : -rad_wave_amplitude[g];
      }

      // The final interface states are just
      // q_s = q_ref - sum(l . dq) r
      // note that the a{mpz}right as defined above have the minus already
      qp(i,j,k,QRHO) = amrex::max(ldensity_floor, ref_density + acoustic_amp_plus + acoustic_amp_minus + entropy_amp_density);
      qp(i,j,k,QUN) = ref_normal_velocity + (acoustic_amp_plus - acoustic_amp_minus)*cc/rho;
      qp(i,j,k,QEINTDENS) = ref_energy_density_gas + (acoustic_amp_plus + acoustic_amp_minus)*h_g*csq + entropy_amp_energy_gas;

      qp(i,j,k,QPRES) = ref_pressure + (acoustic_amp_plus + acoustic_amp_minus)*gas_sound_speed_sq;
      for (int g = 0; g < NGROUPS; g++) {
        qp(i,j,k,QPRES) += -lamp[g]*rad_wave_amplitude[g];
      }
      qp(i,j,k,QPRES) = amrex::max(local_pressure_floor, qp(i,j,k,QPRES));

      qp(i,j,k,QPTOTAL) = ref_total_pressure + (acoustic_amp_plus + acoustic_amp_minus)*csq;
      qp(i,j,k,QETOTDENS) = qp(i,j,k,QEINTDENS);

      Real rad_prim_value;
      for (int g = 0; g < NGROUPS; g++) {
        rad_prim_value = ref_rad_energy[g] + (acoustic_amp_plus + acoustic_amp_minus)*hr[g] + rad_wave_amplitude[g];
        qp(i,j,k,QRADENG+g) = rad_prim_value;
        qp(i,j,k,QETOTDENS) += rad_prim_value;
      }

      for (int g = 0; g < NGROUPS; g++) {
        if (qp(i,j,k,QRADENG+g) < 0.0_rt) {
          Real clipped_rad_energy = -qp(i,j,k,QRADENG+g);
          qp(i,j,k,QRADENG+g) = 0.0_rt;
          qp(i,j,k,QPTOTAL) += lamp[g] * clipped_rad_energy;
          qp(i,j,k,QETOTDENS) += clipped_rad_energy;
        }
      }

      // Transverse velocities -- there's no projection here, so
      // we don't need a reference state.  We only care about
      // the state traced under the middle wave

      // Recall that I already takes the limit of the parabola
      // in the event that the wave is not moving toward the
      // interface
      qp(i,j,k,QUT) = Im[QUT][1] + hdt*TraceM_src[QUT][1];
      qp(i,j,k,QTRANSV2) = Im[QTRANSV2][1] + hdt*TraceM_src[QTRANSV2][1];

    }

    // minus state on face i + 1

    if ((idir == 0 && i <= vhi[0]) ||
        (idir == 1 && j <= vhi[1]) ||
        (idir == 2 && k <= vhi[2])) {

      // Set the reference state
      // This will be the fastest moving state to the right
      Real ref_density = Ip[QRHO][2];
      Real ref_normal_velocity = Ip[QUN][2];

      Real ref_pressure = Ip[QPRES][2];
      Real ref_energy_density_gas = Ip[QEINTDENS][2];

      Real ref_total_pressure = Ip[QPTOTAL][2];
      Real ref_rad_energy[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        ref_rad_energy[g] = Ip[QRADENG+g][2];
      }

      ref_density = amrex::max(ref_density, ldensity_floor);
      ref_pressure = amrex::max(ref_pressure, local_pressure_floor);

      // *m are the jumps carried by u-c
      // *p are the jumps carried by u+c

      Real dum = ref_normal_velocity - Ip[QUN][0] - hdt*TraceP_src[QUN][0];
      Real pressure_jump_minus = ref_total_pressure - Ip[QPTOTAL][0] - hdt*TraceP_src[QPRES][0];

      Real drho = ref_density - Ip[QRHO][1] - hdt*TraceP_src[QRHO][1];
      Real pressure_jump = ref_total_pressure - Ip[QPTOTAL][1] - hdt*TraceP_src[QPRES][1];
      Real energy_density_jump_gas = ref_energy_density_gas - Ip[QEINTDENS][1] - hdt*TraceP_src[QEINTDENS][1];

      Real der[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        der[g]  = ref_rad_energy[g]  - Ip[QRADENG+g][1];
      }

      Real dup = ref_normal_velocity - Ip[QUN][2] - hdt*TraceP_src[QUN][2];
      Real pressure_jump_plus = ref_total_pressure - Ip[QPTOTAL][2] - hdt*TraceP_src[QPRES][2];

      // {rho, u, p, (rho e)} eigensystem

      // These are analogous to the beta's from the original PPM
      // (except we work with rho instead of tau).  This is
      // simply (l . dq), where dq = qref - I(q)

      Real acoustic_amp_minus = 0.5_rt*(pressure_jump_minus/(rho*cc) - dum)*rho/cc;
      Real acoustic_amp_plus = 0.5_rt*(pressure_jump_plus/(rho*cc) + dup)*rho/cc;
      Real entropy_amp_density = drho - pressure_jump/csq;
      Real entropy_amp_energy_gas = energy_density_jump_gas - pressure_jump*h_g;

      Real rad_wave_amplitude[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        rad_wave_amplitude[g] = der[g] - pressure_jump/csq*hr[g];
      }

      acoustic_amp_minus = un-cc > 0.0_rt ? -acoustic_amp_minus : 0.0_rt;
      acoustic_amp_plus = un+cc > 0.0_rt ? -acoustic_amp_plus : 0.0_rt;
      entropy_amp_density = un > 0.0_rt ? -entropy_amp_density : 0.0_rt;
      entropy_amp_energy_gas = un > 0.0_rt ? -entropy_amp_energy_gas : 0.0_rt;

      for (int g = 0; g < NGROUPS; g++) {
        rad_wave_amplitude[g] = un > 0.0_rt ? -rad_wave_amplitude[g] : 0.0_rt;
      }

      // The final interface states are just
      // q_s = q_ref - sum (l . dq) r
      // note that the a{mpz}left as defined above have the minus already

      if (idir == 0) {
        qm(i+1,j,k,QRHO) = amrex::max(ldensity_floor, ref_density + acoustic_amp_plus + acoustic_amp_minus + entropy_amp_density);
        qm(i+1,j,k,QUN) = ref_normal_velocity + (acoustic_amp_plus - acoustic_amp_minus)*cc/rho;
        qm(i+1,j,k,QEINTDENS) = ref_energy_density_gas + (acoustic_amp_plus + acoustic_amp_minus)*h_g*csq + entropy_amp_energy_gas;

        qm(i+1,j,k,QPRES) = ref_pressure + (acoustic_amp_plus + acoustic_amp_minus)*gas_sound_speed_sq;
        for (int g = 0; g < NGROUPS; g++) {
          qm(i+1,j,k,QPRES) += -limiter_minus[g]*rad_wave_amplitude[g];
        }

        qm(i+1,j,k,QPTOTAL) = ref_total_pressure + (acoustic_amp_plus + acoustic_amp_minus)*csq;
        qm(i+1,j,k,QETOTDENS) = qm(i+1,j,k,QEINTDENS);

        Real rad_prim_value;
        for (int g = 0; g < NGROUPS; g++) {
          rad_prim_value = ref_rad_energy[g] + (acoustic_amp_plus + acoustic_amp_minus)*hr[g] + rad_wave_amplitude[g];
          qm(i+1,j,k,QRADENG+g) = rad_prim_value;
          qm(i+1,j,k,QETOTDENS) += rad_prim_value;
        }

        for (int g = 0; g < NGROUPS; g++) {
          if (qm(i+1,j,k,QRADENG+g) < 0.0_rt) {
            Real clipped_rad_energy = -qm(i+1,j,k,QRADENG+g);
            qm(i+1,j,k,QRADENG+g) = 0.0_rt;
            qm(i+1,j,k,QPTOTAL) += limiter_minus[g] * clipped_rad_energy;
            qm(i+1,j,k,QETOTDENS) += clipped_rad_energy;
          }
        }

        // transverse velocities
        qm(i+1,j,k,QUT) = Ip[QUT][1] + hdt*TraceP_src[QUT][1];
        qm(i+1,j,k,QTRANSV2) = Ip[QTRANSV2][1] + hdt*TraceP_src[QTRANSV2][1];

      } else if (idir == 1) {
        qm(i,j+1,k,QRHO) = amrex::max(ldensity_floor, ref_density + acoustic_amp_plus + acoustic_amp_minus + entropy_amp_density);
        qm(i,j+1,k,QUN) = ref_normal_velocity + (acoustic_amp_plus - acoustic_amp_minus)*cc/rho;
        qm(i,j+1,k,QEINTDENS) = ref_energy_density_gas + (acoustic_amp_plus + acoustic_amp_minus)*h_g*csq + entropy_amp_energy_gas;

        qm(i,j+1,k,QPRES) = ref_pressure + (acoustic_amp_plus + acoustic_amp_minus)*gas_sound_speed_sq;
        for (int g = 0; g < NGROUPS; g++) {
          qm(i,j+1,k,QPRES) += -limiter_minus[g]*rad_wave_amplitude[g];
        }

        qm(i,j+1,k,QPTOTAL) = ref_total_pressure + (acoustic_amp_plus + acoustic_amp_minus)*csq;
        qm(i,j+1,k,QETOTDENS) = qm(i,j+1,k,QEINTDENS);

        Real rad_prim_value;
        for (int g = 0; g < NGROUPS; g++) {
          rad_prim_value = ref_rad_energy[g] + (acoustic_amp_plus + acoustic_amp_minus)*hr[g] + rad_wave_amplitude[g];
          qm(i,j+1,k,QRADENG+g) = rad_prim_value;
          qm(i,j+1,k,QETOTDENS) += rad_prim_value;
        }

        for (int g = 0; g < NGROUPS; g++) {
          rad_prim_value = ref_rad_energy[g] + (acoustic_amp_plus + acoustic_amp_minus)*hr[g] + rad_wave_amplitude[g];
          qm(i,j+1,k,QRADENG+g) = rad_prim_value;
          qm(i,j+1,k,QETOTDENS) += rad_prim_value;
        }

        for (int g = 0; g < NGROUPS; g++) {
          if (qm(i,j+1,k,QRADENG+g) < 0.0_rt) {
            Real clipped_rad_energy = -qm(i,j+1,k,QRADENG+g);
            qm(i,j+1,k,QRADENG+g) = 0.0_rt;
            qm(i,j+1,k,QPTOTAL) += limiter_minus[g] * clipped_rad_energy;
            qm(i,j+1,k,QETOTDENS) += clipped_rad_energy;
          }
        }

        // transverse velocities
        qm(i,j+1,k,QUT) = Ip[QUT][1] + hdt*TraceP_src[QUT][1];
        qm(i,j+1,k,QTRANSV2) = Ip[QTRANSV2][1] + hdt*TraceP_src[QTRANSV2][1];


      } else {

        qm(i,j,k+1,QRHO) = amrex::max(ldensity_floor, ref_density + acoustic_amp_plus + acoustic_amp_minus + entropy_amp_density);
        qm(i,j,k+1,QUN) = ref_normal_velocity + (acoustic_amp_plus - acoustic_amp_minus)*cc/rho;
        qm(i,j,k+1,QEINTDENS) = ref_energy_density_gas + (acoustic_amp_plus + acoustic_amp_minus)*h_g*csq + entropy_amp_energy_gas;

        qm(i,j,k+1,QPRES) = ref_pressure + (acoustic_amp_plus + acoustic_amp_minus)*gas_sound_speed_sq;
        for (int g = 0; g < NGROUPS; g++) {
          qm(i,j,k+1,QPRES) += -limiter_minus[g]*rad_wave_amplitude[g];
        }

        qm(i,j,k+1,QPTOTAL) = ref_total_pressure + (acoustic_amp_plus + acoustic_amp_minus)*csq;
        qm(i,j,k+1,QETOTDENS) = qm(i,j,k+1,QEINTDENS);

        Real rad_prim_value;
        for (int g = 0; g < NGROUPS; g++) {
          rad_prim_value = ref_rad_energy[g] + (acoustic_amp_plus + acoustic_amp_minus)*hr[g] + rad_wave_amplitude[g];
          qm(i,j,k+1,QRADENG+g) = rad_prim_value;
          qm(i,j,k+1,QETOTDENS) += rad_prim_value;
        }

        for (int g = 0; g < NGROUPS; g++) {
          rad_prim_value = ref_rad_energy[g] + (acoustic_amp_plus + acoustic_amp_minus)*hr[g] + rad_wave_amplitude[g];
          qm(i,j,k+1,QRADENG+g) = rad_prim_value;
          qm(i,j,k+1,QETOTDENS) += rad_prim_value;
        }

        for (int g = 0; g < NGROUPS; g++) {
          if (qm(i,j,k+1,QRADENG+g) < 0.0_rt) {
            Real clipped_rad_energy = -qm(i,j,k+1,QRADENG+g);
            qm(i,j,k+1,QRADENG+g) = 0.0_rt;
            qm(i,j,k+1,QPTOTAL) += limiter_minus[g] * clipped_rad_energy;
            qm(i,j,k+1,QETOTDENS) += clipped_rad_energy;
          }
        }

        // transverse velocities
        qm(i,j,k+1,QUT) = Ip[QUT][1] + hdt*TraceP_src[QUT][1];
        qm(i,j,k+1,QTRANSV2) = Ip[QTRANSV2][1] + hdt*TraceP_src[QTRANSV2][1];

      }

    }

    // geometry source terms

#if (AMREX_SPACEDIM < 3)
    // these only apply for x states (idir = 0)
    if (idir == 0 && dloga(i,j,k) != 0.0_rt) {
      Real courant_local = dt/dx[0]*(cc+std::abs(un));
      Real eta = (1.0_rt - courant_local)/(cc*dt*std::abs(dloga(i,j,k)));
      Real limited_dlog_area = amrex::min(eta, 1.0_rt)*dloga(i,j,k);
      Real source_density = -0.5_rt*dt*rho*limited_dlog_area*un;
      Real source_pressure = source_density*gas_sound_speed_sq;
      Real source = source_pressure*h_g;
      Real rad_source_terms[NGROUPS];
      for (int g = 0; g < NGROUPS; g++) {
        rad_source_terms[g] = -0.5_rt*dt*limited_dlog_area*un*(lam0[g] + 1.0_rt)*er[g];
      }

      if (i <= vhi[0]) {
        qm(i+1,j,k,QRHO) = qm(i+1,j,k,QRHO) + source_density;
        qm(i+1,j,k,QRHO) = amrex::max(qm(i+1,j,k,QRHO), ldensity_floor);
        qm(i+1,j,k,QPRES ) = qm(i+1,j,k,QPRES ) + source_pressure;
        qm(i+1,j,k,QEINTDENS) = qm(i+1,j,k,QEINTDENS) + source;
        for (int g = 0; g < NGROUPS; g++) {
          qm(i+1,j,k,QRADENG+g) += rad_source_terms[g];
        }
        qm(i+1,j,k,QPTOTAL) = qm(i+1,j,k,QPTOTAL) + source_pressure;
        qm(i+1,j,k,QETOTDENS) = qm(i+1,j,k,QEINTDENS);
        for (int g = 0; g < NGROUPS; g++) {
          qm(i+1,j,k,QPTOTAL) += limiter_minus[g]*rad_source_terms[g];
          qm(i+1,j,k,QETOTDENS) += qm(i+1,j,k,QRADENG+g);
          }
      }

      if (i >= vlo[0]) {
        qp(i,j,k,QRHO) = qp(i,j,k,QRHO) + source_density;
        qp(i,j,k,QRHO) = amrex::max(qp(i,j,k,QRHO), ldensity_floor);
        qp(i,j,k,QPRES) = qp(i,j,k,QPRES) + source_pressure;
        qp(i,j,k,QEINTDENS) = qp(i,j,k,QEINTDENS) + source;
        for (int g = 0; g < NGROUPS; g++) {
          qp(i,j,k,QRADENG+g) += rad_source_terms[g];
        }
        qp(i,j,k,QPTOTAL) = qp(i,j,k,QPTOTAL) + source_pressure;
        qp(i,j,k,QETOTDENS) = qp(i,j,k,QEINTDENS);
        for (int g = 0; g < NGROUPS; g++) {
          qp(i,j,k,QPTOTAL) += lamp[g]*rad_source_terms[g];
          qp(i,j,k,QETOTDENS) += qp(i,j,k,QRADENG+g);
        }
      }
    }
#endif

  });
}
