#ifndef interface_solvers_H
#define interface_solvers_H

#include <HydroSim_helpers.H>
#include <solver_types.H>
#include <shock_pair_solvers.H>
#ifdef HYBRID_MOMENTUM
#include <hybrid.H>
#endif
#ifdef RADIATION
#include <Radiation.H>
#include <radiation_helpers.H>
#endif

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
flux_from_interface_state(const int i, const int j, const int k, const int idir,
               const GeometryData& geomdata,
               const InterfaceState& interface_state,
               Array4<Real> const& F,
#ifdef RADIATION
               Array4<Real> const& rF,
#endif
               Array4<Real> const& godunov_state, const bool keep_full_state) {


    int im1, im2, im3;

    auto coord = geomdata.Coord();
    auto has_pressure_flux = momentum_flux_has_pressure(idir, idir, coord);

    if (idir == 0) {
        im1 = UMX;
        im2 = UMY;
        im3 = UMZ;

    } else if (idir == 1) {
        im1 = UMY;
        im2 = UMX;
        im3 = UMZ;

    } else {
        im1 = UMZ;
        im2 = UMX;
        im3 = UMY;
    }

    F(i,j,k,URHO) = interface_state.rho * interface_state.un;

    F(i,j,k,im1) = F(i,j,k,URHO) * interface_state.un;
    F(i,j,k,im2) = F(i,j,k,URHO) * interface_state.ut;
    F(i,j,k,im3) = F(i,j,k,URHO) * interface_state.utt;

#ifndef RADIATION
    if (has_pressure_flux) {
      F(i,j,k,im1) += interface_state.p;
    }

    Real rho_total_energy = interface_state.rhoe + 0.5_rt * interface_state.rho *
        (interface_state.un * interface_state.un +
         interface_state.ut * interface_state.ut +
         interface_state.utt * interface_state.utt);

    F(i,j,k,UEDEN) = interface_state.un * (rho_total_energy + interface_state.p);
    F(i,j,k,UEINT) = interface_state.un * interface_state.rhoe;
#else
    if (has_pressure_flux) {
      F(i,j,k,im1) += interface_state.p_g;
    }

    Real rho_total_energy = interface_state.energy_density_gas + 0.5_rt * interface_state.rho *
        (interface_state.un * interface_state.un +
         interface_state.ut * interface_state.ut +
         interface_state.utt * interface_state.utt);

    F(i,j,k,UEDEN) = interface_state.un * (rho_total_energy + interface_state.p_g);
    F(i,j,k,UEINT) = interface_state.un * interface_state.energy_density_gas;
#endif

    F(i,j,k,UTEMP) = 0.0;
#ifdef SHOCK_VAR
    F(i,j,k,UFLAG_SHOCK) = 0.0;
#endif

#ifdef NSE_NET
    F(i,j,k,UPROTON_MU) = 0.0;
    F(i,j,k,UNEUTRON_MU) = 0.0;
#endif

#ifdef RADIATION
    if (radiation::frequency_advection_mode == 1) {
        for (int g = 0; g < NGROUPS; g++) {
            Real eddington_factor = Eddington_factor(interface_state.lam[g]);
            Real f1 = 0.5e0_rt*(1.0_rt-eddington_factor);
            rF(i,j,k,g) = (1.0_rt + f1) * interface_state.er[g] * interface_state.un;
        }
    } else {
        for (int g = 0; g < NGROUPS; g++) {
            rF(i,j,k,g) = interface_state.er[g] * interface_state.un;
        }
    }
#endif

#ifdef HYBRID_MOMENTUM
    GpuArray<Real, NGDSTATE> godunov_zone;
    godunov_zone[GDDENSITY] = interface_state.rho;
    if (idir == 0) {
        godunov_zone[GDU] = interface_state.un;
        godunov_zone[GDV] = interface_state.ut;
        godunov_zone[GDW] = interface_state.utt;
    } else if (idir == 1) {
        godunov_zone[GDV] = interface_state.un;
        godunov_zone[GDU] = interface_state.ut;
        godunov_zone[GDW] = interface_state.utt;
    } else {
        godunov_zone[GDW] = interface_state.un;
        godunov_zone[GDU] = interface_state.ut;
        godunov_zone[GDV] = interface_state.utt;
    }
    godunov_zone[GDPRESSURE] = interface_state.p;
#ifdef RADIATION
    for (int g = 0; g < NGROUPS; g++) {
        godunov_zone[GDFLUXLIM+g] = interface_state.lam[g];
        godunov_zone[GDRADENG+g] = interface_state.er[g];
    }
#endif
    GpuArray<Real, NUM_STATE> ZoneFlux;
    for (int n = 0; n < NUM_STATE; n++) {
        ZoneFlux[n] = F(i,j,k,n);
    }
    hybrid_flux_from_state(godunov_zone, geomdata, idir, i, j, k, ZoneFlux);
    for (int n = 0; n < NUM_STATE; n++) {
        F(i,j,k,n) = ZoneFlux[n];
    }
#endif



    if (keep_full_state) {


        godunov_state(i,j,k,QRHO) = interface_state.rho;
        if (idir == 0) {
            godunov_state(i,j,k,QU) = interface_state.un;
            godunov_state(i,j,k,QV) = interface_state.ut;
            godunov_state(i,j,k,QW) = interface_state.utt;
        } else if (idir == 1) {
            godunov_state(i,j,k,QV) = interface_state.un;
            godunov_state(i,j,k,QU) = interface_state.ut;
            godunov_state(i,j,k,QW) = interface_state.utt;
        } else {
            godunov_state(i,j,k,QW) = interface_state.un;
            godunov_state(i,j,k,QU) = interface_state.ut;
            godunov_state(i,j,k,QV) = interface_state.utt;
        }

        godunov_state(i,j,k,QTEMP) = 0.0_rt;

#ifdef RADIATION
        godunov_state(i,j,k,QPRES) = interface_state.p_g;
        godunov_state(i,j,k,QEINTDENS) = interface_state.energy_density_gas;
        godunov_state(i,j,k,QPTOTAL) = interface_state.p;
        godunov_state(i,j,k,QETOTDENS) = interface_state.rhoe;
        for (int g = 0; g < NGROUPS; g++) {
            godunov_state(i,j,k,QRADENG) = interface_state.er[g];
        }
#else
        godunov_state(i,j,k,QPRES) = interface_state.p;
        godunov_state(i,j,k,QEINTDENS) = interface_state.rhoe;
#endif

    } else {


#ifdef HYBRID_MOMENTUM
        godunov_state(i,j,k,GDDENSITY) = interface_state.rho;
#endif
        if (idir == 0) {
            godunov_state(i,j,k,GDU) = interface_state.un;
            godunov_state(i,j,k,GDV) = interface_state.ut;
            godunov_state(i,j,k,GDW) = interface_state.utt;
        } else if (idir == 1) {
            godunov_state(i,j,k,GDV) = interface_state.un;
            godunov_state(i,j,k,GDU) = interface_state.ut;
            godunov_state(i,j,k,GDW) = interface_state.utt;
        } else {
            godunov_state(i,j,k,GDW) = interface_state.un;
            godunov_state(i,j,k,GDU) = interface_state.ut;
            godunov_state(i,j,k,GDV) = interface_state.utt;
        }
#ifndef RADIATION
        godunov_state(i,j,k,GDPRESSURE) = interface_state.p;
#else
        godunov_state(i,j,k,GDPRESSURE) = interface_state.p_g;
        for (int g = 0; g < NGROUPS; g++) {
            godunov_state(i,j,k,GDFLUXLIM+g) = interface_state.lam[g];
            godunov_state(i,j,k,GDRADENG+g) = interface_state.er[g];
        }
#endif

    }

}



AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
solve_interface_state(const int i, const int j, const int k, const int idir,
              Array4<Real> const& qm,
              Array4<Real> const& qp,
              Array4<Real const> const& qthermo_arr,
              InterfaceState& interface_state,
              const bool lo_no_flux_wall, const bool hi_no_flux_wall,
              GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi,
              const bool use_parabolic_temp_fix, const Real temperature_guess) {




  if (use_parabolic_temp_fix) {


      eos_rep_t eos_state;

      eos_state.T = temperature_guess;

      eos_state.rho = qm(i,j,k,QRHO);
      eos_state.e = qm(i,j,k,QEINTDENS)/qm(i,j,k,QRHO);
      for (int n = 0; n < NumSpec; n++) {
          eos_state.xn[n] = qm(i,j,k,QFS+n);
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
          eos_state.aux[n] = qm(i,j,k,QFX+n);
      }
#endif

      eos(eos_input_re, eos_state);

      qm(i,j,k,QEINTDENS) = eos_state.e * eos_state.rho;
      qm(i,j,k,QPRES) = eos_state.p;

      eos_state.rho = qp(i,j,k,QRHO);
      eos_state.e = qp(i,j,k,QEINTDENS)/qp(i,j,k,QRHO);
      for (int n = 0; n < NumSpec; n++) {
          eos_state.xn[n] = qp(i,j,k,QFS+n);
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
          eos_state.aux[n] = qp(i,j,k,QFX+n);
      }
#endif

      eos(eos_input_re, eos_state);

      qp(i,j,k,QEINTDENS) = eos_state.e * eos_state.rho;
      qp(i,j,k,QPRES) = eos_state.p;
  }


  InterfaceState ql;
  InterfaceState qr;
  InterfaceAux interface_aux;


  fetch_interface_inputs(i, j, k, idir,
                    qm, qp, qthermo_arr,
                    ql, qr, interface_aux);

  interface_aux.boundary_factor = 1.0_rt;

  if (idir == 0) {
      if ((i == domlo[0] && lo_no_flux_wall) ||
          (i == domhi[0]+1 && hi_no_flux_wall)) {
          interface_aux.boundary_factor = 0.0_rt;
      }

  } else if (idir == 1) {
      if ((j == domlo[1] && lo_no_flux_wall) ||
          (j == domhi[1]+1 && hi_no_flux_wall)) {
          interface_aux.boundary_factor = 0.0_rt;
      }

  } else {
      if ((k == domlo[2] && lo_no_flux_wall) ||
          (k == domhi[2]+1 && hi_no_flux_wall)) {
          interface_aux.boundary_factor = 0.0_rt;
      }
  }


  if (riemann_method == 0) {

      ShockPairSolver::shock_pair_direct(ql, qr, interface_aux, interface_state);

  } else if (riemann_method == 1) {

#ifndef RADIATION
      ShockPairSolver::shock_pair_iterate(ql, qr, interface_aux, interface_state);
#endif

#ifndef AMREX_USE_GPU
  } else {
      amrex::Error("ERROR: invalid value of riemann_method");
#endif
  }


}


#endif
