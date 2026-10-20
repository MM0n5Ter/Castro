#include <HydroSim.H>

#include <HydroSim_helpers.H>

#include <riemann_states.H>

#include <iterative_riemann.H>

#ifdef HYBRID_MOMENTUM
#include <hybrid_momentum.H>
#endif

#ifdef RADIATION
#include <radiation_helpers.H>
#endif
#include <approximate_riemann.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#include <cmath>

#include <eos.H>
using namespace amrex;

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


    constexpr int mom_comp[3][3] = {{UMX, UMY, UMZ}, {UMY, UMX, UMZ}, {UMZ, UMX, UMY}};
    constexpr int prim_vel[3][3] = {{QU, QV, QW}, {QV, QU, QW}, {QW, QU, QV}};
    constexpr int godunov_vel[3][3] = {{GDU, GDV, GDW}, {GDV, GDU, GDW}, {GDW, GDU, GDV}};

    const int im1 = mom_comp[idir][0];
    const int im2 = mom_comp[idir][1];
    const int im3 = mom_comp[idir][2];

    auto coord = geomdata.Coord();
    auto has_pressure_flux = momentum_flux_has_pressure(idir, idir, coord);

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
    godunov_zone[godunov_vel[idir][0]] = interface_state.un;
    godunov_zone[godunov_vel[idir][1]] = interface_state.ut;
    godunov_zone[godunov_vel[idir][2]] = interface_state.utt;
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
        godunov_state(i,j,k,prim_vel[idir][0]) = interface_state.un;
        godunov_state(i,j,k,prim_vel[idir][1]) = interface_state.ut;
        godunov_state(i,j,k,prim_vel[idir][2]) = interface_state.utt;

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
        godunov_state(i,j,k,godunov_vel[idir][0]) = interface_state.un;
        godunov_state(i,j,k,godunov_vel[idir][1]) = interface_state.ut;
        godunov_state(i,j,k,godunov_vel[idir][2]) = interface_state.utt;
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
              NoFluxWalls const& walls,
              const bool use_parabolic_temp_fix, const Real temperature_guess) {




  if (use_parabolic_temp_fix) {


      eos_rep_t eos_state;

      eos_state.T = temperature_guess;

      auto reload_from_temperature = [&] (Array4<Real> const& edge) {
          eos_state.rho = edge(i,j,k,QRHO);
          eos_state.e = edge(i,j,k,QEINTDENS)/edge(i,j,k,QRHO);
          for (int n = 0; n < NumSpec; n++) {
              eos_state.xn[n] = edge(i,j,k,QFS+n);
          }
#if NAUX_NET > 0
          for (int n = 0; n < NumAux; n++) {
              eos_state.aux[n] = edge(i,j,k,QFX+n);
          }
#endif

          eos(eos_input_re, eos_state);

          edge(i,j,k,QEINTDENS) = eos_state.e * eos_state.rho;
          edge(i,j,k,QPRES) = eos_state.p;
      };

      reload_from_temperature(qm);

      reload_from_temperature(qp);
  }


  InterfaceState ql;
  InterfaceState qr;
  InterfaceAux interface_aux;


  fetch_interface_inputs(i, j, k, idir,
                    qm, qp, qthermo_arr,
                    ql, qr, interface_aux);

  interface_aux.boundary_factor = 1.0_rt;

  const amrex::GpuArray<int, 3> zone{i, j, k};
  if ((zone[idir] == walls.domlo[idir] && walls.lo) ||
      (zone[idir] == walls.domhi[idir]+1 && walls.hi)) {
      interface_aux.boundary_factor = 0.0_rt;
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

void
HydroSim::compute_godunov_flux(const Box& bx,
                            Array4<Real> const& qm,
                            Array4<Real> const& qp,
                            Array4<Real> const& flx,
#ifdef RADIATION
                            Array4<Real> const& radiative_flux_out,
#endif
                            Array4<Real> const& godunov_state,
                            Array4<Real const> const& qthermo_arr,
                            Array4<Real const> const& shk,
                            const int idir, const bool keep_full_state) {






#ifdef RADIATION
    if (keep_full_state == true) {
        amrex::Error("cannot store full interface state with radiation");
    }
#endif

    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();

    const bool lo_no_flux_wall = (lo_bc[idir] == amrex::PhysBCType::symmetry ||
                                 lo_bc[idir] == amrex::PhysBCType::slipwall ||
                                 lo_bc[idir] == amrex::PhysBCType::noslipwall);
    const bool hi_no_flux_wall = (hi_bc[idir] == amrex::PhysBCType::symmetry ||
                                 hi_bc[idir] == amrex::PhysBCType::slipwall ||
                                 hi_bc[idir] == amrex::PhysBCType::noslipwall);

    auto coord = geom.Coord();

    GeometryData geomdata = geom.data();

    const NoFluxWalls walls{lo_no_flux_wall, hi_no_flux_wall,
                            geom.Domain().loVect3d(), geom.Domain().hiVect3d()};

    const bool use_parabolic_temp_fix = ppm_temp_mode == 2;
    const Real temperature_guess = eos_temp_guess;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {


        if (riemann_method == 0 || riemann_method == 1) {


            InterfaceState interface_state{};

            solve_interface_state(i, j, k, idir,
                          qm, qp, qthermo_arr,
                          interface_state,
                          walls,
                          use_parabolic_temp_fix, temperature_guess);


            flux_from_interface_state(i, j, k, idir,
                           geomdata,
                           interface_state, flx,
#ifdef RADIATION
                           radiative_flux_out,
#endif
                           godunov_state, keep_full_state);



            Real sgnm = std::copysign(1.0_rt, interface_state.un);
            if (interface_state.un == 0.0_rt) {
                sgnm = 0.0_rt;
            }

            Real fp = 0.5_rt*(1.0_rt + sgnm);
            Real fm = 0.5_rt*(1.0_rt - sgnm);

            for (int passive_index = 0; passive_index < npassive; passive_index++) {
                int nqp = passive_prim_index(passive_index);
                int n  = passive_cons_index(passive_index);

                Real passive_interface_value = fp * qm(i,j,k,nqp) + fm * qp(i,j,k,nqp);

                flx(i,j,k,n) = flx(i,j,k,URHO) * passive_interface_value;

                if (keep_full_state) {
                    godunov_state(i,j,k,nqp) = passive_interface_value;
                }
            }

        } else if (riemann_method == 2) {
            HLL::CONTACT_FLUX(i, j, k, idir,
                      qm, qp,
                      qthermo_arr,
                      flx,
                      godunov_state, keep_full_state,
                      geomdata,
                      walls);

#ifndef AMREX_USE_GPU
        } else {
            amrex::Error("ERROR: invalid value of riemann_method");
#endif
        }

        if (shock_flux_switch != 1) {
            return;
        }

        const int il = i - (idir == 0);
        const int jl = j - (idir == 1);
        const int kl = k - (idir == 2);

        int shock_count = static_cast<int>(shk(il,jl,kl) + shk(i,j,k));

        if (shock_count < 1) {
            return;
        }

        Real cl = qthermo_arr(il,jl,kl,QC);
        Real cr = qthermo_arr(i,j,k,QC);

        amrex::GpuArray<Real, NQ> zone_left_state;
        amrex::GpuArray<Real, NQ> zone_right_state;
        amrex::GpuArray<Real, NUM_STATE> zone_flux_vec;

        for (int n = 0; n < NQ; n++) {
            zone_left_state[n] = qm(i,j,k,n);
            zone_right_state[n] = qp(i,j,k,n);
        }

        for (int n = 0; n < NUM_STATE; n++) {
            zone_flux_vec[n] = flx(i,j,k,n);
        }

        HLL::HLL(zone_left_state, zone_right_state, cl, cr,
                 idir, coord,
                 zone_flux_vec);

        for (int n = 0; n < NUM_STATE; n++) {
            flx(i,j,k,n) = zone_flux_vec[n];
        }
    });

}
