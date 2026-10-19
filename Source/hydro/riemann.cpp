#include <HydroSim.H>

#include <interface_solvers.H>
#include <two_wave_solvers.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#include <cmath>

#include <eos.H>
using namespace amrex;

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

    const auto domlo = geom.Domain().loVect3d();
    const auto domhi = geom.Domain().hiVect3d();

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
                          lo_no_flux_wall, hi_no_flux_wall,
                          domlo, domhi,
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
                      lo_no_flux_wall, hi_no_flux_wall,
                      domlo, domhi);

#ifndef AMREX_USE_GPU
        } else {
            amrex::Error("ERROR: invalid value of riemann_method");
#endif
        }

        if (shock_flux_switch == 1) {

            int shock_count = 0;

            if (idir == 0) {
                shock_count = static_cast<int>(shk(i-1,j,k) + shk(i,j,k));
            } else if (idir == 1) {
                shock_count = static_cast<int>(shk(i,j-1,k) + shk(i,j,k));
            } else {
                shock_count = static_cast<int>(shk(i,j,k-1) + shk(i,j,k));
            }

            if (shock_count >= 1) {

                Real cl;
                Real cr;
                if (idir == 0) {
                    cl = qthermo_arr(i-1,j,k,QC);
                    cr = qthermo_arr(i,j,k,QC);
                } else if (idir == 1) {
                    cl = qthermo_arr(i,j-1,k,QC);
                    cr = qthermo_arr(i,j,k,QC);
                } else {
                    cl = qthermo_arr(i,j,k-1,QC);
                    cr = qthermo_arr(i,j,k,QC);
                }

                Real zone_left_state[NQ];
                Real zone_right_state[NQ];
                Real zone_flux_vec[NUM_STATE];

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
            }
        }
    });

}
