#include <HydroSim.H>

#include <riemann_solvers.H>
#include <HLL_solvers.H>


#include <cmath>

#include <eos.H>
using namespace amrex;

///
/// Replace the flux on an interface next to a zone flagged as a shock
/// by the HLL flux computed from the same left and right states
///
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
hll_shock_flux(const int i, const int j, const int k, const int idir, const int coord,
               Array4<Real> const& qm,
               Array4<Real> const& qp,
               Array4<Real const> const& qaux_arr,
               Array4<Real const> const& shk,
               Array4<Real> const& flx) {

    int is_shock = 0;

    if (idir == 0) {
        is_shock = static_cast<int>(shk(i-1,j,k) + shk(i,j,k));
    } else if (idir == 1) {
        is_shock = static_cast<int>(shk(i,j-1,k) + shk(i,j,k));
    } else {
        is_shock = static_cast<int>(shk(i,j,k-1) + shk(i,j,k));
    }

    if (is_shock < 1) {
        return;
    }

    Real cl;
    Real cr;
    if (idir == 0) {
        cl = qaux_arr(i-1,j,k,QC);
        cr = qaux_arr(i,j,k,QC);
    } else if (idir == 1) {
        cl = qaux_arr(i,j-1,k,QC);
        cr = qaux_arr(i,j,k,QC);
    } else {
        cl = qaux_arr(i,j,k-1,QC);
        cr = qaux_arr(i,j,k,QC);
    }

    Real ql_zone[NQ];
    Real qr_zone[NQ];
    Real flx_zone[NUM_STATE];

    for (int n = 0; n < NQ; n++) {
        ql_zone[n] = qm(i,j,k,n);
        qr_zone[n] = qp(i,j,k,n);
    }

    // pass in the current flux -- the
    // HLL solver will overwrite this
    // if necessary
    for (int n = 0; n < NUM_STATE; n++) {
        flx_zone[n] = flx(i,j,k,n);
    }

    HLL::HLL(ql_zone, qr_zone, cl, cr,
             idir, coord,
             flx_zone);

    for (int n = 0; n < NUM_STATE; n++) {
        flx(i,j,k,n) = flx_zone[n];
    }
}

void
HydroSim::riemann_flux_and_state(const Box& bx,
                            Array4<Real> const& qm,
                            Array4<Real> const& qp,
                            Array4<Real> const& flx,
                            Array4<Real> const& qface,
                            Array4<Real const> const& qaux_arr,
                            Array4<Real const> const& shk,
                            const int idir, const bool store_full_state) {

    // note: bx is not necessarily the limits of the valid (no ghost
    // cells) domain, but could be hi+1 in some dimensions.  We rely on
    // the caller to specify the interfaces over which to solve the
    // Riemann problems

    // Solve Riemann problem to get the fluxes

    // store_full_state determines what is put into qface.  if
    // store_full_state is True, we put all NQ variables into the
    // qface.  if store_full_state is False, we only store the NIFC
    // needed elsewhere in the algorithm.


    // Note: because the NQ variables do not include lambda and the
    // NIFC do, we only support store_full_state = false for
    // Radiation


    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();

    // do we want to force the flux to zero at the boundary?
    const bool special_bnd_lo = (lo_bc[idir] == amrex::PhysBCType::symmetry ||
                                 lo_bc[idir] == amrex::PhysBCType::slipwall ||
                                 lo_bc[idir] == amrex::PhysBCType::noslipwall);
    const bool special_bnd_hi = (hi_bc[idir] == amrex::PhysBCType::symmetry ||
                                 hi_bc[idir] == amrex::PhysBCType::slipwall ||
                                 hi_bc[idir] == amrex::PhysBCType::noslipwall);

    auto coord = geom.Coord();

    GeometryData geomdata = geom.data();

    const auto domlo = geom.Domain().loVect3d();
    const auto domhi = geom.Domain().hiVect3d();

    const bool do_ppm_temp_fix = ppm_temp_fix == 2;
    const Real temp_guess = T_guess;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {


        if (riemann_solver == 0 || riemann_solver == 1) {
            // approximate state Riemann solvers

            // first find the interface state on the current interface

            RiemannState qint{};

            riemann_state(i, j, k, idir,
                          qm, qp, qaux_arr,
                          qint,
                          special_bnd_lo, special_bnd_hi,
                          domlo, domhi,
                          do_ppm_temp_fix, temp_guess);

            // now use the interface state to compute and store the flux

            compute_flux_q(i, j, k, idir,
                           geomdata,
                           qint, flx,
                           qface, store_full_state);

            // now do the passives -- we didn't include them in qint, qface, or flux above

            // the passives are always just upwinded, so we do that here
            // regardless of the solver

            Real sgnm = std::copysign(1.0_rt, qint.un);
            if (qint.un == 0.0_rt) {
                sgnm = 0.0_rt;
            }

            Real fp = 0.5_rt*(1.0_rt + sgnm);
            Real fm = 0.5_rt*(1.0_rt - sgnm);

            for (int ipassive = 0; ipassive < npassive; ipassive++) {
                int nqp = qpassmap(ipassive);
                int n  = upassmap(ipassive);

                Real X_int = fp * qm(i,j,k,nqp) + fm * qp(i,j,k,nqp);

                flx(i,j,k,n) = flx(i,j,k,URHO) * X_int;

                if (store_full_state) {
                    qface(i,j,k,nqp) = X_int;
                }
            }

        } else if (riemann_solver == 2) {
            // HLLC
            HLL::HLLC(i, j, k, idir,
                      qm, qp,
                      qaux_arr,
                      flx,
                      qface, store_full_state,
                      geomdata,
                      special_bnd_lo, special_bnd_hi,
                      domlo, domhi);

#ifndef AMREX_USE_GPU
        } else {
            amrex::Error("ERROR: invalid value of riemann_solver");
#endif
        }

        if (hybrid_riemann == 1) {
            // correct the fluxes using an HLL scheme if we are in a shock
            // and doing the hybrid approach
            hll_shock_flux(i, j, k, idir, coord,
                           qm, qp, qaux_arr, shk, flx);
        }
    });

}
