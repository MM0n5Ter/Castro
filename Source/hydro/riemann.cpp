#include <HydroSim.H>

#include <riemann_solvers.H>
#include <HLL_solvers.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#include <cmath>

#include <eos.H>
using namespace amrex;

namespace {

// primitive variables of a single zone, as the HLL solver takes them
using PrimitiveState = GpuArray<Real, NQ>;

// the zone on the low side of the idir interface of zone (i,j,k)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Dim3 lower_neighbor (int i, int j, int k, const int idir)
{
    if (idir == 0) {
        return Dim3{i-1, j, k};
    } else if (idir == 1) {
        return Dim3{i, j-1, k};
    } else {
        return Dim3{i, j, k-1};
    }
}

}

void
HydroSim::riemann_flux_and_state(const Box& bx,
                            Array4<Real> const& qm,
                            Array4<Real> const& qp,
                            InterfaceOutput const& out,
                            Array4<Real const> const& qaux_arr,
                            Array4<Real const> const& shk,
                            const int idir, const FaceComponents qface_comps) {

    // note: bx is not necessarily the limits of the valid (no ghost
    // cells) domain, but could be hi+1 in some dimensions.  We rely on
    // the caller to specify the interfaces over which to solve the
    // Riemann problems

    // Solve Riemann problem to get the fluxes

    // qface_comps determines what is put into qface.  if
    // qface_comps is Full, we put all NQ variables into the
    // qface.  if qface_comps is Reduced, we only store the NIFC
    // needed elsewhere in the algorithm.


    // Note: because the NQ variables do not include lambda and the
    // NIFC do, we only support qface_comps = Reduced for
    // Radiation

#ifdef RADIATION
    if (qface_comps == FaceComponents::Full) {
        amrex::Error("cannot store full interface state with radiation");
    }
#endif

    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();

    // do we want to force the flux to zero at the boundary?
    auto is_wall = [] (int bc) {
        return bc == amrex::PhysBCType::symmetry ||
               bc == amrex::PhysBCType::slipwall ||
               bc == amrex::PhysBCType::noslipwall;
    };

    const DomainEdges edges{geom.Domain().loVect3d(), geom.Domain().hiVect3d(),
                            is_wall(lo_bc[idir]), is_wall(hi_bc[idir])};

    auto coord = geom.Coord();

    GeometryData geomdata = geom.data();

    const bool do_ppm_temp_fix = ppm_temp_fix == 2;
    const Real temp_guess = T_guess;

    const int solver_type = riemann_solver;
    const bool do_hybrid_correction = hybrid_riemann;

    Array4<Real> const& flx = out.flux;
    Array4<Real> const& qface = out.qface;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {


        if (solver_type == 0 || solver_type == 1) {
            // approximate state Riemann solvers

            // first find the interface state on the current interface

            const RiemannState qint = riemann_state(i, j, k, idir,
                                                    qm, qp, qaux_arr,
                                                    edges,
                                                    do_ppm_temp_fix, temp_guess);

            // now use the interface state to compute and store the flux

            compute_flux_q(i, j, k, idir,
                           geomdata,
                           qint, flx,
#ifdef RADIATION
                           out.rad_flux,
#endif
                           qface, qface_comps);

            // now do the passives -- we didn't include them in qint, qface, or flux above

            // the passives are always just upwinded, so we do that here
            // regardless of the solver

            Real sgnm = std::copysign(1.0_rt, qint.vel.un);
            if (qint.vel.un == 0.0_rt) {
                sgnm = 0.0_rt;
            }

            Real fp = 0.5_rt*(1.0_rt + sgnm);
            Real fm = 0.5_rt*(1.0_rt - sgnm);

            for (int ipassive = 0; ipassive < npassive; ipassive++) {
                int nqp = qpassmap(ipassive);
                int n  = upassmap(ipassive);

                Real X_int = fp * qm(i,j,k,nqp) + fm * qp(i,j,k,nqp);

                flx(i,j,k,n) = flx(i,j,k,URHO) * X_int;

                if (qface_comps == FaceComponents::Full) {
                    qface(i,j,k,nqp) = X_int;
                }
            }

        } else if (solver_type == 2) {
            // HLLC
            HLL::HLLC(i, j, k, idir,
                      qm, qp,
                      qaux_arr,
                      flx,
                      qface, qface_comps,
                      geomdata,
                      edges);

#ifndef AMREX_USE_GPU
        } else {
            amrex::Error("ERROR: invalid value of riemann_solver");
#endif
        }

        // correct the fluxes using an HLL scheme if we are in a shock
        // and doing the hybrid approach

        if (!do_hybrid_correction) {
            return;
        }

        const Dim3 left = lower_neighbor(i, j, k, idir);

        int is_shock = static_cast<int>(shk(left) + shk(i,j,k));

        if (is_shock < 1) {
            return;
        }

        Real cl = qaux_arr(left, QC);
        Real cr = qaux_arr(i,j,k,QC);

        PrimitiveState ql_zone;
        PrimitiveState qr_zone;
        GpuArray<Real, NUM_STATE> flx_zone;

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
    });

}
