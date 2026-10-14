#ifndef riemann_solvers_H
#define riemann_solvers_H

#include <HydroSim_util.H>
#include <riemann_type.H>
#include <riemann_2shock_solvers.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
compute_flux_q(const int i, const int j, const int k, const int idir,
               const GeometryData& geomdata,
               const RiemannState& qint,
               Array4<Real> const& F,
               Array4<Real> const& qface, const bool store_full_state) {

    // given a primitive state, compute the flux in direction idir
    //

    int im1, im2, im3;

    auto coord = geomdata.Coord();
    auto mom_check = mom_flux_has_p(idir, idir, coord);

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

    // Compute fluxes, order as conserved state (not q)
    F(i,j,k,URHO) = qint.rho * qint.un;

    F(i,j,k,im1) = F(i,j,k,URHO) * qint.un;
    F(i,j,k,im2) = F(i,j,k,URHO) * qint.ut;
    F(i,j,k,im3) = F(i,j,k,URHO) * qint.utt;

    if (mom_check) {
      F(i,j,k,im1) += qint.p;
    }

    Real rhoetot = qint.rhoe + 0.5_rt * qint.rho *
        (qint.un * qint.un +
         qint.ut * qint.ut +
         qint.utt * qint.utt);

    F(i,j,k,UEDEN) = qint.un * (rhoetot + qint.p);
    F(i,j,k,UEINT) = qint.un * qint.rhoe;

    F(i,j,k,UTEMP) = 0.0;
#ifdef SHOCK_VAR
    F(i,j,k,USHOCK) = 0.0;
#endif




    // now we store the interface state in the full 4-d array.  We
    // either store just the NIFC subset or the whole NQ hydro state,
    // depending on the value of store_full_state

    // note: we don't handle the passives here

    if (store_full_state) {

        // this is the primitive variable state on the interface

        qface(i,j,k,QRHO) = qint.rho;
        if (idir == 0) {
            qface(i,j,k,QU) = qint.un;
            qface(i,j,k,QV) = qint.ut;
            qface(i,j,k,QW) = qint.utt;
        } else if (idir == 1) {
            qface(i,j,k,QV) = qint.un;
            qface(i,j,k,QU) = qint.ut;
            qface(i,j,k,QW) = qint.utt;
        } else {
            qface(i,j,k,QW) = qint.un;
            qface(i,j,k,QU) = qint.ut;
            qface(i,j,k,QV) = qint.utt;
        }

        qface(i,j,k,QTEMP) = 0.0_rt;

        qface(i,j,k,QPRES) = qint.p;
        qface(i,j,k,QRHOEI) = qint.rhoe;

    } else {

        // this is the reduced "Godunov" state on the interface,
        // with NIFC components

        if (idir == 0) {
            qface(i,j,k,IFCU) = qint.un;
            qface(i,j,k,IFCV) = qint.ut;
            qface(i,j,k,IFCW) = qint.utt;
        } else if (idir == 1) {
            qface(i,j,k,IFCV) = qint.un;
            qface(i,j,k,IFCU) = qint.ut;
            qface(i,j,k,IFCW) = qint.utt;
        } else {
            qface(i,j,k,IFCW) = qint.un;
            qface(i,j,k,IFCU) = qint.ut;
            qface(i,j,k,IFCV) = qint.utt;
        }
        qface(i,j,k,IFCPRES) = qint.p;

    }

}



///
/// The normal velocity, and with it the flux, vanishes on a domain
/// face that is a symmetry plane or a wall.  Return 0 for an interface
/// on such a face and 1 everywhere else.
///
AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real
wall_flux_factor(const int i, const int j, const int k, const int idir,
                 const bool special_bnd_lo, const bool special_bnd_hi,
                 GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {

  const int n = idir == 0 ? i : (idir == 1 ? j : k);

  if (n == domlo[idir] && special_bnd_lo) {
      return 0.0_rt;
  }

  if (n == domhi[idir]+1 && special_bnd_hi) {
      return 0.0_rt;
  }

  return 1.0_rt;
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
riemann_state(const int i, const int j, const int k, const int idir,
              Array4<Real> const& qm,
              Array4<Real> const& qp,
              Array4<Real const> const& qaux_arr,
              RiemannState& qint,
              const bool special_bnd_lo, const bool special_bnd_hi,
              GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi,
              const bool do_ppm_temp_fix, const Real temp_guess) {

  // just compute the hydrodynamic state on the interfaces
  // don't compute the fluxes

  // note: bx is not necessarily the limits of the valid (no ghost
  // cells) domain, but could be hi+1 in some dimensions.  We rely on
  // the caller to specify the interfaces over which to solve the
  // Riemann problems


  if (do_ppm_temp_fix) {
      // recompute the thermodynamics on the interface to make it
      // all consistent

      // we want to take the edge states of rho, e, and X, and get
      // new values for p on the edges that are
      // thermodynamically consistent.

      eos_rep_t eos_state;

      // this is an initial guess for iterations, since we
      // can't be certain what temp is on interfaces
      eos_state.T = temp_guess;

      // minus state
      eos_state.rho = qm(i,j,k,QRHO);
      eos_state.e = qm(i,j,k,QRHOEI)/qm(i,j,k,QRHO);
      for (int n = 0; n < NumSpec; n++) {
          eos_state.xn[n] = qm(i,j,k,QSPEC+n);
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
          eos_state.aux[n] = qm(i,j,k,QAUXV+n);
      }
#endif

      eos(eos_input_re, eos_state);

      qm(i,j,k,QRHOEI) = eos_state.e * eos_state.rho;
      qm(i,j,k,QPRES) = eos_state.p;

      // plus state
      eos_state.rho = qp(i,j,k,QRHO);
      eos_state.e = qp(i,j,k,QRHOEI)/qp(i,j,k,QRHO);
      for (int n = 0; n < NumSpec; n++) {
          eos_state.xn[n] = qp(i,j,k,QSPEC+n);
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
          eos_state.aux[n] = qp(i,j,k,QAUXV+n);
      }
#endif

      eos(eos_input_re, eos_state);

      qp(i,j,k,QRHOEI) = eos_state.e * eos_state.rho;
      qp(i,j,k,QPRES) = eos_state.p;
  }


  RiemannState ql;
  RiemannState qr;
  RiemannAux raux;


  load_input_states(i, j, k, idir,
                    qm, qp, qaux_arr,
                    ql, qr, raux);

  // deal with hard walls
  raux.bnd_fac = wall_flux_factor(i, j, k, idir,
                                  special_bnd_lo, special_bnd_hi,
                                  domlo, domhi);


  // Solve Riemann problem
  if (riemann_solver == 0) {
      // Colella, Glaz, & Ferguson solver

      TwoShock::riemann_cgf_approx(ql, qr, raux, qint);

  } else if (riemann_solver == 1) {
      // Colella & Glaz solver

      TwoShock::riemann_colella_glaz(ql, qr, raux, qint);

#ifndef AMREX_USE_GPU
  } else {
      amrex::Error("ERROR: invalid value of riemann_solver");
#endif
  }


}


#endif
