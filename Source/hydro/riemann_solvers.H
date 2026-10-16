#ifndef riemann_solvers_H
#define riemann_solvers_H

#include <HydroSim_util.H>
#include <riemann_type.H>
#include <two_shock_riemann.H>
#ifdef HYBRID_MOMENTUM
#include <hybrid.H>
#endif
#ifdef RADIATION
#include <Radiation.H>
#include <rad_util.H>
#endif

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
flux_from_interface_state(const int i, const int j, const int k, const int idir,
               const GeometryData& geomdata,
               const RiemannVars& q_riemann,
               Array4<Real> const& F,
#ifdef RADIATION
               Array4<Real> const& rF,
#endif
               Array4<Real> const& qface, const FaceComponents qface_comps) {

    // given a primitive state, compute the flux in direction idir
    //

    int im1, im2, im3;

    auto coord = geomdata.Coord();
    auto mom_check = momentum_flux_has_pressure(idir, idir, coord);

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
    F(i,j,k,URHO) = q_riemann.rho * q_riemann.vel.un;

    F(i,j,k,im1) = F(i,j,k,URHO) * q_riemann.vel.un;
    F(i,j,k,im2) = F(i,j,k,URHO) * q_riemann.vel.ut;
    F(i,j,k,im3) = F(i,j,k,URHO) * q_riemann.vel.utt;

#ifndef RADIATION
    if (mom_check) {
      F(i,j,k,im1) += q_riemann.p;
    }

    Real rhoetot = q_riemann.rhoe + 0.5_rt * q_riemann.rho *
        (q_riemann.vel.un * q_riemann.vel.un +
         q_riemann.vel.ut * q_riemann.vel.ut +
         q_riemann.vel.utt * q_riemann.vel.utt);

    F(i,j,k,UEDEN) = q_riemann.vel.un * (rhoetot + q_riemann.p);
    F(i,j,k,UEINT) = q_riemann.vel.un * q_riemann.rhoe;
#else
    if (mom_check) {
      F(i,j,k,im1) += q_riemann.rad.p_g;
    }

    Real rhoetot = q_riemann.rad.rhoe_g + 0.5_rt * q_riemann.rho *
        (q_riemann.vel.un * q_riemann.vel.un +
         q_riemann.vel.ut * q_riemann.vel.ut +
         q_riemann.vel.utt * q_riemann.vel.utt);

    F(i,j,k,UEDEN) = q_riemann.vel.un * (rhoetot + q_riemann.rad.p_g);
    F(i,j,k,UEINT) = q_riemann.vel.un * q_riemann.rad.rhoe_g;
#endif

    F(i,j,k,UTEMP) = 0.0;
#ifdef SHOCK_VAR
    F(i,j,k,USHOCK) = 0.0;
#endif

#ifdef NSE_NET
    F(i,j,k,UMU_P) = 0.0;
    F(i,j,k,UMU_N) = 0.0;
#endif

#ifdef RADIATION
    if (radiation::fspace_advection_type == 1) {
        for (int g = 0; g < NGROUPS; g++) {
            Real eddf = Edd_factor(q_riemann.rad.lam[g]);
            Real f1 = 0.5e0_rt*(1.0_rt-eddf);
            rF(i,j,k,g) = (1.0_rt + f1) * q_riemann.rad.er[g] * q_riemann.vel.un;
        }
    } else {
        // type 2
        for (int g = 0; g < NGROUPS; g++) {
            rF(i,j,k,g) = q_riemann.rad.er[g] * q_riemann.vel.un;
        }
    }
#endif

#ifdef HYBRID_MOMENTUM
    // the hybrid routine uses the Godunov indices, not the full NQ state
    GpuArray<Real, NIFC> qface_zone;
    qface_zone[IFCRHO] = q_riemann.rho;
    if (idir == 0) {
        qface_zone[IFCU] = q_riemann.vel.un;
        qface_zone[IFCV] = q_riemann.vel.ut;
        qface_zone[IFCW] = q_riemann.vel.utt;
    } else if (idir == 1) {
        qface_zone[IFCV] = q_riemann.vel.un;
        qface_zone[IFCU] = q_riemann.vel.ut;
        qface_zone[IFCW] = q_riemann.vel.utt;
    } else {
        qface_zone[IFCW] = q_riemann.vel.un;
        qface_zone[IFCU] = q_riemann.vel.ut;
        qface_zone[IFCV] = q_riemann.vel.utt;
    }
    qface_zone[IFCPRES] = q_riemann.p;
#ifdef RADIATION
    for (int g = 0; g < NGROUPS; g++) {
        qface_zone[IFCFLXLIM+g] = q_riemann.rad.lam[g];
        qface_zone[IFCERAD+g] = q_riemann.rad.er[g];
    }
#endif
    GpuArray<Real, NUM_STATE> F_zone;
    for (int n = 0; n < NUM_STATE; n++) {
        F_zone[n] = F(i,j,k,n);
    }
    compute_hybrid_flux(qface_zone, geomdata, idir, i, j, k, F_zone);
    for (int n = 0; n < NUM_STATE; n++) {
        F(i,j,k,n) = F_zone[n];
    }
#endif

    // now we store the interface state in the full 4-d array.  We
    // either store just the NIFC subset or the whole NQ hydro state,
    // depending on the value of qface_comps

    // note: we don't handle the passives here

    if (qface_comps == FaceComponents::Full) {

        // this is the primitive variable state on the interface

        qface(i,j,k,QRHO) = q_riemann.rho;
        if (idir == 0) {
            qface(i,j,k,QU) = q_riemann.vel.un;
            qface(i,j,k,QV) = q_riemann.vel.ut;
            qface(i,j,k,QW) = q_riemann.vel.utt;
        } else if (!(idir == 1)) {
            qface(i,j,k,QW) = q_riemann.vel.un;
            qface(i,j,k,QU) = q_riemann.vel.ut;
            qface(i,j,k,QV) = q_riemann.vel.utt;
        } else {
            qface(i,j,k,QV) = q_riemann.vel.un;
            qface(i,j,k,QU) = q_riemann.vel.ut;
            qface(i,j,k,QW) = q_riemann.vel.utt;
        }

        qface(i,j,k,QTEMP) = 0.0_rt;

#ifdef RADIATION
        qface(i,j,k,QPRES) = q_riemann.rad.p_g;
        qface(i,j,k,QRHOEI) = q_riemann.rad.rhoe_g;
        qface(i,j,k,QPTOTAL) = q_riemann.p;
        qface(i,j,k,QRHOEITOT) = q_riemann.rhoe;
        for (int g = 0; g < NGROUPS; g++) {
            qface(i,j,k,QERAD) = q_riemann.rad.er[g];
        }
#else
        qface(i,j,k,QPRES) = q_riemann.p;
        qface(i,j,k,QRHOEI) = q_riemann.rhoe;
#endif

    } else {

        // this is the reduced "Godunov" state on the interface,
        // with NIFC components

#ifdef HYBRID_MOMENTUM
        qface(i,j,k,IFCRHO) = q_riemann.rho;
#endif
        if (idir == 0) {
            qface(i,j,k,IFCU) = q_riemann.vel.un;
            qface(i,j,k,IFCV) = q_riemann.vel.ut;
            qface(i,j,k,IFCW) = q_riemann.vel.utt;
        } else if (idir == 1) {
            qface(i,j,k,IFCV) = q_riemann.vel.un;
            qface(i,j,k,IFCU) = q_riemann.vel.ut;
            qface(i,j,k,IFCW) = q_riemann.vel.utt;
        } else {
            qface(i,j,k,IFCW) = q_riemann.vel.un;
            qface(i,j,k,IFCU) = q_riemann.vel.ut;
            qface(i,j,k,IFCV) = q_riemann.vel.utt;
        }
#ifndef RADIATION
        qface(i,j,k,IFCPRES) = q_riemann.p;
#else
        qface(i,j,k,IFCPRES) = q_riemann.rad.p_g;
        for (int g = 0; g < NGROUPS; g++) {
            qface(i,j,k,IFCFLXLIM+g) = q_riemann.rad.lam[g];
            qface(i,j,k,IFCERAD+g) = q_riemann.rad.er[g];
        }
#endif

    }

}



AMREX_GPU_HOST_DEVICE AMREX_INLINE
RiemannVars
solve_interface_state(const int i, const int j, const int k, const int idir,
              Array4<Real> const& qm,
              Array4<Real> const& qp,
              Array4<Real const> const& qaux_arr,
              DomainEdges const& edges,
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


  auto [ql, qr, riemann_aux] = load_input_states(i, j, k, idir,
                                          qm, qp, qaux_arr);

  // deal with hard walls
  riemann_aux.boundary_scale = 1.0_rt;

  if (idir == 0) {
      if ((i == edges.lo[0] && edges.wall_lo) ||
          (i == edges.hi[0]+1 && edges.wall_hi)) {
          riemann_aux.boundary_scale = 0.0_rt;
      }

  } else if (idir == 1) {
      if ((j == edges.lo[1] && edges.wall_lo) ||
          (j == edges.hi[1]+1 && edges.wall_hi)) {
          riemann_aux.boundary_scale = 0.0_rt;
      }

  } else {
      if ((k == edges.lo[2] && edges.wall_lo) ||
          (k == edges.hi[2]+1 && edges.wall_hi)) {
          riemann_aux.boundary_scale = 0.0_rt;
      }
  }


  // Solve Riemann problem
  RiemannVars q_riemann{};

  if (riemann_solver == 0) {
      TwoShock::riemann_two_shock_approx(ql, qr, riemann_aux, q_riemann);

  } else if (riemann_solver == 1) {
#ifndef RADIATION
      TwoShock::riemann_two_shock_iterative(ql, qr, riemann_aux, q_riemann);
#endif

#ifndef AMREX_USE_GPU
  } else {
      amrex::Error("ERROR: invalid value of riemann_solver");
#endif
  }

  return q_riemann;
}


#endif
