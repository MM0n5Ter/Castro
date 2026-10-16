#include <HydroSim.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef GRAVITY
#include <Gravity.H>
#endif

#ifdef HYBRID_MOMENTUM
#include <hybrid.H>
#endif

#include <ppm.H>
#include <slope.H>
#include <advection_util.H>

using namespace amrex;
using namespace reconstruction;

void
HydroSim::plm_face_states_mol(const Box& bx,
                            const int idir,
                            Array4<Real const> const& q_arr,
                            Array4<Real const> const& flatcoef_arr,
                            Array4<Real const> const& src_q_arr,
                            Array4<Real> const& dq,
                            Array4<Real> const& qm,
                            Array4<Real> const& qp) {

  const auto dx = geom.CellSizeArray();

  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  bool lo_symm = lo_bc[idir] == amrex::PhysBCType::symmetry;
  bool hi_symm = hi_bc[idir] == amrex::PhysBCType::symmetry;

  const auto domlo = geom.Domain().loVect3d();
  const auto domhi = geom.Domain().hiVect3d();

  // piecewise linear slopes
  amrex::ParallelFor(bx, NQ,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {

    bool lo_bc_test = lo_symm && ((idir == 0 && i == domlo[0]) ||
                                  (idir == 1 && j == domlo[1]) ||
                                  (idir == 2 && k == domlo[2]));

    bool hi_bc_test = hi_symm && ((idir == 0 && i == domhi[0]) ||
                                  (idir == 1 && j == domhi[1]) ||
                                  (idir == 2 && k == domhi[2]));

    Real flat = flatcoef_arr(i,j,k);

    const ZoneStencil s = load_stencil(q_arr, idir, i, j, k, n);

    // normal velocity?
    bool vtest = n == QU+idir;

    dq(i,j,k,n) = uslope(s, flat, reflect_edge(lo_bc_test && vtest, hi_bc_test && vtest));
  });

  if (use_pslope == 1) {

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

      Real flat = flatcoef_arr(i,j,k);

      bool lo_bc_test = lo_symm && ((idir == 0 && i == domlo[0]) ||
                                    (idir == 1 && j == domlo[1]) ||
                                    (idir == 2 && k == domlo[2]));

      bool hi_bc_test = hi_symm && ((idir == 0 && i == domhi[0]) ||
                                    (idir == 1 && j == domhi[1]) ||
                                    (idir == 2 && k == domhi[2]));

      const HydrostaticStencils hse{load_stencil(q_arr, idir, i, j, k, QRHO),
                                    load_stencil(q_arr, idir, i, j, k, QPRES),
                                    load_stencil(src_q_arr, idir, i, j, k, QU+idir)};

      Real dp = dq(i,j,k,QPRES);
      pslope(hse, flat, reflect_edge(lo_bc_test, hi_bc_test), dx[idir], dp);
      dq(i,j,k,QPRES) = dp;

    });
  }

  amrex::ParallelFor(bx, NQ,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {


   // this is a loop over zones.  For each slope in the zone, fill the
   // two adjacent edge states (e.g., the right state at i-1/2 and the
   // left state at i+1/2

   if (idir == 0) {

     // left state at i+1/2 interface
     qm(i+1,j,k,n) = q_arr(i,j,k,n) + 0.5_rt*dq(i,j,k,n);

     // right state at i-1/2 interface
     qp(i,j,k,n) = q_arr(i,j,k,n) - 0.5_rt*dq(i,j,k,n);


#if AMREX_SPACEDIM >= 2
   } else if (idir == 1) {

     // left state at j+1/2 interface
     qm(i,j+1,k,n) = q_arr(i,j,k,n) + 0.5_rt*dq(i,j,k,n);

     // right state at j-1/2 interface
     qp(i,j,k,n) = q_arr(i,j,k,n) - 0.5_rt*dq(i,j,k,n);
#endif

#if AMREX_SPACEDIM == 3
   } else {

     // left state at k+1/2 interface
     qm(i,j,k+1,n) = q_arr(i,j,k,n) + 0.5_rt*dq(i,j,k,n);

     // right state at k-1/2 interface
     qp(i,j,k,n) = q_arr(i,j,k,n) - 0.5_rt*dq(i,j,k,n);

#endif
   }

  });


  // special care for reflecting BCs

  // we have to do this after the loops above, since here we will
  // consider interfaces, not zones

  mirror_states_at_walls(bx, idir, qm, qp);

}


void
HydroSim::ppm_face_states_mol(const Box& bx,
                            const int idir,
                            Array4<Real const> const& q_arr,
                            Array4<Real const> const& flatcoef_arr,
                            Array4<Real> const& qm,
                            Array4<Real> const& qp) {

  amrex::ParallelFor(bx, NQ,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {

    Real flat = flatcoef_arr(i,j,k);

    const ZoneStencil s = load_stencil(q_arr, idir, i, j, k, n);
    const auto [sm, sp] = ppm_reconstruct(s, flat);

    if (idir == 0) {
      // right state at i-1/2
      qp(i,j,k,n) = sm;

      // left state at i+1/2
      qm(i+1,j,k,n) = sp;

    } else if (idir == 1) {
      // right state at j-1/2
      qp(i,j,k,n) = sm;

      // left state at j+1/2
      qm(i,j+1,k,n) = sp;

    } else {
      // right state at k-1/2
      qp(i,j,k,n) = sm;

      // left state at k+1/2
      qm(i,j,k+1,n) = sp;

    }

  });

  // special care for reflecting BCs

  // we have to do this after the loops above, since here we will
  // consider interfaces, not zones

  mirror_states_at_walls(bx, idir, qm, qp);

}


void
HydroSim::mol_cons_update(const Box& bx,  // NOLINT(readability-convert-member-functions-to-static)
#ifdef SHOCK_VAR
                   Array4<Real const> const& shk,
#endif
                   Array4<Real const> const& srcU,
                   Array4<Real> const& update,
                   const Real dt,
                   GpuArray<Array4<Real const>, AMREX_SPACEDIM> const& flux,
                   GpuArray<Array4<Real const>, AMREX_SPACEDIM> const& face_area,
                   GpuArray<Array4<Real const>, AMREX_SPACEDIM> const& qe,
                   Array4<Real const> const& vol) {

  amrex::ignore_unused(dt);

  // For hydro, we will create an update source term that is
  // essentially the flux divergence.  This can be added with dt to
  // get the update

#if AMREX_SPACEDIM <= 2
  const auto dx = geom.CellSizeArray();
  auto coord = geom.Coord();
#if AMREX_SPACEDIM == 2
  auto prob_lo = geom.ProbLoArray();
#endif
#endif

  amrex::ParallelFor(bx, NUM_STATE,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {

#if AMREX_SPACEDIM == 1
    update(i,j,k,n) += (flux[0](i,j,k,n) * face_area[0](i,j,k) - flux[0](i+1,j,k,n) * face_area[0](i+1,j,k) ) / vol(i,j,k);

#elif AMREX_SPACEDIM == 2
    update(i,j,k,n) += (flux[0](i,j,k,n) * face_area[0](i,j,k) - flux[0](i+1,j,k,n) * face_area[0](i+1,j,k) +
                        flux[1](i,j,k,n) * face_area[1](i,j,k) - flux[1](i,j+1,k,n) * face_area[1](i,j+1,k) ) / vol(i,j,k);

#else
    update(i,j,k,n) += (flux[0](i,j,k,n) * face_area[0](i,j,k) - flux[0](i+1,j,k,n) * face_area[0](i+1,j,k) +
                        flux[1](i,j,k,n) * face_area[1](i,j,k) - flux[1](i,j+1,k,n) * face_area[1](i,j+1,k) +
                        flux[2](i,j,k,n) * face_area[2](i,j,k) - flux[2](i,j,k+1,n) * face_area[2](i,j,k+1) ) / vol(i,j,k);
#endif

#if AMREX_SPACEDIM <= 2
    if (do_hydro == 1) {
        if (n == UMX && !momentum_flux_has_pressure(0, 0, coord)) {
            // Add gradp term to radial momentum equation -- only for axisymmetric
            // coords.

            update(i,j,k,UMX) -= (qe[0](i+1,j,k,IFCPRES) - qe[0](i,j,k,IFCPRES)) / dx[0];

#if AMREX_SPACEDIM == 2
        } else if (n == UMY && !momentum_flux_has_pressure(1, 1, coord)) {
            // Add gradp term to polar(theta) momentum equation for Spherical 2D geometry

            Real r = prob_lo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
            update(i,j,k,UMY) -= (qe[1](i,j+1,k,IFCPRES) - qe[1](i,j,k,IFCPRES)) / (r * dx[1]);
#endif
        }
    }
#endif

    // this assumes that the species are at the end of the conserved state
    if (n < NSRC) {
      update(i,j,k,n) += srcU(i,j,k,n);
    }

  });

#ifdef SHOCK_VAR
  // We'll update the shock data for future use in the burning step.
  // For the update, we are starting from USHOCK == 0 (set at the
  // beginning of the timestep) and we need to divide by dt since
  // we'll be multiplying that for the update calculation.

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    update(i,j,k,USHOCK) = shk(i,j,k) / dt;
  });
#endif

}


void
HydroSim::add_mol_conduction_flux(const Box& bx,
                           const int idir,
                           Array4<Real const> const& U,
                           Array4<Real const> const& cond,
                           Array4<Real> const& flux) {

  const auto dx = geom.CellSizeArray();

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real cond_int;
    Real diff_term;

    if (idir == 0) {
      cond_int = 0.5_rt * (cond(i,j,k) + cond(i-1,j,k));
      diff_term = -cond_int * (U(i,j,k,UTEMP) - U(i-1,j,k,UTEMP)) / dx[0];

    } else if (idir == 1) {
      cond_int = 0.5_rt * (cond(i,j,k) + cond(i,j-1,k));
      diff_term = -cond_int * (U(i,j,k,UTEMP) - U(i,j-1,k,UTEMP)) / dx[1];

    } else {
      cond_int = 0.5_rt * (cond(i,j,k) + cond(i,j,k-1));
      diff_term = -cond_int * (U(i,j,k,UTEMP) - U(i,j,k-1,UTEMP)) / dx[2];
    }

    flux(i,j,k,UEINT) += diff_term;
    flux(i,j,k,UEDEN) += diff_term;

  });
}


void
HydroSim::flux_from_face_state(const Box& bx,
                            Array4<Real const> const& q_riemann,
                            Array4<Real> const& F,
                            const int idir) {

  // given a primitive state, compute the flux in direction idir
  //

  int iu, iv1, iv2;
  int im1, im2, im3;

  auto coord = geom.Coord();
  auto mom_check = momentum_flux_has_pressure(idir, idir, coord);

  if (idir == 0) {
    iu = QU;
    iv1 = QV;
    iv2 = QW;
    im1 = UMX;
    im2 = UMY;
    im3 = UMZ;

  } else if (idir == 1) {
    iu = QV;
    iv1 = QU;
    iv2 = QW;
    im1 = UMY;
    im2 = UMX;
    im3 = UMZ;

  } else {
    iu = QW;
    iv1 = QU;
    iv2 = QV;
    im1 = UMZ;
    im2 = UMX;
    im3 = UMY;
  }

#ifdef HYBRID_MOMENTUM
  GeometryData geomdata = geom.data();
#endif

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real u_adv = q_riemann(i,j,k,iu);
    Real rhoeint = q_riemann(i,j,k,QRHOEI);

    // Compute fluxes, order as conserved state (not q)
    F(i,j,k,URHO) = q_riemann(i,j,k,QRHO)*u_adv;

    F(i,j,k,im1) = F(i,j,k,URHO)*q_riemann(i,j,k,iu);
    if (mom_check) {
      F(i,j,k,im1) += q_riemann(i,j,k,QPRES);
    }
    F(i,j,k,im2) = F(i,j,k,URHO)*q_riemann(i,j,k,iv1);
    F(i,j,k,im3) = F(i,j,k,URHO)*q_riemann(i,j,k,iv2);

    Real rhoetot = rhoeint + 0.5_rt * q_riemann(i,j,k,QRHO)*
      (q_riemann(i,j,k,iu)*q_riemann(i,j,k,iu) +
       q_riemann(i,j,k,iv1)*q_riemann(i,j,k,iv1) +
       q_riemann(i,j,k,iv2)*q_riemann(i,j,k,iv2));

    F(i,j,k,UEDEN) = u_adv*(rhoetot + q_riemann(i,j,k,QPRES));
    F(i,j,k,UEINT) = u_adv*rhoeint;

    F(i,j,k,UTEMP) = 0.0;
#ifdef SHOCK_VAR
    F(i,j,k,USHOCK) = 0.0;
#endif

#ifdef NSE_NET
    F(i,j,k,UMU_P) = 0.0;
    F(i,j,k,UMU_N) = 0.0;
#endif
    // passively advected quantities
    for (int ipassive = 0; ipassive < npassive; ipassive++) {
      int n  = cons_passive_index(ipassive);
      int nqp = prim_passive_index(ipassive);

      F(i,j,k,n) = F(i,j,k,URHO)*q_riemann(i,j,k,nqp);
    }

#ifdef HYBRID_MOMENTUM
    // the hybrid routine uses the Godunov indices, not the full NQ state
    GpuArray<Real, NIFC> qface_zone;
    qface_zone[IFCRHO] = q_riemann(i,j,k,QRHO);
    qface_zone[IFCU] = q_riemann(i,j,k,QU);
    qface_zone[IFCV] = q_riemann(i,j,k,QV);
    qface_zone[IFCW] = q_riemann(i,j,k,QW);
    qface_zone[IFCPRES] = q_riemann(i,j,k,QPRES);
    GpuArray<Real, NUM_STATE> F_zone;
    for (int n = 0; n < NUM_STATE; n++) {
        F_zone[n] = F(i,j,k,n);
    }
    compute_hybrid_flux(qface_zone, geomdata, idir, i, j, k, F_zone);
    for (int n = 0; n < NUM_STATE; n++) {
        F(i,j,k,n) = F_zone[n];
    }
#endif
  });
}

void
hydro::store_face_state(const Box& bx,
                           Array4<Real const> const& q_riemann,
#ifdef RADIATION
                           Array4<Real const> const& lambda,
#endif
                           Array4<Real> const& qface) {

  // this copies the full interface state (NQ -- one for each primitive
  // variable) over to a smaller subset of size NIFC for use later in the
  // hydro advancement.

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {


#ifdef HYBRID_MOMENTUM
    qface(i,j,k,IFCRHO) = q_riemann(i,j,k,QRHO);
#endif
    qface(i,j,k,IFCU) = q_riemann(i,j,k,QU);
    qface(i,j,k,IFCV) = q_riemann(i,j,k,QV);
    qface(i,j,k,IFCW) = q_riemann(i,j,k,QW);
    qface(i,j,k,IFCPRES) = q_riemann(i,j,k,QPRES);
#ifdef RADIATION
    for (int g = 0; g < NGROUPS; g++) {
      qface(i,j,k,IFCFLXLIM+g) = lambda(i,j,k,g);
      qface(i,j,k,IFCERAD+g) = q_riemann(i,j,k,QERAD+g);
    }
#endif
  });
}
