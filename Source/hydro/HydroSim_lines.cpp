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

using namespace amrex;
using namespace reconstruction;

void
HydroSim::lines_plm_reconstruct(const Box& bx,
                            const int idir,
                            Array4<Real const> const& prim_arr,
                            Array4<Real const> const& flatten_arr,
                            Array4<Real const> const& prim_source_arr,
                            Array4<Real> const& dq,
                            Array4<Real> const& qm,
                            Array4<Real> const& qp) {

  const auto dx = geom.CellSizeArray();

  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  bool lo_symmetric = lo_bc[idir] == amrex::PhysBCType::symmetry;
  bool hi_symmetric = hi_bc[idir] == amrex::PhysBCType::symmetry;

  const auto domlo = geom.Domain().loVect3d();
  const auto domhi = geom.Domain().hiVect3d();

  // piecewise linear slopes
  amrex::ParallelFor(bx, NQ,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {

    bool at_lo_symmetry = lo_symmetric && ((idir == 0 && i == domlo[0]) ||
                                  (idir == 1 && j == domlo[1]) ||
                                  (idir == 2 && k == domlo[2]));

    bool at_hi_symmetry = hi_symmetric && ((idir == 0 && i == domhi[0]) ||
                                  (idir == 1 && j == domhi[1]) ||
                                  (idir == 2 && k == domhi[2]));

    Real s[nslp];
    Real flat = flatten_arr(i,j,k);

    gather_stencil(prim_arr, idir, i, j, k, n, s);

    // normal velocity?
    bool is_normal_velocity = n == QU+idir;

    dq(i,j,k,n) = limited_slope(s, flat, at_lo_symmetry && is_normal_velocity, at_hi_symmetry && is_normal_velocity);
  });

  if (use_pressure_slope == 1) {

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

      Real s[nslp];
      Real flat = flatten_arr(i,j,k);

      Real rho_stencil[nslp];
      Real src[nslp];

      bool at_lo_symmetry = lo_symmetric && ((idir == 0 && i == domlo[0]) ||
                                    (idir == 1 && j == domlo[1]) ||
                                    (idir == 2 && k == domlo[2]));

      bool at_hi_symmetry = hi_symmetric && ((idir == 0 && i == domhi[0]) ||
                                    (idir == 1 && j == domhi[1]) ||
                                    (idir == 2 && k == domhi[2]));

      gather_stencil(prim_arr, idir, i, j, k, QPRES, s);
      gather_stencil(prim_arr, idir, i, j, k, QRHO, rho_stencil);
      gather_stencil(prim_source_arr, idir, i, j, k, QU+idir, src);

      Real dp = dq(i,j,k,QPRES);
      pressure_slope(rho_stencil, s, src, flat, at_lo_symmetry, at_hi_symmetry, dx[idir], dp);
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
     qm(i+1,j,k,n) = prim_arr(i,j,k,n) + 0.5_rt*dq(i,j,k,n);

     // right state at i-1/2 interface
     qp(i,j,k,n) = prim_arr(i,j,k,n) - 0.5_rt*dq(i,j,k,n);


#if AMREX_SPACEDIM >= 2
   } else if (idir == 1) {

     // left state at j+1/2 interface
     qm(i,j+1,k,n) = prim_arr(i,j,k,n) + 0.5_rt*dq(i,j,k,n);

     // right state at j-1/2 interface
     qp(i,j,k,n) = prim_arr(i,j,k,n) - 0.5_rt*dq(i,j,k,n);
#endif

#if AMREX_SPACEDIM == 3
   } else {

     // left state at k+1/2 interface
     qm(i,j,k+1,n) = prim_arr(i,j,k,n) + 0.5_rt*dq(i,j,k,n);

     // right state at k-1/2 interface
     qp(i,j,k,n) = prim_arr(i,j,k,n) - 0.5_rt*dq(i,j,k,n);

#endif
   }

  });


  // special care for reflecting BCs

  // we have to do this after the loops above, since here we will
  // consider interfaces, not zones

  set_reflecting_boundary_states(bx, idir, qm, qp);

}


void
HydroSim::lines_ppm_reconstruct(const Box& bx,
                            const int idir,
                            Array4<Real const> const& prim_arr,
                            Array4<Real const> const& flatten_arr,
                            Array4<Real> const& qm,
                            Array4<Real> const& qp) {

  amrex::ParallelFor(bx, NQ,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {

    Real s[nslp];
    Real flat = flatten_arr(i,j,k);
    Real sm;
    Real sp;

    gather_stencil(prim_arr, idir, i, j, k, n, s);
    parabolic_reconstruct(s, flat, sm, sp);

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

  set_reflecting_boundary_states(bx, idir, qm, qp);

}


void
HydroSim::lines_cons_update(const Box& bx,  // NOLINT(readability-convert-member-functions-to-static)
#ifdef SHOCK_VAR
                   Array4<Real const> const& shk,
#endif
                   Array4<Real const> const& sourceTerms,
                   Array4<Real> const& update,
                   const Real dt,
                   Array4<Real const> const& xface_flux,
#if AMREX_SPACEDIM >= 2
                   Array4<Real const> const& flux1,
#endif
#if AMREX_SPACEDIM == 3
                   Array4<Real const> const& flux2,
#endif
                   Array4<Real const> const& face_area0,
#if AMREX_SPACEDIM >= 2
                   Array4<Real const> const& area1,
#endif
#if AMREX_SPACEDIM == 3
                   Array4<Real const> const& area2,
#endif
#if AMREX_SPACEDIM <= 2
                   Array4<Real const> const& q0,
#if AMREX_SPACEDIM == 2
                   Array4<Real const> const& q1,
#endif
#endif
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
    update(i,j,k,n) += (xface_flux(i,j,k,n) * face_area0(i,j,k) - xface_flux(i+1,j,k,n) * face_area0(i+1,j,k) ) / vol(i,j,k);

#elif AMREX_SPACEDIM == 2
    update(i,j,k,n) += (xface_flux(i,j,k,n) * face_area0(i,j,k) - xface_flux(i+1,j,k,n) * face_area0(i+1,j,k) +
                        flux1(i,j,k,n) * area1(i,j,k) - flux1(i,j+1,k,n) * area1(i,j+1,k) ) / vol(i,j,k);

#else
    update(i,j,k,n) += (xface_flux(i,j,k,n) * face_area0(i,j,k) - xface_flux(i+1,j,k,n) * face_area0(i+1,j,k) +
                        flux1(i,j,k,n) * area1(i,j,k) - flux1(i,j+1,k,n) * area1(i,j+1,k) +
                        flux2(i,j,k,n) * area2(i,j,k) - flux2(i,j,k+1,n) * area2(i,j,k+1) ) / vol(i,j,k);
#endif

#if AMREX_SPACEDIM <= 2
    if (do_hydro == 1) {
        if (n == UMX && !momentum_flux_has_pressure(0, 0, coord)) {
            // Add gradp term to radial momentum equation -- only for axisymmetric
            // coords.

            update(i,j,k,UMX) -= (q0(i+1,j,k,GDPRESSURE) - q0(i,j,k,GDPRESSURE)) / dx[0];

#if AMREX_SPACEDIM == 2
        } else if (n == UMY && !momentum_flux_has_pressure(1, 1, coord)) {
            // Add gradp term to polar(theta) momentum equation for Spherical 2D geometry

            Real r = prob_lo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
            update(i,j,k,UMY) -= (q1(i,j+1,k,GDPRESSURE) - q1(i,j,k,GDPRESSURE)) / (r * dx[1]);
#endif
        }
    }
#endif

    // this assumes that the species are at the end of the conserved state
    if (n < NUM_SOURCE) {
      update(i,j,k,n) += sourceTerms(i,j,k,n);
    }

  });

#ifdef SHOCK_VAR
  // We'll update the shock data for future use in the burning step.
  // For the update, we are starting from UFLAG_SHOCK == 0 (set at the
  // beginning of the timestep) and we need to divide by dt since
  // we'll be multiplying that for the update calculation.

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    update(i,j,k,UFLAG_SHOCK) = shk(i,j,k) / dt;
  });
#endif

}


void
HydroSim::lines_diffusion_flux(const Box& bx,
                           const int idir,
                           Array4<Real const> const& U,
                           Array4<Real const> const& cond,
                           Array4<Real> const& flux) {

  const auto dx = geom.CellSizeArray();

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real face_conductivity;
    Real diffusion_term;

    if (idir == 0) {
      face_conductivity = 0.5_rt * (cond(i,j,k) + cond(i-1,j,k));
      diffusion_term = -face_conductivity * (U(i,j,k,UTEMP) - U(i-1,j,k,UTEMP)) / dx[0];

    } else if (idir == 1) {
      face_conductivity = 0.5_rt * (cond(i,j,k) + cond(i,j-1,k));
      diffusion_term = -face_conductivity * (U(i,j,k,UTEMP) - U(i,j-1,k,UTEMP)) / dx[1];

    } else {
      face_conductivity = 0.5_rt * (cond(i,j,k) + cond(i,j,k-1));
      diffusion_term = -face_conductivity * (U(i,j,k,UTEMP) - U(i,j,k-1,UTEMP)) / dx[2];
    }

    flux(i,j,k,UEINT) += diffusion_term;
    flux(i,j,k,UEDEN) += diffusion_term;

  });
}


void
HydroSim::flux_from_state(const Box& bx,
                            Array4<Real const> const& interface_state,
                            Array4<Real> const& F,
                            const int idir) {

  // given a primitive state, compute the flux in direction idir
  //

  int iu, iv1, iv2;
  int im1, im2, im3;

  auto coord = geom.Coord();
  auto has_pressure_flux = momentum_flux_has_pressure(idir, idir, coord);

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

    Real advect_velocity = interface_state(i,j,k,iu);
    Real rho_internal_energy = interface_state(i,j,k,QEINTDENS);

    // Compute fluxes, order as conserved state (not q)
    F(i,j,k,URHO) = interface_state(i,j,k,QRHO)*advect_velocity;

    F(i,j,k,im1) = F(i,j,k,URHO)*interface_state(i,j,k,iu);
    if (has_pressure_flux) {
      F(i,j,k,im1) += interface_state(i,j,k,QPRES);
    }
    F(i,j,k,im2) = F(i,j,k,URHO)*interface_state(i,j,k,iv1);
    F(i,j,k,im3) = F(i,j,k,URHO)*interface_state(i,j,k,iv2);

    Real rho_total_energy = rho_internal_energy + 0.5_rt * interface_state(i,j,k,QRHO)*
      (interface_state(i,j,k,iu)*interface_state(i,j,k,iu) +
       interface_state(i,j,k,iv1)*interface_state(i,j,k,iv1) +
       interface_state(i,j,k,iv2)*interface_state(i,j,k,iv2));

    F(i,j,k,UEDEN) = advect_velocity*(rho_total_energy + interface_state(i,j,k,QPRES));
    F(i,j,k,UEINT) = advect_velocity*rho_internal_energy;

    F(i,j,k,UTEMP) = 0.0;
#ifdef SHOCK_VAR
    F(i,j,k,UFLAG_SHOCK) = 0.0;
#endif

#ifdef NSE_NET
    F(i,j,k,UPROTON_MU) = 0.0;
    F(i,j,k,UNEUTRON_MU) = 0.0;
#endif
    // passively advected quantities
    for (int passive_index = 0; passive_index < npassive; passive_index++) {
      int n  = passive_cons_index(passive_index);
      int nqp = passive_prim_index(passive_index);

      F(i,j,k,n) = F(i,j,k,URHO)*interface_state(i,j,k,nqp);
    }

#ifdef HYBRID_MOMENTUM
    // the hybrid routine uses the Godunov indices, not the full NQ state
    GpuArray<Real, NGDSTATE> godunov_zone;
    godunov_zone[GDDENSITY] = interface_state(i,j,k,QRHO);
    godunov_zone[GDU] = interface_state(i,j,k,QU);
    godunov_zone[GDV] = interface_state(i,j,k,QV);
    godunov_zone[GDW] = interface_state(i,j,k,QW);
    godunov_zone[GDPRESSURE] = interface_state(i,j,k,QPRES);
    GpuArray<Real, NUM_STATE> ZoneFlux;
    for (int n = 0; n < NUM_STATE; n++) {
        ZoneFlux[n] = F(i,j,k,n);
    }
    hybrid_flux_from_state(godunov_zone, geomdata, idir, i, j, k, ZoneFlux);
    for (int n = 0; n < NUM_STATE; n++) {
        F(i,j,k,n) = ZoneFlux[n];
    }
#endif
  });
}

void
HydroSim::save_godunov_state(const Box& bx,
                            Array4<Real const> const& interface_state,
#ifdef RADIATION
                            Array4<Real const> const& lambda,
#endif
                            Array4<Real> const& godunov_state) {

  // this copies the full interface state (NQ -- one for each primitive
  // variable) over to a smaller subset of size NGDSTATE for use later in the
  // hydro advancement.

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {


#ifdef HYBRID_MOMENTUM
    godunov_state(i,j,k,GDDENSITY) = interface_state(i,j,k,QRHO);
#endif
    godunov_state(i,j,k,GDU) = interface_state(i,j,k,QU);
    godunov_state(i,j,k,GDV) = interface_state(i,j,k,QV);
    godunov_state(i,j,k,GDW) = interface_state(i,j,k,QW);
    godunov_state(i,j,k,GDPRESSURE) = interface_state(i,j,k,QPRES);
#ifdef RADIATION
    for (int g = 0; g < NGROUPS; g++) {
      godunov_state(i,j,k,GDFLUXLIM+g) = lambda(i,j,k,g);
      godunov_state(i,j,k,GDRADENG+g) = interface_state(i,j,k,QRADENG+g);
    }
#endif
  });
}
