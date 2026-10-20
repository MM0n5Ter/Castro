#include <HydroSim.H>
#include <HydroSim_helpers.H>

#ifdef RADIATION
#include <Radiation.H>
#endif
#ifdef GRAVITY
#include <Gravity.H>
#endif

#ifdef HYBRID_MOMENTUM
#include <hybrid_momentum.H>
#endif

#include <parabolic_profile.H>

#include <limiters.H>

using namespace amrex;

void
HydroSim::corner_ppm_states(const Box& bx, const Box& vbx,
                       Array4<Real const> const& field_arr,
                       Array4<Real const> const& inv_density_arr,
                       Array4<Real const> const& prim_arr,
                       Array4<Real const> const& qthermo_arr,
                       Array4<Real const> const& prim_source,
                       Array4<Real> const& qxm,
                       Array4<Real> const& qxp,
#if AMREX_SPACEDIM >= 2
                       Array4<Real> const& qym,
                       Array4<Real> const& qyp,
#endif
#if AMREX_SPACEDIM == 3
                       Array4<Real> const& qzm,
                       Array4<Real> const& qzp,
#endif
#if AMREX_SPACEDIM < 3
                       Array4<Real const> const& logAreaRateX,
#endif
#if AMREX_SPACEDIM == 2
                       Array4<Real const> const& logAreaRateY,
#endif
                       const Real dt) {

  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {

    if (idir == 0) {
      trace_piecewise_parabolic(bx,
                idir,
                field_arr, inv_density_arr, prim_arr, qthermo_arr, prim_source,
                qxm, qxp,
#if AMREX_SPACEDIM <= 2
                logAreaRateX,
#endif
                vbx, dt);

      set_reflecting_boundary_states(bx, 0, qxm, qxp);

#if AMREX_SPACEDIM >= 2
    } else if (idir == 1) {
      trace_piecewise_parabolic(bx,
                idir,
                field_arr, inv_density_arr, prim_arr, qthermo_arr, prim_source,
                qym, qyp,
#if AMREX_SPACEDIM <= 2
                logAreaRateY,
#endif
                vbx, dt);

      set_reflecting_boundary_states(bx, 1, qym, qyp);
#endif

#if AMREX_SPACEDIM == 3
    } else {
      trace_piecewise_parabolic(bx,
                idir,
                field_arr, inv_density_arr, prim_arr, qthermo_arr, prim_source,
                qzm, qzp,
                vbx, dt);

      set_reflecting_boundary_states(bx, 2, qzm, qzp);
#endif
    }
  }
}

#ifdef RADIATION
void
HydroSim::corner_ppm_rad_states(const Box& bx, const Box& vbx,
                           Array4<Real const> const& field_arr,
                           Array4<Real const> const& inv_density_arr,
                           Array4<Real const> const& prim_arr,
                           Array4<Real const> const& qthermo_arr,
                           Array4<Real const> const& prim_source,
                           Array4<Real> const& qxm,
                           Array4<Real> const& qxp,
#if AMREX_SPACEDIM >= 2
                           Array4<Real> const& qym,
                           Array4<Real> const& qyp,
#endif
#if AMREX_SPACEDIM == 3
                           Array4<Real> const& qzm,
                           Array4<Real> const& qzp,
#endif
#if AMREX_SPACEDIM < 3
                           Array4<Real const> const& logAreaRateX,
#endif
#if AMREX_SPACEDIM == 2
                           Array4<Real const> const& logAreaRateY,
#endif
                           const Real dt) {

  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {

    if (idir == 0) {

      trace_parabolic_radiative(bx,
                    idir,
                    field_arr, inv_density_arr, prim_arr, qthermo_arr, prim_source,
                    qxm, qxp,
#if AMREX_SPACEDIM <= 2
                    logAreaRateX,
#endif
                    vbx, dt);

      set_reflecting_boundary_states(bx, 0, qxm, qxp);

#if AMREX_SPACEDIM >= 2
    } else if (idir == 1) {
      trace_parabolic_radiative(bx,
                    idir,
                    field_arr, inv_density_arr, prim_arr, qthermo_arr, prim_source,
                    qym, qyp,
#if AMREX_SPACEDIM <= 2
                    logAreaRateY,
#endif
                    vbx, dt);

      set_reflecting_boundary_states(bx, 1, qym, qyp);
#endif

#if AMREX_SPACEDIM == 3
    } else {
      trace_parabolic_radiative(bx,
                    idir,
                    field_arr, inv_density_arr, prim_arr, qthermo_arr, prim_source,
                    qzm, qzp,
                    vbx, dt);

      set_reflecting_boundary_states(bx, 2, qzm, qzp);

#endif
    }
  }
}
#endif

void
HydroSim::corner_plm_states(const Box& bx, const Box& vbx,
                       Array4<Real const> const& field_arr,
                       Array4<Real const> const& inv_density_arr,
                       Array4<Real const> const& prim_arr,
                       Array4<Real const> const& qthermo_arr,
                       Array4<Real const> const& prim_source,
                       Array4<Real> const& qxm,
                       Array4<Real> const& qxp,
#if AMREX_SPACEDIM >= 2
                       Array4<Real> const& qym,
                       Array4<Real> const& qyp,
#endif
#if AMREX_SPACEDIM == 3
                       Array4<Real> const& qzm,
                       Array4<Real> const& qzp,
#endif
#if AMREX_SPACEDIM < 3
                       Array4<Real const> const& logAreaRateX,
#endif
#if AMREX_SPACEDIM == 2
                       Array4<Real const> const& logAreaRateY,
#endif
                       const Real dt) {

#ifdef RADIATION
#ifndef AMREX_USE_GPU
  amrex::Error("ppm_recon_kind <=0 is not supported in with radiation");
#endif
#endif

  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {

    if (idir == 0) {
      trace_piecewise_linear(bx, 0,
                field_arr, inv_density_arr, prim_arr, qthermo_arr,
                qxm, qxp,
#if AMREX_SPACEDIM < 3
                logAreaRateX,
#endif
                prim_source, vbx, dt);

      set_reflecting_boundary_states(bx, 0, qxm, qxp);

#if AMREX_SPACEDIM >= 2
    } else if (idir == 1) {
      trace_piecewise_linear(bx, 1,
                field_arr, inv_density_arr, prim_arr, qthermo_arr,
                qym, qyp,
#if AMREX_SPACEDIM < 3
                logAreaRateY,
#endif
                prim_source, vbx, dt);

      set_reflecting_boundary_states(bx, 1, qym, qyp);
#endif

#if AMREX_SPACEDIM == 3
    } else {
      trace_piecewise_linear(bx, 2,
                field_arr, inv_density_arr, prim_arr, qthermo_arr,
                qzm, qzp,
                prim_source, vbx, dt);

      set_reflecting_boundary_states(bx, 2, qzm, qzp);
#endif
    }
  }
}

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

   if (idir == 0) {

     qm(i+1,j,k,n) = prim_arr(i,j,k,n) + 0.5_rt*dq(i,j,k,n);

     qp(i,j,k,n) = prim_arr(i,j,k,n) - 0.5_rt*dq(i,j,k,n);

#if AMREX_SPACEDIM >= 2
   } else if (idir == 1) {

     qm(i,j+1,k,n) = prim_arr(i,j,k,n) + 0.5_rt*dq(i,j,k,n);

     qp(i,j,k,n) = prim_arr(i,j,k,n) - 0.5_rt*dq(i,j,k,n);
#endif

#if AMREX_SPACEDIM == 3
   } else {

     qm(i,j,k+1,n) = prim_arr(i,j,k,n) + 0.5_rt*dq(i,j,k,n);

     qp(i,j,k,n) = prim_arr(i,j,k,n) - 0.5_rt*dq(i,j,k,n);

#endif
   }

  });

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
      qp(i,j,k,n) = sm;

      qm(i+1,j,k,n) = sp;

    } else if (idir == 1) {
      qp(i,j,k,n) = sm;

      qm(i,j+1,k,n) = sp;

    } else {
      qp(i,j,k,n) = sm;

      qm(i,j,k+1,n) = sp;

    }

  });

  set_reflecting_boundary_states(bx, idir, qm, qp);

}
