#include <HydroSim.H>
#include <HydroSim_util.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef HYBRID_MOMENTUM
#include <hybrid.H>
#endif

#include <advection_util.H>

using namespace amrex;

advance_status
HydroSim::construct_ctu_hydro_source(Real time, Real dt)  // NOLINT(readability-convert-member-functions-to-static)
{
  amrex::ignore_unused(time);
  amrex::ignore_unused(dt);

  advance_status status {};

  if (!do_hydro) {
      return status;
  }

#ifndef TRUE_SDC

  BL_PROFILE("HydroSim::construct_ctu_hydro_source()");

  const Real strt_time = ParallelDescriptor::second();

  // this constructs the hydrodynamic source (essentially the flux
  // divergence) using the CTU framework for unsplit hydrodynamics

  if (verbose) {
      amrex::Print() << "... Entering construct_ctu_hydro_source() on level " << level << std::endl << std::endl;
  }

#ifdef HYBRID_MOMENTUM
  GeometryData geomdata = geom.data();
#endif

#if AMREX_SPACEDIM <= 2
  int coord = geom.Coord();
#endif

#if AMREX_SPACEDIM >= 2
  const Real *dx = geom.CellSize();
#endif

  MultiFab& S_new = get_new_data(State_Type);

#ifdef REACTIONS
  MultiFab tmp_SDC_mf;
  MultiFab& SDC_react_source = hydrosim::time_integration_method == SimplifiedSpectralDeferredCorrections ?
                               get_new_data(Simplified_SDC_React_Type) : tmp_SDC_mf;
#endif

  // we will treat the hydro source as any other source term

#ifdef RADIATION
  MultiFab& Er_new = get_new_data(Rad_Type);

  if (!Radiation::rad_hydro_combined) {
    amrex::Abort("HydroSim::construct_ctu_hydro_source -- we don't implement a mode where we have radiation, but it is not coupled to hydro");
  }

  int nstep_fsp = -1;

  AmrLevel::FillPatch(*this, Erborder, NUM_GROW, time, Rad_Type, 0, Radiation::nGroups);

  MultiFab lamborder(grids, dmap, Radiation::nGroups, NUM_GROW);
  if (radiation->pure_hydro) {
      lamborder.setVal(0.0, NUM_GROW);
  }
  else {
      radiation->compute_limiter(level, grids, Sborder, Erborder, lamborder);
  }
#endif

  // Record a running total of the number of bytes allocated as temporary Fab data.

  size_t fab_size = 0;
#ifdef AMREX_USE_GPU
  size_t mf_size = 0;
#endif
  IntVect maximum_tile_size{0};

  // Our strategy for launching work on GPUs in the hydro is incompatible with OpenMP,
  // throw an error if the user is trying this. If this were to ever change, it would
  // require at minimum doing a safe atomic update on the running fab_size total.

#if defined(AMREX_USE_OMP) && defined(AMREX_USE_GPU)
  amrex::Error("USE_OMP=TRUE and USE_GPU=TRUE are not concurrently supported in HydroSim");
#endif

#ifdef AMREX_USE_GPU
   if (hydrosim::hydro_memory_footprint_ratio > 0.0) {
       // If we haven't done any tuning yet, set the tile size to an arbitrary
       // small value to start with.

       if (hydro_tile_size_has_been_tuned == 0) {
           hydro_tile_size[0] = 16;
#if AMREX_SPACEDIM >= 2
           hydro_tile_size[1] = 16;
#endif
#if AMREX_SPACEDIM == 3
           hydro_tile_size[2] = 16;
#endif
       }

       // Run through boxes on this level and see if any of them are
       // bigger than the biggest box from our previous tuning. If so,
       // we need to re-compute the tile size.
       for (MFIter mfi(S_new, false); mfi.isValid(); ++mfi) {
           if (mfi.validbox().numPts() > largest_box_from_hydro_tile_size_tuning) {
               hydro_tile_size_has_been_tuned = 0;
           }

           // Also, sum up the number of bytes in the state data.
           mf_size += S_new[mfi].nBytes();
       }
   }
#endif

#ifdef _OPENMP
#ifdef RADIATION
#pragma omp parallel reduction(max:nstep_fsp)
#else
#pragma omp parallel
#endif
#endif
  {

#ifdef RADIATION
    int priv_nstep_fsp = -1;
#endif

    // Declare local storage now. This should be done outside the
    // MFIter loop, and then we will resize the Fabs in each MFIter
    // loop iteration. We use the async arenato ensure that their
    // memory is saved until it is no longer needed (only relevant for
    // the asynchronous case, usually on GPUs).

    FArrayBox shk(The_Async_Arena());
    FArrayBox q(The_Async_Arena()), qaux(The_Async_Arena());
    FArrayBox rho_inv(The_Async_Arena());
    FArrayBox src_q(The_Async_Arena());
    FArrayBox qxm(The_Async_Arena()), qxp(The_Async_Arena());
#if AMREX_SPACEDIM >= 2
    FArrayBox qym(The_Async_Arena()), qyp(The_Async_Arena());
#endif
#if AMREX_SPACEDIM == 3
    FArrayBox qzm(The_Async_Arena()), qzp(The_Async_Arena());
#endif
    FArrayBox div(The_Async_Arena());
#if AMREX_SPACEDIM >= 2
    FArrayBox ftmp1(The_Async_Arena()), ftmp2(The_Async_Arena());
#ifdef RADIATION
    FArrayBox rftmp1(The_Async_Arena()), rftmp2(The_Async_Arena());
#endif
    FArrayBox qfacetmp1(The_Async_Arena()), qfacetmp2(The_Async_Arena());
    FArrayBox ql(The_Async_Arena()), qr(The_Async_Arena());
#endif
    Vector<FArrayBox> flux, qe;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        flux.push_back(FArrayBox(The_Async_Arena()));
        qe.push_back(FArrayBox(The_Async_Arena()));
    }

#ifdef RADIATION
    Vector<FArrayBox> rad_flux;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        rad_flux.push_back(FArrayBox(The_Async_Arena()));
    }
#endif
#if AMREX_SPACEDIM <= 2
    FArrayBox pradial(The_Async_Arena());
#endif
#if AMREX_SPACEDIM == 3
    FArrayBox qmyx(The_Async_Arena()), qpyx(The_Async_Arena());
    FArrayBox qmzx(The_Async_Arena()), qpzx(The_Async_Arena());
    FArrayBox qmxy(The_Async_Arena()), qpxy(The_Async_Arena());
    FArrayBox qmzy(The_Async_Arena()), qpzy(The_Async_Arena());
    FArrayBox qmxz(The_Async_Arena()), qpxz(The_Async_Arena());
    FArrayBox qmyz(The_Async_Arena()), qpyz(The_Async_Arena());
#endif

    MultiFab& old_source = get_old_data(Source_Type);

    // Size a temporary Fab for the current tile, add it to the running
    // total of temporary storage, and return a view of its data.

    auto resize_tile_fab = [&fab_size] (FArrayBox& fab, const Box& box, int ncomp) -> Array4<Real> {
        fab.resize(box, ncomp);
        fab_size += fab.nBytes();
        return fab.array();
    };

    for (MFIter mfi(S_new, hydro_tile_size); mfi.isValid(); ++mfi) {

      // the valid region box
      const Box& bx = mfi.tilebox();

      const Box& obx = amrex::grow(bx, 1);

      // Compute the primitive variables (both q and qaux) from
      // the conserved variables.

      const Box& qbx = amrex::grow(bx, NUM_GROW);
      const Box& qbx3 = amrex::grow(bx, 3);

#ifdef RADIATION
      Array4<Real> const q_arr = resize_tile_fab(q, qbx, NQ);
#else
      // note: we won't store the passives in q, so we'll compute their
      // primitive versions on demand as needed
      Array4<Real> const q_arr = resize_tile_fab(q, qbx, NQTHERM);
#endif

      Array4<Real> const qaux_arr = resize_tile_fab(qaux, qbx, NQAUX);

      Array4<Real const> const U_old_arr = Sborder.array(mfi);

      Array4<Real> const rho_inv_arr = resize_tile_fab(rho_inv, qbx3, 1);

      amrex::ParallelFor(qbx3,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
          rho_inv_arr(i,j,k) = 1.0 / U_old_arr(i,j,k,URHO);
      });

      ctoprim(qbx, time, U_old_arr,
#ifdef RADIATION
              Erborder.array(mfi), lamborder.array(mfi),
#endif
              q_arr, qaux_arr);

#if AMREX_SPACEDIM == 2
      Array4<Real const> const areax_arr = area[0].array(mfi);
      Array4<Real const> const areay_arr = area[1].array(mfi);
      Array4<Real> const vol_arr = volume.array(mfi);
#endif

#if AMREX_SPACEDIM < 3
      Array4<Real const> const dLogAreaX_arr = (dLogArea[0]).array(mfi);
#endif
#if AMREX_SPACEDIM == 2
      Array4<Real const> const dLogAreaY_arr = (dLogArea[1]).array(mfi);
#endif

      const Box& xbx = amrex::surroundingNodes(bx, 0);
      const Box& gxbx = amrex::grow(xbx, 1);
#if AMREX_SPACEDIM >= 2
      const Box& ybx = amrex::surroundingNodes(bx, 1);
      const Box& gybx = amrex::grow(ybx, 1);
#endif
#if AMREX_SPACEDIM == 3
      const Box& zbx = amrex::surroundingNodes(bx, 2);
      const Box& gzbx = amrex::grow(zbx, 1);
#endif

      // Multidimensional shock detection

      // this is a local shock variable used only for the Riemann
      // solver -- this will never be used to update the value in the
      // conserved state.

      Array4<Real> const shk_arr = resize_tile_fab(shk, obx, 1);

      // get the primitive variable hydro sources

      Array4<Real> const src_q_arr = resize_tile_fab(src_q, qbx3, NQSRC);

      Array4<Real> const old_src_arr = old_source.array(mfi);
      Array4<Real> const src_corr_arr = source_corrector.array(mfi);

      amrex::ParallelFor(qbx3,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
          hydro::src_to_prim(i, j, k, dt, U_old_arr, q_arr, old_src_arr, src_corr_arr, src_q_arr);
      });

      if (hybrid_riemann == 1) {
        shock(obx, q_arr, old_src_arr, shk_arr);
      }
      else {
        amrex::ParallelFor(obx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
          shk_arr(i,j,k) = 0.0;
        });
      }

      // work on the interface states

      Array4<Real> const qxm_arr = resize_tile_fab(qxm, obx, NQ);
      Array4<Real> const qxp_arr = resize_tile_fab(qxp, obx, NQ);

#if AMREX_SPACEDIM >= 2
      Array4<Real> const qym_arr = resize_tile_fab(qym, obx, NQ);
      Array4<Real> const qyp_arr = resize_tile_fab(qyp, obx, NQ);

#endif

#if AMREX_SPACEDIM == 3
      Array4<Real> const qzm_arr = resize_tile_fab(qzm, obx, NQ);
      Array4<Real> const qzp_arr = resize_tile_fab(qzp, obx, NQ);

#endif

      if (ppm_type == 0) {

        ctu_plm_states(obx, bx,
                       U_old_arr, rho_inv_arr,
                       q_arr,
                       qaux_arr,
                       src_q_arr,
                       qxm_arr, qxp_arr,
#if AMREX_SPACEDIM >= 2
                       qym_arr, qyp_arr,
#endif
#if AMREX_SPACEDIM == 3
                       qzm_arr, qzp_arr,
#endif
#if AMREX_SPACEDIM < 3
                       dLogAreaX_arr,
#endif
#if AMREX_SPACEDIM == 2
                       dLogAreaY_arr,
#endif
                       dt);

      } else {

#ifdef RADIATION
        ctu_ppm_rad_states(obx, bx,
                           U_old_arr, rho_inv_arr,
                           q_arr, qaux_arr, src_q_arr,
                           qxm_arr, qxp_arr,
#if AMREX_SPACEDIM >= 2
                           qym_arr, qyp_arr,
#endif
#if AMREX_SPACEDIM == 3
                           qzm_arr, qzp_arr,
#endif
#if AMREX_SPACEDIM < 3
                           dLogAreaX_arr,
#endif
#if AMREX_SPACEDIM == 2
                           dLogAreaY_arr,
#endif
                           dt);
#else

        ctu_ppm_states(obx, bx,
                       U_old_arr, rho_inv_arr,
                       q_arr, qaux_arr, src_q_arr,
                       qxm_arr, qxp_arr,
#if AMREX_SPACEDIM >= 2
                       qym_arr, qyp_arr,
#endif
#if AMREX_SPACEDIM == 3
                       qzm_arr, qzp_arr,
#endif
#if AMREX_SPACEDIM < 3
                       dLogAreaX_arr,
#endif
#if AMREX_SPACEDIM == 2
                       dLogAreaY_arr,
#endif
                       dt);
#endif

      }

      auto div_arr = resize_tile_fab(div, obx, 1);

      // compute divu -- we'll use this later when doing the artificial viscosity
      divu(obx, q_arr, div_arr);

      Array4<Real> const flux0_arr = resize_tile_fab(flux[0], gxbx, NUM_STATE);
      auto qex_arr = resize_tile_fab(qe[0], gxbx, NIFC);

#ifdef RADIATION
      auto rad_flux0_arr = resize_tile_fab(rad_flux[0], gxbx, Radiation::nGroups);
#endif

#if AMREX_SPACEDIM >= 2
      Array4<Real> const flux1_arr = resize_tile_fab(flux[1], gybx, NUM_STATE);
      auto qey_arr = resize_tile_fab(qe[1], gybx, NIFC);

#ifdef RADIATION
      auto const rad_flux1_arr = resize_tile_fab(rad_flux[1], gybx, Radiation::nGroups);
#endif
#endif

#if AMREX_SPACEDIM == 3
      Array4<Real> const flux2_arr = resize_tile_fab(flux[2], gzbx, NUM_STATE);
      auto qez_arr = resize_tile_fab(qe[2], gzbx, NIFC);

#ifdef RADIATION
      auto const rad_flux2_arr = resize_tile_fab(rad_flux[2], gzbx, Radiation::nGroups);
#endif
#endif

#if AMREX_SPACEDIM <= 2
      if (!Geom().IsCartesian()) {
          pradial.resize(xbx, 1);
      }
      fab_size += pradial.nBytes();
#endif

#ifdef REACTIONS
      auto sdc_src_arr = hydrosim::time_integration_method == SimplifiedSpectralDeferredCorrections ?
                         SDC_react_source.array(mfi) : Array4<Real const>{};
#endif

#if AMREX_SPACEDIM == 1

#ifdef REACTIONS
      if (hydrosim::time_integration_method == SimplifiedSpectralDeferredCorrections) {
          add_sdc_source_to_states(xbx, 0, dt,
                                   qxm_arr, qxp_arr, sdc_src_arr);
      }
#endif

      // compute the fluxes through the x-interface

      riemann_flux_and_state(xbx,
                          qxm_arr, qxp_arr,
                          flux0_arr,
#ifdef RADIATION
                          rad_flux0_arr,
#endif
                          qex_arr,
                          qaux_arr,
                          shk_arr,
                          0, false);

#endif // 1-d



#if AMREX_SPACEDIM >= 2
      auto ftmp1_arr = resize_tile_fab(ftmp1, obx, NUM_STATE);
      auto ftmp2_arr = resize_tile_fab(ftmp2, obx, NUM_STATE);

#ifdef RADIATION
      auto rftmp1_arr = resize_tile_fab(rftmp1, obx, Radiation::nGroups);
      auto rftmp2_arr = resize_tile_fab(rftmp2, obx, Radiation::nGroups);
#endif

      auto qfacetmp1_arr = resize_tile_fab(qfacetmp1, obx, NIFC);

#if AMREX_SPACEDIM == 3
      auto qfacetmp2_arr = resize_tile_fab(qfacetmp2, obx, NIFC);
#endif

      auto ql_arr = resize_tile_fab(ql, obx, NQ);
      auto qr_arr = resize_tile_fab(qr, obx, NQ);

      // Make the final interface states in ql/qr thermodynamically
      // consistent, add the reaction source for simplified SDC, and
      // solve the Riemann problem across the idir faces of nbx.

      auto final_riemann_solve = [&] (const Box& nbx, int idir,
                                      Array4<Real> const& flux_arr,
#ifdef RADIATION
                                      Array4<Real> const& rad_flux_arr,
#endif
                                      Array4<Real> const& qface_arr) {
          reset_edge_state_thermo(nbx, ql_arr);

          reset_edge_state_thermo(nbx, qr_arr);

#ifdef REACTIONS
          if (hydrosim::time_integration_method == SimplifiedSpectralDeferredCorrections) {
              add_sdc_source_to_states(nbx, idir, dt,
                                       ql_arr, qr_arr, sdc_src_arr);
          }
#endif

          riemann_flux_and_state(nbx,
                                 ql_arr, qr_arr,
                                 flux_arr,
#ifdef RADIATION
                                 rad_flux_arr,
#endif
                                 qface_arr,
                                 qaux_arr, shk_arr,
                                 idir, false);
      };
#endif



#if AMREX_SPACEDIM == 2

      const amrex::Real hdt = 0.5*dt;
      const amrex::Real hdtdx = 0.5*dt/dx[0];
      const amrex::Real hdtdy = 0.5*dt/dx[1];

      // compute F^x
      // [lo(1), lo(2)-1, 0], [hi(1)+1, hi(2)+1, 0]
      const Box& cxbx = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,1,0)));

      // ftmp1 = fx
      // rftmp1 = rfx
      // qfacetmp1 = qgdnxv
      riemann_flux_and_state(cxbx,
                          qxm_arr, qxp_arr,
                          ftmp1_arr,
#ifdef RADIATION
                          rftmp1_arr,
#endif
                          qfacetmp1_arr,
                          qaux_arr, shk_arr,
                          0, false);

      // compute F^y
      // [lo(1)-1, lo(2), 0], [hi(1)+1, hi(2)+1, 0]
      const Box& cybx = amrex::grow(ybx, IntVect(AMREX_D_DECL(1,0,0)));

      // ftmp2 = fy
      // rftmp2 = rfy
      riemann_flux_and_state(cybx,
                          qym_arr, qyp_arr,
                          ftmp2_arr,
#ifdef RADIATION
                          rftmp2_arr,
#endif
                          qey_arr,
                          qaux_arr, shk_arr,
                          1, false);

      // add the transverse flux difference in y to the x states
      // [lo(1), lo(2), 0], [hi(1)+1, hi(2), 0]

      // ftmp2 = fy
      // rftmp2 = rfy
      transverse_correct_single(xbx, 1, 0,
                   qxm_arr, ql_arr,
                   qxp_arr, qr_arr,
                   qaux_arr,
                   ftmp2_arr,
#ifdef RADIATION
                   rftmp2_arr,
#endif
                   qey_arr,
                   areay_arr,
                   vol_arr,
                   hdt, hdtdy);

      // solve the final Riemann problem axross the x-interfaces

      final_riemann_solve(xbx, 0, flux0_arr,
#ifdef RADIATION
                          rad_flux0_arr,
#endif
                          qex_arr);

      // add the transverse flux difference in x to the y states
      // [lo(1), lo(2), 0], [hi(1), hi(2)+1, 0]

      // ftmp1 = fx
      // rftmp1 = rfx
      // qfacetmp1 = qfacex

      transverse_correct_single(ybx, 0, 1,
                   qym_arr, ql_arr,
                   qyp_arr, qr_arr,
                   qaux_arr,
                   ftmp1_arr,
#ifdef RADIATION
                   rftmp1_arr,
#endif
                   qfacetmp1_arr,
                   areax_arr,
                   vol_arr,
                   hdt, hdtdx);

      // solve the final Riemann problem axross the y-interfaces

      final_riemann_solve(ybx, 1, flux1_arr,
#ifdef RADIATION
                          rad_flux1_arr,
#endif
                          qey_arr);
#endif // 2-d



#if AMREX_SPACEDIM == 3

      const amrex::Real hdt = 0.5*dt;

      const amrex::Real hdtdx = 0.5*dt/dx[0];
      const amrex::Real hdtdy = 0.5*dt/dx[1];
      const amrex::Real hdtdz = 0.5*dt/dx[2];

      const amrex::Real cdtdx = dt/dx[0]/3.0;
      const amrex::Real cdtdy = dt/dx[1]/3.0;
      const amrex::Real cdtdz = dt/dx[2]/3.0;

      // compute F^x
      // [lo(1), lo(2)-1, lo(3)-1], [hi(1)+1, hi(2)+1, hi(3)+1]
      const Box& cxbx = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,1,1)));

      // ftmp1 = fx
      // rftmp1 = rfx
      // qfacetmp1 = qgdnxv
      riemann_flux_and_state(cxbx,
                          qxm_arr, qxp_arr,
                          ftmp1_arr,
#ifdef RADIATION
                          rftmp1_arr,
#endif
                          qfacetmp1_arr,
                          qaux_arr, shk_arr,
                          0, false);


      // [lo(1), lo(2), lo(3)-1], [hi(1), hi(2)+1, hi(3)+1]
      const Box& tyxbx = amrex::grow(ybx, IntVect(AMREX_D_DECL(0,0,1)));

      auto qmyx_arr = resize_tile_fab(qmyx, tyxbx, NQ);
      auto qpyx_arr = resize_tile_fab(qpyx, tyxbx, NQ);

      // ftmp1 = fx
      // rftmp1 = rfx
      // qfacetmp1 = qfacex
      transverse_correct_single(tyxbx, 0, 1,
                   qym_arr, qmyx_arr,
                   qyp_arr, qpyx_arr,
                   qaux_arr,
                   ftmp1_arr,
#ifdef RADIATION
                   rftmp1_arr,
#endif
                   qfacetmp1_arr,
                   hdt, cdtdx);
      reset_edge_state_thermo(tyxbx, qmyx.array());

      reset_edge_state_thermo(tyxbx, qpyx.array());

      // [lo(1), lo(2)-1, lo(3)], [hi(1), hi(2)+1, hi(3)+1]
      const Box& tzxbx = amrex::grow(zbx, IntVect(AMREX_D_DECL(0,1,0)));

      auto qmzx_arr = resize_tile_fab(qmzx, tzxbx, NQ);
      auto qpzx_arr = resize_tile_fab(qpzx, tzxbx, NQ);

      transverse_correct_single(tzxbx, 0, 2,
                   qzm_arr, qmzx_arr,
                   qzp_arr, qpzx_arr,
                   qaux_arr,
                   ftmp1_arr,
#ifdef RADIATION
                   rftmp1_arr,
#endif
                   qfacetmp1_arr,
                   hdt, cdtdx);
      reset_edge_state_thermo(tzxbx, qmzx.array());

      reset_edge_state_thermo(tzxbx, qpzx.array());

      // compute F^y
      // [lo(1)-1, lo(2), lo(3)-1], [hi(1)+1, hi(2)+1, hi(3)+1]
      const Box& cybx = amrex::grow(ybx, IntVect(AMREX_D_DECL(1,0,1)));

      // ftmp1 = fy
      // rftmp1 = rfy
      // qfacetmp1 = qfacey
      riemann_flux_and_state(cybx,
                          qym_arr, qyp_arr,
                          ftmp1_arr,
#ifdef RADIATION
                          rftmp1_arr,
#endif
                          qfacetmp1_arr,
                          qaux_arr, shk_arr,
                          1, false);
      // [lo(1), lo(2), lo(3)-1], [hi(1)+1, hi(2), lo(3)+1]
      const Box& txybx = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,0,1)));

      auto qmxy_arr = resize_tile_fab(qmxy, txybx, NQ);
      auto qpxy_arr = resize_tile_fab(qpxy, txybx, NQ);

      // ftmp1 = fy
      // rftmp1 = rfy
      // qfacetmp1 = qfacey
      transverse_correct_single(txybx, 1, 0,
                   qxm_arr, qmxy_arr,
                   qxp_arr, qpxy_arr,
                   qaux_arr,
                   ftmp1_arr,
#ifdef RADIATION
                   rftmp1_arr,
#endif
                   qfacetmp1_arr,
                   hdt, cdtdy);
      reset_edge_state_thermo(txybx, qmxy.array());

      reset_edge_state_thermo(txybx, qpxy.array());

      // [lo(1)-1, lo(2), lo(3)], [hi(1)+1, hi(2), lo(3)+1]
      const Box& tzybx = amrex::grow(zbx, IntVect(AMREX_D_DECL(1,0,0)));

      auto qmzy_arr = resize_tile_fab(qmzy, tzybx, NQ);
      auto qpzy_arr = resize_tile_fab(qpzy, tzybx, NQ);

      // ftmp1 = fy
      // rftmp1 = rfy
      // qfacetmp1 = qfacey
      transverse_correct_single(tzybx, 1, 2,
                   qzm_arr, qmzy_arr,
                   qzp_arr, qpzy_arr,
                   qaux_arr,
                   ftmp1_arr,
#ifdef RADIATION
                   rftmp1_arr,
#endif
                   qfacetmp1_arr,
                   hdt, cdtdy);
      reset_edge_state_thermo(tzybx, qmzy.array());

      reset_edge_state_thermo(tzybx, qpzy.array());

      // compute F^z
      // [lo(1)-1, lo(2)-1, lo(3)], [hi(1)+1, hi(2)+1, hi(3)+1]
      const Box& czbx = amrex::grow(zbx, IntVect(AMREX_D_DECL(1,1,0)));

      // ftmp1 = fz
      // rftmp1 = rfz
      // qfacetmp1 = qfacez
      riemann_flux_and_state(czbx,
                          qzm_arr, qzp_arr,
                          ftmp1_arr,
#ifdef RADIATION
                          rftmp1_arr,
#endif
                          qfacetmp1_arr,
                          qaux_arr, shk_arr,
                          2, false);
      // [lo(1)-1, lo(2)-1, lo(3)], [hi(1)+1, hi(2)+1, lo(3)]
      const Box& txzbx = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,1,0)));

      auto qmxz_arr = resize_tile_fab(qmxz, txzbx, NQ);
      auto qpxz_arr = resize_tile_fab(qpxz, txzbx, NQ);

      // ftmp1 = fz
      // rftmp1 = rfz
      // qfacetmp1 = qfacez
      transverse_correct_single(txzbx, 2, 0,
                   qxm_arr, qmxz_arr,
                   qxp_arr, qpxz_arr,
                   qaux_arr,
                   ftmp1_arr,
#ifdef RADIATION
                   rftmp1_arr,
#endif
                   qfacetmp1_arr,
                   hdt, cdtdz);
      reset_edge_state_thermo(txzbx, qmxz.array());

      reset_edge_state_thermo(txzbx, qpxz.array());

      // [lo(1)-1, lo(2), lo(3)], [hi(1)+1, hi(2)+1, lo(3)]
      const Box& tyzbx = amrex::grow(ybx, IntVect(AMREX_D_DECL(1,0,0)));

      auto qmyz_arr = resize_tile_fab(qmyz, tyzbx, NQ);
      auto qpyz_arr = resize_tile_fab(qpyz, tyzbx, NQ);

      // ftmp1 = fz
      // rftmp1 = rfz
      // qfacetmp1 = qfacez
      transverse_correct_single(tyzbx, 2, 1,
                   qym_arr, qmyz_arr,
                   qyp_arr, qpyz_arr,
                   qaux_arr,
                   ftmp1_arr,
#ifdef RADIATION
                   rftmp1_arr,
#endif
                   qfacetmp1_arr,
                   hdt, cdtdz);
      reset_edge_state_thermo(tyzbx, qmyz.array());

      reset_edge_state_thermo(tyzbx, qpyz.array());

      // we now have q?zx, q?yx, q?zy, q?xy, q?yz, q?xz

      //
      // Use qx?, q?yz, q?zy to compute final x-flux
      //

      // compute F^{y|z}
      // [lo(1)-1, lo(2), lo(3)], [hi(1)+1, hi(2)+1, hi(3)]
      const Box& cyzbx = amrex::grow(ybx, IntVect(AMREX_D_DECL(1,0,0)));

      // ftmp1 = fyz
      // rftmp1 = rfyz
      // qfacetmp1 = qfaceyz
      riemann_flux_and_state(cyzbx,
                          qmyz_arr, qpyz_arr,
                          ftmp1_arr,
#ifdef RADIATION
                          rftmp1_arr,
#endif
                          qfacetmp1_arr,
                          qaux_arr, shk_arr,
                          1, false);

      // compute F^{z|y}
      // [lo(1)-1, lo(2), lo(3)], [hi(1)+1, hi(2), hi(3)+1]
      const Box& czybx = amrex::grow(zbx, IntVect(AMREX_D_DECL(1,0,0)));

      // ftmp2 = fzy
      // rftmp2 = rfzy
      // qfacetmp2 = qfacezy
      riemann_flux_and_state(czybx,
                          qmzy_arr, qpzy_arr,
                          ftmp2_arr,
#ifdef RADIATION
                          rftmp2_arr,
#endif
                          qfacetmp2_arr,
                          qaux_arr, shk_arr,
                          2, false);

      // compute the corrected x interface states and fluxes
      // [lo(1), lo(2), lo(3)], [hi(1)+1, hi(2), hi(3)]

      transverse_correct_final(xbx, 0, 1, 2,
                  qxm_arr, ql_arr,
                  qxp_arr, qr_arr,
                  qaux_arr,
                  ftmp1_arr,
#ifdef RADIATION
                  rftmp1_arr,
#endif
                  ftmp2_arr,
#ifdef RADIATION
                  rftmp2_arr,
#endif
                  qfacetmp1_arr,
                  qfacetmp2_arr,
                  hdtdy, hdtdz);

      final_riemann_solve(xbx, 0, flux0_arr,
#ifdef RADIATION
                          rad_flux0_arr,
#endif
                          qex_arr);

      //
      // Use qy?, q?zx, q?xz to compute final y-flux
      //

      // compute F^{z|x}
      // [lo(1), lo(2)-1, lo(3)], [hi(1), hi(2)+1, hi(3)+1]
      const Box& czxbx = amrex::grow(zbx, IntVect(AMREX_D_DECL(0,1,0)));

      // ftmp1 = fzx
      // rftmp1 = rfzx
      // qfacetmp1 = qfacezx
      riemann_flux_and_state(czxbx,
                          qmzx_arr, qpzx_arr,
                          ftmp1_arr,
#ifdef RADIATION
                          rftmp1_arr,
#endif
                          qfacetmp1_arr,
                          qaux_arr, shk_arr,
                          2, false);

      // compute F^{x|z}
      // [lo(1), lo(2)-1, lo(3)], [hi(1)+1, hi(2)+1, hi(3)]
      const Box& cxzbx = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,1,0)));

      // ftmp2 = fxz
      // rftmp2 = rfxz
      // qfacetmp2 = qfacexz
      riemann_flux_and_state(cxzbx,
                          qmxz_arr, qpxz_arr,
                          ftmp2_arr,
#ifdef RADIATION
                          rftmp2_arr,
#endif
                          qfacetmp2_arr,
                          qaux_arr, shk_arr,
                          0, false);

      // Compute the corrected y interface states and fluxes
      // [lo(1), lo(2), lo(3)], [hi(1), hi(2)+1, hi(3)]

      transverse_correct_final(ybx, 1, 0, 2,
                  qym_arr, ql_arr,
                  qyp_arr, qr_arr,
                  qaux_arr,
                  ftmp2_arr,
#ifdef RADIATION
                  rftmp2_arr,
#endif
                  ftmp1_arr,
#ifdef RADIATION
                  rftmp1_arr,
#endif
                  qfacetmp2_arr,
                  qfacetmp1_arr,
                  hdtdx, hdtdz);

      // Compute the final F^y
      // [lo(1), lo(2), lo(3)], [hi(1), hi(2)+1, hi(3)]
      final_riemann_solve(ybx, 1, flux1_arr,
#ifdef RADIATION
                          rad_flux1_arr,
#endif
                          qey_arr);

      //
      // Use qz?, q?xy, q?yx to compute final z-flux
      //

      // compute F^{x|y}
      // [lo(1), lo(2), lo(3)-1], [hi(1)+1, hi(2), hi(3)+1]
      const Box& cxybx = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,0,1)));

      // ftmp1 = fxy
      // rftmp1 = rfxy
      // qfacetmp1 = qfacexy
      riemann_flux_and_state(cxybx,
                          qmxy_arr, qpxy_arr,
                          ftmp1_arr,
#ifdef RADIATION
                          rftmp1_arr,
#endif
                          qfacetmp1_arr,
                          qaux_arr, shk_arr,
                          0, false);

      // compute F^{y|x}
      // [lo(1), lo(2), lo(3)-1], [hi(1), hi(2)+dg(2), hi(3)+1]
      const Box& cyxbx = amrex::grow(ybx, IntVect(AMREX_D_DECL(0,0,1)));

      // ftmp2 = fyx
      // rftmp2 = rfyx
      // qfacetmp2 = qfaceyx
      riemann_flux_and_state(cyxbx,
                          qmyx_arr, qpyx_arr,
                          ftmp2_arr,
#ifdef RADIATION
                          rftmp2_arr,
#endif
                          qfacetmp2_arr,
                          qaux_arr, shk_arr,
                          1, false);

      // compute the corrected z interface states and fluxes
      // [lo(1), lo(2), lo(3)], [hi(1), hi(2), hi(3)+1]

      transverse_correct_final(zbx, 2, 0, 1,
                  qzm_arr, ql_arr,
                  qzp_arr, qr_arr,
                  qaux_arr,
                  ftmp1_arr,
#ifdef RADIATION
                  rftmp1_arr,
#endif
                  ftmp2_arr,
#ifdef RADIATION
                  rftmp2_arr,
#endif
                  qfacetmp1_arr,
                  qfacetmp2_arr,
                  hdtdx, hdtdy);

      // compute the final z fluxes F^z
      // [lo(1), lo(2), lo(3)], [hi(1), hi(2), hi(3)+1]

      final_riemann_solve(zbx, 2, flux2_arr,
#ifdef RADIATION
                          rad_flux2_arr,
#endif
                          qez_arr);

#endif // 3-d



      // clean the fluxes

      for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

          const Box& nbx = amrex::surroundingNodes(bx, idir);

          Array4<Real> const flux_arr = (flux[idir]).array();

          // Zero out shock and temp fluxes -- these are physically meaningless here
          amrex::ParallelFor(nbx,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              flux_arr(i,j,k,UTEMP) = 0.e0;
#ifdef SHOCK_VAR
              flux_arr(i,j,k,USHOCK) = 0.e0;
#endif
#ifdef NSE_NET
              flux_arr(i,j,k,UMU_P) = 0.e0;
              flux_arr(i,j,k,UMU_N) = 0.e0;
#endif
          });

          apply_av(nbx, idir, div_arr, U_old_arr, flux_arr);

#ifdef RADIATION
          Array4<Real> const rad_flux_arr = (rad_flux[idir]).array();
          Array4<Real const> const Erin_arr = Erborder.array(mfi);

          apply_av_rad(nbx, idir, div_arr, Erin_arr, rad_flux_arr);
#endif

          if (limit_fluxes_on_small_dens == 1) {
              limit_hydro_fluxes_on_small_dens
                  (nbx, idir,
                   Sborder.array(mfi),
                   volume.array(mfi),
                   flux[idir].array(),
                   area[idir].array(mfi),
                   dt);
          }

          normalize_species_fluxes(nbx, flux_arr);

      }



      // conservative update
      Array4<Real> const update_arr = S_new.array(mfi);

      Array4<Real> const flx_arr = (flux[0]).array();
      Array4<Real> const qx_arr = (qe[0]).array();

#if AMREX_SPACEDIM >= 2
      Array4<Real> const fly_arr = (flux[1]).array();
      Array4<Real> const qy_arr = (qe[1]).array();
#endif

#if AMREX_SPACEDIM == 3
      Array4<Real> const flz_arr = (flux[2]).array();
      Array4<Real> const qz_arr = (qe[2]).array();
#endif

      cons_update_hydro(bx,
                   update_arr,
                   flx_arr, qx_arr,
#if AMREX_SPACEDIM >= 2
                   fly_arr, qy_arr,
#endif
#if AMREX_SPACEDIM == 3
                   flz_arr, qz_arr,
#endif
                   dt);


#ifdef HYBRID_MOMENTUM
      auto dx_arr = geom.CellSizeArray();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

          GpuArray<Real, 3> loc;

          position(i, j, k, geomdata, loc);

          for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
              loc[dir] -= problem::center[dir];

          Real R = amrex::max(std::sqrt(loc[0] * loc[0] + loc[1] * loc[1]),
                              std::numeric_limits<Real>::min());
          Real RInv = 1.0_rt / R;

          update_arr(i,j,k,UMR) -= dt * (loc[0] * RInv) * (qx_arr(i+1,j,k,IFCPRES) - qx_arr(i,j,k,IFCPRES)) / dx_arr[0];
#if AMREX_SPACEDIM >= 2
          update_arr(i,j,k,UMR) -= dt * (loc[1] * RInv) * (qy_arr(i,j+1,k,IFCPRES) - qy_arr(i,j,k,IFCPRES)) / dx_arr[1];
#endif
      });
#endif

#ifdef RADIATION
      ctu_rad_cons_update(bx,
                     update_arr,
                     Erborder.array(mfi),
                     Er_new.array(mfi),
                     (rad_flux[0]).array(),
                     (qe[0]).array(),
                     (area[0]).array(mfi),
#if AMREX_SPACEDIM >= 2
                     (rad_flux[1]).array(),
                     (qe[1]).array(),
                     (area[1]).array(mfi),
#endif
#if AMREX_SPACEDIM == 3
                     (rad_flux[2]).array(),
                     (qe[2]).array(),
                     (area[2]).array(mfi),
#endif
                     priv_nstep_fsp,
                     volume.array(mfi),
                     dt);

      nstep_fsp = std::max(nstep_fsp, priv_nstep_fsp);
#endif


      for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

        const Box& nbx = amrex::surroundingNodes(bx, idir);

        Array4<Real> const flux_arr = (flux[idir]).array();
        Array4<Real const> const area_arr = (area[idir]).array(mfi);

        scale_flux(nbx, flux_arr, area_arr, dt);

#ifdef RADIATION
        Array4<Real> const rad_flux_arr = (rad_flux[idir]).array();
        scale_rad_flux(nbx, rad_flux_arr, area_arr, dt);
#endif

#if AMREX_SPACEDIM <= 2
        // get the scaled radial pressure -- we need to treat this specially

        if (idir == 0 && !mom_flux_has_p(0, 0, coord)) {
            Array4<Real> pradial_fab = pradial.array();

            amrex::ParallelFor(nbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                pradial_fab(i,j,k) = qex_arr(i,j,k,IFCPRES) * dt;
            });
        }
#endif
        // Store the fluxes from this advance. For simplified SDC integration we
        // only need to do this on the last iteration.

        bool add_fluxes = true;

        if (time_integration_method == SimplifiedSpectralDeferredCorrections &&
            sdc_iteration != sdc_iters - 1) {
            add_fluxes = false;
        }

        if (add_fluxes) {

            Array4<Real> const flux_fab = (flux[idir]).array();
            Array4<Real> fluxes_fab = (*fluxes[idir]).array(mfi);

            amrex::ParallelFor(mfi.nodaltilebox(idir), NUM_STATE,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                fluxes_fab(i,j,k,n) += flux_fab(i,j,k,n);
            });

#ifdef RADIATION
            Array4<Real> const rad_flux_fab = (rad_flux[idir]).array();
            Array4<Real> rad_fluxes_fab = (*rad_fluxes[idir]).array(mfi);

            amrex::ParallelFor(mfi.nodaltilebox(idir), Radiation::nGroups,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                rad_fluxes_fab(i,j,k,n) += rad_flux_fab(i,j,k,n);
            });
#endif

#if AMREX_SPACEDIM <= 2
            if (idir == 0 && !mom_flux_has_p(0, 0, coord)) {
                Array4<Real> pradial_fab = pradial.array();
                Array4<Real> P_radial_fab = P_radial.array(mfi);

                amrex::ParallelFor(mfi.nodaltilebox(0),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    P_radial_fab(i,j,k,0) += pradial_fab(i,j,k,0);
                });
            }

#endif

        } // add_fluxes

        Array4<Real> const flux_fab = (flux[idir]).array();
        Array4<Real> mass_fluxes_fab = (*mass_fluxes[idir]).array(mfi);

        amrex::ParallelFor(mfi.nodaltilebox(idir),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // This is a copy, not an add, since we need mass_fluxes to be
            // only this subcycle's data when we evaluate the gravitational
            // forces.

            mass_fluxes_fab(i,j,k,0) = flux_fab(i,j,k,URHO);
        });

      } // idir loop

#ifdef AMREX_USE_GPU
      if (hydrosim::hydro_memory_footprint_ratio > 0.0) {

          if (hydro_tile_size_has_been_tuned == 0) {

              // Keep a running record of the largest box we've encountered.

              largest_box_from_hydro_tile_size_tuning = amrex::max(largest_box_from_hydro_tile_size_tuning,
                                                                   mfi.validbox().numPts());

              // If we're tuning the hydro tile size during this timestep, we will record
              // the total amount of additional memory allocated, relative to the size of S_new.
              // Then we will reset the tile size so that it is no larger than the requested
              // memory footprint.

              // This could be generalized in the future to operate with more granularity
              // than the MFIter loop boundary. We could have potential synchronization
              // points prior to each of the above kernel launches.

              maximum_tile_size[0] = amrex::max(maximum_tile_size[0], bx.bigEnd(0) - mfi.validbox().smallEnd(0) + 1);
#if AMREX_SPACEDIM >= 2
              maximum_tile_size[1] = amrex::max(maximum_tile_size[1], bx.bigEnd(1) - mfi.validbox().smallEnd(1) + 1);
#endif
#if AMREX_SPACEDIM ==3
              maximum_tile_size[2] = amrex::max(maximum_tile_size[2], bx.bigEnd(2) - mfi.validbox().smallEnd(2) + 1);
#endif

              if (fab_size >= hydrosim::hydro_memory_footprint_ratio * mf_size) {
                  // If we reached the memory limit, set the tile size to the current
                  // maximum tile size.
                  Gpu::synchronize();
                  hydro_tile_size = maximum_tile_size;
                  hydro_tile_size_has_been_tuned = 1;
              }
              else if (mfi.tileIndex() == mfi.length() - 1) {
                  // If we reached the last tile and we haven't gone over the
                  // memory limit, effectively disable tiling.
                  hydro_tile_size[0] = 1024;
#if AMREX_SPACEDIM >= 2
                  hydro_tile_size[1] = 1024;
#endif
#if AMREX_SPACEDIM == 3
                  hydro_tile_size[2] = 1024;
#endif
                  hydro_tile_size_has_been_tuned = 1;
              }

          }
          else {
              // If we have already tuned the parameter, then synchronize each time the
              // outstanding number of active bytes is larger than our ratio.

              if (fab_size >= hydrosim::hydro_memory_footprint_ratio * mf_size) {
                  Gpu::synchronize();

                  // Reset the counter for the next sequence of tiles.
                  fab_size = 0;
              }
          }
      }
#endif

    } // MFIter loop

  } // OMP loop

#ifdef RADIATION
  if (radiation->verbose>=1) {
      amrex::Real llevel = level;
#ifdef BL_LAZY
    Lazy::QueueReduction( [=] () mutable {
#endif
       ParallelDescriptor::ReduceIntMax(nstep_fsp, ParallelDescriptor::IOProcessorNumber());
       if (ParallelDescriptor::IOProcessor() && nstep_fsp > 0) {
         std::cout << "Radiation f-space advection on level " << llevel
                   << " takes as many as " << nstep_fsp;
         if (nstep_fsp == 1) {
           std::cout<< " substep.\n";
         }
         else {
           std::cout<< " substeps.\n";
         }
       }
#ifdef BL_LAZY
     });
#endif
  }
#endif

  // Check for small/negative densities and X > 1 or X < 0.

  status = check_for_negative_density();

  if (status.success == false) {
      return status;
  }

  // Sync up state after hydro source.

  clean_state(
#ifdef MHD
               Bx_new, By_new, Bz_new,
#endif
               S_new, time + dt, 0);

  // Check for NaN's.

  check_for_nan(S_new);

#ifdef GRAVITY
  // Must define new value of "center" after advecting on the grid

  if (moving_center == 1) {
      define_new_center(S_new, time);
  }
#endif

  // Perform reflux (for non-subcycling advances).

  if (parent->subcyclingMode() == "None") {
      if (do_reflux == 1) {
          FluxRegCrseInit();
          FluxRegFineAdd();
      }
  }

  if (verbose) {
      amrex::Print() << "... Leaving construct_ctu_hydro_source() on level " << level << std::endl << std::endl;
  }

  if (verbose > 0)
    {
      const int IOProc = ParallelDescriptor::IOProcessorNumber();
      amrex::Real run_time = ParallelDescriptor::second() - strt_time;
      amrex::Real llevel = level;

#ifdef BL_LAZY
      Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::construct_ctu_hydro_source() time = " << run_time
                       << " on level " << llevel << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }

#endif

  return status;
}
