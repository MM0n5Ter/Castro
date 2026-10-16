#include <HydroSim.H>
#include <HydroSim_util.H>



#include <advection_util.H>

using namespace amrex;

advance_status
HydroSim::assemble_ctu_hydro_source(Real time, Real dt)  // NOLINT(readability-convert-member-functions-to-static)
{
  amrex::ignore_unused(time);
  amrex::ignore_unused(dt);

  advance_status status {};

  if (!do_hydro) {
      return status;
  }


  BL_PROFILE("HydroSim::assemble_ctu_hydro_source()");

  const Real strt_time = ParallelDescriptor::second();

  // this constructs the hydrodynamic source (essentially the flux
  // divergence) using the CTU framework for unsplit hydrodynamics

  if (verbose) {
      amrex::Print() << "... Entering assemble_ctu_hydro_source() on level " << level << std::endl << std::endl;
  }


#if AMREX_SPACEDIM <= 2
  int coord = geom.Coord();
#endif

#if AMREX_SPACEDIM >= 2
  const Real *dx = geom.CellSize();
#endif

  MultiFab& S_new = get_new_data(State_Type);


  // we will treat the hydro source as any other source term


  // Whether to store the fluxes from this advance. For simplified SDC
  // integration we only need to do this on the last iteration.

  const bool add_fluxes = time_integration_method != SimplifiedSpectralDeferredCorrections ||
                          sdc_iteration == sdc_iters - 1;

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

       if (tile_size_autotuned == 0) {
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
           if (mfi.validbox().numPts() > autotuned_max_box_cells) {
               tile_size_autotuned = 0;
           }

           // Also, sum up the number of bytes in the state data.
           mf_size += S_new[mfi].nBytes();
       }
   }
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
  {


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
    FArrayBox qfacetmp1(The_Async_Arena()), qfacetmp2(The_Async_Arena());
    FArrayBox ql(The_Async_Arena()), qr(The_Async_Arena());
#endif
    Vector<FArrayBox> flux, qe;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        flux.push_back(FArrayBox(The_Async_Arena()));
        qe.push_back(FArrayBox(The_Async_Arena()));
    }

#if AMREX_SPACEDIM <= 2
    FArrayBox pradial(The_Async_Arena());
#endif
#if AMREX_SPACEDIM == 3
    FArrayBox qym_tx(The_Async_Arena()), qyp_tx(The_Async_Arena());
    FArrayBox qzm_tx(The_Async_Arena()), qzp_tx(The_Async_Arena());
    FArrayBox qxm_ty(The_Async_Arena()), qxp_ty(The_Async_Arena());
    FArrayBox qzm_ty(The_Async_Arena()), qzp_ty(The_Async_Arena());
    FArrayBox qxm_tz(The_Async_Arena()), qxp_tz(The_Async_Arena());
    FArrayBox qym_tz(The_Async_Arena()), qyp_tz(The_Async_Arena());
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

      // note: we won't store the passives in q, so we'll compute their
      // primitive versions on demand as needed
      Array4<Real> const q_arr = resize_tile_fab(q, qbx, NQTHERM);

      Array4<Real> const qaux_arr = resize_tile_fab(qaux, qbx, NQAUX);

      Array4<Real const> const U_old_arr = Sborder.array(mfi);

      Array4<Real> const rho_inv_arr = resize_tile_fab(rho_inv, qbx3, 1);

      amrex::ParallelFor(qbx3,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
          rho_inv_arr(i,j,k) = 1.0 / U_old_arr(i,j,k,URHO);
      });

      hydro::CoupledFields coupled{};

      hydro::ctoprim(qbx, time, U_old_arr, coupled, q_arr, qaux_arr);

#if AMREX_SPACEDIM == 2
      Array4<Real const> const areax_arr = area[0].array(mfi);
      Array4<Real const> const areay_arr = area[1].array(mfi);
      Array4<Real> const vol_arr = volume.array(mfi);
#endif

      Array<Array4<Real const>, AMREX_SPACEDIM> dloga_arrs{};
#if AMREX_SPACEDIM < 3
      for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
        dloga_arrs[idir] = dLogArea[idir].array(mfi);
      }
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

      const Array<Array4<Real>, AMREX_SPACEDIM> qm_arrs{AMREX_D_DECL(qxm_arr, qym_arr, qzm_arr)};
      const Array<Array4<Real>, AMREX_SPACEDIM> qp_arrs{AMREX_D_DECL(qxp_arr, qyp_arr, qzp_arr)};

      if (ppm_type == 0) {

        plm_interface_states(obx, bx,
                       U_old_arr, rho_inv_arr,
                       q_arr,
                       qaux_arr,
                       src_q_arr,
                       qm_arrs, qp_arrs, dloga_arrs,
                       dt);

      } else {


        ppm_interface_states(obx, bx,
                       U_old_arr, rho_inv_arr,
                       q_arr, qaux_arr, src_q_arr,
                       qm_arrs, qp_arrs, dloga_arrs,
                       dt);

      }

      auto div_arr = resize_tile_fab(div, obx, 1);

      // compute divu -- we'll use this later when doing the artificial viscosity
      divu(obx, q_arr, div_arr);

      Array4<Real> const flux0_arr = resize_tile_fab(flux[0], gxbx, NUM_STATE);
      auto qex_arr = resize_tile_fab(qe[0], gxbx, NIFC);
      InterfaceOutput x_out{flux0_arr, qex_arr};


#if AMREX_SPACEDIM >= 2
      Array4<Real> const flux1_arr = resize_tile_fab(flux[1], gybx, NUM_STATE);
      auto qey_arr = resize_tile_fab(qe[1], gybx, NIFC);
      InterfaceOutput y_out{flux1_arr, qey_arr};

#endif

#if AMREX_SPACEDIM == 3
      Array4<Real> const flux2_arr = resize_tile_fab(flux[2], gzbx, NUM_STATE);
      auto qez_arr = resize_tile_fab(qe[2], gzbx, NIFC);
      InterfaceOutput z_out{flux2_arr, qez_arr};

#endif

#if AMREX_SPACEDIM <= 2
      if (!Geom().IsCartesian()) {
          pradial.resize(xbx, 1);
      }
      fab_size += pradial.nBytes();
#endif


#if AMREX_SPACEDIM == 1


      // compute the fluxes through the x-interface

      riemann_flux_and_state(xbx,
                          qxm_arr, qxp_arr,
                          x_out,
                          qaux_arr,
                          shk_arr,
                          0, FaceComponents::Reduced);

#endif // 1-d



#if AMREX_SPACEDIM >= 2
      auto ftmp1_arr = resize_tile_fab(ftmp1, obx, NUM_STATE);
      auto ftmp2_arr = resize_tile_fab(ftmp2, obx, NUM_STATE);


      auto qfacetmp1_arr = resize_tile_fab(qfacetmp1, obx, NIFC);
      InterfaceOutput tmp1_out{ftmp1_arr, qfacetmp1_arr};

#if AMREX_SPACEDIM == 3
      auto qfacetmp2_arr = resize_tile_fab(qfacetmp2, obx, NIFC);
      InterfaceOutput tmp2_out{ftmp2_arr, qfacetmp2_arr};

      // the transverse corrections read back what these solves store
      TransverseFaces tmp1_faces{ftmp1_arr, qfacetmp1_arr};
      TransverseFaces tmp2_faces{ftmp2_arr, qfacetmp2_arr};
#else
      // in 2-d, the second transverse solve writes the y interface
      // state directly
      InterfaceOutput tmp2_out{ftmp2_arr, qey_arr};

      // the transverse corrections read back what these solves store
      TransverseFaces tmp1_faces{ftmp1_arr, qfacetmp1_arr, areax_arr, vol_arr};
      TransverseFaces tmp2_faces{ftmp2_arr, qey_arr, areay_arr, vol_arr};
#endif


      auto ql_arr = resize_tile_fab(ql, obx, NQ);
      auto qr_arr = resize_tile_fab(qr, obx, NQ);

      // Make the final interface states in ql/qr thermodynamically
      // consistent, add the reaction source for simplified SDC, and
      // solve the Riemann problem across the idir faces of nbx.

      auto final_riemann_solve = [&] (const Box& nbx, int idir,
                                      InterfaceOutput const& out) {
          hydro::recompute_face_thermo(nbx, ql_arr);

          hydro::recompute_face_thermo(nbx, qr_arr);


          riemann_flux_and_state(nbx,
                                 ql_arr, qr_arr,
                                 out,
                                 qaux_arr, shk_arr,
                                 idir, FaceComponents::Reduced);
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
                          tmp1_out,
                          qaux_arr, shk_arr,
                          0, FaceComponents::Reduced);

      // compute F^y
      // [lo(1)-1, lo(2), 0], [hi(1)+1, hi(2)+1, 0]
      const Box& cybx = amrex::grow(ybx, IntVect(AMREX_D_DECL(1,0,0)));

      // ftmp2 = fy
      // rftmp2 = rfy
      riemann_flux_and_state(cybx,
                          qym_arr, qyp_arr,
                          tmp2_out,
                          qaux_arr, shk_arr,
                          1, FaceComponents::Reduced);

      // add the transverse flux difference in y to the x states
      // [lo(1), lo(2), 0], [hi(1)+1, hi(2), 0]

      // ftmp2 = fy
      // rftmp2 = rfy
      transverse_correct_single(xbx, 1, 0,
                   qxm_arr, ql_arr,
                   qxp_arr, qr_arr,
                   qaux_arr,
                   tmp2_faces,
                   hdt, hdtdy);

      // solve the final Riemann problem axross the x-interfaces

      final_riemann_solve(xbx, 0, x_out);

      // add the transverse flux difference in x to the y states
      // [lo(1), lo(2), 0], [hi(1), hi(2)+1, 0]

      // ftmp1 = fx
      // rftmp1 = rfx
      // qfacetmp1 = qfacex

      transverse_correct_single(ybx, 0, 1,
                   qym_arr, ql_arr,
                   qyp_arr, qr_arr,
                   qaux_arr,
                   tmp1_faces,
                   hdt, hdtdx);

      // solve the final Riemann problem axross the y-interfaces

      final_riemann_solve(ybx, 1, y_out);
#endif // 2-d



#if AMREX_SPACEDIM == 3

      const amrex::Real hdt = 0.5*dt;

      const amrex::Real hdtdx = 0.5*dt/dx[0];
      const amrex::Real hdtdy = 0.5*dt/dx[1];
      const amrex::Real hdtdz = 0.5*dt/dx[2];

      const amrex::Real coef_dtdx = dt/dx[0]/3.0;
      const amrex::Real coef_dtdy = dt/dx[1]/3.0;
      const amrex::Real coef_dtdz = dt/dx[2]/3.0;

      // compute F^x
      // [lo(1), lo(2)-1, lo(3)-1], [hi(1)+1, hi(2)+1, hi(3)+1]
      const Box& cxbx = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,1,1)));

      // ftmp1 = fx
      // rftmp1 = rfx
      // qfacetmp1 = qgdnxv
      riemann_flux_and_state(cxbx,
                          qxm_arr, qxp_arr,
                          tmp1_out,
                          qaux_arr, shk_arr,
                          0, FaceComponents::Reduced);


      // [lo(1), lo(2), lo(3)-1], [hi(1), hi(2)+1, hi(3)+1]
      const Box& ybx_tx = amrex::grow(ybx, IntVect(AMREX_D_DECL(0,0,1)));

      auto qym_tx_arr = resize_tile_fab(qym_tx, ybx_tx, NQ);
      auto qyp_tx_arr = resize_tile_fab(qyp_tx, ybx_tx, NQ);

      // ftmp1 = fx
      // rftmp1 = rfx
      // qfacetmp1 = qfacex
      transverse_correct_single(ybx_tx, 0, 1,
                   qym_arr, qym_tx_arr,
                   qyp_arr, qyp_tx_arr,
                   qaux_arr,
                   tmp1_faces,
                   hdt, coef_dtdx);
      hydro::recompute_face_thermo(ybx_tx, qym_tx.array());

      hydro::recompute_face_thermo(ybx_tx, qyp_tx.array());

      // [lo(1), lo(2)-1, lo(3)], [hi(1), hi(2)+1, hi(3)+1]
      const Box& zbx_tx = amrex::grow(zbx, IntVect(AMREX_D_DECL(0,1,0)));

      auto qzm_tx_arr = resize_tile_fab(qzm_tx, zbx_tx, NQ);
      auto qzp_tx_arr = resize_tile_fab(qzp_tx, zbx_tx, NQ);

      transverse_correct_single(zbx_tx, 0, 2,
                   qzm_arr, qzm_tx_arr,
                   qzp_arr, qzp_tx_arr,
                   qaux_arr,
                   tmp1_faces,
                   hdt, coef_dtdx);
      hydro::recompute_face_thermo(zbx_tx, qzm_tx.array());

      hydro::recompute_face_thermo(zbx_tx, qzp_tx.array());

      // compute F^y
      // [lo(1)-1, lo(2), lo(3)-1], [hi(1)+1, hi(2)+1, hi(3)+1]
      const Box& cybx = amrex::grow(ybx, IntVect(AMREX_D_DECL(1,0,1)));

      // ftmp1 = fy
      // rftmp1 = rfy
      // qfacetmp1 = qfacey
      riemann_flux_and_state(cybx,
                          qym_arr, qyp_arr,
                          tmp1_out,
                          qaux_arr, shk_arr,
                          1, FaceComponents::Reduced);
      // [lo(1), lo(2), lo(3)-1], [hi(1)+1, hi(2), lo(3)+1]
      const Box& xbx_ty = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,0,1)));

      auto qxm_ty_arr = resize_tile_fab(qxm_ty, xbx_ty, NQ);
      auto qxp_ty_arr = resize_tile_fab(qxp_ty, xbx_ty, NQ);

      // ftmp1 = fy
      // rftmp1 = rfy
      // qfacetmp1 = qfacey
      transverse_correct_single(xbx_ty, 1, 0,
                   qxm_arr, qxm_ty_arr,
                   qxp_arr, qxp_ty_arr,
                   qaux_arr,
                   tmp1_faces,
                   hdt, coef_dtdy);
      hydro::recompute_face_thermo(xbx_ty, qxm_ty.array());

      hydro::recompute_face_thermo(xbx_ty, qxp_ty.array());

      // [lo(1)-1, lo(2), lo(3)], [hi(1)+1, hi(2), lo(3)+1]
      const Box& zbx_ty = amrex::grow(zbx, IntVect(AMREX_D_DECL(1,0,0)));

      auto qzm_ty_arr = resize_tile_fab(qzm_ty, zbx_ty, NQ);
      auto qzp_ty_arr = resize_tile_fab(qzp_ty, zbx_ty, NQ);

      // ftmp1 = fy
      // rftmp1 = rfy
      // qfacetmp1 = qfacey
      transverse_correct_single(zbx_ty, 1, 2,
                   qzm_arr, qzm_ty_arr,
                   qzp_arr, qzp_ty_arr,
                   qaux_arr,
                   tmp1_faces,
                   hdt, coef_dtdy);
      hydro::recompute_face_thermo(zbx_ty, qzm_ty.array());

      hydro::recompute_face_thermo(zbx_ty, qzp_ty.array());

      // compute F^z
      // [lo(1)-1, lo(2)-1, lo(3)], [hi(1)+1, hi(2)+1, hi(3)+1]
      const Box& czbx = amrex::grow(zbx, IntVect(AMREX_D_DECL(1,1,0)));

      // ftmp1 = fz
      // rftmp1 = rfz
      // qfacetmp1 = qfacez
      riemann_flux_and_state(czbx,
                          qzm_arr, qzp_arr,
                          tmp1_out,
                          qaux_arr, shk_arr,
                          2, FaceComponents::Reduced);
      // [lo(1)-1, lo(2)-1, lo(3)], [hi(1)+1, hi(2)+1, lo(3)]
      const Box& xbx_tz = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,1,0)));

      auto qxm_tz_arr = resize_tile_fab(qxm_tz, xbx_tz, NQ);
      auto qxp_tz_arr = resize_tile_fab(qxp_tz, xbx_tz, NQ);

      // ftmp1 = fz
      // rftmp1 = rfz
      // qfacetmp1 = qfacez
      transverse_correct_single(xbx_tz, 2, 0,
                   qxm_arr, qxm_tz_arr,
                   qxp_arr, qxp_tz_arr,
                   qaux_arr,
                   tmp1_faces,
                   hdt, coef_dtdz);
      hydro::recompute_face_thermo(xbx_tz, qxm_tz.array());

      hydro::recompute_face_thermo(xbx_tz, qxp_tz.array());

      // [lo(1)-1, lo(2), lo(3)], [hi(1)+1, hi(2)+1, lo(3)]
      const Box& ybx_tz = amrex::grow(ybx, IntVect(AMREX_D_DECL(1,0,0)));

      auto qym_tz_arr = resize_tile_fab(qym_tz, ybx_tz, NQ);
      auto qyp_tz_arr = resize_tile_fab(qyp_tz, ybx_tz, NQ);

      // ftmp1 = fz
      // rftmp1 = rfz
      // qfacetmp1 = qfacez
      transverse_correct_single(ybx_tz, 2, 1,
                   qym_arr, qym_tz_arr,
                   qyp_arr, qyp_tz_arr,
                   qaux_arr,
                   tmp1_faces,
                   hdt, coef_dtdz);
      hydro::recompute_face_thermo(ybx_tz, qym_tz.array());

      hydro::recompute_face_thermo(ybx_tz, qyp_tz.array());

      // we now have q?zx, q?yx, q?zy, q?xy, q?yz, q?xz

      //
      // Use qx?, q?yz, q?zy to compute final x-flux
      //

      // compute F^{y|z}
      // [lo(1)-1, lo(2), lo(3)], [hi(1)+1, hi(2)+1, hi(3)]
      const Box& ybx_cz = amrex::grow(ybx, IntVect(AMREX_D_DECL(1,0,0)));

      // ftmp1 = fyz
      // rftmp1 = rfyz
      // qfacetmp1 = qfaceyz
      riemann_flux_and_state(ybx_cz,
                          qym_tz_arr, qyp_tz_arr,
                          tmp1_out,
                          qaux_arr, shk_arr,
                          1, FaceComponents::Reduced);

      // compute F^{z|y}
      // [lo(1)-1, lo(2), lo(3)], [hi(1)+1, hi(2), hi(3)+1]
      const Box& zbx_cy = amrex::grow(zbx, IntVect(AMREX_D_DECL(1,0,0)));

      // ftmp2 = fzy
      // rftmp2 = rfzy
      // qfacetmp2 = qfacezy
      riemann_flux_and_state(zbx_cy,
                          qzm_ty_arr, qzp_ty_arr,
                          tmp2_out,
                          qaux_arr, shk_arr,
                          2, FaceComponents::Reduced);

      // compute the corrected x interface states and fluxes
      // [lo(1), lo(2), lo(3)], [hi(1)+1, hi(2), hi(3)]

      transverse_correct_final(xbx, 0, 1, 2,
                  qxm_arr, ql_arr,
                  qxp_arr, qr_arr,
                  qaux_arr,
                  tmp1_faces,
                  tmp2_faces,
                  hdtdy, hdtdz);

      final_riemann_solve(xbx, 0, x_out);

      //
      // Use qy?, q?zx, q?xz to compute final y-flux
      //

      // compute F^{z|x}
      // [lo(1), lo(2)-1, lo(3)], [hi(1), hi(2)+1, hi(3)+1]
      const Box& zbx_cx = amrex::grow(zbx, IntVect(AMREX_D_DECL(0,1,0)));

      // ftmp1 = fzx
      // rftmp1 = rfzx
      // qfacetmp1 = qfacezx
      riemann_flux_and_state(zbx_cx,
                          qzm_tx_arr, qzp_tx_arr,
                          tmp1_out,
                          qaux_arr, shk_arr,
                          2, FaceComponents::Reduced);

      // compute F^{x|z}
      // [lo(1), lo(2)-1, lo(3)], [hi(1)+1, hi(2)+1, hi(3)]
      const Box& xbx_cz = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,1,0)));

      // ftmp2 = fxz
      // rftmp2 = rfxz
      // qfacetmp2 = qfacexz
      riemann_flux_and_state(xbx_cz,
                          qxm_tz_arr, qxp_tz_arr,
                          tmp2_out,
                          qaux_arr, shk_arr,
                          0, FaceComponents::Reduced);

      // Compute the corrected y interface states and fluxes
      // [lo(1), lo(2), lo(3)], [hi(1), hi(2)+1, hi(3)]

      transverse_correct_final(ybx, 1, 0, 2,
                  qym_arr, ql_arr,
                  qyp_arr, qr_arr,
                  qaux_arr,
                  tmp2_faces,
                  tmp1_faces,
                  hdtdx, hdtdz);

      // Compute the final F^y
      // [lo(1), lo(2), lo(3)], [hi(1), hi(2)+1, hi(3)]
      final_riemann_solve(ybx, 1, y_out);

      //
      // Use qz?, q?xy, q?yx to compute final z-flux
      //

      // compute F^{x|y}
      // [lo(1), lo(2), lo(3)-1], [hi(1)+1, hi(2), hi(3)+1]
      const Box& xbx_cy = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,0,1)));

      // ftmp1 = fxy
      // rftmp1 = rfxy
      // qfacetmp1 = qfacexy
      riemann_flux_and_state(xbx_cy,
                          qxm_ty_arr, qxp_ty_arr,
                          tmp1_out,
                          qaux_arr, shk_arr,
                          0, FaceComponents::Reduced);

      // compute F^{y|x}
      // [lo(1), lo(2), lo(3)-1], [hi(1), hi(2)+dg(2), hi(3)+1]
      const Box& ybx_cx = amrex::grow(ybx, IntVect(AMREX_D_DECL(0,0,1)));

      // ftmp2 = fyx
      // rftmp2 = rfyx
      // qfacetmp2 = qfaceyx
      riemann_flux_and_state(ybx_cx,
                          qym_tx_arr, qyp_tx_arr,
                          tmp2_out,
                          qaux_arr, shk_arr,
                          1, FaceComponents::Reduced);

      // compute the corrected z interface states and fluxes
      // [lo(1), lo(2), lo(3)], [hi(1), hi(2), hi(3)+1]

      transverse_correct_final(zbx, 2, 0, 1,
                  qzm_arr, ql_arr,
                  qzp_arr, qr_arr,
                  qaux_arr,
                  tmp1_faces,
                  tmp2_faces,
                  hdtdx, hdtdy);

      // compute the final z fluxes F^z
      // [lo(1), lo(2), lo(3)], [hi(1), hi(2), hi(3)+1]

      final_riemann_solve(zbx, 2, z_out);

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
          });

          apply_art_visc(nbx, idir, div_arr, U_old_arr, flux_arr);


          if (limit_fluxes_on_small_dens == 1) {
              hydro::limit_fluxes_for_density_floor
                  (nbx, idir,
                   Sborder.array(mfi),
                   volume.array(mfi),
                   flux[idir].array(),
                   area[idir].array(mfi),
                   dt, hydro::FluxForm::PerUnitArea);
          }

          hydro::normalize_species_fluxes(nbx, flux_arr);

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
                   {AMREX_D_DECL(flx_arr, fly_arr, flz_arr)},
                   {AMREX_D_DECL(qx_arr, qy_arr, qz_arr)},
                   dt);





      for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

        const Box& nbx = amrex::surroundingNodes(bx, idir);

        Array4<Real> const flux_arr = (flux[idir]).array();
        Array4<Real const> const area_arr = (area[idir]).array(mfi);

        scale_flux(nbx, flux_arr, area_arr, dt);


#if AMREX_SPACEDIM <= 2
        // get the scaled radial pressure -- we need to treat this specially

        if (idir == 0 && !momentum_flux_has_pressure(0, 0, coord)) {
            Array4<Real> pradial_fab = pradial.array();

            amrex::ParallelFor(nbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                pradial_fab(i,j,k) = qex_arr(i,j,k,IFCPRES) * dt;
            });
        }
#endif

        if (add_fluxes) {

            Array4<Real> const flux_fab = (flux[idir]).array();
            Array4<Real> fluxes_fab = (*fluxes[idir]).array(mfi);

            amrex::ParallelFor(mfi.nodaltilebox(idir), NUM_STATE,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                fluxes_fab(i,j,k,n) += flux_fab(i,j,k,n);
            });


#if AMREX_SPACEDIM <= 2
            if (idir == 0 && !momentum_flux_has_pressure(0, 0, coord)) {
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

          if (tile_size_autotuned == 0) {

              // Keep a running record of the largest box we've encountered.

              autotuned_max_box_cells = amrex::max(autotuned_max_box_cells,
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
                  tile_size_autotuned = 1;
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
                  tile_size_autotuned = 1;
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


  // Check for small/negative densities and X > 1 or X < 0.

  status = check_for_negative_density();

  if (status.success == false) {
      return status;
  }

  // Sync up state after hydro source.

  clean_state(
               S_new, time + dt, 0);

  // Check for NaN's.

  check_for_nan(S_new);


  // Perform reflux (for non-subcycling advances).

  if (parent->subcyclingMode() == "None" && do_reflux == 1) {
      CoarseFluxRegInit();
      FineFluxRegAdd();
  }

  if (verbose) {
      amrex::Print() << "... Leaving assemble_ctu_hydro_source() on level " << level << std::endl << std::endl;
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

        amrex::Print() << "HydroSim::assemble_ctu_hydro_source() time = " << run_time
                       << " on level " << llevel << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }


  return status;
}
