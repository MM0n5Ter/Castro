#include <HydroSim.H>
#include <HydroSim_helpers.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef HYBRID_MOMENTUM
#include <hybrid.H>
#endif

#include <transport_util.H>

using namespace amrex;

step_result
HydroSim::assemble_ctu_hydro_source(Real time, Real dt)
{
  amrex::ignore_unused(time);
  amrex::ignore_unused(dt);

  step_result status {};

  if (!do_hydro) {
      return status;
  }

#ifndef TRUE_SDC

  BL_PROFILE("HydroSim::assemble_ctu_hydro_source()");

  const Real strt_time = ParallelDescriptor::second();


  if (verbose) {
      amrex::Print() << "... Entering assemble_ctu_hydro_source() on level " << level << std::endl << std::endl;
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
  MultiFab scratch_deferred_mf;
  MultiFab& deferred_burn_source = hydrosim::time_scheme == SimplifiedIterCorrection ?
                               get_new_data(Burn_Corr_Type) : scratch_deferred_mf;
#endif


#ifdef RADIATION
  MultiFab& rad_energy_new = get_new_data(Radiation_Type);

  if (!Radiation::rad_hydro_coupled) {
    amrex::Abort("HydroSim::assemble_ctu_hydro_source -- we don't implement a mode where we have radiation, but it is not coupled to hydro");
  }

  int rad_step_count = -1;

  AmrLevel::FillPatch(*this, RadEnergyGrown, NUM_GROW, time, Radiation_Type, 0, Radiation::nGroups);

  MultiFab rad_lambda_grown(grids, dmap, Radiation::nGroups, NUM_GROW);
  if (radiation->hydro_only) {
      rad_lambda_grown.setVal(0.0, NUM_GROW);
  }
  else {
      radiation->build_flux_limiter(level, grids, Sborder, RadEnergyGrown, rad_lambda_grown);
  }
#endif


  size_t fab_bytes = 0;
#ifdef AMREX_USE_GPU
  size_t state_bytes = 0;
#endif
  IntVect largest_tile_size{0};


#if defined(AMREX_USE_OMP) && defined(AMREX_USE_GPU)
  amrex::Error("USE_OMP=TRUE and USE_GPU=TRUE are not concurrently supported in HydroSim");
#endif

#ifdef AMREX_USE_GPU
   if (hydrosim::hydro_tile_mem_ratio > 0.0) {

       if (sweep_tile_tuned == 0) {
           hydro_tile_size[0] = 16;
#if AMREX_SPACEDIM >= 2
           hydro_tile_size[1] = 16;
#endif
#if AMREX_SPACEDIM == 3
           hydro_tile_size[2] = 16;
#endif
       }

       for (MFIter mfi(S_new, false); mfi.isValid(); ++mfi) {
           if (mfi.validbox().numPts() > largest_tuned_box_pts) {
               sweep_tile_tuned = 0;
           }

           state_bytes += S_new[mfi].nBytes();
       }
   }
#endif

#ifdef _OPENMP
#ifdef RADIATION
#pragma omp parallel reduction(max:rad_step_count)
#else
#pragma omp parallel
#endif
#endif
  {

#ifdef RADIATION
    int local_rad_step_count = -1;
#endif


    FArrayBox shk(The_Async_Arena());
    FArrayBox q(The_Async_Arena()), qthermo(The_Async_Arena());
    FArrayBox rho_inv(The_Async_Arena());
    FArrayBox prim_source_fab(The_Async_Arena());
    FArrayBox qxm(The_Async_Arena()), qxp(The_Async_Arena());
#if AMREX_SPACEDIM >= 2
    FArrayBox qym(The_Async_Arena()), qyp(The_Async_Arena());
#endif
#if AMREX_SPACEDIM == 3
    FArrayBox qzm(The_Async_Arena()), qzp(The_Async_Arena());
#endif
    FArrayBox div(The_Async_Arena());
#if AMREX_SPACEDIM >= 2
    FArrayBox flux_temp1(The_Async_Arena()), flux_temp2(The_Async_Arena());
#ifdef RADIATION
    FArrayBox radiation_temp1(The_Async_Arena()), radiation_temp2(The_Async_Arena());
#endif
    FArrayBox godunov_temp1(The_Async_Arena()), godunov_temp2(The_Async_Arena());
    FArrayBox ql(The_Async_Arena()), qr(The_Async_Arena());
#endif
    Vector<FArrayBox> flux, qe;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        flux.push_back(FArrayBox(The_Async_Arena()));
        qe.push_back(FArrayBox(The_Async_Arena()));
    }

#ifdef RADIATION
    Vector<FArrayBox> radiation_dir_fluxes;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        radiation_dir_fluxes.push_back(FArrayBox(The_Async_Arena()));
    }
#endif
#if AMREX_SPACEDIM <= 2
    FArrayBox scaled_pressure(The_Async_Arena());
#endif
#if AMREX_SPACEDIM == 3
    FArrayBox edge_minus_yx(The_Async_Arena()), edge_plus_yx(The_Async_Arena());
    FArrayBox edge_minus_zx(The_Async_Arena()), edge_plus_zx(The_Async_Arena());
    FArrayBox edge_minus_xy(The_Async_Arena()), edge_plus_xy(The_Async_Arena());
    FArrayBox edge_minus_zy(The_Async_Arena()), edge_plus_zy(The_Async_Arena());
    FArrayBox edge_minus_xz(The_Async_Arena()), edge_plus_xz(The_Async_Arena());
    FArrayBox edge_minus_yz(The_Async_Arena()), edge_plus_yz(The_Async_Arena());
#endif

    MultiFab& old_forcing = get_old_data(Forcing_Type);

    for (MFIter mfi(S_new, hydro_tile_size); mfi.isValid(); ++mfi) {

      const Box& bx = mfi.tilebox();

      const Box& obx = amrex::grow(bx, 1);


      const Box& qbx = amrex::grow(bx, NUM_GROW);
      const Box& ghost3_box = amrex::grow(bx, 3);

#ifdef RADIATION
      q.resize(qbx, NQ);
#else
      q.resize(qbx, NQPRIMBASE);
#endif
      fab_bytes += q.nBytes();
      Array4<Real> const prim_arr = q.array();

      qthermo.resize(qbx, NQEXTRAQ);
      fab_bytes += qthermo.nBytes();
      Array4<Real> const qthermo_arr = qthermo.array();

      Array4<Real const> const cons_old_arr = Sborder.array(mfi);

      rho_inv.resize(ghost3_box, 1);
      fab_bytes += rho_inv.nBytes();
      Array4<Real> const inv_density_arr = rho_inv.array();

      amrex::ParallelFor(ghost3_box,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
          inv_density_arr(i,j,k) = 1.0 / cons_old_arr(i,j,k,URHO);
      });

      ctoprim(qbx, time, cons_old_arr,
#ifdef RADIATION
              RadEnergyGrown.array(mfi), rad_lambda_grown.array(mfi),
#endif
              prim_arr, qthermo_arr);

#if AMREX_SPACEDIM == 2
      Array4<Real const> const face_area_x_arr = area[0].array(mfi);
      Array4<Real const> const face_area_y_arr = area[1].array(mfi);
      Array4<Real> const cell_volume_arr = volume.array(mfi);
#endif

#if AMREX_SPACEDIM < 3
      Array4<Real const> const logAreaSlopeX_arr = (dLogFaceMeasure[0]).array(mfi);
#endif
#if AMREX_SPACEDIM == 2
      Array4<Real const> const logAreaSlopeY_arr = (dLogFaceMeasure[1]).array(mfi);
#endif

      const Box& xbx = amrex::surroundingNodes(bx, 0);
      const Box& grown_x_box = amrex::grow(xbx, 1);
#if AMREX_SPACEDIM >= 2
      const Box& ybx = amrex::surroundingNodes(bx, 1);
      const Box& grown_y_box = amrex::grow(ybx, 1);
#endif
#if AMREX_SPACEDIM == 3
      const Box& zbx = amrex::surroundingNodes(bx, 2);
      const Box& grown_z_box = amrex::grow(zbx, 1);
#endif

      shk.resize(obx, 1);
      fab_bytes += shk.nBytes();



      Array4<Real> const shock_flag_arr = shk.array();


      prim_source_fab.resize(ghost3_box, NQSOURCE);
      fab_bytes += prim_source_fab.nBytes();
      Array4<Real> const prim_source_arr = prim_source_fab.array();

      Array4<Real> const prev_source_arr = old_forcing.array(mfi);
      Array4<Real> const source_corrector_arr = forcing_corrector.array(mfi);

      amrex::ParallelFor(ghost3_box,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
          hydro::source_to_primitive(i, j, k, dt, cons_old_arr, prim_arr, prev_source_arr, source_corrector_arr, prim_source_arr);
      });

      if (shock_flux_switch == 1) {
        shock(obx, prim_arr, prev_source_arr, shock_flag_arr);
      }
      else {
        amrex::ParallelFor(obx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
          shock_flag_arr(i,j,k) = 0.0;
        });
      }


      qxm.resize(obx, NQ);
      fab_bytes += qxm.nBytes();

      qxp.resize(obx, NQ);
      fab_bytes += qxp.nBytes();

      Array4<Real> const x_minus_arr = qxm.array();
      Array4<Real> const x_plus_arr = qxp.array();

#if AMREX_SPACEDIM >= 2
      qym.resize(obx, NQ);
      fab_bytes += qym.nBytes();

      qyp.resize(obx, NQ);
      fab_bytes += qyp.nBytes();

      Array4<Real> const y_minus_arr = qym.array();
      Array4<Real> const y_plus_arr = qyp.array();

#endif

#if AMREX_SPACEDIM == 3
      qzm.resize(obx, NQ);
      fab_bytes += qzm.nBytes();

      qzp.resize(obx, NQ);
      fab_bytes += qzp.nBytes();

      Array4<Real> const z_minus_arr = qzm.array();
      Array4<Real> const z_plus_arr = qzp.array();

#endif

      if (ppm_recon_kind == 0) {

        corner_plm_states(obx, bx,
                       cons_old_arr, inv_density_arr,
                       prim_arr,
                       qthermo_arr,
                       prim_source_arr,
                       x_minus_arr, x_plus_arr,
#if AMREX_SPACEDIM >= 2
                       y_minus_arr, y_plus_arr,
#endif
#if AMREX_SPACEDIM == 3
                       z_minus_arr, z_plus_arr,
#endif
#if AMREX_SPACEDIM < 3
                       logAreaSlopeX_arr,
#endif
#if AMREX_SPACEDIM == 2
                       logAreaSlopeY_arr,
#endif
                       dt);

      } else {

#ifdef RADIATION
        corner_ppm_rad_states(obx, bx,
                           cons_old_arr, inv_density_arr,
                           prim_arr, qthermo_arr, prim_source_arr,
                           x_minus_arr, x_plus_arr,
#if AMREX_SPACEDIM >= 2
                           y_minus_arr, y_plus_arr,
#endif
#if AMREX_SPACEDIM == 3
                           z_minus_arr, z_plus_arr,
#endif
#if AMREX_SPACEDIM < 3
                           logAreaSlopeX_arr,
#endif
#if AMREX_SPACEDIM == 2
                           logAreaSlopeY_arr,
#endif
                           dt);
#else

        corner_ppm_states(obx, bx,
                       cons_old_arr, inv_density_arr,
                       prim_arr, qthermo_arr, prim_source_arr,
                       x_minus_arr, x_plus_arr,
#if AMREX_SPACEDIM >= 2
                       y_minus_arr, y_plus_arr,
#endif
#if AMREX_SPACEDIM == 3
                       z_minus_arr, z_plus_arr,
#endif
#if AMREX_SPACEDIM < 3
                       logAreaSlopeX_arr,
#endif
#if AMREX_SPACEDIM == 2
                       logAreaSlopeY_arr,
#endif
                       dt);
#endif

      }

      div.resize(obx, 1);
      fab_bytes += div.nBytes();
      auto divergence_arr = div.array();

      divu(obx, prim_arr, divergence_arr);

      flux[0].resize(grown_x_box, NUM_STATE);
      fab_bytes += flux[0].nBytes();
      Array4<Real> const x_flux_arr = (flux[0]).array();

      qe[0].resize(grown_x_box, NGDSTATE);
      auto x_edge_state_arr = qe[0].array();
      fab_bytes += qe[0].nBytes();

#ifdef RADIATION
      radiation_dir_fluxes[0].resize(grown_x_box, Radiation::nGroups);
      fab_bytes += radiation_dir_fluxes[0].nBytes();
      auto radiation_x_flux_arr = (radiation_dir_fluxes[0]).array();
#endif

#if AMREX_SPACEDIM >= 2
      flux[1].resize(grown_y_box, NUM_STATE);
      fab_bytes += flux[1].nBytes();
      Array4<Real> const y_flux_arr = (flux[1]).array();

      qe[1].resize(grown_y_box, NGDSTATE);
      auto y_edge_state_arr = qe[1].array();
      fab_bytes += qe[1].nBytes();

#ifdef RADIATION
      radiation_dir_fluxes[1].resize(grown_y_box, Radiation::nGroups);
      fab_bytes += radiation_dir_fluxes[1].nBytes();
      auto const radiation_y_flux_arr = (radiation_dir_fluxes[1]).array();
#endif
#endif

#if AMREX_SPACEDIM == 3
      flux[2].resize(grown_z_box, NUM_STATE);
      fab_bytes += flux[2].nBytes();
      Array4<Real> const z_flux_arr = (flux[2]).array();

      qe[2].resize(grown_z_box, NGDSTATE);
      auto z_edge_state_arr = qe[2].array();
      fab_bytes += qe[2].nBytes();

#ifdef RADIATION
      radiation_dir_fluxes[2].resize(grown_z_box, Radiation::nGroups);
      fab_bytes += radiation_dir_fluxes[2].nBytes();
      auto const radiation_z_flux_arr = (radiation_dir_fluxes[2]).array();
#endif
#endif

#if AMREX_SPACEDIM <= 2
      if (!Geom().IsCartesian()) {
          scaled_pressure.resize(xbx, 1);
      }
      fab_bytes += scaled_pressure.nBytes();
#endif

#ifdef REACTIONS
      auto deferred_source_arr = hydrosim::time_scheme == SimplifiedIterCorrection ?
                         deferred_burn_source.array(mfi) : Array4<Real const>{};
#endif

#if AMREX_SPACEDIM == 1

#ifdef REACTIONS
      if (hydrosim::time_scheme == SimplifiedIterCorrection) {
          inject_deferred_source(xbx, 0, dt,
                                   x_minus_arr, x_plus_arr, deferred_source_arr);
      }
#endif


      compute_godunov_flux(xbx,
                          x_minus_arr, x_plus_arr,
                          x_flux_arr,
#ifdef RADIATION
                          radiation_x_flux_arr,
#endif
                          x_edge_state_arr,
                          qthermo_arr,
                          shock_flag_arr,
                          0, false);

#endif



#if AMREX_SPACEDIM >= 2
      flux_temp1.resize(obx, NUM_STATE);
      auto flux_temp1_arr = flux_temp1.array();
      fab_bytes += flux_temp1.nBytes();

      flux_temp2.resize(obx, NUM_STATE);
      auto flux_temp2_arr = flux_temp2.array();
      fab_bytes += flux_temp2.nBytes();

#ifdef RADIATION
      radiation_temp1.resize(obx, Radiation::nGroups);
      auto radiation_temp1_arr = radiation_temp1.array();
      fab_bytes += radiation_temp1.nBytes();

      radiation_temp2.resize(obx, Radiation::nGroups);
      auto radiation_temp2_arr = radiation_temp2.array();
      fab_bytes += radiation_temp2.nBytes();
#endif

      godunov_temp1.resize(obx, NGDSTATE);
      auto godunov_temp1_arr = godunov_temp1.array();
      fab_bytes += godunov_temp1.nBytes();

#if AMREX_SPACEDIM == 3
      godunov_temp2.resize(obx, NGDSTATE);
      auto godunov_temp2_arr = godunov_temp2.array();
      fab_bytes += godunov_temp2.nBytes();
#endif

      ql.resize(obx, NQ);
      auto prim_left_arr = ql.array();
      fab_bytes += ql.nBytes();

      qr.resize(obx, NQ);
      auto prim_right_arr = qr.array();
      fab_bytes += qr.nBytes();
#endif



#if AMREX_SPACEDIM == 2

      const amrex::Real hdt = 0.5*dt;
      const amrex::Real hdtdx = 0.5*dt/dx[0];
      const amrex::Real hdtdy = 0.5*dt/dx[1];

      const Box& flux_box_x = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,1,0)));

      compute_godunov_flux(flux_box_x,
                          x_minus_arr, x_plus_arr,
                          flux_temp1_arr,
#ifdef RADIATION
                          radiation_temp1_arr,
#endif
                          godunov_temp1_arr,
                          qthermo_arr, shock_flag_arr,
                          0, false);

      const Box& flux_box_y = amrex::grow(ybx, IntVect(AMREX_D_DECL(1,0,0)));

      compute_godunov_flux(flux_box_y,
                          y_minus_arr, y_plus_arr,
                          flux_temp2_arr,
#ifdef RADIATION
                          radiation_temp2_arr,
#endif
                          y_edge_state_arr,
                          qthermo_arr, shock_flag_arr,
                          1, false);


      tangential_pass(xbx, 1, 0,
                   x_minus_arr, prim_left_arr,
                   x_plus_arr, prim_right_arr,
                   qthermo_arr,
                   flux_temp2_arr,
#ifdef RADIATION
                   radiation_temp2_arr,
#endif
                   y_edge_state_arr,
                   face_area_y_arr,
                   cell_volume_arr,
                   hdt, hdtdy);
      reset_edge_thermo(xbx, prim_left_arr);

      reset_edge_thermo(xbx, prim_right_arr);


#ifdef REACTIONS
      if (hydrosim::time_scheme == SimplifiedIterCorrection) {
          inject_deferred_source(xbx, 0, dt,
                                   prim_left_arr, prim_right_arr, deferred_source_arr);
      }
#endif

      compute_godunov_flux(xbx,
                          prim_left_arr, prim_right_arr,
                          x_flux_arr,
#ifdef RADIATION
                          radiation_x_flux_arr,
#endif
                          x_edge_state_arr,
                          qthermo_arr, shock_flag_arr,
                          0, false);



      tangential_pass(ybx, 0, 1,
                   y_minus_arr, prim_left_arr,
                   y_plus_arr, prim_right_arr,
                   qthermo_arr,
                   flux_temp1_arr,
#ifdef RADIATION
                   radiation_temp1_arr,
#endif
                   godunov_temp1_arr,
                   face_area_x_arr,
                   cell_volume_arr,
                   hdt, hdtdx);
      reset_edge_thermo(ybx, prim_left_arr);

      reset_edge_thermo(ybx, prim_right_arr);



#ifdef REACTIONS
      if (hydrosim::time_scheme == SimplifiedIterCorrection) {
          inject_deferred_source(ybx, 1, dt,
                                   prim_left_arr, prim_right_arr, deferred_source_arr);
      }
#endif

      compute_godunov_flux(ybx,
                          prim_left_arr, prim_right_arr,
                          y_flux_arr,
#ifdef RADIATION
                          radiation_y_flux_arr,
#endif
                          y_edge_state_arr,
                          qthermo_arr, shock_flag_arr,
                          1, false);
#endif



#if AMREX_SPACEDIM == 3

      const amrex::Real hdt = 0.5*dt;

      const amrex::Real hdtdx = 0.5*dt/dx[0];
      const amrex::Real hdtdy = 0.5*dt/dx[1];
      const amrex::Real hdtdz = 0.5*dt/dx[2];

      const amrex::Real third_dt_dx = dt/dx[0]/3.0;
      const amrex::Real third_dt_dy = dt/dx[1]/3.0;
      const amrex::Real third_dt_dz = dt/dx[2]/3.0;

      const Box& flux_box_x = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,1,1)));

      compute_godunov_flux(flux_box_x,
                          x_minus_arr, x_plus_arr,
                          flux_temp1_arr,
#ifdef RADIATION
                          radiation_temp1_arr,
#endif
                          godunov_temp1_arr,
                          qthermo_arr, shock_flag_arr,
                          0, false);


      const Box& trans_box_yx = amrex::grow(ybx, IntVect(AMREX_D_DECL(0,0,1)));

      edge_minus_yx.resize(trans_box_yx, NQ);
      auto edge_minus_yx_arr = edge_minus_yx.array();
      fab_bytes += edge_minus_yx.nBytes();

      edge_plus_yx.resize(trans_box_yx, NQ);
      auto edge_plus_yx_arr = edge_plus_yx.array();
      fab_bytes += edge_plus_yx.nBytes();

      tangential_pass(trans_box_yx, 0, 1,
                   y_minus_arr, edge_minus_yx_arr,
                   y_plus_arr, edge_plus_yx_arr,
                   qthermo_arr,
                   flux_temp1_arr,
#ifdef RADIATION
                   radiation_temp1_arr,
#endif
                   godunov_temp1_arr,
                   hdt, third_dt_dx);
      reset_edge_thermo(trans_box_yx, edge_minus_yx.array());

      reset_edge_thermo(trans_box_yx, edge_plus_yx.array());

      const Box& trans_box_zx = amrex::grow(zbx, IntVect(AMREX_D_DECL(0,1,0)));

      edge_minus_zx.resize(trans_box_zx, NQ);
      auto edge_minus_zx_arr = edge_minus_zx.array();
      fab_bytes += edge_minus_zx.nBytes();

      edge_plus_zx.resize(trans_box_zx, NQ);
      auto edge_plus_zx_arr = edge_plus_zx.array();
      fab_bytes += edge_plus_zx.nBytes();

      tangential_pass(trans_box_zx, 0, 2,
                   z_minus_arr, edge_minus_zx_arr,
                   z_plus_arr, edge_plus_zx_arr,
                   qthermo_arr,
                   flux_temp1_arr,
#ifdef RADIATION
                   radiation_temp1_arr,
#endif
                   godunov_temp1_arr,
                   hdt, third_dt_dx);
      reset_edge_thermo(trans_box_zx, edge_minus_zx.array());

      reset_edge_thermo(trans_box_zx, edge_plus_zx.array());

      const Box& flux_box_y = amrex::grow(ybx, IntVect(AMREX_D_DECL(1,0,1)));

      compute_godunov_flux(flux_box_y,
                          y_minus_arr, y_plus_arr,
                          flux_temp1_arr,
#ifdef RADIATION
                          radiation_temp1_arr,
#endif
                          godunov_temp1_arr,
                          qthermo_arr, shock_flag_arr,
                          1, false);
      const Box& trans_box_xy = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,0,1)));

      edge_minus_xy.resize(trans_box_xy, NQ);
      auto edge_minus_xy_arr = edge_minus_xy.array();
      fab_bytes += edge_minus_xy.nBytes();

      edge_plus_xy.resize(trans_box_xy, NQ);
      auto edge_plus_xy_arr = edge_plus_xy.array();
      fab_bytes += edge_plus_xy.nBytes();

      tangential_pass(trans_box_xy, 1, 0,
                   x_minus_arr, edge_minus_xy_arr,
                   x_plus_arr, edge_plus_xy_arr,
                   qthermo_arr,
                   flux_temp1_arr,
#ifdef RADIATION
                   radiation_temp1_arr,
#endif
                   godunov_temp1_arr,
                   hdt, third_dt_dy);
      reset_edge_thermo(trans_box_xy, edge_minus_xy.array());

      reset_edge_thermo(trans_box_xy, edge_plus_xy.array());

      const Box& trans_box_zy = amrex::grow(zbx, IntVect(AMREX_D_DECL(1,0,0)));

      edge_minus_zy.resize(trans_box_zy, NQ);
      auto edge_minus_zy_arr = edge_minus_zy.array();
      fab_bytes += edge_minus_zy.nBytes();

      edge_plus_zy.resize(trans_box_zy, NQ);
      auto edge_plus_zy_arr = edge_plus_zy.array();
      fab_bytes += edge_plus_zy.nBytes();

      tangential_pass(trans_box_zy, 1, 2,
                   z_minus_arr, edge_minus_zy_arr,
                   z_plus_arr, edge_plus_zy_arr,
                   qthermo_arr,
                   flux_temp1_arr,
#ifdef RADIATION
                   radiation_temp1_arr,
#endif
                   godunov_temp1_arr,
                   hdt, third_dt_dy);
      reset_edge_thermo(trans_box_zy, edge_minus_zy.array());

      reset_edge_thermo(trans_box_zy, edge_plus_zy.array());

      const Box& flux_box_z = amrex::grow(zbx, IntVect(AMREX_D_DECL(1,1,0)));

      compute_godunov_flux(flux_box_z,
                          z_minus_arr, z_plus_arr,
                          flux_temp1_arr,
#ifdef RADIATION
                          radiation_temp1_arr,
#endif
                          godunov_temp1_arr,
                          qthermo_arr, shock_flag_arr,
                          2, false);
      const Box& trans_box_xz = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,1,0)));

      edge_minus_xz.resize(trans_box_xz, NQ);
      auto edge_minus_xz_arr = edge_minus_xz.array();
      fab_bytes += edge_minus_xz.nBytes();

      edge_plus_xz.resize(trans_box_xz, NQ);
      auto edge_plus_xz_arr = edge_plus_xz.array();
      fab_bytes += edge_plus_xz.nBytes();

      tangential_pass(trans_box_xz, 2, 0,
                   x_minus_arr, edge_minus_xz_arr,
                   x_plus_arr, edge_plus_xz_arr,
                   qthermo_arr,
                   flux_temp1_arr,
#ifdef RADIATION
                   radiation_temp1_arr,
#endif
                   godunov_temp1_arr,
                   hdt, third_dt_dz);
      reset_edge_thermo(trans_box_xz, edge_minus_xz.array());

      reset_edge_thermo(trans_box_xz, edge_plus_xz.array());

      const Box& trans_box_yz = amrex::grow(ybx, IntVect(AMREX_D_DECL(1,0,0)));

      edge_minus_yz.resize(trans_box_yz, NQ);
      auto edge_minus_yz_arr = edge_minus_yz.array();
      fab_bytes += edge_minus_yz.nBytes();

      edge_plus_yz.resize(trans_box_yz, NQ);
      auto edge_plus_yz_arr = edge_plus_yz.array();
      fab_bytes += edge_plus_yz.nBytes();

      tangential_pass(trans_box_yz, 2, 1,
                   y_minus_arr, edge_minus_yz_arr,
                   y_plus_arr, edge_plus_yz_arr,
                   qthermo_arr,
                   flux_temp1_arr,
#ifdef RADIATION
                   radiation_temp1_arr,
#endif
                   godunov_temp1_arr,
                   hdt, third_dt_dz);
      reset_edge_thermo(trans_box_yz, edge_minus_yz.array());

      reset_edge_thermo(trans_box_yz, edge_plus_yz.array());



      const Box& flux_box_yz = amrex::grow(ybx, IntVect(AMREX_D_DECL(1,0,0)));

      compute_godunov_flux(flux_box_yz,
                          edge_minus_yz_arr, edge_plus_yz_arr,
                          flux_temp1_arr,
#ifdef RADIATION
                          radiation_temp1_arr,
#endif
                          godunov_temp1_arr,
                          qthermo_arr, shock_flag_arr,
                          1, false);

      const Box& flux_box_zy = amrex::grow(zbx, IntVect(AMREX_D_DECL(1,0,0)));

      compute_godunov_flux(flux_box_zy,
                          edge_minus_zy_arr, edge_plus_zy_arr,
                          flux_temp2_arr,
#ifdef RADIATION
                          radiation_temp2_arr,
#endif
                          godunov_temp2_arr,
                          qthermo_arr, shock_flag_arr,
                          2, false);


      tangential_correction(xbx, 0, 1, 2,
                  x_minus_arr, prim_left_arr,
                  x_plus_arr, prim_right_arr,
                  qthermo_arr,
                  flux_temp1_arr,
#ifdef RADIATION
                  radiation_temp1_arr,
#endif
                  flux_temp2_arr,
#ifdef RADIATION
                  radiation_temp2_arr,
#endif
                  godunov_temp1_arr,
                  godunov_temp2_arr,
                  hdtdy, hdtdz);

      reset_edge_thermo(xbx, prim_left_arr);

      reset_edge_thermo(xbx, prim_right_arr);

#ifdef REACTIONS
      if (hydrosim::time_scheme == SimplifiedIterCorrection) {
          inject_deferred_source(xbx, 0, dt,
                                   prim_left_arr, prim_right_arr, deferred_source_arr);
      }
#endif


      compute_godunov_flux(xbx,
                          prim_left_arr, prim_right_arr,
                          x_flux_arr,
#ifdef RADIATION
                          radiation_x_flux_arr,
#endif
                          x_edge_state_arr,
                          qthermo_arr, shock_flag_arr,
                          0, false);


      const Box& flux_box_zx = amrex::grow(zbx, IntVect(AMREX_D_DECL(0,1,0)));

      compute_godunov_flux(flux_box_zx,
                          edge_minus_zx_arr, edge_plus_zx_arr,
                          flux_temp1_arr,
#ifdef RADIATION
                          radiation_temp1_arr,
#endif
                          godunov_temp1_arr,
                          qthermo_arr, shock_flag_arr,
                          2, false);

      const Box& flux_box_xz = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,1,0)));

      compute_godunov_flux(flux_box_xz,
                          edge_minus_xz_arr, edge_plus_xz_arr,
                          flux_temp2_arr,
#ifdef RADIATION
                          radiation_temp2_arr,
#endif
                          godunov_temp2_arr,
                          qthermo_arr, shock_flag_arr,
                          0, false);


      tangential_correction(ybx, 1, 0, 2,
                  y_minus_arr, prim_left_arr,
                  y_plus_arr, prim_right_arr,
                  qthermo_arr,
                  flux_temp2_arr,
#ifdef RADIATION
                  radiation_temp2_arr,
#endif
                  flux_temp1_arr,
#ifdef RADIATION
                  radiation_temp1_arr,
#endif
                  godunov_temp2_arr,
                  godunov_temp1_arr,
                  hdtdx, hdtdz);

      reset_edge_thermo(ybx, prim_left_arr);

      reset_edge_thermo(ybx, prim_right_arr);

#ifdef REACTIONS
      if (hydrosim::time_scheme == SimplifiedIterCorrection) {
          inject_deferred_source(ybx, 1, dt,
                                   prim_left_arr, prim_right_arr, deferred_source_arr);
      }
#endif


      compute_godunov_flux(ybx,
                          prim_left_arr, prim_right_arr,
                          y_flux_arr,
#ifdef RADIATION
                          radiation_y_flux_arr,
#endif
                          y_edge_state_arr,
                          qthermo_arr, shock_flag_arr,
                          1, false);


      const Box& flux_box_xy = amrex::grow(xbx, IntVect(AMREX_D_DECL(0,0,1)));

      compute_godunov_flux(flux_box_xy,
                          edge_minus_xy_arr, edge_plus_xy_arr,
                          flux_temp1_arr,
#ifdef RADIATION
                          radiation_temp1_arr,
#endif
                          godunov_temp1_arr,
                          qthermo_arr, shock_flag_arr,
                          0, false);

      const Box& flux_box_yx = amrex::grow(ybx, IntVect(AMREX_D_DECL(0,0,1)));

      compute_godunov_flux(flux_box_yx,
                          edge_minus_yx_arr, edge_plus_yx_arr,
                          flux_temp2_arr,
#ifdef RADIATION
                          radiation_temp2_arr,
#endif
                          godunov_temp2_arr,
                          qthermo_arr, shock_flag_arr,
                          1, false);


      tangential_correction(zbx, 2, 0, 1,
                  z_minus_arr, prim_left_arr,
                  z_plus_arr, prim_right_arr,
                  qthermo_arr,
                  flux_temp1_arr,
#ifdef RADIATION
                  radiation_temp1_arr,
#endif
                  flux_temp2_arr,
#ifdef RADIATION
                  radiation_temp2_arr,
#endif
                  godunov_temp1_arr,
                  godunov_temp2_arr,
                  hdtdx, hdtdy);

      reset_edge_thermo(zbx, prim_left_arr);

      reset_edge_thermo(zbx, prim_right_arr);

#ifdef REACTIONS
      if (hydrosim::time_scheme == SimplifiedIterCorrection) {
          inject_deferred_source(zbx, 2, dt,
                                   prim_left_arr, prim_right_arr, deferred_source_arr);
      }
#endif


      compute_godunov_flux(zbx,
                          prim_left_arr, prim_right_arr,
                          z_flux_arr,
#ifdef RADIATION
                          radiation_z_flux_arr,
#endif
                          z_edge_state_arr,
                          qthermo_arr, shock_flag_arr,
                          2, false);

#endif




      for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

          const Box& nbx = amrex::surroundingNodes(bx, idir);

          Array4<Real> const dir_flux_arr = (flux[idir]).array();

          amrex::ParallelFor(nbx,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              dir_flux_arr(i,j,k,UTEMP) = 0.e0;
#ifdef SHOCK_VAR
              dir_flux_arr(i,j,k,UFLAG_SHOCK) = 0.e0;
#endif
#ifdef NSE_NET
              dir_flux_arr(i,j,k,UPROTON_MU) = 0.e0;
              dir_flux_arr(i,j,k,UNEUTRON_MU) = 0.e0;
#endif
          });

          apply_viscosity(nbx, idir, divergence_arr, cons_old_arr, dir_flux_arr);

#ifdef RADIATION
          Array4<Real> const radiation_dir_flux_arr = (radiation_dir_fluxes[idir]).array();
          Array4<Real const> const rad_energy_in_arr = RadEnergyGrown.array(mfi);

          apply_viscosity_radiative(nbx, idir, divergence_arr, rad_energy_in_arr, radiation_dir_flux_arr);
#endif

          if (clip_small_rho_flux == 1) {
              clip_fluxes_at_density_floor
                  (nbx, idir,
                   Sborder.array(mfi),
                   volume.array(mfi),
                   flux[idir].array(),
                   area[idir].array(mfi),
                   dt);
          }

          balance_species_flux(nbx, dir_flux_arr);

      }



      Array4<Real> const updated_state_arr = S_new.array(mfi);

      Array4<Real> const x_flux_view = (flux[0]).array();
      Array4<Real> const x_edge_state_view = (qe[0]).array();

#if AMREX_SPACEDIM >= 2
      Array4<Real> const y_flux_view = (flux[1]).array();
      Array4<Real> const y_edge_state_view = (qe[1]).array();
#endif

#if AMREX_SPACEDIM == 3
      Array4<Real> const z_flux_view = (flux[2]).array();
      Array4<Real> const z_edge_state_view = (qe[2]).array();
#endif

      cons_update_hydro(bx,
                   updated_state_arr,
                   x_flux_view, x_edge_state_view,
#if AMREX_SPACEDIM >= 2
                   y_flux_view, y_edge_state_view,
#endif
#if AMREX_SPACEDIM == 3
                   z_flux_view, z_edge_state_view,
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
          Real InvRadius = 1.0_rt / R;

          updated_state_arr(i,j,k,UMR) -= dt * (loc[0] * InvRadius) * (x_edge_state_view(i+1,j,k,GDPRESSURE) - x_edge_state_view(i,j,k,GDPRESSURE)) / dx_arr[0];
#if AMREX_SPACEDIM >= 2
          updated_state_arr(i,j,k,UMR) -= dt * (loc[1] * InvRadius) * (y_edge_state_view(i,j+1,k,GDPRESSURE) - y_edge_state_view(i,j,k,GDPRESSURE)) / dx_arr[1];
#endif
      });
#endif

#ifdef RADIATION
      rad_cons_update(bx,
                     updated_state_arr,
                     RadEnergyGrown.array(mfi),
                     rad_energy_new.array(mfi),
                     (radiation_dir_fluxes[0]).array(),
                     (qe[0]).array(),
                     (area[0]).array(mfi),
#if AMREX_SPACEDIM >= 2
                     (radiation_dir_fluxes[1]).array(),
                     (qe[1]).array(),
                     (area[1]).array(mfi),
#endif
#if AMREX_SPACEDIM == 3
                     (radiation_dir_fluxes[2]).array(),
                     (qe[2]).array(),
                     (area[2]).array(mfi),
#endif
                     local_rad_step_count,
                     volume.array(mfi),
                     dt);

      rad_step_count = std::max(rad_step_count, local_rad_step_count);
#endif


      for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

        const Box& nbx = amrex::surroundingNodes(bx, idir);

        Array4<Real> const dir_flux_arr = (flux[idir]).array();
        Array4<Real const> const face_area_arr = (area[idir]).array(mfi);

        scale_flux_by_area(nbx, dir_flux_arr, face_area_arr, dt);

#ifdef RADIATION
        Array4<Real> const radiation_dir_flux_arr = (radiation_dir_fluxes[idir]).array();
        scale_radiative_flux(nbx, radiation_dir_flux_arr, face_area_arr, dt);
#endif

#if AMREX_SPACEDIM <= 2

        if (idir == 0 && !momentum_flux_has_pressure(0, 0, coord)) {
            Array4<Real> scaled_pressure_fab = scaled_pressure.array();

            amrex::ParallelFor(nbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                scaled_pressure_fab(i,j,k) = x_edge_state_arr(i,j,k,GDPRESSURE) * dt;
            });
        }
#endif

        bool store_fluxes = true;

        if (time_scheme == SimplifiedIterCorrection &&
            correction_iter != deferred_iters - 1) {
            store_fluxes = false;
        }

        if (store_fluxes) {

            Array4<Real> const dir_flux_fab = (flux[idir]).array();
            Array4<Real> stored_fluxes_fab = (*fluxes[idir]).array(mfi);

            amrex::ParallelFor(mfi.nodaltilebox(idir), NUM_STATE,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                stored_fluxes_fab(i,j,k,n) += dir_flux_fab(i,j,k,n);
            });

#ifdef RADIATION
            Array4<Real> const radiation_dir_flux_fab = (radiation_dir_fluxes[idir]).array();
            Array4<Real> radiation_stored_flux_fab = (*rad_face_flux[idir]).array(mfi);

            amrex::ParallelFor(mfi.nodaltilebox(idir), Radiation::nGroups,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                radiation_stored_flux_fab(i,j,k,n) += radiation_dir_flux_fab(i,j,k,n);
            });
#endif

#if AMREX_SPACEDIM <= 2
            if (idir == 0 && !momentum_flux_has_pressure(0, 0, coord)) {
                Array4<Real> scaled_pressure_fab = scaled_pressure.array();
                Array4<Real> radial_pressure_fab = Pressure_radial.array(mfi);

                amrex::ParallelFor(mfi.nodaltilebox(0),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    radial_pressure_fab(i,j,k,0) += scaled_pressure_fab(i,j,k,0);
                });
            }

#endif

        }

        Array4<Real> const dir_flux_fab = (flux[idir]).array();
        Array4<Real> density_flux_fab = (*mass_face_flux[idir]).array(mfi);

        amrex::ParallelFor(mfi.nodaltilebox(idir),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {

            density_flux_fab(i,j,k,0) = dir_flux_fab(i,j,k,URHO);
        });

      }

#ifdef AMREX_USE_GPU
      if (hydrosim::hydro_tile_mem_ratio > 0.0) {

          if (sweep_tile_tuned == 0) {


              largest_tuned_box_pts = amrex::max(largest_tuned_box_pts,
                                                                   mfi.validbox().numPts());



              largest_tile_size[0] = amrex::max(largest_tile_size[0], bx.bigEnd(0) - mfi.validbox().smallEnd(0) + 1);
#if AMREX_SPACEDIM >= 2
              largest_tile_size[1] = amrex::max(largest_tile_size[1], bx.bigEnd(1) - mfi.validbox().smallEnd(1) + 1);
#endif
#if AMREX_SPACEDIM ==3
              largest_tile_size[2] = amrex::max(largest_tile_size[2], bx.bigEnd(2) - mfi.validbox().smallEnd(2) + 1);
#endif

              if (fab_bytes >= hydrosim::hydro_tile_mem_ratio * state_bytes) {
                  Gpu::synchronize();
                  hydro_tile_size = largest_tile_size;
                  sweep_tile_tuned = 1;
              }
              else if (mfi.tileIndex() == mfi.length() - 1) {
                  hydro_tile_size[0] = 1024;
#if AMREX_SPACEDIM >= 2
                  hydro_tile_size[1] = 1024;
#endif
#if AMREX_SPACEDIM == 3
                  hydro_tile_size[2] = 1024;
#endif
                  sweep_tile_tuned = 1;
              }

          }
          else {

              if (fab_bytes >= hydrosim::hydro_tile_mem_ratio * state_bytes) {
                  Gpu::synchronize();

                  fab_bytes = 0;
              }
          }
      }
#endif

    }

  }

#ifdef RADIATION
  if (radiation->verbose>=1) {
      amrex::Real level_real = level;
#ifdef BL_LAZY
    Lazy::QueueReduction( [=] () mutable {
#endif
       ParallelDescriptor::ReduceIntMax(rad_step_count, ParallelDescriptor::IOProcessorNumber());
       if (ParallelDescriptor::IOProcessor() && rad_step_count > 0) {
         std::cout << "Radiation f-space advection on level " << level_real
                   << " takes as many as " << rad_step_count;
         if (rad_step_count == 1) {
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


  status = scan_for_negative_mass_density();

  if (status.success == false) {
      return status;
  }


  clean_state(
#ifdef MHD
               magfield_x_new, magfield_y_new, magfield_z_new,
#endif
               S_new, time + dt, 0);


  scan_for_nan(S_new);

#ifdef GRAVITY

  if (track_center == 1) {
      define_tracked_center(S_new, time);
  }
#endif


  if (parent->subcyclingMode() == "None") {
      if (do_reflux == 1) {
          FluxRegCoarseInit();
          FluxRegFineAccum();
      }
  }

  if (verbose) {
      amrex::Print() << "... Leaving assemble_ctu_hydro_source() on level " << level << std::endl << std::endl;
  }

  if (verbose > 0)
    {
      const int IOProc = ParallelDescriptor::IOProcessorNumber();
      amrex::Real run_time = ParallelDescriptor::second() - strt_time;
      amrex::Real level_real = level;

#ifdef BL_LAZY
      Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::assemble_ctu_hydro_source() time = " << run_time
                       << " on level " << level_real << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }

#endif

  return status;
}

void
HydroSim::verify_courant_limit(const MultiFab& State, const Real dt)
{
    BL_PROFILE("HydroSim::verify_courant_limit()");

    int cfl_breach = 0;

    auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
    const auto coord = geom.Coord();
    amrex::ignore_unused(problo, coord);

    Real dtdx = dt / dx[0];

    Real dtdy = 0.0_rt;
    if (AMREX_SPACEDIM >= 2) {
      dtdy = dt / dx[1];
      if (coord == 2) {
          dtdy /= problo[0] + 0.5_rt * dx[0];
      }
    }

    Real dtdz = 0.0_rt;
    if (AMREX_SPACEDIM == 3) {
      dtdz = dt / dx[2];
    }

    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(State, hydro_tile_size); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        auto U = State.array(mfi);

        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
        {

            Real rho = U(i,j,k,URHO);
            Real rhoInv = 1.0 / rho;

            Real u = U(i,j,k,UMX) * rhoInv;
            Real v = U(i,j,k,UMY) * rhoInv;
            Real w = U(i,j,k,UMZ) * rhoInv;

            eos_rep_t eos_state;

            eos_state.rho = U(i,j,k,URHO);
            eos_state.T = U(i,j,k,UTEMP);
            eos_state.e = U(i,j,k,UEINT) * rhoInv;
            for (int n = 0; n < NumSpec; n++) {
                eos_state.xn[n] = U(i,j,k,UFS+n) * rhoInv;
            }
#if NAUX_NET > 0
            for (int n = 0; n < NumAux; n++) {
                eos_state.aux[n] = U(i,j,k,UFX+n) * rhoInv;
            }
#endif

            eos(eos_input_re, eos_state);

            Real cs = eos_state.cs;

            Real courant_x = (cs + std::abs(u)) * dtdx;
            Real courant_y = (cs + std::abs(v)) * dtdy;
            Real courant_z = (cs + std::abs(w)) * dtdz;

            if (hydrosim::time_scheme == 0) {

#ifndef AMREX_USE_GPU
                if (verbose == 1) {

                    if (courant_x > 1.0_rt) {
                        std::cout << std::endl;
                        std::cout << Font::Bold << FGColor::Yellow;
                        std::cout << "[WARNING] CFL violation in verify_courant_limit" << std::endl;
                        std::cout << ">>> ... (u+c) * dt / dx > 1 " << courant_x << std::endl;
                        std::cout << ">>> ... at cell i = " << i << " j = " << j << " k = " << k << std::endl;
                        std::cout << ">>> ... u = " << u << " c = " << cs << std::endl;
                        std::cout << ">>> ... density = " << rho << std::endl;
                        std::cout << ResetDisplay;
                    }

                    if (courant_y > 1.0_rt) {
                        std::cout << std::endl;
                        std::cout << Font::Bold << FGColor::Yellow;
                        std::cout << "[WARNING] CFL violation in verify_courant_limit" << std::endl;
                        std::cout << ">>> ... (v+c) * dt / dx > 1 " << courant_y << std::endl;
                        std::cout << ">>> ... at cell i = " << i << " j = " << j << " k = " << k << std::endl;
                        std::cout << ">>> ... v = " << v << " c = " << cs << std::endl;
                        std::cout << ">>> ... density = " << rho << std::endl;
                        std::cout << ResetDisplay;
                    }

                    if (courant_z > 1.0_rt) {
                        std::cout << std::endl;
                        std::cout << Font::Bold << FGColor::Yellow;
                        std::cout << "[WARNING] CFL violation in verify_courant_limit" << std::endl;
                        std::cout << ">>> ... (w+c) * dt / dx > 1 " << courant_z << std::endl;
                        std::cout << ">>> ... at cell i = " << i << " j = " << j << " k = " << k << std::endl;
                        std::cout << ">>> ... w = " << w << " c = " << cs << std::endl;
                        std::cout << ">>> ... density = " << rho << std::endl;
                        std::cout << ResetDisplay;
                    }

                }
#endif


                return {amrex::max(courant_x, courant_y, courant_z)};

            }
            else {

                Real courant_combined = courant_x;
                if (AMREX_SPACEDIM >= 2) {
                    courant_combined += courant_y;
                }
                if (AMREX_SPACEDIM == 3) {
                    courant_combined += courant_z;
                }

#ifndef AMREX_USE_GPU
                if (verbose == 1) {

                    if (courant_combined > 1.0_rt) {
                        std::cout << std::endl;
                        std::cout << Font::Bold << FGColor::Yellow;
                        std::cout << "[WARNING] CFL violation in verify_courant_limit" << std::endl;
                        std::cout << ">>> ... at cell i = " << i << " j = " << j << " k = " << k << std::endl;
                        std::cout << ">>> ... u = " << u << " v = " << v
                                  << " w = " << w << " c = " << cs << std::endl;
                        std::cout << ">>> ... density = " << rho << std::endl;
                        std::cout << ResetDisplay;
                    }

                }
#endif

                return {courant_combined};

            }

        });

    }

    ReduceTuple hv = reduce_data.value();
    Real courant_max = amrex::get<0>(hv);

    ParallelDescriptor::ReduceRealMax(courant_max);

    if (courant_max > 1.0) {
        amrex::Print() << Font::Bold << FGColor::Yellow << "[WARNING] EFFECTIVE CFL AT LEVEL " << level << " IS " << courant_max << ResetDisplay << std::endl << std::endl;
        cfl_breach = 1;
    }

    if (cfl_breach) {
        amrex::Abort("CFL is too high at this level; go back to a checkpoint and restart with lower CFL number");
    }
}
