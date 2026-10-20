#include <HydroSim.H>

#include <primitive_conversion.H>
#include <limiters.H>
#include <HydroSim_helpers.H>

#include <mhd_zone_ops.H>

using namespace amrex;

step_result
HydroSim::assemble_ctu_mhd_source(Real time, Real dt)
{
      step_result status {};

      if (!do_hydro) {
          return status;
      }

      if (verbose && ParallelDescriptor::IOProcessor())
        std::cout << "... mhd ...!!! " << std::endl << std::endl;

      const auto dx = geom.CellSizeArray();

      MultiFab& S_new = get_new_data(State_Type);

      MultiFab& magfield_x_new= get_new_data(Bfield_Type_x);
      MultiFab& magfield_y_new= get_new_data(Bfield_Type_y);
      MultiFab& magfield_z_new= get_new_data(Bfield_Type_z);

      MultiFab& old_forcing = get_old_data(Forcing_Type);




      BL_ASSERT(NUM_GROW == 6);


#ifdef _OPENMP
#pragma omp parallel
#endif
    {

      FArrayBox flux[AMREX_SPACEDIM], E[AMREX_SPACEDIM];

      FArrayBox q;
      FArrayBox qthermo;
      FArrayBox prim_source;

      FArrayBox flatten_factor;
      FArrayBox flatten_scratch;

      FArrayBox prim_left[AMREX_SPACEDIM];
      FArrayBox prim_right[AMREX_SPACEDIM];

      FArrayBox flux_x_1d;
      FArrayBox flux_y_1d;
      FArrayBox flux_z_1d;

      FArrayBox cons_x_left;
      FArrayBox cons_x_right;
      FArrayBox cons_y_left;
      FArrayBox cons_y_right;
      FArrayBox cons_z_left;
      FArrayBox cons_z_right;

      FArrayBox corner_left_state;
      FArrayBox corner_right_state;

      FArrayBox trans_flux_xy;
      FArrayBox trans_flux_xz;

      FArrayBox trans_flux_yx;
      FArrayBox trans_flux_yz;

      FArrayBox trans_flux_zx;
      FArrayBox trans_flux_zy;

      FArrayBox q2D;

      FArrayBox div;

      for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {

          const Box& bx = mfi.tilebox();
          const Box& obx = amrex::grow(bx, 1);
          const Box& gbx = amrex::grow(bx, 2);

          const Box& ghost_box = amrex::grow(bx, NUM_GROW);

          FArrayBox &statein  = Sborder[mfi];
          auto state_view = statein.array();

          FArrayBox &new_state_fab = S_new[mfi];
          auto updated_state_arr = new_state_fab.array();

          FArrayBox& Bx  = Bx_old_grown[mfi];
          auto magfield_x_arr = Bx.array();
          auto Bx_hold = Bx.elixir();

          FArrayBox& By  = By_old_grown[mfi];
          auto magfield_y_arr = By.array();
          auto By_hold = By.elixir();

          FArrayBox& Bz  = Bz_old_grown[mfi];
          auto magfield_z_arr = Bz.array();
          auto Bz_hold = Bz.elixir();

          FArrayBox& Bx_new_fab = magfield_x_new[mfi];
          auto Bx_new_arr = Bx_new_fab.array();

          FArrayBox& By_new_fab = magfield_y_new[mfi];
          auto By_new_arr = By_new_fab.array();

          FArrayBox& Bz_new_fab = magfield_z_new[mfi];
          auto Bz_new_arr = Bz_new_fab.array();



          const Box& nbx = amrex::surroundingNodes(bx, 0);
          const Box& nby = amrex::surroundingNodes(bx, 1);
          const Box& nbz = amrex::surroundingNodes(bx, 2);

          const Box& nodal_flux_box_x = amrex::grow(nbx, IntVect(0, 1, 1));
          const Box& nodal_flux_box_y = amrex::grow(nby, IntVect(1, 0, 1));
          const Box& nodal_flux_box_z = amrex::grow(nbz, IntVect(1, 1, 0));

          const Box& nodal_emf_box_x = amrex::grow(nbx, IntVect(2, 3, 3));
          const Box& nodal_emf_box_y = amrex::grow(nby, IntVect(3, 2, 3));
          const Box& nodal_emf_box_z = amrex::grow(nbz, IntVect(3, 3, 2));

          flux[0].resize(nodal_flux_box_x, NUM_STATE+3);
          auto flux_x_arr = flux[0].array();
          auto flux_x_hold = flux[0].elixir();

          E[0].resize(nodal_emf_box_x);
          auto emf_x_arr = E[0].array();
          auto emf_x_hold = E[0].elixir();

          flux[1].resize(nodal_flux_box_y, NUM_STATE+3);
          auto flux_y_arr = flux[1].array();
          auto flux_y_hold = flux[1].elixir();

          E[1].resize(nodal_emf_box_y);
          auto emf_y_arr = E[1].array();
          auto emf_y_hold = E[1].elixir();

          flux[2].resize(nodal_flux_box_z, NUM_STATE+3);
          auto flux_z_arr = flux[2].array();
          auto flux_z_hold = flux[2].elixir();

          E[2].resize(nodal_emf_box_z);
          auto emf_z_arr = E[2].array();
          auto emf_z_hold = E[2].elixir();


          q.resize(ghost_box, NQ);
          auto prim_arr = q.array();
          auto prim_state_hold = q.elixir();

          qthermo.resize(ghost_box, NQEXTRAQ);
          auto qthermo_arr = qthermo.array();
          auto qthermo_hold = qthermo.elixir();

          prim_source.resize(ghost_box, NQSOURCE);
          auto prim_source_arr = prim_source.array();
          auto prim_source_hold = prim_source.elixir();

          Array4<Real> const prev_source_arr = old_forcing.array(mfi);
          Array4<Real> const source_corrector_arr = forcing_corrector.array(mfi);

          ctoprim(ghost_box, time,
                  state_view,
                  magfield_x_arr, magfield_y_arr, magfield_z_arr,
                  prim_arr, qthermo_arr);

          amrex::ParallelFor(ghost_box,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              hydro::source_to_primitive(i, j, k, dt, state_view, prim_arr, prev_source_arr, source_corrector_arr, prim_source_arr);
          });


          const Box& bxi = amrex::grow(bx, IntVect(3, 3, 3));

          flatten_factor.resize(bxi, 1);
          auto flatten_arr = flatten_factor.array();
          auto flatten_factor_hold = flatten_factor.elixir();

          if (use_shock_flatten == 0) {
            amrex::ParallelFor(bxi,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
              flatten_arr(i,j,k) = 1.0;
            });

          } else {

            amrex::ParallelFor(bxi,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                flatten_arr(i,j,k) = hydro::flatten(i, j, k, prim_arr, QPRES) *
                                   hydro::flatten(i, j, k, prim_arr, QPTOTAL);
            });

          }

          prim_left[0].resize(ghost_box, NQ);
          auto prim_x_left_arr = prim_left[0].array();
          auto prim_x_left_hold = prim_left[0].elixir();

          prim_right[0].resize(ghost_box, NQ);
          auto prim_x_right_arr = prim_right[0].array();
          auto prim_x_right_hold = prim_right[0].elixir();

          prim_left[1].resize(ghost_box, NQ);
          auto prim_y_left_arr = prim_left[1].array();
          auto prim_y_left_hold = prim_left[1].elixir();

          prim_right[1].resize(ghost_box, NQ);
          auto prim_y_right_arr = prim_right[1].array();
          auto prim_y_right_hold = prim_right[1].elixir();

          prim_left[2].resize(ghost_box, NQ);
          auto prim_z_left_arr = prim_left[2].array();
          auto prim_z_left_hold = prim_left[2].elixir();

          prim_right[2].resize(ghost_box, NQ);
          auto prim_z_right_arr = prim_right[2].array();
          auto prim_z_right_hold = prim_right[2].elixir();


          for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {

            if (ppm_recon_kind == 0) {
              plm(bxi, idir,
                  prim_arr, qthermo_arr, flatten_arr,
                  magfield_x_arr, magfield_y_arr, magfield_z_arr,
                  prim_left[idir].array(), prim_right[idir].array(),
                  prim_source_arr, dt);

            } else {
              parabolic_reconstruct_mhd(bxi, idir,
                      prim_arr, qthermo_arr, flatten_arr,
                      magfield_x_arr, magfield_y_arr, magfield_z_arr,
                      prim_left[idir].array(), prim_right[idir].array(),
                      prim_source_arr, dt);
            }
          }




          const Box& bfx = amrex::grow(nbx, IntVect(2, 3, 3));

          flux_x_1d.resize(bfx, NUM_STATE+3);
          auto flux_x_1d_arr = flux_x_1d.array();
          auto flux_x_1d_hold = flux_x_1d.elixir();

          mhd_riemann_flux(bfx, prim_left[0].array(), prim_right[0].array(), flux_x_1d_arr, 0);

          const Box& bfy = amrex::grow(nby, IntVect(3, 2, 3));

          flux_y_1d.resize(bfy, NUM_STATE+3);
          auto flux_y_1d_arr = flux_y_1d.array();
          auto flux_y_1d_hold = flux_y_1d.elixir();

          mhd_riemann_flux(bfy, prim_left[1].array(), prim_right[1].array(), flux_y_1d_arr, 1);

          const Box& bfz = amrex::grow(nbz, IntVect(3, 3, 2));

          flux_z_1d.resize(bfz, NUM_STATE+3);
          auto flux_z_1d_arr = flux_z_1d.array();
          auto flux_z_1d_hold = flux_z_1d.elixir();

          mhd_riemann_flux(bfz, prim_left[2].array(), prim_right[2].array(), flux_z_1d_arr, 2);



          cons_x_left.resize(gbx, NUM_STATE+3);
          auto cons_x_left_arr = cons_x_left.array();
          auto cons_x_left_hold = cons_x_left.elixir();

          cons_x_right.resize(gbx, NUM_STATE+3);
          auto cons_x_right_arr = cons_x_right.array();
          auto cons_x_right_hold = cons_x_right.elixir();

          PrimitiveToConserved(gbx, prim_x_left_arr, cons_x_left_arr);
          PrimitiveToConserved(gbx, prim_x_right_arr, cons_x_right_arr);

          cons_y_left.resize(gbx, NUM_STATE+3);
          auto cons_y_left_arr = cons_y_left.array();
          auto cons_y_left_hold = cons_y_left.elixir();

          cons_y_right.resize(gbx, NUM_STATE+3);
          auto cons_y_right_arr = cons_y_right.array();
          auto cons_y_right_hold = cons_y_right.elixir();

          PrimitiveToConserved(gbx, prim_y_left_arr, cons_y_left_arr);
          PrimitiveToConserved(gbx, prim_y_right_arr, cons_y_right_arr);

          cons_z_left.resize(gbx, NUM_STATE+3);
          auto cons_z_left_arr = cons_z_left.array();
          auto cons_z_left_hold = cons_z_left.elixir();

          cons_z_right.resize(gbx, NUM_STATE+3);
          auto cons_z_right_arr = cons_z_right.array();
          auto cons_z_right_hold = cons_z_right.elixir();

          PrimitiveToConserved(gbx, prim_z_left_arr, cons_z_left_arr);
          PrimitiveToConserved(gbx, prim_z_right_arr, cons_z_right_arr);


          Box emf_box_x_wide = amrex::grow(bx, 2);
          emf_box_x_wide.growHi(1);
          emf_box_x_wide.growHi(2);

          edge_emf_x(emf_box_x_wide, prim_arr, emf_x_arr, flux_y_1d_arr, flux_z_1d_arr);


          Box emf_box_y_wide = amrex::grow(bx, 2);
          emf_box_y_wide.growHi(0);
          emf_box_y_wide.growHi(2);

          edge_emf_y(emf_box_y_wide, prim_arr, emf_y_arr, flux_x_1d_arr, flux_z_1d_arr);


          Box emf_box_z_wide = amrex::grow(bx, 2);
          emf_box_z_wide.growHi(0);
          emf_box_z_wide.growHi(1);

          edge_emf_z(emf_box_z_wide, prim_arr, emf_z_arr, flux_x_1d_arr, flux_y_1d_arr);




          const Box& ccbx = amrex::grow(nbx, IntVect(1, 2, 2));

          corner_left_state.resize(gbx, NQ);
          auto corner_left_state_arr = corner_left_state.array();
          auto corner_left_state_hold = corner_left_state.elixir();

          corner_right_state.resize(gbx, NQ);
          auto corner_right_state_arr = corner_right_state.array();
          auto corner_right_state_hold = corner_right_state.elixir();

          corner_coupling(ccbx,
                        corner_right_state_arr, corner_left_state_arr,
                        cons_x_right_arr, cons_x_left_arr,
                        flux_y_1d_arr, emf_x_arr, emf_z_arr,
                        0, 1, 2, dt);

          trans_flux_xy.resize(ccbx, NUM_STATE+3);
          auto flux_xy_arr = trans_flux_xy.array();
          auto flux_xy_hold = trans_flux_xy.elixir();

          mhd_riemann_flux(ccbx, corner_left_state_arr, corner_right_state_arr, flux_xy_arr, 0);

          corner_coupling(ccbx,
                        corner_right_state_arr, corner_left_state_arr,
                        cons_x_right_arr, cons_x_left_arr,
                        flux_z_1d_arr, emf_x_arr, emf_y_arr,
                        0, 2, 1, dt);

          trans_flux_xz.resize(ccbx, NUM_STATE+3);
          auto flux_xz_arr = trans_flux_xz.array();
          auto flux_xz_hold = trans_flux_xz.elixir();

          mhd_riemann_flux(ccbx, corner_left_state_arr, corner_right_state_arr, flux_xz_arr, 0);



          const Box& corner_box_y = amrex::grow(nby, IntVect(2, 1, 2));

          corner_coupling(corner_box_y,
                        corner_right_state_arr, corner_left_state_arr,
                        cons_y_right_arr, cons_y_left_arr,
                        flux_x_1d_arr, emf_y_arr, emf_z_arr,
                        1, 0, 2, dt);

          trans_flux_yx.resize(corner_box_y, NUM_STATE+3);
          auto flux_yx_arr = trans_flux_yx.array();
          auto flux_yx_hold = trans_flux_yx.elixir();

          mhd_riemann_flux(corner_box_y, corner_left_state_arr, corner_right_state_arr, flux_yx_arr, 1);


          corner_coupling(corner_box_y,
                        corner_right_state_arr, corner_left_state_arr,
                        cons_y_right_arr, cons_y_left_arr,
                        flux_z_1d_arr, emf_y_arr, emf_x_arr,
                        1, 2, 0, dt);

          trans_flux_yz.resize(corner_box_y, NUM_STATE+3);
          auto flux_yz_arr = trans_flux_yz.array();
          auto flux_yz_hold = trans_flux_yz.elixir();

          mhd_riemann_flux(corner_box_y, corner_left_state_arr, corner_right_state_arr, flux_yz_arr, 1);


          const Box& corner_box_z = amrex::grow(nbz, IntVect(2, 2, 1));

          corner_coupling(corner_box_z,
                        corner_right_state_arr, corner_left_state_arr,
                        cons_z_right_arr, cons_z_left_arr,
                        flux_x_1d_arr, emf_z_arr, emf_y_arr,
                        2, 0, 1, dt);

          trans_flux_zx.resize(corner_box_z, NUM_STATE+3);
          auto flux_zx_arr = trans_flux_zx.array();
          auto flux_zx_hold = trans_flux_zx.elixir();

          mhd_riemann_flux(corner_box_z, corner_left_state_arr, corner_right_state_arr, flux_zx_arr, 2);


          corner_coupling(corner_box_z,
                        corner_right_state_arr, corner_left_state_arr,
                        cons_z_right_arr, cons_z_left_arr,
                        flux_y_1d_arr, emf_z_arr, emf_x_arr,
                        2, 1, 0, dt);

          trans_flux_zy.resize(corner_box_z, NUM_STATE+3);
          auto flux_zy_arr = trans_flux_zy.array();
          auto flux_zy_hold = trans_flux_zy.elixir();

          mhd_riemann_flux(corner_box_z, corner_left_state_arr, corner_right_state_arr, flux_zy_arr, 2);



          amrex::ParallelFor(ccbx, NUM_STATE+3,
          [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
          {
            flux_x_1d_arr(i,j,k,n) = 0.5_rt * (flux_xy_arr(i,j,k,n) + flux_xz_arr(i,j,k,n));
          });

          amrex::ParallelFor(corner_box_y, NUM_STATE+3,
          [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
          {
            flux_y_1d_arr(i,j,k,n) = 0.5_rt * (flux_yx_arr(i,j,k,n) + flux_yz_arr(i,j,k,n));
          });

          amrex::ParallelFor(corner_box_z, NUM_STATE+3,
          [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
          {
            flux_z_1d_arr(i,j,k,n) = 0.5_rt * (flux_zx_arr(i,j,k,n) + flux_zy_arr(i,j,k,n));
          });


          Box emf_box_x_narrow = amrex::grow(bx, 1);
          emf_box_x_narrow.growHi(1);
          emf_box_x_narrow.growHi(2);

          edge_emf_x(emf_box_x_narrow, prim_arr, emf_x_arr, flux_y_1d_arr, flux_z_1d_arr);


          Box emf_box_y_narrow = amrex::grow(bx, 1);
          emf_box_y_narrow.growHi(0);
          emf_box_y_narrow.growHi(2);

          edge_emf_y(emf_box_y_narrow, prim_arr, emf_y_arr, flux_x_1d_arr, flux_z_1d_arr);


          Box emf_box_z_narrow = amrex::grow(bx, 1);
          emf_box_z_narrow.growHi(0);
          emf_box_z_narrow.growHi(1);

          edge_emf_z(emf_box_z_narrow, prim_arr, emf_z_arr, flux_x_1d_arr, flux_y_1d_arr);



          const Box& nodal_grown_box = amrex::grow(nbx, IntVect(0, 1, 1));

          predict_half_step(nodal_grown_box,
                    corner_right_state_arr, corner_left_state_arr,
                    cons_x_right_arr, cons_x_left_arr,
                    flux_yz_arr, flux_zy_arr,
                    emf_x_arr, emf_y_arr, emf_z_arr,
                    0, 1, 2, dt);



          mhd_riemann_flux(nodal_grown_box, corner_left_state_arr, corner_right_state_arr, flux[0].array(), 0);

          const Box& nodal_box_y_half = amrex::grow(nby, IntVect(1, 0, 1));

          predict_half_step(nodal_box_y_half,
                    corner_right_state_arr, corner_left_state_arr,
                    cons_y_right_arr, cons_y_left_arr,
                    flux_xz_arr, flux_zx_arr,
                    emf_y_arr, emf_x_arr, emf_z_arr,
                    1, 0, 2, dt);

          mhd_riemann_flux(nodal_box_y_half, corner_left_state_arr, corner_right_state_arr, flux[1].array(), 1);

          const Box& nodal_box_z_half = amrex::grow(nbz, IntVect(1, 1, 0));

          predict_half_step(nodal_box_z_half,
                    corner_right_state_arr, corner_left_state_arr,
                    cons_z_right_arr, cons_z_left_arr,
                    flux_xy_arr, flux_yx_arr,
                    emf_z_arr, emf_x_arr, emf_y_arr,
                    2, 0, 1, dt);

          mhd_riemann_flux(nodal_box_z_half, corner_left_state_arr, corner_right_state_arr, flux[2].array(), 2);


          q2D.resize(obx, NQ);
          auto prim_half_state_arr = q2D.array();
          auto prim_half_state_hold = q2D.elixir();

          primitive_half_update(obx, prim_half_state_arr, prim_arr,
                    flux_x_1d_arr, flux_y_1d_arr, flux_z_1d_arr, dt);


          Box emf_box_x_final = mfi.tilebox();
          emf_box_x_final.growHi(1, 1);
          emf_box_x_final.growHi(2, 1);

          edge_emf_x(emf_box_x_final, prim_half_state_arr, emf_x_arr, flux_y_arr, flux_z_arr);

          Box emf_box_y_final = mfi.tilebox();
          emf_box_y_final.growHi(0, 1);
          emf_box_y_final.growHi(2, 1);

          edge_emf_y(emf_box_y_final, prim_half_state_arr, emf_y_arr, flux_x_arr, flux_z_arr);

          Box emf_box_z_final = mfi.tilebox();
          emf_box_z_final.growHi(0, 1);
          emf_box_z_final.growHi(1, 1);

          edge_emf_z(emf_box_z_final, prim_half_state_arr, emf_z_arr, flux_x_arr, flux_y_arr);


          div.resize(obx, 1);
          Elixir div_hold = div.elixir();
          auto divergence_arr = div.array();

          divu(obx, prim_arr, divergence_arr);

          for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

            const Box& nbox = amrex::surroundingNodes(bx, idir);

            Array4<Real> const dir_flux_arr = (flux[idir]).array();

            amrex::ParallelFor(nbox,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
              dir_flux_arr(i,j,k,UTEMP) = 0.e0;
#ifdef SHOCK_VAR
              dir_flux_arr(i,j,k,UFLAG_SHOCK) = 0.e0;
#endif
            });

            apply_viscosity(nbox, idir, divergence_arr, state_view, dir_flux_arr);

            balance_species_flux(nbox, dir_flux_arr);

          }



          cons_update_mhd(bx, dt, updated_state_arr, flux_x_arr, flux_y_arr, flux_z_arr);


          Real dtdx = dt / dx[0];

          amrex::ParallelFor(nbx,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
            Bx_new_arr(i,j,k) = magfield_x_arr(i,j,k) + dtdx *
              ((emf_y_arr(i,j,k+1) - emf_y_arr(i,j,k)) - (emf_z_arr(i,j+1,k) - emf_z_arr(i,j,k)));
          });

#if AMREX_SPACEDIM >= 2
          dtdx = dt / dx[1];
#else
          dtdx = 0.0_rt;
#endif

          amrex::ParallelFor(nby,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
            By_new_arr(i,j,k) = magfield_y_arr(i,j,k) + dtdx *
              ((emf_z_arr(i+1,j,k) - emf_z_arr(i,j,k)) - (emf_x_arr(i,j,k+1) - emf_x_arr(i,j,k)));
          });

#if AMREX_SPACEDIM == 3
          dtdx = dt / dx[2];
#else
          dtdx = 0.0_rt;
#endif

          amrex::ParallelFor(nbz,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
            Bz_new_arr(i,j,k) = magfield_z_arr(i,j,k) + dtdx *
              ((emf_x_arr(i,j+1,k) - emf_x_arr(i,j,k)) - (emf_y_arr(i+1,j,k) - emf_y_arr(i,j,k)));
          });






          for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {

            Array4<Real> const dir_flux_fab = (flux[idir]).array();
            Array4<Real> stored_fluxes_fab = (*fluxes[idir]).array(mfi);
            const int numcomp = NUM_STATE;

            if (time_scheme == SimplifiedIterCorrection) {

              AMREX_HOST_DEVICE_FOR_4D(mfi.nodaltilebox(idir), numcomp, i, j, k, n,
              {
                stored_fluxes_fab(i,j,k,n) = dir_flux_fab(i,j,k,n);
              });

            } else {

              AMREX_HOST_DEVICE_FOR_4D(mfi.nodaltilebox(idir), numcomp, i, j, k, n,
              {
                stored_fluxes_fab(i,j,k,n) += dir_flux_fab(i,j,k,n);
              });

            }


            Array4<Real> density_flux_fab = (*mass_face_flux[idir]).array(mfi);

            AMREX_HOST_DEVICE_FOR_4D(mfi.nodaltilebox(idir), 1, i, j, k, n,
            {
              density_flux_fab(i,j,k,0) = dir_flux_fab(i,j,k,URHO);
            });

          }

        }

    }


    status = scan_for_negative_mass_density();

    if (status.success == false) {
        return status;
    }


    clean_state(magfield_x_new, magfield_y_new, magfield_z_new, S_new, time + dt, 0);


    scan_for_nan(S_new);


    if (parent->subcyclingMode() == "None") {
        if (do_reflux == 1) {
            FluxRegCoarseInit();
            FluxRegFineAccum();
        }
    }

    return status;
}

void
HydroSim::cons_update_mhd(const Box& bx, const Real dt,
                   Array4<Real> const& cons_new,
                   Array4<Real const> const& xface_flux,
                   Array4<Real const> const& flux1,
                   Array4<Real const> const& flux2) {


  const auto dx = geom.CellSizeArray();

  Real dxinv = 1.0_rt/dx[0];
#if AMREX_SPACEDIM >= 2
  Real dyinv = 1.0_rt/dx[1];
#endif
#if AMREX_SPACEDIM == 3
  Real dzinv = 1.0_rt/dx[2];
#endif

  amrex::ParallelFor(bx, NUM_STATE,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {

    if (n == UTEMP) {
      cons_new(i,j,k,n) = 0.0_rt;
#ifdef SHOCK_VAR
    } else if (n == UFLAG_SHOCK) {
      cons_new(i,j,k,n) = 0.0_rt;
#endif
    } else {
      cons_new(i,j,k,n) += dt * (xface_flux(i,j,k,n) - xface_flux(i+1,j,k,n)) * dxinv;
#if AMREX_SPACEDIM >= 2
      cons_new(i,j,k,n) += dt * (flux1(i,j,k,n) - flux1(i,j+1,k,n)) * dyinv;
#endif
#if AMREX_SPACEDIM == 3
      cons_new(i,j,k,n) += dt * (flux2(i,j,k,n) - flux2(i,j,k+1,n)) * dzinv;
#endif
    }

  });

}
