#include <HydroSim.H>
#include <HydroSim_helpers.H>

#ifdef DIFFUSION
#include <conduction_util.H>
#endif

#include <transport_util.H>

#include <fourth_order_center.H>
#include <flatten.H>

using namespace amrex;

void
HydroSim::assemble_mol_hydro_source(Real time, Real dt, MultiFab& A_divergence)
{

    amrex::ignore_unused(time);

#ifdef RADIATION
  amrex::Abort("Error: radiation not supported for the MOL hydro source term");
#else

  BL_PROFILE("HydroSim::assemble_mol_hydro_source()");


  const Real strt_time = ParallelDescriptor::second();

  if (verbose && ParallelDescriptor::IOProcessor()) {
    std::cout << "... construct advection term, SDC iteration: " << correction_iter << "; current node: " << current_colloc_node << std::endl;
  }


  MultiFab& S_new = get_new_data(State_Type);

#if AMREX_SPACEDIM <= 2
  int coord = geom.Coord();
#endif

  BL_PROFILE_VAR("HydroSim::advance_hydro_ca_umdrv()", HS_ADVANCE_HYDRO);

#if AMREX_SPACEDIM >= 2
  GpuArray<int, 3> domain_lo = geom.Domain().loVect3d();
  GpuArray<int, 3> domain_hi = geom.Domain().hiVect3d();
#endif

#ifdef HYBRID_MOMENTUM
  GeometryData geomdata = geom.data();
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
  {


    FArrayBox flatten_factor(The_Async_Arena());
    FArrayBox cond(The_Async_Arena());
    FArrayBox dq(The_Async_Arena());
    FArrayBox prim_source_fab(The_Async_Arena());
    FArrayBox shk(The_Async_Arena());
    FArrayBox qm(The_Async_Arena()), qp(The_Async_Arena());
    FArrayBox div(The_Async_Arena());
    FArrayBox interp_state(The_Async_Arena());
    FArrayBox averaged_state(The_Async_Arena());
    FArrayBox face_center_state(The_Async_Arena());
    FArrayBox averaged_flux(The_Async_Arena());
    Vector<FArrayBox> flux, qe;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        flux.push_back(FArrayBox(The_Async_Arena()));
        qe.push_back(FArrayBox(The_Async_Arena()));
    }
#if AMREX_SPACEDIM <= 2
    FArrayBox scaled_pressure(The_Async_Arena());
#endif
    FArrayBox viscosity_field(The_Async_Arena());

    MultiFab& old_forcing = get_old_data(Forcing_Type);

    for (MFIter mfi(S_new, (sdc_accuracy_order == 4) ? untiled_size : hydro_tile_size); mfi.isValid(); ++mfi)
      {
        const Box& bx  = mfi.tilebox();

        const Box& obx = amrex::grow(bx, 1);
        const Box& box_grow2 = amrex::grow(bx, 2);
        const Box& grown_src_box = amrex::grow(bx, old_forcing.nGrow());

        Array4<Real const> const input_state_arr = Sborder.array(mfi);

        auto previous_source_arr = old_forcing.array(mfi);

        auto update_out_arr = A_divergence.array(mfi);

        Real quadrature_weight = 1.0;

        if (time_scheme == IterativeCorrection) {
          quadrature_weight = node_quad_weights[current_colloc_node];
        }

        flatten_factor.resize(obx, 1);

        Array4<Real const> const prim_arr = q.array(mfi);
        Array4<Real> const flatten_arr = flatten_factor.array();

        if (first_order_scheme == 1) {
            amrex::ParallelFor(obx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                flatten_arr(i,j,k) = 0.0;
            });
        } else if (use_shock_flatten == 1) {
            amrex::ParallelFor(obx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                flatten_arr(i,j,k) = hydro::flatten(i, j, k, prim_arr, QPRES);
            });
        } else {
            amrex::ParallelFor(obx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                flatten_arr(i,j,k) = 1.0;
            });
        }


#ifdef SHOCK_VAR
        bool shock_enabled = true;
#else
        bool shock_enabled = false;
#endif


        prim_source_fab.resize(grown_src_box, NQSOURCE);
        Array4<Real> const prim_source_arr = prim_source_fab.array();

        if (sdc_accuracy_order == 2) {

            amrex::ParallelFor(grown_src_box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                hydro::source_to_primitive(i, j, k, dt, input_state_arr, prim_arr, previous_source_arr, prim_source_arr);
            });
        }


        shk.resize(obx, 1);
        Array4<Real> const shock_flag_arr = shk.array();



        if (shock_flux_switch == 1 || shock_enabled) {
            shock(obx, prim_arr, previous_source_arr, shock_flag_arr);
        }
        else {
            amrex::ParallelFor(obx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                shock_flag_arr(i,j,k) = 0.0;
            });
        }

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

        auto qthermo_arr = qthermo.array(mfi);

        flux[0].resize(xbx, NUM_STATE);
        qe[0].resize(grown_x_box, NGDSTATE);

#if AMREX_SPACEDIM >= 2
        flux[1].resize(ybx, NUM_STATE);
        qe[1].resize(grown_y_box, NGDSTATE);
#endif

#if AMREX_SPACEDIM == 3
        flux[2].resize(zbx, NUM_STATE);
        qe[2].resize(grown_z_box, NGDSTATE);
#endif

        viscosity_field.resize(obx, 1);
        auto viscosity_field_arr = viscosity_field.array();

#ifndef AMREX_USE_GPU
        if (sdc_accuracy_order == 4) {


          Box ibx[AMREX_SPACEDIM];
          ibx[0] = amrex::grow(amrex::surroundingNodes(bx, 0), IntVect(AMREX_D_DECL(0,1,1)));
#if AMREX_SPACEDIM >= 2
          ibx[1] = amrex::grow(amrex::surroundingNodes(bx, 1), IntVect(AMREX_D_DECL(1,0,1)));
#endif
#if AMREX_SPACEDIM == 3
          ibx[2] = amrex::grow(amrex::surroundingNodes(bx, 2), IntVect(AMREX_D_DECL(1,1,0)));
#endif
          for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

            const Box& nbx = amrex::surroundingNodes(bx, idir);
            const Box& nodal_grown_box = amrex::grow(nbx, 1);

            qm.resize(box_grow2, NQ);
            auto face_minus_arr = qm.array();

            qp.resize(box_grow2, NQ);
            auto face_plus_arr = qp.array();

            interp_state.resize(nodal_grown_box, 1);
            auto interp_state_arr = interp_state.array();

            averaged_state.resize(ibx[idir], NQ);
            auto averaged_state_arr = averaged_state.array();

#if AMREX_SPACEDIM >= 2
            face_center_state.resize(nbx, NQ);
            auto face_center_state_arr = face_center_state.array();
#endif

            averaged_flux.resize(ibx[idir], NUM_STATE);
            auto averaged_flux_arr = averaged_flux.array();

            for (int n = 0; n < NQ; n++) {

              fourth_face_states(nodal_grown_box,
                                idir, n,
                                prim_arr, interp_state_arr);

              states(obx,
                     idir, n,
                     prim_arr, interp_state_arr, flatten_arr,
                     face_minus_arr, face_plus_arr);
            }


            compute_godunov_flux(ibx[idir],
                                face_minus_arr, face_plus_arr,
                                averaged_flux_arr, averaged_state_arr,
                                qthermo_arr,
                                shock_flag_arr,
                                idir, true);

            if (do_hydro == 0) {
                amrex::ParallelFor(ibx[idir], NUM_STATE,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    averaged_flux_arr(i,j,k,n) = 0.0;
                });

            }

#ifdef DIFFUSION
            if (conduction_enabled == 1) {
              bool holds_cell_average = true;
              fourth_accumulate_diffusion(ibx[idir],
                                        prim_arr, QTEMP,
                                        averaged_state_arr, averaged_flux_arr,
                                        idir, holds_cell_average);
            }
#endif


#if AMREX_SPACEDIM == 1
            Array4<Real> const dir_flux_arr = (flux[0]).array();

            amrex::ParallelFor(nbx, NUM_STATE,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
              dir_flux_arr(i,j,k,n) = averaged_flux_arr(i,j,k,n);
            });

#endif

#if AMREX_SPACEDIM >= 2
            const int* lo_bc = phys_bc.lo();
            const int* hi_bc = phys_bc.hi();

            GpuArray<bool, AMREX_SPACEDIM> lo_wrap;
            GpuArray<bool, AMREX_SPACEDIM> hi_wrap;
            for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
              lo_wrap[idir] = lo_bc[idir] == amrex::PhysBCType::interior;
              hi_wrap[idir] = hi_bc[idir] == amrex::PhysBCType::interior;
            }

            amrex::ParallelFor(nbx, NQ,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
              bool test = (n == QGC) || (n == QTEMP);

              if (test) return;

              Real lap = tangential_laplacian(i, j, k, n, idir, averaged_state_arr,
                                         lo_wrap, hi_wrap, domain_lo, domain_hi);

              face_center_state_arr(i,j,k,n) = averaged_state_arr(i,j,k,n) - (1.0_rt/24.0_rt) * lap;

            });

            Array4<Real> const face_flux_arr = (flux[idir]).array();

            flux_from_state(nbx, face_center_state_arr, face_flux_arr, idir);

            if (do_hydro == 0) {

              amrex::ParallelFor(nbx, NUM_STATE,
              [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
              {
                face_flux_arr(i,j,k,n) = 0.0;
              });

            }


#ifdef DIFFUSION
            if (conduction_enabled == 1) {
              bool holds_cell_average = false;
              fourth_accumulate_diffusion(nbx,
                                        Temperature_cc.array(mfi), 0,
                                        face_center_state_arr, flux[idir].array(),
                                        idir, holds_cell_average);
            }
#endif


            Array4<Real> const dir_flux_arr = (flux[idir]).array();

            amrex::ParallelFor(nbx, NUM_STATE,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {

              Real lap = tangential_laplacian(i, j, k, n, idir, averaged_flux_arr,
                                         lo_wrap, hi_wrap, domain_lo, domain_hi);

              dir_flux_arr(i,j,k,n) += (1.0_rt/24.0_rt) * lap;

            });

#endif

            if (do_hydro == 1) {

              const Real alpha = 0.3;

              Real viscosity_coef = alpha * (viscosity_coeff / 0.1);

              auto q_mean_arr = q_second.array(mfi);
              auto qthermo_mean_arr = qthermo_second.array(mfi);

              fourth_viscosity(nbx,
                           q_mean_arr, qthermo_mean_arr,
                           viscosity_field_arr, idir);

              amrex::ParallelFor(nbx, NUM_STATE,
              [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
              {
                  if (n == UTEMP) {
                      dir_flux_arr(i,j,k,n) = 0.0;
#ifdef SHOCK_VAR
                  } else if (n == UFLAG_SHOCK) {
                      dir_flux_arr(i,j,k,n) = 0.0;
#endif
#ifdef NSE_NET
                  } else if (n == UPROTON_MU || n == UNEUTRON_MU) {
                      dir_flux_arr(i,j,k,n) = 0.0;
#endif
                  } else {
                      if (idir == 0) {
                          dir_flux_arr(i,j,k,n) += viscosity_coef * viscosity_field_arr(i,j,k,0) *
                              (input_state_arr(i,j,k,n) - input_state_arr(i-1,j,k,n));

                      } else if (idir == 1) {
                          dir_flux_arr(i,j,k,n) += viscosity_coef * viscosity_field_arr(i,j,k,0) *
                              (input_state_arr(i,j,k,n) - input_state_arr(i,j-1,k,n));

                      } else {
                          dir_flux_arr(i,j,k,n) += viscosity_coef * viscosity_field_arr(i,j,k,0) *
                              (input_state_arr(i,j,k,n) - input_state_arr(i,j,k-1,n));

                      }
                  }

                });

            }

            auto godunov_state_arr = (qe[idir]).array();
            save_godunov_state(nbx, averaged_state_arr, godunov_state_arr);

          }


        } else {
#endif


          div.resize(obx, 1);
          auto divergence_arr = div.array();

          if (do_hydro) {
            divu(obx, prim_arr, divergence_arr);
          }

          const Box& tbx = amrex::grow(bx, 2);

          qm.resize(tbx, NQ);
          Array4<Real> const face_minus_arr = qm.array();

          qp.resize(tbx, NQ);
          Array4<Real> const face_plus_arr = qp.array();


          for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

            if (do_hydro) {

              if (ppm_recon_kind == 0) {

                dq.resize(obx, NQ);
                auto delta_q_arr = dq.array();

                lines_plm_reconstruct(obx, idir,
                                    prim_arr, flatten_arr, prim_source_arr,
                                    delta_q_arr,
                                    face_minus_arr, face_plus_arr);

              } else {

                lines_ppm_reconstruct(obx, idir,
                                    prim_arr, flatten_arr,
                                    face_minus_arr, face_plus_arr);
              }

              if (ppm_temp_mode == 1) {
                  edge_temp_to_pressure(obx, qm.array(), qp.array());
              }

              const Box& nbx = amrex::surroundingNodes(bx, idir);

              auto godunov_state_arr = (qe[idir]).array();
              auto dir_flux_arr = (flux[idir]).array();

              compute_godunov_flux
                (nbx,
                 face_minus_arr, face_plus_arr,
                 dir_flux_arr, godunov_state_arr,
                 qthermo_arr, shock_flag_arr,
                 idir, false);

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

              apply_viscosity(nbx, idir, divergence_arr, input_state_arr, dir_flux_arr);

            } else {
                const Box& nbx = amrex::surroundingNodes(bx, idir);
                const Box& gbx = amrex::grow(nbx, 1);

                Array4<Real> const dir_flux_arr = (flux[idir]).array();
                Array4<Real> const godunov_state_arr = (qe[idir]).array();

                amrex::ParallelFor(nbx, NUM_STATE,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    dir_flux_arr(i,j,k,n) = 0.e0;
                });

                amrex::ParallelFor(gbx, NGDSTATE,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {

                    godunov_state_arr(i,j,k,n) = 0.e0;
                });

            }

#ifdef DIFFUSION
            cond.resize(obx, 1);
            auto conductivity_arr = cond.array();

            fill_cell_conductivity(obx, Sborder.array(mfi), conductivity_arr);

            const Box& nbx = amrex::surroundingNodes(bx, idir);

            lines_diffusion_flux(nbx, idir,
                               input_state_arr, conductivity_arr,
                               flux[idir].array());
#endif

          }

#ifndef AMREX_USE_GPU
        }
#endif

        if (do_hydro) {

            for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

                const Box& nbx = amrex::surroundingNodes(bx, idir);

                Array4<Real> const dir_flux_arr = (flux[idir]).array();

                if (clip_small_rho_flux == 1) {
                    clip_fluxes_at_density_floor(nbx, idir,
                                                     Sborder.array(mfi),
                                                     volume.array(mfi),
                                                     flux[idir].array(),
                                                     area[idir].array(mfi),
                                                     dt);
                }

                balance_species_flux(nbx, dir_flux_arr);

            }

        }

        lines_cons_update(bx,
#ifdef SHOCK_VAR
                   shock_flag_arr,
#endif
                   previous_source_arr,
                   update_out_arr,
                   dt,
                   flux[0].array(),
#if AMREX_SPACEDIM >= 2
                   flux[1].array(),
#endif
#if AMREX_SPACEDIM == 3
                   flux[2].array(),
#endif
                   area[0].array(mfi),
#if AMREX_SPACEDIM >= 2
                   area[1].array(mfi),
#endif
#if AMREX_SPACEDIM == 3
                   area[2].array(mfi),
#endif
#if AMREX_SPACEDIM <= 2
                   qe[0].array(),
#if AMREX_SPACEDIM == 2
                   qe[1].array(),
#endif
#endif
                   volume.array(mfi));


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

          update_out_arr(i,j,k,UMR) -= ((loc[0] * InvRadius) * (x_edge_state_view(i+1,j,k,GDPRESSURE) - x_edge_state_view(i,j,k,GDPRESSURE)) / dx_arr[0] +
                                        (loc[1] * InvRadius) * (y_edge_state_view(i,j+1,k,GDPRESSURE) - y_edge_state_view(i,j,k,GDPRESSURE)) / dx_arr[1]);

        });
#endif

#if AMREX_SPACEDIM <= 2
        if (!Geom().IsCartesian()) {
          scaled_pressure.resize(xbx, 1);
        }

        Array4<Real> scaled_pressure_fab = scaled_pressure.array();
#endif

        for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

          const Box& nbx = amrex::surroundingNodes(bx, idir);

          Array4<Real> const dir_flux_arr = (flux[idir]).array();
          Array4<Real const> const face_area_arr = (area[idir]).array(mfi);

          scale_flux_by_area(nbx, dir_flux_arr, face_area_arr, dt);

#if AMREX_SPACEDIM <= 2

          if (idir == 0 && !momentum_flux_has_pressure(0, 0, coord)) {
              Array4<Real> const x_edge_state_arr = qe[idir].array();

              amrex::ParallelFor(nbx,
              [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
              {
                  scaled_pressure_fab(i,j,k) = x_edge_state_arr(i,j,k,GDPRESSURE) * dt;
              });
          }
#endif
        }



        if (time_scheme == IterativeCorrection &&
             (current_colloc_node == 0 || correction_iter == sdc_accuracy_order+sdc_extra_passes-1)) {

          for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

            Array4<Real> const dir_flux_fab = (flux[idir]).array();
            Array4<Real> stored_fluxes_fab = (*fluxes[idir]).array(mfi);
            const int numcomp = NUM_STATE;

            AMREX_HOST_DEVICE_FOR_4D(mfi.nodaltilebox(idir), numcomp, i, j, k, n,
            {
                stored_fluxes_fab(i,j,k,n) += quadrature_weight * dir_flux_fab(i,j,k,n);
            });

          }

#if AMREX_SPACEDIM <= 2
          if (!Geom().IsCartesian()) {

            Array4<Real> radial_pressure_fab = Pressure_radial.array(mfi);
            const Real scale = quadrature_weight;

            AMREX_HOST_DEVICE_FOR_4D(mfi.nodaltilebox(0), 1, i, j, k, n,
            {
                radial_pressure_fab(i,j,k,0) += scale * scaled_pressure_fab(i,j,k,0);
            });

          }
#endif
        }

      }

  }


  BL_PROFILE_VAR_STOP(HS_ADVANCE_HYDRO);

  if (log_source_changes) {
      measure_and_log_source_change(A_divergence, dt, "hydro source");
    }

    if (verbose > 0)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        if (ParallelDescriptor::IOProcessor())
          std::cout << "HydroSim::assemble_mol_hydro_source() time = " << run_time << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }

#endif
}
