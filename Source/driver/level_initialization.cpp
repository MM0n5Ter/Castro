#ifndef WIN32
#include <unistd.h>
#endif

#include <iomanip>

#include <algorithm>
#include <cstdio>
#include <vector>
#include <iostream>
#include <string>
#include <ctime>
#include <memory>
#include <numbers>
#include <format>

#include <AMReX_Utility.H>
#include <AMReX_CONSTANTS.H>
#include <HydroSim.H>
#include <global.H>
#include <runtime_param_init.H>
#include <AMReX_VisMF.H>
#include <AMReX_TagBox.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_ParmParse.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef AMREX_PARTICLES
#include <AMReX_Particles.H>
#endif

#ifdef GRAVITY
#include <Gravity.H>
#endif

#ifdef DIFFUSION
#include <diffusion_operator.H>
#endif

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <extern_parameters.H>
#include <prob_parameters.H>

#include <user_initialize.H>
#include <user_initialize_state_data.H>
#ifdef MHD
#include <user_initialize_mhd_data.H>
#endif
#ifdef RADIATION
#include <user_initialize_rad_data.H>
#endif

#include <ambient.H>
#include <hydrosim_limits.H>

#include <solver_constants.H>

using namespace amrex;

void
HydroSim::init (AmrLevel &old)
{
    BL_PROFILE("HydroSim::init(old)");

    auto* oldlev = dynamic_cast<HydroSim*>(&old);

    Real dt_new    = parent->dtLevel(level);
    Real cur_time  = oldlev->state[State_Type].curTime();
    Real prev_time = oldlev->state[State_Type].prevTime();
    Real dt_old    = cur_time - prev_time;
    setTimeLevel(cur_time,dt_old,dt_new);

    for (int s = 0; s < num_field_types; ++s) {
        MultiFab& state_field = get_new_data(s);
        FillPatch(old, state_field, state_field.nGrow(), cur_time, s, 0, state_field.nComp());
        if (oldlev->state[s].hasOldData()) {
            if (!state[s].hasOldData()) {
                state[s].allocOldData();
            }
            MultiFab& old_state_field = get_old_data(s);
            FillPatch(old, old_state_field, old_state_field.nGrow(), prev_time, s, 0, old_state_field.nComp());
        }
    }

    iteration = oldlev->iteration;
    sub_pass = oldlev->sub_pass;

    sub_cycles = oldlev->sub_cycles;
    dt_cycle_step = oldlev->dt_cycle_step;
    dt_level_step = oldlev->dt_level_step;

    retain_prev_field = oldlev->retain_prev_field;

    retry_active = oldlev->retry_active;

}

void
HydroSim::init ()
{
    BL_PROFILE("HydroSim::init()");

    Real dt        = parent->dtLevel(level);
    Real cur_time  = getLevel(level-1).state[State_Type].curTime();
    Real prev_time = getLevel(level-1).state[State_Type].prevTime();

    Real dt_old = (cur_time - prev_time) / static_cast<Real>(parent->MaxRefRatio(level-1));

    Real time = cur_time;

    if (getLevel(level-1).post_step_regrid) {
      time = prev_time;
    }

    setTimeLevel(time,dt_old,dt);

    for (int s = 0; s < num_field_types; ++s) {
        MultiFab& state_field = get_new_data(s);
        FillCoarsePatch(state_field, 0, time, s, 0, state_field.nComp(), state_field.nGrow());
    }
}

void
HydroSim::initData ()
{
    BL_PROFILE("HydroSim::initData()");

#if AMREX_SPACEDIM > 1
    const Real* dx  = geom.CellSize();
#endif
    MultiFab& S_new = get_new_data(State_Type);
    Real cur_time   = state[State_Type].curTime();

    S_new.setVal(0.);

    if (! allow_anisotropic_cells) {

#if (AMREX_SPACEDIM == 2)
        const Real SMALL = 1.e-13;
        if (std::abs(dx[0] - dx[1]) > SMALL*dx[0])
          {
            amrex::Abort("We don't support dx != dy");
          }
#elif (AMREX_SPACEDIM == 3)
        const Real SMALL = 1.e-13;
        if ( (std::abs(dx[0] - dx[1]) > SMALL*dx[0]) || (std::abs(dx[0] - dx[2]) > SMALL*dx[0]) )
          {
            amrex::Abort("We don't support dx != dy != dz");
          }
#endif

    }

    if (verbose && ParallelDescriptor::IOProcessor()) {
      std::cout << "Initializing the data at level " << level << std::endl;
    }

#ifdef MHD
   MultiFab& magfield_x_new   = get_new_data(Bfield_Type_x);
   magfield_x_new.setVal(0.0);

   MultiFab& magfield_y_new   = get_new_data(Bfield_Type_y);
   magfield_y_new.setVal(0.0);

   MultiFab& magfield_z_new  =  get_new_data(Bfield_Type_z);
   magfield_z_new.setVal(0.0);

#endif

    Gpu::Device::profilerStop();

#ifdef RADIATION
    MultiFab &radiation_new = get_new_data(Radiation_Type);
    radiation_new.setVal(0.);
#endif

#ifdef REACTIONS
    MultiFab &reaction_new = get_new_data(Burning_Type);
    reaction_new.setVal(0.);
#endif

#ifdef REACTIONS
   if (time_scheme == SimplifiedIterCorrection) {
       MultiFab& reaction_src_new = get_new_data(Burn_Corr_Type);
       reaction_src_new.setVal(0.0, reaction_src_new.nGrow());
   }
#endif

#ifdef LOWMACH_INIT
    LOWMACH_init();
#else
    {

#ifdef MHD
       magfield_x_new.setVal(0.0);
       magfield_y_new.setVal(0.0);
       magfield_z_new.setVal(0.0);

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

          auto geomdata = geom.data();

          auto magfield_x_arr = magfield_x_new.array(mfi);
          auto magfield_y_arr = magfield_y_new.array(mfi);
          auto magfield_z_arr = magfield_z_new.array(mfi);

          const Box& face_box_x = mfi.nodaltilebox(0);

          amrex::ParallelFor(face_box_x,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              user_initialize_mhd_data(i, j, k, magfield_x_arr, 0, geomdata);
          });

          const Box& face_box_y = mfi.nodaltilebox(1);

          amrex::ParallelFor(face_box_y,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              user_initialize_mhd_data(i, j, k, magfield_y_arr, 1, geomdata);
          });

          const Box& face_box_z = mfi.nodaltilebox(2);

          amrex::ParallelFor(face_box_z,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              user_initialize_mhd_data(i, j, k, magfield_z_arr, 2, geomdata);
          });

       }

#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
       {
          const Box& box = mfi.tilebox();

          auto s = S_new[mfi].array();
          auto geomdata = geom.data();

#ifdef RNG_STATE_INIT
          amrex::ParallelForRNG(box,
          [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
          {
              user_initialize_state_data(i, j, k, s, geomdata, engine);
          });
#else
          amrex::ParallelFor(box,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              user_initialize_state_data(i, j, k, s, geomdata);
          });
#endif
       }

#ifdef MHD
      add_bfield_energy(magfield_x_new, magfield_y_new, magfield_z_new, S_new);

      check_div_bfield(magfield_x_new, magfield_y_new, magfield_z_new, S_new);

#endif

       computeTemp(
#ifdef MHD
                   magfield_x_new, magfield_y_new, magfield_z_new,
#endif
                   S_new, cur_time, 0);

       ReduceOps<ReduceOpSum, ReduceOpSum> reduce_op;
       ReduceData<int, int> reduce_data(reduce_op);
       using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
       {
           const Box& bx = mfi.tilebox();

           auto cons_state_arr = S_new.array(mfi);

           Real ltemp_floor = small_temp;
           Real ldensity_floor = small_dens;

           reduce_op.eval(bx, reduce_data,
           [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
           {
               int temp_failed = 0;
               if (cons_state_arr(i,j,k,UTEMP) < ltemp_floor * 1.001) {
                   temp_failed = 1;
               }

               int density_failed = 0;
               if (cons_state_arr(i,j,k,URHO) < ldensity_floor * 1.001) {
                   density_failed = 1;
               }

               return {temp_failed, density_failed};
           });

       }

       ReduceTuple hv = reduce_data.value();
       int init_temp_failed   = amrex::get<0>(hv);
       int init_density_failed = amrex::get<1>(hv);

       if (init_density_failed != 0) {
         amrex::Error("Error: initial data has rho <~ small_dens");
       }

       if (init_temp_failed != 0) {
         amrex::Error("Error: initial data has T <~ small_temp");
       }

#ifdef HYBRID_MOMENTUM

       linear_to_split_momentum(S_new, 0);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
       for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
         const Box& bx = mfi.tilebox();

         auto cons_state_arr = S_new.array(mfi);

         amrex::ParallelFor(bx,
         [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
         {
           Real species_sum = 0.0_rt;
           for (int n = 0; n < NumSpec; n++) {
             species_sum += cons_state_arr(i,j,k,UFS+n);
           }
           if (std::abs(cons_state_arr(i,j,k,URHO) - species_sum) > 1.e-8_rt * cons_state_arr(i,j,k,URHO)) {
#ifndef AMREX_USE_GPU
             std::cout << "Sum of (rho X)_i vs rho at (i,j,k): "
                       << i << " " << j << " " << k << " "
                       << species_sum << " " << cons_state_arr(i,j,k,URHO) << std::endl;
#endif
             amrex::Error("Error: failed check of initial species summing to 1");
           }
         });
       }

#ifdef TRUE_SDC
       auto convert_to_fourth_order = [&] (MultiFab& mf, const MultiFab& layout, const Box& domain) {
           auto domain_lo = domain.loVect3d();
           auto domain_hi = domain.hiVect3d();

           FArrayBox tmp;

           for (MFIter mfi(layout); mfi.isValid(); ++mfi)
             {
               const Box& box = mfi.validbox();

               tmp.resize(box, 1, The_Async_Arena());
               auto tmp_arr = tmp.array();

               build_fourth_in_place(box, mf.array(mfi), tmp_arr, domain_lo, domain_hi);
             }
       };

       if (init_cell_averaged == 0) {

         reconcile_energy(
#ifdef MHD
                            magfield_x_new, magfield_y_new,magfield_z_new,
#endif
                            S_new);

#ifndef AMREX_USE_GPU
         if (sdc_accuracy_order == 4) {
           Sborder.define(grids, dmap, NUM_STATE, NUM_GROW);
           AmrLevel::FillPatch(*this, Sborder, NUM_GROW, cur_time, State_Type, 0, NUM_STATE);

           convert_to_fourth_order(Sborder, S_new, geom.Domain());

           MultiFab::Copy(S_new, Sborder, 0, 0, NUM_STATE, 0);
           Sborder.clear();
         }
#endif
       } else {

         Sborder.define(grids, dmap, NUM_STATE, NUM_GROW);
         AmrLevel::FillPatch(*this, Sborder, NUM_GROW, cur_time, State_Type, 0, NUM_STATE);

         auto domain_lo = geom.Domain().loVect3d();
         auto domain_hi = geom.Domain().hiVect3d();

         FArrayBox tmp;

         for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
           {
             const Box& box = mfi.growntilebox(2);

             tmp.resize(box, 1, The_Async_Arena());
             auto tmp_arr = tmp.array();

             build_center_state_in_place(box, Sborder.array(mfi), tmp_arr, domain_lo, domain_hi);
           }

         for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
           {
             const Box& box = mfi.growntilebox(1);

             auto cons_state_arr = Sborder.array(mfi);

             amrex::ParallelFor(box,
             [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
             {

               Real rhoInv = 1.0_rt / cons_state_arr(i,j,k,URHO);
               Real u = cons_state_arr(i,j,k,UMX) * rhoInv;
               Real v = cons_state_arr(i,j,k,UMY) * rhoInv;
               Real w = cons_state_arr(i,j,k,UMZ) * rhoInv;

               eos_re_t eos_state;
               eos_state.rho = cons_state_arr(i,j,k,URHO);
               eos_state.T = cons_state_arr(i,j,k,UTEMP);
               eos_state.e = cons_state_arr(i,j,k,UEDEN) * rhoInv - 0.5_rt * (u*u + v*v + w*w);
               for (int n = 0; n < NumSpec; n++) {
                 eos_state.xn[n] = cons_state_arr(i,j,k,UFS+n) * rhoInv;
               }
#if NAUX_NET > 0
               for (int n = 0; n < NumAux; n++) {
                 eos_state.aux[n] = cons_state_arr(i,j,k,UFX+n) * rhoInv;
               }
#endif

               eos(eos_input_re, eos_state);

               cons_state_arr(i,j,k,UTEMP) = eos_state.T;

               cons_state_arr(i,j,k,UEINT) = eos_state.rho * eos_state.e;
             });
           }

         convert_to_fourth_order(Sborder, S_new, geom.Domain());

         MultiFab::Copy(S_new, Sborder, UEINT, UEINT, 1, 0);
         MultiFab::Copy(S_new, Sborder, UTEMP, UTEMP, 1, 0);
         Sborder.clear();

       }
#else
       reconcile_energy(
#ifdef MHD
                            magfield_x_new, magfield_y_new,magfield_z_new,
#endif
                            S_new);
#endif

       int ng = S_new.nGrow();

       if (ng > 0) {
         AmrLevel::FillPatch(*this, S_new, ng, cur_time, State_Type, 0, S_new.nComp());
       }
    }

    clean_state(
#ifdef MHD
                    magfield_x_new, magfield_y_new, magfield_z_new,
#endif
                    S_new, cur_time, S_new.nGrow());

#ifdef RADIATION
    if (do_radiative_transfer) {
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
      for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
          const Box& box = mfi.tilebox();

          auto r = radiation_new[mfi].array();
          auto geomdata = geom.data();

          GpuArray<Real, NGROUPS+1> nu_edge_pass = {0.0};
          GpuArray<Real, NGROUPS> nu_center_pass = {0.0};
          GpuArray<Real, NGROUPS> nu_width_pass = {0.0};
#if NGROUPS > 1
          for (int g = 0; g <= NGROUPS; g++) {
              nu_edge_pass[g] = radiation->xnu[g];
          }
          for (int g = 0; g < NGROUPS; g++) {
              nu_center_pass[g] = radiation->freq_center[g];
              nu_width_pass[g] = radiation->freq_width[g];
          }
#endif

          amrex::ParallelFor(box,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {

              user_initialize_rad_data(i, j, k, r, nu_edge_pass, nu_center_pass, nu_width_pass, geomdata);
          });
      }
    }
#endif

#endif

#ifdef GRAVITY
    MultiFab& gravity_accel_new = get_new_data(Accel_Type);
    gravity_accel_new.setVal(0.);

    MultiFab& phi_new = get_new_data(Potential_Type);
    phi_new.setVal(0.);
#endif

    MultiFab& source_terms_new = get_new_data(Forcing_Type);
    source_terms_new.setVal(0., source_terms_new.nGrow());

#ifdef AMREX_PARTICLES
    if (level == 0) {
      init_particles();
    }
#endif

    Gpu::Device::profilerStart();

    if (verbose && ParallelDescriptor::IOProcessor()) {
      std::cout << "Done initializing the level " << level << " data " << std::endl;
    }
}

#ifdef GRAVITY
int
HydroSim::count_points_1d ()
{
     int grid_points_1d;

     Box bx(geom.Domain());
     long nx = bx.size()[0];

#if (AMREX_SPACEDIM == 1)
     grid_points_1d = static_cast<int>(nx);
#elif (AMREX_SPACEDIM == 2)
     long ny = bx.size()[1];
     Real diag_length_sq = Real(nx*nx + ny*ny);
     grid_points_1d = int(std::sqrt(diag_length_sq))+2*NUM_GROW;
#elif (AMREX_SPACEDIM == 3)
     long ny = bx.size()[1];
     long nz = bx.size()[2];
     Real diag_length_sq = Real(nx*nx + ny*ny + nz*nz);
     grid_points_1d = int(std::sqrt(diag_length_sq))+2*NUM_GROW;
#endif

     if (verbose && ParallelDescriptor::IOProcessor()) {
         std::cout << "HydroSim::grid_points_1d at level  " << level << " is " << grid_points_1d << std::endl;
     }

     return grid_points_1d;
}
#endif
