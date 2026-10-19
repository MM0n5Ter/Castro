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
#include <runtime_parameters.H>
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
#include <Diffusion.H>
#endif

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <extern_parameters.H>
#include <prob_parameters.H>

#include <ambient.H>
#include <hydrosim_limits.H>

#include <solver_constants.H>

using namespace amrex;

void
HydroSim::computeTemp(
#ifdef MHD
                    MultiFab& Bx,
                    MultiFab& By,
                    MultiFab& Bz,
#endif

                    MultiFab& State, [[maybe_unused]] Real time, int ng)

{

  BL_PROFILE("HydroSim::computeTemp()");

  MultiFab StateCopy;

  MultiFab energy_laplacian;

#ifdef TRUE_SDC
  if (sdc_accuracy_order == 4) {

    StateCopy.define(State.boxArray(), State.DistributionMap(), NUM_STATE, 2);
    fill_ghost_field(StateCopy, time, StateCopy.nGrow());

    energy_laplacian.define(State.boxArray(), State.DistributionMap(), 1, 0);

    auto domain_lo = geom.Domain().loVect3d();
    auto domain_hi = geom.Domain().hiVect3d();

    FArrayBox tmp;

    for (MFIter mfi(StateCopy); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.growntilebox(1);
      const Box& bx0 = mfi.tilebox();

      laplacian_of_state(bx0, StateCopy.array(mfi), energy_laplacian.array(mfi), UEINT,
                       domain_lo, domain_hi);

      tmp.resize(bx, 1, The_Async_Arena());
      auto tmp_arr = tmp.array();

      build_center_state_in_place(bx, StateCopy.array(mfi), tmp_arr, domain_lo, domain_hi);

    }

  }
#endif

#ifdef TRUE_SDC
  if (sdc_accuracy_order == 4) {
    clamp_min_density(StateCopy, StateCopy.nGrow());
    repair_internal_energy(StateCopy, StateCopy.nGrow());
  } else {
#endif
    repair_internal_energy(
#ifdef MHD
                          Bx, By, Bz,
#endif
                          State, ng);
#ifdef TRUE_SDC
  }
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
  for (MFIter mfi(State, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
      int num_ghost = ng;
#ifdef TRUE_SDC
      if (sdc_accuracy_order == 4) {
          num_ghost = 1;
      }
#endif

      const Box& bx = mfi.growntilebox(num_ghost);

#ifdef TRUE_SDC
      FArrayBox& state_box_data = (sdc_accuracy_order == 4) ? StateCopy[mfi] : State[mfi];
#else
      FArrayBox& state_box_data = State[mfi];
#endif

      Array4<Real> const u = state_box_data.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

          Real rhoInv = 1.0_rt / u(i,j,k,URHO);

          eos_re_t eos_state;

          eos_state.rho = u(i,j,k,URHO);
          eos_state.T   = u(i,j,k,UTEMP);
          eos_state.e   = u(i,j,k,UEINT) * rhoInv;
          for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = u(i,j,k,UFS+n) * rhoInv;
          }
#if NAUX_NET > 0
          for (int n = 0; n < NumAux; ++n) {
            eos_state.aux[n] = u(i,j,k,UFX+n) * rhoInv;
          }
#endif

          eos(eos_input_re, eos_state);

          u(i,j,k,UTEMP) = eos_state.T;

      });

      if (clamp_ambient_temperature == 1) {
          amrex::ParallelFor(bx,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
          {
              Real rhoInv = 1.0_rt / u(i,j,k,URHO);

              if (u(i,j,k,URHO) <= hydrosim::ambient_threshold_factor * ambient::ambient_reference_state[URHO]) {
                  u(i,j,k,UTEMP) = ambient::ambient_reference_state[UTEMP];
                  u(i,j,k,UEINT) = ambient::ambient_reference_state[UEINT] * (u(i,j,k,URHO) * rhoInv);
                  u(i,j,k,UEDEN) = u(i,j,k,UEINT) + 0.5_rt * rhoInv * (u(i,j,k,UMX) * u(i,j,k,UMX) +
                                                                       u(i,j,k,UMY) * u(i,j,k,UMY) +
                                                                       u(i,j,k,UMZ) * u(i,j,k,UMZ));
              }
          });
      }
  }

#ifdef TRUE_SDC
  if (sdc_accuracy_order == 4) {

    auto domain_lo = geom.Domain().loVect3d();
    auto domain_hi = geom.Domain().hiVect3d();

    FArrayBox tmp;

    for (MFIter mfi(StateCopy); mfi.isValid(); ++mfi) {

      const Box& bx = mfi.tilebox();

      tmp.resize(bx, 1, The_Async_Arena());
      auto tmp_arr = tmp.array();

      build_fourth_in_place_n(bx, StateCopy.array(mfi), UTEMP, tmp_arr, domain_lo, domain_hi);
    }

    MultiFab::Add(StateCopy, energy_laplacian, 0, UEINT, 1, 0);

    MultiFab::Copy(State, StateCopy, UTEMP, UTEMP, 1, 0);
    MultiFab::Copy(State, StateCopy, UEINT, UEINT, 1, 0);

    if (ng > 0) {
      AmrLevel::FillPatch(*this, State, State.nGrow(), time, State_Type, 0, NUM_STATE);
    }

    StateCopy.clear();
  }
#endif

}

void
HydroSim::repair_internal_energy(const Box& bx,
#ifdef MHD
                              Array4<Real> const& Bx, Array4<Real> const& By, Array4<Real> const& Bz,
#endif
                              Array4<Real> const& u)
{
    BL_PROFILE("HydroSim::repair_internal_energy(Fab)");

    Real ltemp_floor = small_temp;
    Real ldual_energy_switch = dual_energy_switch;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

        auto cons_cell = u.cellData(i, j, k);

        const Real rhoInv = 1.0_rt / cons_cell[URHO];
        const Real Up = cons_cell[UMX] * rhoInv;
        const Real Vp = cons_cell[UMY] * rhoInv;
        const Real Wp = cons_cell[UMZ] * rhoInv;
        const Real ke = 0.5_rt * (Up * Up + Vp * Vp + Wp * Wp);

        eos_re_t eos_state;

        eos_state.rho = cons_cell[URHO];
        eos_state.T   = ltemp_floor;
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = cons_cell[UFS+n] * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; ++n) {
            eos_state.aux[n] = cons_cell[UFX+n] * rhoInv;
        }
#endif

        eos(eos_input_rt, eos_state);

        const Real min_eint = eos_state.e;

#ifdef MHD
        const Real magfield_x_cell = 0.5_rt * (Bx(i,j,k) + Bx(i+1,j,k));
        const Real magfield_y_cell = 0.5_rt * (By(i,j,k) + By(i,j+1,k));
        const Real magfield_z_cell = 0.5_rt * (Bz(i,j,k) + Bz(i,j,k+1));

        const Real magnetic_energy = 0.5_rt * (magfield_x_cell*magfield_x_cell +
                                      magfield_y_cell*magfield_y_cell +
                                      magfield_z_cell*magfield_z_cell);
#else
        const Real magnetic_energy = 0.0_rt;
#endif

        cons_cell[UEINT] = amrex::max(cons_cell[UEINT], cons_cell[URHO] * min_eint);
        cons_cell[UEDEN] = amrex::max(cons_cell[UEDEN], cons_cell[URHO] * (min_eint + ke) + magnetic_energy);

        const Real internal_energy_density = cons_cell[UEDEN] - cons_cell[URHO] * ke - magnetic_energy;

        if (internal_energy_density > ldual_energy_switch * cons_cell[UEDEN]) {
            cons_cell[UEINT] = internal_energy_density;
        }
    });
}

void
HydroSim::repair_internal_energy(
#ifdef MHD
                              MultiFab& Bx,
                              MultiFab& By,
                              MultiFab& Bz,
#endif
                              MultiFab& State, int ng)

{

    BL_PROFILE("HydroSim::repair_internal_energy(MultiFab)");

    MultiFab old_state;

    if (log_source_changes)
    {
        old_state.define(State.boxArray(), State.DistributionMap(), State.nComp(), 0);
        MultiFab::Copy(old_state, State, 0, 0, State.nComp(), 0);
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(State, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(ng);

        repair_internal_energy(bx,
#ifdef MHD
                              Bx.array(mfi), By.array(mfi), Bz.array(mfi),
#endif
                              State.array(mfi));
    }

    if (log_source_changes)
    {

        MultiFab reset_delta(State.boxArray(), State.DistributionMap(), State.nComp(), 0);

        MultiFab::Copy(reset_delta, State, 0, 0, State.nComp(), 0);

        MultiFab::Subtract(reset_delta, old_state, 0, 0, old_state.nComp(), 0);

        measure_and_log_source_change(reset_delta, 1.0, "negative energy resets");
    }
}

void
HydroSim::reconcile_energy (
#ifdef MHD
                              MultiFab& Bx,
                              MultiFab& By,
                              MultiFab& Bz,
#endif
                              MultiFab& S)
{

    BL_PROFILE("HydroSim::reconcile_energy()");

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& box = mfi.tilebox();

        auto cons_state_arr = S.array(mfi);

#ifdef MHD
        auto magfield_x_arr = Bx.array(mfi);
        auto magfield_y_arr = By.array(mfi);
        auto magfield_z_arr = Bz.array(mfi);
#endif

        ParallelFor(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
          Real rhoInv = 1.0_rt / cons_state_arr(i,j,k,URHO);
          Real u = cons_state_arr(i,j,k,UMX) * rhoInv;
          Real v = cons_state_arr(i,j,k,UMY) * rhoInv;
          Real w = cons_state_arr(i,j,k,UMZ) * rhoInv;

          cons_state_arr(i,j,k,UEDEN) = cons_state_arr(i,j,k,UEINT) +
            0.5_rt * cons_state_arr(i,j,k,URHO) * (u*u + v*v + w*w);

#ifdef MHD
          Real magfield_x_cell = 0.5_rt * (magfield_x_arr(i,j,k) + magfield_x_arr(i+1,j,k));
          Real magfield_y_cell = 0.5_rt * (magfield_y_arr(i,j,k) + magfield_y_arr(i,j+1,k));
          Real magfield_z_cell = 0.5_rt * (magfield_z_arr(i,j,k) + magfield_z_arr(i,j,k+1));

          cons_state_arr(i,j,k,UEDEN) += 0.5_rt * (magfield_x_cell * magfield_x_cell +
                                          magfield_y_cell * magfield_y_cell +
                                          magfield_z_cell * magfield_z_cell);
#endif

        });

    }
}

void
HydroSim::renorm_species (MultiFab& S_new, int ng)
{
    BL_PROFILE("HydroSim::renorm_species()");

    Real lmassfrac_floor = network_rp::small_x;

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<int> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(ng);

        auto u = S_new.array(mfi);

        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
        {

            auto cons_cell = u.cellData(i, j, k);
            Real rhoX_sum = 0.0_rt;
            Real rho = cons_cell[URHO];
            Real rhoInv = 1.0_rt / rho;

            int failed{};

            for (int n = 0; n < NumSpec; ++n) {
                Real X = cons_cell[UFS+n] * rhoInv;

                if (rho >= hydrosim::massfrac_check_min_density) {
                    if (X < -hydrosim::massfrac_check_tolerance ||
                        X > 1.0_rt + hydrosim::massfrac_check_tolerance) {
#ifndef AMREX_USE_GPU
                        std::cout << "(i, j, k) = " << i << " " << j << " " << k << " " << ", X[" << n << "] = " << X
                                  << "  (density here is: " << rho << ")" << std::endl;
#elif defined(ALLOW_GPU_PRINTF)
                        AMREX_DEVICE_PRINTF("(i, j, k) = %d %d %d, X[%d] = %g  (density here is: %g)\n",
                                            i, j, k, n, X, rho);
#endif
                        failed = 1;
                    }
                }

                cons_cell[UFS+n] = amrex::Clamp(cons_cell[UFS+n], lmassfrac_floor * rho, rho);
                rhoX_sum += cons_cell[UFS+n];
            }

            Real fac = rho / rhoX_sum;

            amrex::constexpr_for<0, NumSpec>([&] (auto n) {
                cons_cell[UFS+n] *= fac;
            });

            return {failed};
        });
    }

    ReduceTuple hv = reduce_data.value();
    int num_failed = amrex::get<0>(hv);

    if (num_failed > 0) {
        amrex::Error("Invalid mass fraction in HydroSim::renorm_species()");
    }
}

void
HydroSim::clamp_min_density (MultiFab& state_in, int ng)
{

    BL_PROFILE("HydroSim::clamp_min_density()");

    MultiFab reset_delta;

    if (log_source_changes)
    {

        reset_delta.define(state_in.boxArray(), state_in.DistributionMap(), state_in.nComp(), 0);

        MultiFab::Copy(reset_delta, state_in, 0, 0, state_in.nComp(), 0);

    }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox(ng);

        clamp_state_density(bx, state_in.array(mfi), verbose);
    }

    if (log_source_changes)
    {

        MultiFab::Subtract(reset_delta, state_in, 0, 0, state_in.nComp(), 0);

        measure_and_log_source_change(reset_delta, 1.0, "negative density resets");
    }

}

void
HydroSim::clamp_speed_limit (MultiFab& state_in, int ng)
{
    BL_PROFILE("HydroSim::clamp_speed_limit()");

    if (hydrosim::velocity_cap <= 0.0_rt) {
        return;
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox(ng);

        auto u = state_in[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real rho = u(i,j,k,URHO);
            Real rhoInv = 1.0_rt / rho;

            Real vx = u(i,j,k,UMX) * rhoInv;
            Real vy = u(i,j,k,UMY) * rhoInv;
            Real vz = u(i,j,k,UMZ) * rhoInv;

            Real v = std::sqrt(vx * vx + vy * vy + vz * vz);

            if (v > hydrosim::velocity_cap) {
                Real velocity_scale = hydrosim::velocity_cap / v;

                u(i,j,k,UMX) *= velocity_scale;
                u(i,j,k,UMY) *= velocity_scale;
                u(i,j,k,UMZ) *= velocity_scale;

                u(i,j,k,UEDEN) -= 0.5_rt * rhoInv * (rho * vx * rho * vx - u(i,j,k,UMX) * u(i,j,k,UMX) +
                                                     rho * vy * rho * vy - u(i,j,k,UMY) * u(i,j,k,UMY) +
                                                     rho * vz * rho * vz - u(i,j,k,UMZ) * u(i,j,k,UMZ));
            }
        });
    }
}

step_result
HydroSim::scan_for_negative_mass_density ()
{
    BL_PROFILE("scan_for_negative_mass_density()");

    MultiFab& S_old = get_old_data(State_Type);
    MultiFab& S_new = get_new_data(State_Type);

    ReduceOps<ReduceOpMax, ReduceOpMax> reduce_op;
    ReduceData<int, int> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        const auto state_old_arr = S_old.array(mfi);
        const auto new_state_view = S_new.array(mfi);

        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
        {
            int density_check_failed = 0;
            int massfrac_check_failed = 0;

            const Real rho = new_state_view(i,j,k,URHO);
            const Real rhoInv = 1.0_rt / rho;

            if (state_old_arr(i,j,k,URHO) >= retry_density_cutoff && rho < small_dens) {
#ifndef AMREX_USE_GPU
                std::cout << "Invalid density = " << rho << " at index " << i << ", " << j << ", " << k << "\n";
#endif
                density_check_failed = 1;
            }

            if (rho >= hydrosim::massfrac_check_min_density) {

                for (int n = 0; n < NumSpec; ++n) {
                    Real X = new_state_view(i,j,k,UFS+n) * rhoInv;

                    if (X < -hydrosim::massfrac_check_tolerance ||
                        X > 1.0_rt + hydrosim::massfrac_check_tolerance) {
#ifndef AMREX_USE_GPU
                        std::cout << "Invalid X[" << n << "] = " << X << " in zone "
                                  << i << ", " << j << ", " << k
                                  << " with density = " << rho << std::endl;
#elif defined(ALLOW_GPU_PRINTF)
                        AMREX_DEVICE_PRINTF("Invalid X[%d] = %g in zone (%d,%d,%d) with density = %g\n",
                                            n, X, i, j, k, rho);
#endif
                        massfrac_check_failed = 1;
                    }
                }

            }

            return {density_check_failed, massfrac_check_failed};
        });

#ifdef ALLOW_GPU_PRINTF
        std::fflush(nullptr);
#endif

    }

    ReduceTuple hv = reduce_data.value();
    int density_check_failed = amrex::get<0>(hv);
    int massfrac_check_failed = amrex::get<1>(hv);

    int fails[] = {density_check_failed, massfrac_check_failed};
    ParallelDescriptor::ReduceIntMax(fails, 2);

    step_result status {};

    if (fails[0] == 1) {
        status.success = false;
        status.reason = "invalid density";
    }

    if (fails[1] == 1) {
        status.success = false;
        status.reason = "invalid X";
    }

    return status;
}

void
HydroSim::scan_for_nan(const MultiFab& state_in, int include_ghosts)
{
  BL_PROFILE("HydroSim::scan_for_nan()");

  int ng = 0;
  if (include_ghosts == 1) {
    ng = state_in.nGrow();
  }

  if (state_in.contains_nan(URHO,state_in.nComp(),ng,true)) {
#if defined(AMREX_USE_GPU) && !defined(ALLOW_GPU_PRINTF)
      std::string abort_message = std::string("State has NaNs in scan_for_nan()");
      amrex::Abort(abort_message.c_str());
#else
      for (int i = 0; i < state_in.nComp(); i++) {
          if (state_in.contains_nan(URHO + i, 1, ng, true)) {
              std::string abort_message = std::string("State has NaNs in the ") + desc_lst[State_Type].name(i) + std::string(" component::scan_for_nan()");
              amrex::Abort(abort_message.c_str());
          }
      }
#endif
  }
}

void
HydroSim::clean_state(
#ifdef MHD
                    MultiFab& bx,
                    MultiFab& by,
                    MultiFab& bz,
#endif
                    MultiFab& state_in, Real time, int ng) {

    BL_PROFILE("HydroSim::clean_state()");

    clamp_min_density(state_in, ng);

    clamp_speed_limit(state_in, ng);

    renorm_species(state_in, ng);

#ifdef HYBRID_MOMENTUM
    if (split_momentum_hydro) {
        split_to_linear_momentum(state_in, ng);
    }
#endif

    computeTemp(
#ifdef MHD
                bx, by, bz,
#endif
                state_in, time, ng);

}

#ifdef MHD
void
HydroSim::add_bfield_energy( MultiFab& Bx,
                        MultiFab& By,
                        MultiFab& Bz,
                        MultiFab& State)
{

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
  for (MFIter mfi(State, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
      const Box& box = mfi.tilebox();
      auto cons_state_arr = State.array(mfi);
      auto magfield_x_arr = Bx.array(mfi);
      auto magfield_y_arr = By.array(mfi);
      auto magfield_z_arr = Bz.array(mfi);

      ParallelFor(box,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

          Real magfield_x_cell = 0.5_rt * (magfield_x_arr(i,j,k) + magfield_x_arr(i+1,j,k));
          Real magfield_y_cell = 0.5_rt * (magfield_y_arr(i,j,k) + magfield_y_arr(i,j+1,k));
          Real magfield_z_cell = 0.5_rt * (magfield_z_arr(i,j,k) + magfield_z_arr(i,j,k+1));

          cons_state_arr(i,j,k,UEDEN) += 0.5_rt * (magfield_x_cell * magfield_x_cell +
                                          magfield_y_cell * magfield_y_cell +
                                          magfield_z_cell * magfield_z_cell);

      });

  }

}

void
HydroSim::check_div_bfield( MultiFab& Bx,
                     MultiFab& By,
                     MultiFab& Bz,
                     MultiFab& State)
{

  ReduceOps<ReduceOpSum> reduce_op;
  ReduceData<int> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
  for (MFIter mfi(State, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
      const Box& box = mfi.tilebox();
      auto magfield_x_arr = Bx.array(mfi);
      auto magfield_y_arr = By.array(mfi);
      auto magfield_z_arr = Bz.array(mfi);

      const auto dx = geom.CellSizeArray();

      reduce_op.eval(box, reduce_data,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
      {

          Real magDivergence = (magfield_x_arr(i+1,j,k) - magfield_x_arr(i,j,k))/dx[0] +
                      (magfield_y_arr(i,j+1,k) - magfield_y_arr(i,j,k))/dx[1] +
                      (magfield_z_arr(i,j,k+1) - magfield_z_arr(i,j,k))/dx[2];

          Real magfield_x_cell = 0.5_rt * (magfield_x_arr(i,j,k) + magfield_x_arr(i+1,j,k));
          Real magfield_y_cell = 0.5_rt * (magfield_y_arr(i,j,k) + magfield_y_arr(i,j+1,k));
          Real magfield_z_cell = 0.5_rt * (magfield_z_arr(i,j,k) + magfield_z_arr(i,j,k+1));

          Real magNorm = std::sqrt(magfield_x_cell * magfield_x_cell +
                                magfield_y_cell * magfield_y_cell +
                                magfield_z_cell * magfield_z_cell);

          int mag_div_failed = 0;

          if (std::abs(magDivergence) > 1.0e-10*magNorm){
             mag_div_failed = 1;
          }

          return {mag_div_failed};
      });

  }

  ReduceTuple hv = reduce_data.value();
  int init_mag_div_failed = amrex::get<0>(hv);

  if (init_mag_div_failed != 0) {
     amrex::Error("Error: initial data has divergence of B not zero");
  }

}
#endif
