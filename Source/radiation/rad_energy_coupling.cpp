#include <AMReX_LO_BCTYPES.H>
#include <AMReX_ParmParse.H>
#include <Radiation.H>
#include <RadTransportSolver.H>
#include <radiation_helpers.H>

#include <opacity.H>

#include <format>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <HydroSim.H>

using namespace amrex;
using std::string;

void Radiation::compute_coupling_factors(MultiFab& eta, MultiFab& coupling_inverse,
                            MultiFab& state, MultiFab& temp,
                            MultiFab& fkp, MultiFab& Er,
                            Real delta_t, Real c,
                            Real underrelax_scale, PlanckCoupling planck_coupling, int rad_group_idx)
{
    BL_PROFILE("Radiation::compute_coupling_factors");

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        FArrayBox c_v;

        for (MFIter mfi(eta, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

            const Box& bx = mfi.tilebox();

            if (planck_coupling == PlanckCoupling::Frozen) {

                Array4<Real> const coupling_arr = eta.array(mfi);
                Array4<Real> const scaled_planck_arr = fkp.array(mfi);
                AMREX_PARALLEL_FOR_3D(bx, i, j, k,
                                      { coupling_arr(i,j,k) = scaled_planck_arr(i,j,k); });

            }
            else {

                auto coupling_arr = eta[mfi].array();
                auto state_arr = state[mfi].array();

                const Real nu = freq_center[rad_group_idx];
                const Real local_temp_step = dT;

                amrex::ParallelFor(bx,
                [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
                {
                    Real rho = state_arr(i,j,k,URHO);
                    Real temp = state_arr(i,j,k,UTEMP) + local_temp_step;
                    Real Ye;
                    if (NumAux > 0)  {
                        Ye = state_arr(i,j,k,UFX);
                    } else {
                        Ye = 0.e0_rt;
                    }

                    Real kp, kr;
                    bool need_planck_opac = true;
                    bool need_ross_opac = false;
                    opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

                    coupling_arr(i,j,k,rad_group_idx) = kp;
                });

            }

            c_v.resize(bx);
            Elixir heat_capacity_hold = c_v.elixir();

            get_heat_capacity(c_v, temp[mfi], state[mfi], bx);

            auto coupling_arr = eta[mfi].array();
            auto coupling_inverse_arr = coupling_inverse[mfi].array();
            auto fluid_density_arr = state[mfi].array(URHO);
            auto temp_field_arr = temp[mfi].array();
            auto heat_capacity_arr = c_v.array();
            auto scaled_planck_arr = fkp[mfi].array();
            auto rad_eden_arr = Er[mfi].array(rad_group_idx);

            Real local_temp_step = dT;

            const Real fac1 = 16.e0_rt * sigma * delta_t;
            const Real fac0 = 0.25e0_rt * fac1 / dT;
            const Real fac2 = delta_t * c / dT;

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                Real d;

                if (planck_coupling == PlanckCoupling::Frozen)
                {
                    d = fac1 * scaled_planck_arr(i,j,k) * std::pow(temp_field_arr(i,j,k), 3);
                }
                else
                {
                    d = fac0 * (coupling_arr(i,j,k) * std::pow(temp_field_arr(i,j,k) + local_temp_step, 4) -
                                scaled_planck_arr(i,j,k) * std::pow(temp_field_arr(i,j,k), 4)) -
                        fac2 * (coupling_arr(i,j,k) - scaled_planck_arr(i,j,k)) * rad_eden_arr(i,j,k);
                }

                Real frc = fluid_density_arr(i,j,k) * heat_capacity_arr(i,j,k) + 1.0e-50_rt;
                coupling_arr(i,j,k) = d / (d + frc);
                coupling_inverse_arr(i,j,k) = underrelax_scale * frc / (d + frc);
                coupling_arr(i,j,k) = 1.e0_rt - coupling_inverse_arr(i,j,k);
            });
        }
    }
}

void Radiation::evaluate_energy_transfer(MultiFab& exch,
                                 MultiFab& Er,
                                 MultiFab& fkp, int rad_group_idx)
{
    BL_PROFILE("Radiation::evaluate_energy_transfer");

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(exch, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto energy_exchange_arr = exch[mfi].array();
        auto rad_eden_arr = Er[mfi].array();
        auto scaled_planck_arr = fkp[mfi].array();

        Real local_sb_const = sigma;
        Real lc = c;

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            energy_exchange_arr(i,j,k) = scaled_planck_arr(i,j,k) * (4.e0_rt * local_sb_const * std::pow(energy_exchange_arr(i,j,k), 4) - lc * rad_eden_arr(i,j,k));
        });
    }
}

Radiation::UpdateNorms Radiation::apply_implicit_energy(MultiFab& state_energy_density,
                                       MultiFab& matter_energy_density,
                                       MultiFab& eta,
                                       MultiFab& coupling_inverse,
                                       MultiFab& flux_div_old,
                                       MultiFab& flux_div_new,
                                       MultiFab& exch,
                                       Real delta_t)
{
  BL_PROFILE("Radiation::apply_implicit_energy");

  ReduceOps<ReduceOpMax, ReduceOpMax> reduce_op;
  ReduceData<Real, Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

  Real theta = 1.0;

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(eta, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      const auto coupling_arr = eta[mfi].array();
      const auto coupling_inverse_arr = coupling_inverse[mfi].array();
      const auto density_matter_energy_arr = matter_energy_density[mfi].array();
      const auto energy_exchange_arr = exch[mfi].array();
      const auto dfo = flux_div_old[mfi].array();
      const auto dfn = flux_div_new[mfi].array();
      auto density_state_energy_arr = state_energy_density[mfi].array();

      reduce_op.eval(bx, reduce_data,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real chg = 0.e0_rt;
          Real tot = 0.e0_rt;

          Real tmp = coupling_arr(i,j,k) * density_state_energy_arr(i,j,k) +
                     coupling_inverse_arr(i,j,k) *
                     (density_matter_energy_arr(i,j,k) -
                      delta_t * ((1.e0_rt - theta) *
                                 (dfo(i,j,k) - dfn(i,j,k)) +
                                 energy_exchange_arr(i,j,k)));

          chg = std::abs(tmp - density_state_energy_arr(i,j,k));
          tot = std::abs(density_state_energy_arr(i,j,k));

          density_state_energy_arr(i,j,k) = tmp;

          Real abs_residual = chg;
          Real relative_residual = chg / (tot + 1.e-50_rt);

          return {relative_residual, abs_residual};
      });
  }

  ReduceTuple hv = reduce_data.value();

  Real relative = amrex::get<0>(hv);
  Real absolute = amrex::get<1>(hv);

  ParallelDescriptor::ReduceRealMax(relative);
  ParallelDescriptor::ReduceRealMax(absolute);

  return {relative, absolute};
}

Radiation::UpdateNorms Radiation::apply_implicit_energy(MultiFab& state_energy_density,
                                       MultiFab& matter_energy_density,
                                       MultiFab& eta,
                                       MultiFab& coupling_inverse,
                                       MultiFab& flux_div_old,
                                       MultiFab& flux_div_new,
                                       MultiFab& exch,
                                       MultiFab& diffusive_face_term,
                                       Real delta_t)
{
  BL_PROFILE("Radiation::internal_energy_update_d");

  Real relative = 0.0;
  Real absolute = 0.0;

  ReduceOps<ReduceOpMax, ReduceOpMax> reduce_op;
  ReduceData<Real, Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

  const Real theta = 1.0;
  const Real tiny = 1.e-50_rt;

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(eta,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box &reg = mfi.tilebox();

      const auto coupling_arr = eta[mfi].array();
      const auto coupling_inverse_arr = coupling_inverse[mfi].array();
      const auto density_matter_energy_arr = matter_energy_density[mfi].array();
      const auto energy_exchange_arr = exch[mfi].array();
      const auto dfo = flux_div_old[mfi].array();
      const auto dfn = flux_div_new[mfi].array();
      const auto diffusion_term_arr = diffusive_face_term[mfi].array();
      auto density_state_energy_arr = state_energy_density[mfi].array();

      reduce_op.eval(reg, reduce_data,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
      {

          Real chg = 0.e0_rt;
          Real tot = 0.e0_rt;

          Real tmp = coupling_arr(i,j,k) * density_state_energy_arr(i,j,k) +
              coupling_inverse_arr(i,j,k) * (density_matter_energy_arr(i,j,k) -
                                   delta_t * ((1.e0_rt - theta) *
                                              (dfo(i,j,k) - dfn(i,j,k)) +
                                              energy_exchange_arr(i,j,k))) +
              delta_t * diffusion_term_arr(i,j,k);

          chg = std::abs(tmp - density_state_energy_arr(i,j,k));
          tot = std::abs(density_state_energy_arr(i,j,k));

          density_state_energy_arr(i,j,k) = tmp;

          Real abs_residual = chg;
          Real relative_residual = chg / (tot + tiny);

          return {relative_residual, abs_residual};

      });

  }

  ReduceTuple hv = reduce_data.value();

  relative = amrex::get<0>(hv);
  absolute = amrex::get<1>(hv);

  ParallelDescriptor::ReduceRealMax(relative);
  ParallelDescriptor::ReduceRealMax(absolute);

  return {relative, absolute};
}

Radiation::UpdateNorms Radiation::apply_nonconservative_correction(MultiFab& state_energy_density,
                                              MultiFab& matter_energy_density,
                                              MultiFab& eta,
                                              MultiFab& coupling_inverse,
                                              MultiFab& rad_energy_new,
                                              MultiFab& flux_div_old,
                                              MultiFab& flux_div_new,
                                              MultiFab& temp,
                                              MultiFab& fkp,
                                              MultiFab& state,
                                              Real delta_t)
{
  BL_PROFILE("Radiation::apply_nonconservative_correction");

  ReduceOps<ReduceOpMax, ReduceOpMax> reduce_op;
  ReduceData<Real, Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

  const Real theta = 1.0;
  const Real tiny = 1.e-50_rt;

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
      FArrayBox c_v;

      for (MFIter mfi(eta,true); mfi.isValid(); ++mfi) {
          const Box &reg = mfi.tilebox();

          c_v.resize(reg);
          Elixir hold_heat_capacity = c_v.elixir();

          get_heat_capacity(c_v, temp[mfi], state[mfi], reg);

          const auto state_arr = state[mfi].array();
          const auto temp_field_arr = temp[mfi].array();
          const auto scaled_planck_arr = fkp[mfi].array();
          const auto rad_energy_arr = rad_energy_new[mfi].array();
          const auto coupling_arr = eta[mfi].array();
          const auto coupling_inverse_arr = coupling_inverse[mfi].array();
          const auto density_matter_energy_arr = matter_energy_density[mfi].array();
          const auto density_state_energy_arr = state_energy_density[mfi].array();
          const auto flux_div_old_view = flux_div_old[mfi].array();
          const auto flux_div_new_arr = flux_div_new[mfi].array();
          const auto heat_capacity_arr = c_v.array();

          reduce_op.eval(reg, reduce_data,
          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
          {

              Real chg = 0.e0_rt;
              Real tot = 0.e0_rt;

              Real density_heat_capacity = state_arr(i,j,k,URHO) * heat_capacity_arr(i,j,k);

              Real planck_temp_slope = 16.0_rt * C::sigma_SB * std::pow(temp_field_arr(i,j,k), 3);
              Real b = 4.0_rt * C::sigma_SB * std::pow(temp_field_arr(i,j,k), 4);
              Real exch = scaled_planck_arr(i,j,k) * (b - C::c_light * rad_energy_arr(i,j,k));
              Real tmp = coupling_arr(i,j,k) * density_state_energy_arr(i,j,k) +
                  coupling_inverse_arr(i,j,k) * (density_matter_energy_arr(i,j,k) -
                                       delta_t * ((1.0_rt - theta) * (flux_div_old_view(i,j,k) - flux_div_new_arr(i,j,k)) + exch));

              if (density_heat_capacity > tiny && tmp > density_state_energy_arr(i,j,k)) {
                  Real db = (tmp - density_state_energy_arr(i,j,k)) * planck_temp_slope / density_heat_capacity;
                  tmp = std::pow((b + db) / (4.0_rt * C::sigma_SB), 0.25_rt);
                  tmp = density_state_energy_arr(i,j,k) + density_heat_capacity * (tmp - temp_field_arr(i,j,k));
              }

              chg = std::abs(tmp - density_state_energy_arr(i,j,k));
              tot = std::abs(density_state_energy_arr(i,j,k));
              density_state_energy_arr(i,j,k) = tmp;

              Real abs_residual = chg;
              Real relative_residual = chg / (tot + tiny);

              return {abs_residual, relative_residual};
          });

      }

  }

  ReduceTuple hv = reduce_data.value();

  Real absolute = amrex::get<0>(hv);
  Real relative = amrex::get<1>(hv);

  ParallelDescriptor::ReduceRealMax(relative);
  ParallelDescriptor::ReduceRealMax(absolute);

  return {relative, absolute};
}

void Radiation::sync_state_from_energy(MultiFab& state, MultiFab& state_energy_density)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto state_arr = state[mfi].array();
        auto density_state_energy_arr = state_energy_density[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Real kin = state_arr(i,j,k,UEDEN) - state_arr(i,j,k,UEINT);
            state_arr(i,j,k,UEINT) = density_state_energy_arr(i,j,k);
            state_arr(i,j,k,UEDEN) = density_state_energy_arr(i,j,k) + kin;

            if (state_arr(i,j,k,UEINT) <= 0.e0_rt)
            {
                density_state_energy_arr(i,j,k) = small_temp;
            }
            else
            {
                Real rhoInv = 1.e0_rt / state_arr(i,j,k,URHO);

                eos_re_t eos_state;
                eos_state.rho = state_arr(i,j,k,URHO);
                eos_state.T   = state_arr(i,j,k,UTEMP);
                eos_state.e   = state_arr(i,j,k,UEINT) * rhoInv;
                for (int n = 0; n < NumSpec; ++n) {
                    eos_state.xn[n] = state_arr(i,j,k,UFS+n) * rhoInv;
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; ++n) {
                    eos_state.aux[n] = state_arr(i,j,k,UFX+n) * rhoInv;
                }
#endif

                eos(eos_input_re, eos_state);

                density_state_energy_arr(i,j,k) = eos_state.T;

                state_arr(i,j,k,UTEMP) = density_state_energy_arr(i,j,k);
            }
        });
    }
}

void Radiation::get_density_energy(MultiFab& fluid_rho_eint,
                          MultiFab& state)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter si(state,TilingIfNotGPU()); si.isValid(); ++si) {
        const Box& reg = si.tilebox();

        auto density_energy_arr = fluid_rho_eint[si].array();
        auto state_arr = state[si].array();

        amrex::ParallelFor(reg,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            density_energy_arr(i,j,k) = state_arr(i,j,k,UEINT);
        });
    }
}

void Radiation::get_heat_capacity(FArrayBox& c_v, FArrayBox& temp, FArrayBox& state,
                        const Box& reg)
{
    BL_PROFILE("Radiation::get_heat_capacity");

    auto heat_capacity_arr = c_v.array();
    auto temp_field_arr = temp.array();
    auto state_arr = state.array();

    amrex::ParallelFor(reg,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        Real rhoInv = 1.e0_rt / state_arr(i,j,k,URHO);

        eos_re_t eos_state;
        eos_state.rho = state_arr(i,j,k,URHO);
        eos_state.T   = temp_field_arr(i,j,k);
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = state_arr(i,j,k,UFS+n) * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; ++n) {
            eos_state.aux[n] = state_arr(i,j,k,UFX+n) * rhoInv;
        }
#endif

        eos(eos_input_rt, eos_state);

        heat_capacity_arr(i,j,k) = eos_state.cv;
    });
    Gpu::synchronize();
}

void Radiation::reset_flux_records(int level, int ncycle)
{
  BL_PROFILE("Radiation::reset_flux_records");
  if (verbose > 0 && ParallelDescriptor::IOProcessor()) {
    std::cout << "Radiation flux initialization at level " << level
         << "..." << std::endl;
  }

  int fine_level = parent->finestLevel();

  flux_div[level]->setVal(0.0);

  if (level < fine_level) {
      flux_accum_reg[level+1]->setVal(0.0);
  }

  if (verbose > 1 && ParallelDescriptor::IOProcessor()) {
    std::cout << "                                           done" << std::endl;
  }
}

#ifdef RADIATION
void
HydroSim::close_radiation_step (MultiFab& S_new, int iteration, int ncycle)
{
    if (!do_radiative_transfer) {
        return;
    }

    if (Radiation::hydro_only) {
      HydroSim::computeTemp(S_new, state[State_Type].curTime(), S_new.nGrow());
      return;
    }

    HydroSim::computeTemp(S_new, state[State_Type].curTime(), S_new.nGrow());

    if (Radiation::filter_primitive_interval > 0 && Radiation::filter_prim_temperature>0
        && iteration==ncycle) {
        int nstep = parent->levelSteps(0);
        if (nstep % Radiation::filter_primitive_interval == 0) {
            radiation->filter_primitives(level, S_new);
        }
    }

    if (Radiation::SolverKind == Radiation::MultiGroupFLDSolver) {
        if (! Radiation::rad_hydro_coupled) {
            MultiFab& rad_energy_old = get_old_data(Radiation_Type);
            MultiFab& rad_energy_new = get_new_data(Radiation_Type);
            MultiFab::Copy(rad_energy_new, rad_energy_old, 0, 0, rad_energy_old.nComp(), 0);
        }
        radiation->MultigroupDiffusion_implicit_update(level, iteration, ncycle);
    }
    else if (Radiation::SolverKind == Radiation::SingleGroupFLDSolver) {
        if (! Radiation::rad_hydro_coupled) {
            MultiFab& rad_energy_old = get_old_data(Radiation_Type);
            MultiFab& rad_energy_new = get_new_data(Radiation_Type);
            MultiFab::Copy(rad_energy_new, rad_energy_old, 0, 0, rad_energy_old.nComp(), 0);
        }
        radiation->advance_single_group(level, iteration, ncycle);
    }
    else {
        MultiFab& rad_energy_old = get_old_data(Radiation_Type);
        MultiFab& rad_energy_new = get_new_data(Radiation_Type);
        MultiFab::Copy(rad_energy_new, rad_energy_old, 0, 0, rad_energy_old.nComp(), 0);
        radiation->advance_single_group(level, iteration, ncycle);
    }
}
#endif
