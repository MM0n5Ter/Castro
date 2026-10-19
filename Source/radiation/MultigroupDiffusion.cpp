#include <Radiation.H>
#include <radiation_helpers.H>
#include <filter.H>
#include <planck_integrals.H>
#include <opacity.H>
#include <user_emissivity.H>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace amrex;

void Radiation::check_rad_convergence(Real& relative, Real& absolute, Real& rad_energy_err,
                                     const MultiFab& rad_energy_new, const MultiFab& rad_energy_iter,
                                     const MultiFab& kappa_p,
                                     const MultiFab& thermal_frac_z,
                                     const MultiFab& temp_next,
                                     Real delta_t)
{
  BL_PROFILE("Radiation::check_rad_convergence (MultigroupDiffusion)");

  ReduceOps<ReduceOpMax, ReduceOpMax, ReduceOpMax> reduce_op;
  ReduceData<Real, Real, Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(rad_energy_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      const auto Ern = rad_energy_new[mfi].array();
      const auto Erl = rad_energy_iter[mfi].array();
      const auto kap = kappa_p[mfi].array();
      const auto thermal_frac_z_view = thermal_frac_z[mfi].array();
      const auto temp = temp_next[mfi].array();

      reduce_op.eval(bx, reduce_data,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real chg = 0.e0_rt;
          Real tot = 0.e0_rt;
          Real kde = 0.e0_rt;

          for (int g = 0; g < NGROUPS; ++g) {
              Real der = Ern(i,j,k,g) - Erl(i,j,k,g);
              chg = chg + std::abs(der);
              tot = tot + std::abs(Ern(i,j,k,g));
              kde = kde + kap(i,j,k,g) * der;
          }

          Real abs_change = chg;
          Real frac_chg = chg / (tot + 1.e-50_rt);

          Real temp_err = thermal_frac_z_view(i,j,k) * kde;
          Real temp_err_rel = std::abs(temp_err / (temp(i,j,k) + 1.e-50_rt));

          return {frac_chg, abs_change, temp_err_rel};
      });
  }

  ReduceTuple hv = reduce_data.value();

  relative = amrex::get<0>(hv);
  absolute = amrex::get<1>(hv);
  rad_energy_err   = amrex::get<2>(hv);

  Real data[3] = {relative, absolute, rad_energy_err};

  ParallelDescriptor::ReduceRealMax(data, 3);

  relative = data[0];
  absolute = data[1];
  rad_energy_err   = data[2];
}


void Radiation::check_matter_convergence(const MultiFab& rhoe_new, const MultiFab& gas_energy_star,
                                       const MultiFab& gas_energy_step, const MultiFab& rad_energy_new,
                                       const MultiFab& temp_next, const MultiFab& temp_iterate,
                                       const MultiFab& rho,
                                       const MultiFab& kappa_p, const MultiFab& jg,
                                       const MultiFab& dedT,
                                       Real& rel_gas_energy, Real& abs_gas_energy,
                                       Real& rel_resid,   Real& abs_resid,
                                       Real& rel_temp,    Real& abs_temp,
                                       Real delta_t)
{
  ReduceOps<ReduceOpMax, ReduceOpMax, ReduceOpMax, ReduceOpMax, ReduceOpMax, ReduceOpMax> reduce_op;
  ReduceData<Real, Real, Real, Real, Real, Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(rhoe_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      auto ren = rhoe_new[mfi].array();
      auto res = gas_energy_star[mfi].array();
      auto re2 = gas_energy_step[mfi].array();
      auto Ern = rad_energy_new[mfi].array();
      auto Tmn = temp_next[mfi].array();
      auto Tms = temp_iterate[mfi].array();
      auto mass_density_arr = rho[mfi].array();
      auto kap = kappa_p[mfi].array();
      auto emission_arr = jg[mfi].array();
      auto deT = dedT[mfi].array();

      int ng = Radiation::nGroups;
      Real cdt = C::c_light * delta_t;

      reduce_op.eval(bx, reduce_data,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real gas_energy_abs_chg = std::abs(ren(i,j,k) - res(i,j,k));
          Real gas_energy_rel_chg = std::abs(gas_energy_abs_chg / (ren(i,j,k) + 1.e-50_rt));

          Real temp_abs_chg = std::abs(Tmn(i,j,k) - Tms(i,j,k));
          Real temp_rel_chg = std::abs(temp_abs_chg / (Tmn(i,j,k) + 1.e-50_rt));

          Real FT = ren(i,j,k) - re2(i,j,k);
          for (int g = 0; g < ng; ++g) {
              FT -= cdt * (kap(i,j,k,g) * Ern(i,j,k,g) - emission_arr(i,j,k,g));
          }
          FT = std::abs(FT);

          Real dTe = Tmn(i,j,k);
          Real resid_scale = mass_density_arr(i,j,k) * std::abs(deT(i,j,k) * dTe);

          Real resid_abs_chg = FT;
          Real resid_rel_chg = FT / (resid_scale + 1.e-50_rt);

          return {gas_energy_rel_chg, gas_energy_abs_chg, resid_rel_chg, resid_abs_chg, temp_rel_chg, temp_abs_chg};
      });
  }

  ReduceTuple hv = reduce_data.value();
  rel_gas_energy = amrex::get<0>(hv);
  abs_gas_energy = amrex::get<1>(hv);
  rel_resid   = amrex::get<2>(hv);
  abs_resid   = amrex::get<3>(hv);
  rel_temp    = amrex::get<4>(hv);
  abs_temp    = amrex::get<5>(hv);

  int ndata = 6;
  Real data[6] = {rel_gas_energy, abs_gas_energy, rel_resid, abs_resid, rel_temp, abs_temp};

  ParallelDescriptor::ReduceRealMax(data, ndata);

  rel_gas_energy = data[0];
  abs_gas_energy = data[1];
  rel_resid   = data[2];
  abs_resid   = data[3];
  rel_temp    = data[4];
  abs_temp    = data[5];
}


void Radiation::build_coupling_term(MultiFab& couple_source,
                                 const MultiFab& kpp, const MultiFab& Eg,
                                 const MultiFab& jg)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(kpp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto couple_source_arr = couple_source[mfi].array();
        auto kappa_group_arr = kpp[mfi].array();
        auto group_energy_arr = Eg[mfi].array();
        auto emission_arr = jg[mfi].array();

        int ng = Radiation::nGroups;

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            couple_source_arr(i,j,k) = 0.0_rt;

            for (int g = 0; g < ng; ++g) {
                couple_source_arr(i,j,k) = couple_source_arr(i,j,k) + (kappa_group_arr(i,j,k,g) * group_energy_arr(i,j,k,g) - emission_arr(i,j,k,g));
            }
        });
    }
}


void Radiation::compute_thermal_fraction(MultiFab& thermal_frac, MultiFab& thermal_frac_z,
                             MultiFab& rest_frac, MultiFab& dEmitT,
                             const MultiFab& dOpacT, const MultiFab& dedT,
                             const MultiFab& rad_energy_star, const MultiFab& rho,
                             Real delta_t, Real pseudo_transient_tau)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(rho, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto thermal_frac_arr = thermal_frac[mfi].array();
        auto thermal_frac_z_arr = thermal_frac_z[mfi].array();
        auto rest_frac_arr = rest_frac[mfi].array();
        auto emission_dtemp_arr = dEmitT[mfi].array();
        auto kappa_dtemp_arr = dOpacT[mfi].array();
        auto energy_dtemp_arr = dedT[mfi].array();
        auto rad_energy_star_arr = rad_energy_star[mfi].array();
        auto mass_density_arr = rho[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Real sigma = 1.e0_rt + pseudo_transient_tau;
            Real cdt = C::c_light * delta_t;

            Real dz_dtemp[NGROUPS];
            Real sum_dz_dtemp = 0.0_rt;
            for (int g = 0; g < NGROUPS; ++g) {
                dz_dtemp[g] = emission_dtemp_arr(i,j,k,g) - kappa_dtemp_arr(i,j,k,g) * rad_energy_star_arr(i,j,k,g);
                sum_dz_dtemp += dz_dtemp[g];
            }

            if (sum_dz_dtemp == 0.0_rt) {
                sum_dz_dtemp = 1.e-50_rt;
            }

            Real foo = cdt * sum_dz_dtemp;
            Real bar = sigma * mass_density_arr(i,j,k) * energy_dtemp_arr(i,j,k);
            thermal_frac_arr(i,j,k) = foo / (foo + bar);
            thermal_frac_z_arr(i,j,k) = thermal_frac_arr(i,j,k) / sum_dz_dtemp;
            rest_frac_arr(i,j,k) = bar / (foo + bar);
            for (int g = 0; g < NGROUPS; ++g) {
                emission_dtemp_arr(i,j,k,g) = dz_dtemp[g] / sum_dz_dtemp;
            }
        });
    }
}


void Radiation::eos_fill_opacity_emission(const MultiFab& S_new,
                                       const MultiFab& temp_next,
                                       const MultiFab& temp_iterate,
                                       MultiFab& kappa_p, MultiFab& kappa_r, MultiFab& jg,
                                       MultiFab& dEmitT, MultiFab& dOpacT, MultiFab& dedT,
                                       int level, int it, int ngrow)
{
  int star_valid_margin = 1 - ngrow;

  int lagged_opac;
  if (it == 1) {
    lagged_opac = 0;
  }
  else if (it <= opacity_lag_iter) {
    lagged_opac = 0;
  }
  else {
    lagged_opac = 1;
  }

  const Geometry& geom = parent->Geom(level);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& box = mfi.tilebox();

      auto energy_dtemp_arr = dedT[mfi].array();
      auto temp_field_arr = temp_next[mfi].array();
      auto new_state_view = S_new[mfi].array();

      amrex::ParallelFor(box,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          Real rhoInv = 1.e0_rt / new_state_view(i,j,k,URHO);

          eos_re_t eos_state;
          eos_state.rho = new_state_view(i,j,k,URHO);
          eos_state.T   = temp_field_arr(i,j,k);
          for (int n = 0; n < NumSpec; ++n) {
              eos_state.xn[n] = new_state_view(i,j,k,UFS+n) * rhoInv;
          }
#if NAUX_NET > 0
          for (int n = 0; n < NumAux; ++n) {
              eos_state.aux[n] = new_state_view(i,j,k,UFX+n) * rhoInv;
          }
#endif

          eos(eos_input_rt, eos_state);

          energy_dtemp_arr(i,j,k) = eos_state.cv;
      });
  }

  if (dedT_safety > 1.0) {
    dedT.mult(dedT_safety);
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.growntilebox(ngrow);

      auto new_state_view = S_new[mfi].array();
      auto temp_next_arr = temp_next[mfi].array();
      auto temp_iterate_arr = temp_iterate[mfi].array();
      auto kappa_planck_arr = kappa_p[mfi].array();
      auto kappa_ross_arr = kappa_r[mfi].array();
      auto kappa_dtemp_arr = dOpacT[mfi].array();
      auto emission_arr = jg[mfi].array();
      auto emission_dtemp_arr = dEmitT[mfi].array();

      bool local_use_kappa_dtemp = use_kappa_deriv;

      GpuArray<Real, NGROUPS> group_freq_loc;
      for (int g = 0; g < NGROUPS; ++g) {
          group_freq_loc[g] = freq_center[g];
      }
      GpuArray<Real, NGROUPS+1> freq_edges_loc;
      for (int g = 0; g < NGROUPS+1; ++g) {
          freq_edges_loc[g] = xnu[g];
      }

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          const Real fac = 0.5e0_rt;
          const Real min_temp_step_frac = 1.e-8_rt;

          if (lagged_opac) {
              kappa_dtemp_arr(i,j,k) = 0.0_rt;
              return;
          }

          Real rho = new_state_view(i,j,k,URHO);
          Real temp = temp_next_arr(i,j,k);

          Real Ye;
          if (NumAux > 0) {
              Real Ye = new_state_view(i,j,k,UFX);
          } else {
              Ye = 0.e0_rt;
          }

          Real dT;
          if (star_valid_margin > 0) {
              dT = fac * std::abs(temp_iterate_arr(i,j,k) - temp_next_arr(i,j,k));
              dT = amrex::max(dT, min_temp_step_frac * temp_next_arr(i,j,k));
          } else {
              dT = temp_next_arr(i,j,k) * 1.e-3_rt + 1.e-50_rt;
          }

          for (int g = 0; g < NGROUPS; ++g) {
              Real nu = group_freq_loc[g];

              bool need_planck_opac = true;
              bool need_ross_opac = true;

              Real kp, kr;

              opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

              kappa_planck_arr(i,j,k,g) = kp;
              kappa_ross_arr(i,j,k,g) = kr;

              if (local_use_kappa_dtemp == 0) {

                  kappa_dtemp_arr(i,j,k,g) = 0.e0_rt;

              } else {

                  bool need_planck_opac = true;
                  bool need_ross_opac = false;

                  Real kp1, kr1, kp2, kr2;

                  opacity(kp1, kr1, rho, temp-dT, Ye, nu, need_planck_opac, need_ross_opac);
                  opacity(kp2, kr2, rho, temp+dT, Ye, nu, need_planck_opac, need_ross_opac);

                  kappa_dtemp_arr(i,j,k,g) = (kp2 - kp1) / (2.e0_rt * dT);
              }
          }
      });

      const Box& reg = mfi.tilebox();

      amrex::ParallelFor(reg,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {

          Real temp_floored = amrex::max(temp_next_arr(i,j,k), 1.e-50_rt);

          Real B1, dplanck_curr;
          PlanckAntiderivSlope(temp_floored, 0.0_rt, B1, dplanck_curr);

          for (int g = 0; g < NGROUPS; ++g) {

              Real freq_upper = freq_edges_loc[g+1];


              if (g == NGROUPS - 1) {
                  freq_upper = amrex::max(freq_upper, 1.e25_rt);
              }

              Real B0 = B1;
              Real dplanck_prev = dplanck_curr;
              PlanckAntiderivSlope(temp_floored, freq_upper, B1, dplanck_curr);
              Real Bg = B1 - B0;
              Real dplanck_group_diff = dplanck_curr - dplanck_prev;

              emission_arr(i,j,k,g) = Bg * kappa_planck_arr(i,j,k,g);
              emission_dtemp_arr(i,j,k,g) = kappa_dtemp_arr(i,j,k,g) * Bg + dplanck_group_diff * kappa_planck_arr(i,j,k,g);


              user_emissivity(i, j, k, g,
                                 group_freq_loc, freq_edges_loc,
                                 temp_next_arr(i,j,k), kappa_planck_arr(i,j,k,g),
                                 kappa_dtemp_arr(i,j,k,g), emission_arr(i,j,k,g), emission_dtemp_arr(i,j,k,g));
          }
      });
  }

  if (ngrow == 0 && !lagged_opac) {
      kappa_r.FillBoundary(geom.periodicity());
  }
}


void Radiation::gray_correction(MultiFab& rad_energy_new, MultiFab& rad_energy_iter,
                           MultiFab& kappa_p, MultiFab& kappa_r,
                           MultiFab& thermal_frac, MultiFab& rest_frac,
                           MultiFab& group_weight_mf,
                           Array<MultiFab, AMREX_SPACEDIM>& lambda,
                           RadTransportSolver* solver, MultiGroupBoundary& multigroup_boundary,
                           const BoxArray& grids, int level, Real time,
                           Real delta_t, Real pseudo_transient_tau)
{
  const Geometry& geom = parent->Geom(level);
  auto dx = parent->Geom(level).CellSizeArray();
  const HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
  const DistributionMapping& dmap = hydrosim->DistributionMap();

  if (nGroups > 1) {
    solver->setSolverScale(1.0);
  }
  else {
    solver->setSolverScale(0.0);
  }
  multigroup_boundary.enterCorrection();

  MultiFab rad_energy_zeros(grids, dmap, 1, 0);
  rad_energy_zeros.setVal(0.0);
  fetchMultigridBndryZeroRad(multigroup_boundary, rad_energy_zeros, level);

  MultiFab spec(grids, dmap, nGroups, 1);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(spec, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.tilebox();

    auto kappa_planck_arr = kappa_p[mfi].array();
    auto group_weight_arr = group_weight_mf[mfi].array();
    auto spectral_weight_arr = spec[mfi].array();

    Real inv_light_dt = 1.e0_rt / (C::c_light * delta_t);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        Real epsilon[NGROUPS];
        Real sum_epsilon = 0.0;
        for (int g = 0; g < NGROUPS; ++g) {
            Real kappa_shift = kappa_planck_arr(i,j,k,g) + (1.e0_rt + pseudo_transient_tau) * inv_light_dt;
            epsilon[g] = group_weight_arr(i,j,k,g) / kappa_shift;
            sum_epsilon += epsilon[g];
        }

        if (sum_epsilon == 0.e0_rt) {
            for (int g = 0; g < NGROUPS; ++g) {
                spectral_weight_arr(i,j,k,g) = 0.e0_rt;
            }
        } else {
            for (int g = 0; g < NGROUPS; ++g) {
                spectral_weight_arr(i,j,k,g) = epsilon[g] / sum_epsilon;
            }
        }
    });
  }

  for (int indx = 0; indx < nGroups; indx++) {
    extendBorderValues(spec, indx);
  }
  spec.FillBoundary(parent->Geom(level).periodicity());

  solver->levelBoundary(multigroup_boundary,0);

  MultiFab acoefs(grids, dmap, 1, 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(acoefs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      auto rest_frac_arr = rest_frac[mfi].array();
      auto spectral_weight_arr = spec[mfi].array();
      auto kappa_planck_arr = kappa_p[mfi].array();
      auto diag_coef_arr = acoefs[mfi].array();

      const Real dt1 = (1.e0_rt + pseudo_transient_tau) / delta_t;

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          Real kappa_avg = 0.0;
          for (int g = 0; g < NGROUPS; ++g) {
              kappa_avg += spectral_weight_arr(i,j,k,g) * kappa_planck_arr(i,j,k,g);
          }

          Real H1 = rest_frac_arr(i,j,k);

          diag_coef_arr(i,j,k) = H1 * kappa_avg * C::c_light + dt1;
      });
  }

  solver->applyCellMetrics(level, acoefs);
  solver->setLevelDiagCoeffs(level, acoefs);

  const DistributionMapping& dm = hydrosim->DistributionMap();

  Array<MultiFab, AMREX_SPACEDIM> bcoefs, cell_coefs, edge_group_coef;
  for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
    const BoxArray& edge_boxes = hydrosim->getEdgeBoxArray(idim);

    bcoefs[idim].define(edge_boxes, dm, 1, 0);
    bcoefs[idim].setVal(0.0);

    edge_group_coef [idim].define(edge_boxes, dm, 1, 0);

    if (nGroups > 1) {
        cell_coefs[idim].define(edge_boxes, dm, 2, 0);
        cell_coefs[idim].setVal(0.0);
    }
  }

  for (int rad_group_idx = 0; rad_group_idx < nGroups; rad_group_idx++) {
    for (int idim=0; idim<AMREX_SPACEDIM; idim++) {
      solver->computeFaceCoeffs(edge_group_coef[idim], idim, kappa_r, rad_group_idx,
                            lambda[idim], rad_group_idx, c, geom);

#ifdef _OPENMP
#pragma omp parallel
#endif
      for (MFIter mfi(spec, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
          const Box&  bx  = mfi.nodaltilebox(idim);

          auto dir_coef_arr = bcoefs[idim][mfi].array();
          auto edge_group_coef_arr = edge_group_coef[idim][mfi].array();
          auto spectral_weight_arr = spec[mfi].array(rad_group_idx);
          auto grad_coef_arr = cell_coefs[idim][mfi].array();

          amrex::ParallelFor(bx,
          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
          {
              if (idim == 0) {
                  dir_coef_arr(i,j,k) += 0.5e0_rt * (spectral_weight_arr(i-1,j,k) + spectral_weight_arr(i,j,k)) * edge_group_coef_arr(i,j,k);
              }
              else if (idim == 1) {
                  dir_coef_arr(i,j,k) += 0.5e0_rt * (spectral_weight_arr(i,j-1,k) + spectral_weight_arr(i,j,k)) * edge_group_coef_arr(i,j,k);
              }
              else {
                  dir_coef_arr(i,j,k) += 0.5e0_rt * (spectral_weight_arr(i,j,k-1) + spectral_weight_arr(i,j,k)) * edge_group_coef_arr(i,j,k);
              }
          });

          if (nGroups > 1) {
              amrex::ParallelFor(bx,
              [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
              {
                  int ioff, joff, koff;
                  Real h1;

                  if (idim == 0) {
                      ioff = 1;
                      joff = 0;
                      koff = 0;
                      h1 = 1.e0_rt / dx[0];
                  }
                  else if (idim == 1) {
                      ioff = 0;
                      joff = 1;
                      koff = 0;
                      h1 = 1.e0_rt / dx[1];
                  }
                  else {
                      ioff = 0;
                      joff = 0;
                      koff = 1;
                      h1 = 1.e0_rt / dx[2];
                  }

                  Real spec_grad = (spectral_weight_arr(i,j,k) - spectral_weight_arr(i-ioff,j-joff,k-koff)) * h1;
                  Real foo = -0.5e0_rt * edge_group_coef_arr(i,j,k) * spec_grad;
                  grad_coef_arr(i,j,k,0) += foo;
                  grad_coef_arr(i,j,k,1) += foo;
              });
          }
      }
    }
  }

  for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
    solver->setLevelFaceCoeffs(level, bcoefs[idim], idim);

    if (nGroups > 1) {
      solver->setLevelCellCoeffs(level, cell_coefs[idim], idim);
    }
  }

  MultiFab rhs(grids,dmap,1,0);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      auto Ern = rad_energy_new[mfi].array();
      auto Erl = rad_energy_iter[mfi].array();
      auto kap = kappa_p[mfi].array();
      auto thermal_frac_arr = thermal_frac[mfi].array();
      auto rhs_arr = rhs[mfi].array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          Real rad_source_term = 0.0;
          for (int g = 0; g < NGROUPS; ++g) {
              rad_source_term += kap(i,j,k,g) * (Ern(i,j,k,g) - Erl(i,j,k,g));
          }

          Real H = thermal_frac_arr(i,j,k);

          rhs_arr(i,j,k) = C::c_light * H * rad_source_term;
      });
  }

  solver->applyCellMetrics(level, rhs);

  MultiFab accel(grids,dmap,1,0);
  accel.setVal(0.0);
  solver->levelSolveSystem(level, accel, 0, rhs, 0.01);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(spec, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& reg  = mfi.tilebox();

      auto rad_energy_new_arr = rad_energy_new[mfi].array();
      auto spectral_weight_arr = spec[mfi].array();
      auto correction_arr = accel[mfi].array();

      amrex::ParallelFor(reg,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          for (int n = 0; n < NGROUPS; ++n) {
              rad_energy_new_arr(i,j,k,n) = rad_energy_new_arr(i,j,k,n) + spectral_weight_arr(i,j,k,n) * correction_arr(i,j,k);
          }
      });
  }

  multigroup_boundary.leaveCorrection();
  fetchMultigridBndry(multigroup_boundary, rad_energy_new, time, level);

  solver->restoreSolverScale();
}


void Radiation::local_convergence_boost(MultiFab& rad_energy_new, const MultiFab& rad_energy_iter,
                            const MultiFab& kappa_p,
                            const MultiFab& thermal_frac,
                            const MultiFab& group_weight_mf,
                            Real delta_t, Real pseudo_transient_tau)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(rad_energy_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto Ern = rad_energy_new[mfi].array();
        auto Erl = rad_energy_iter[mfi].array();
        auto kap = kappa_p[mfi].array();
        auto thermal_frac_arr = thermal_frac[mfi].array();
        auto group_weight_arr = group_weight_mf[mfi].array();

        Real inv_light_dt = 1.0_rt / (C::c_light * delta_t);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Real Hg[NGROUPS], kappa_shift[NGROUPS];
            Real rad_source_term = 0.0_rt;
            Real p = 1.0_rt;

            for (int g = 0; g < NGROUPS; ++g) {
                Hg[g] = group_weight_arr(i,j,k,g) * thermal_frac_arr(i,j,k);
                kappa_shift[g] = kap(i,j,k,g) + (1.e0_rt + pseudo_transient_tau) * inv_light_dt;
                Real kk = kap(i,j,k,g) / kappa_shift[g];

                p -= Hg[g] * kk;
                rad_source_term += kap(i,j,k,g) * (Ern(i,j,k,g) - Erl(i,j,k,g));
            }

            for (int g = 0; g < NGROUPS; ++g) {
                Ern(i,j,k,g) = Ern(i,j,k,g) + (Hg[g] * rad_source_term) / (kappa_shift[g] * p + 1.e-50_rt);
            }
        });
    }
}


void Radiation::update_state_energy(MultiFab& state, const MultiFab& rhoe,
                                    const MultiFab& temp,
                                    const BoxArray& grids,
                                    Real& energy_rel_chg, Real& dT, int level)
{
  BL_PROFILE("Radiation::update_state_energy (MultigroupDiffusion)");

  const DistributionMapping& dmap = state.DistributionMap();

  MultiFab msk(grids,dmap,1,0);

  {
      BoxArray baf;

      if (level < parent->finestLevel()) {
          baf = parent->boxArray(level+1);
          baf.coarsen(parent->refRatio(level));
      }

#ifdef _OPENMP
#pragma omp parallel
#endif
      for (MFIter mfi(msk); mfi.isValid(); ++mfi)
      {
          msk[mfi].setVal<RunOn::Device>(1.0);

          if (level < parent->finestLevel()) {
              std::vector< std::pair<int,Box> > isects = baf.intersections(mfi.validbox());
              for (int ii = 0; ii < isects.size(); ii++) {
                  msk[mfi].setVal<RunOn::Device>(0.0, isects[ii].second, 0);
              }
          }
      }
  }

  ReduceOps<ReduceOpMax, ReduceOpMax> reduce_op;
  ReduceData<Real, Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(state,true); mfi.isValid(); ++mfi)
  {
      const Box& reg  = mfi.tilebox();

      auto state_arr = state[mfi].array();
      auto temp_field_arr = temp[mfi].array();
      auto gas_energy_arr = rhoe[mfi].array();
      auto valid_mask_arr = msk[mfi].array();

      reduce_op.eval(reg, reduce_data,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real ei = state_arr(i,j,k,UEINT);
          Real energy_rel_loc = std::abs((gas_energy_arr(i,j,k) - ei) *
                                    valid_mask_arr(i,j,k) / (ei + 1.e-50_rt));
          Real ek = state_arr(i,j,k,UEDEN) - state_arr(i,j,k,UEINT);
          state_arr(i,j,k,UEINT) = gas_energy_arr(i,j,k);
          state_arr(i,j,k,UEDEN) = gas_energy_arr(i,j,k) + ek;

          Real temp_old = state_arr(i,j,k,UTEMP);
          Real temp_rel_loc = std::abs((temp_field_arr(i,j,k) - temp_old) *
                                    valid_mask_arr(i,j,k) / (temp_old + 1.e-50_rt));
          state_arr(i,j,k,UTEMP) = temp_field_arr(i,j,k);

          return {energy_rel_loc, temp_rel_loc};
      });

  }

  ReduceTuple hv = reduce_data.value();
  energy_rel_chg = amrex::get<0>(hv);
  dT = amrex::get<1>(hv);

  ParallelDescriptor::ReduceRealMax(energy_rel_chg);
  ParallelDescriptor::ReduceRealMax(dT);
}


void Radiation::advance_matter(MultiFab& rhoe_new, MultiFab& temp_next,
                              const MultiFab& rad_energy_new, const MultiFab& rad_energy_iter,
                              const MultiFab& gas_energy_star,
                              const MultiFab& gas_energy_step,
                              const MultiFab& thermal_frac, const MultiFab& thermal_frac_z,
                              const MultiFab& rest_frac,
                              const MultiFab& couple_source,
                              const MultiFab& kappa_p, const MultiFab& jg,
                              const MultiFab& group_weight_mf,
                              const MultiFab& S_new,
                              int level, Real delta_t, Real pseudo_transient_tau,
                              int it, bool energy_conserving_mode)
{
    const Real cdt = C::c_light * delta_t;
    const Real inv_light_dt = 1.e0_rt / (C::c_light * delta_t);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(rhoe_new,true); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto gas_energy_new_view = rhoe_new[mfi].array();
        auto new_state_view = S_new[mfi].array();
        auto temp_next_view = temp_next[mfi].array();
        auto rad_energy_new_view = rad_energy_new[mfi].array();
        auto rad_energy_prev = rad_energy_iter[mfi].array();
        auto gas_energy_star_view = gas_energy_star[mfi].array();
        auto gas_energy_step_view = gas_energy_step[mfi].array();
        auto rest_frac_arr = rest_frac[mfi].array();
        auto cpT = couple_source[mfi].array();
        auto thermal_frac_z_view = thermal_frac_z[mfi].array();
        auto kpp  = kappa_p[mfi].array();
        auto emission_arr = jg[mfi].array();

        if (energy_conserving_mode) {

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                Real H1 = rest_frac_arr(i,j,k);

                Real kappa_weighted_rad_chg = 0.0;
                for (int g = 0; g < NGROUPS; ++g) {
                    kappa_weighted_rad_chg += kpp(i,j,k,g) * (rad_energy_new_view(i,j,k,g) - rad_energy_prev(i,j,k,g));
                }

                Real chg = cdt * kappa_weighted_rad_chg + H1 * ((gas_energy_step_view(i,j,k) - gas_energy_star_view(i,j,k)) + cdt * cpT(i,j,k));

                gas_energy_new_view(i,j,k) = gas_energy_star_view(i,j,k) + chg;

                gas_energy_new_view(i,j,k) = (gas_energy_new_view(i,j,k) + pseudo_transient_tau * gas_energy_star_view(i,j,k)) / (1.e0_rt + pseudo_transient_tau);
            });

            temp_next[mfi].copy<RunOn::Device>(rhoe_new[mfi],bx);
            Gpu::synchronize();

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {

                if (temp_next_view(i,j,k) <= 0.e0_rt)
                {
                    temp_next_view(i,j,k) = small_temp;
                }
                else
                {
                    Real rhoInv = 1.e0_rt / new_state_view(i,j,k,URHO);

                    eos_re_t eos_state;
                    eos_state.rho = new_state_view(i,j,k,URHO);
                    eos_state.T   = new_state_view(i,j,k,UTEMP);
                    eos_state.e   = temp_next_view(i,j,k) * rhoInv;
                    for (int n = 0; n < NumSpec; ++n) {
                        eos_state.xn[n] = new_state_view(i,j,k,UFS+n) * rhoInv;
                    }
#if NAUX_NET > 0
                    for (int n = 0; n < NumAux; ++n) {
                        eos_state.aux[n] = new_state_view(i,j,k,UFX+n) * rhoInv;
                    }
#endif

                    eos(eos_input_re, eos_state);

                    temp_next_view(i,j,k) = eos_state.T;
                }
            });
            Gpu::synchronize();
        }
        else {

            const Real fac = 0.01_rt;

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                Real cpT = 0.e0_rt;

                for (int g = 0; g < NGROUPS; ++g) {
                    cpT = cpT + kpp(i,j,k,g) * rad_energy_new_view(i,j,k,g) - emission_arr(i,j,k,g);
                }

                Real scratch_energy = cpT - (gas_energy_star_view(i,j,k) - gas_energy_step_view(i,j,k)) * inv_light_dt;

                Real temp_update = thermal_frac_z_view(i,j,k) * scratch_energy;

                if (std::abs(temp_update / (temp_next_view(i,j,k) + 1.e-50_rt)) > fac) {
                    temp_update = std::copysign(fac * temp_next_view(i,j,k), temp_update);
                }

                temp_next_view(i,j,k) = temp_next_view(i,j,k) + temp_update;

                Real rhoInv = 1.e0_rt / new_state_view(i,j,k,URHO);

                eos_re_t eos_state;
                eos_state.rho = new_state_view(i,j,k,URHO);
                eos_state.T   = temp_next_view(i,j,k);
                for (int n = 0; n < NumSpec; ++n) {
                    eos_state.xn[n] = new_state_view(i,j,k,UFS+n) * rhoInv;
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; ++n) {
                    eos_state.aux[n] = new_state_view(i,j,k,UFX+n) * rhoInv;
                }
#endif

                eos(eos_input_rt, eos_state);

                gas_energy_new_view(i,j,k) = eos_state.rho * eos_state.e;
            });
        }
    }
}


void Radiation::build_flux_limiter(int level, const BoxArray& grids,
                                const MultiFab &Sborder,
                                const MultiFab &RadEnergyGrown,
                                MultiFab &rad_lambda_grown)
{
  int ngrow = rad_lambda_grown.nGrow();
  const DistributionMapping& dmap = rad_lambda_grown.DistributionMap();

  if (filter_limiter_T) {
    BL_ASSERT(ngrow == 4);
  }

  if (radiation::limiter == 0) {

    rad_lambda_grown.setVal(1./3., ngrow);

  }
  else {

    MultiFab kpr(grids,dmap,Radiation::nGroups,ngrow);

    if (groups_enabled) {
        MultigroupDiffusion_compute_rosseland(kpr, Sborder);
    }
    else {
      compute_rosseland_opacity(kpr, Sborder);
    }

    MultiFab rad_energy_padded(grids, dmap, nGroups, ngrow+1);
    rad_energy_padded.setVal(-1.0);
    MultiFab::Copy(rad_energy_padded, RadEnergyGrown, 0, 0, nGroups, 0);

    rad_energy_padded.FillBoundary(parent->Geom(level).periodicity());

    const Real* dx = parent->Geom(level).CellSize();

    using namespace filter;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(rad_energy_padded,false); mfi.isValid(); ++mfi) {
        FArrayBox limiter_filt_x_fab;
#if AMREX_SPACEDIM >= 2
        FArrayBox limiter_filt_y_fab;
#endif
#if AMREX_SPACEDIM == 3
        FArrayBox limiter_filt_z_fab;
#endif

        Box limiter_box = rad_lambda_grown[mfi].box();
        Box bx = grow(limiter_box, -ngrow);

        if (filter_limiter_T > 0) {
            limiter_filt_x_fab.resize(bx);
#if AMREX_SPACEDIM >= 2
            limiter_filt_y_fab.resize(bx);
#endif
#if AMREX_SPACEDIM == 3
            limiter_filt_z_fab.resize(bx);
#endif
        }

        Array4<Real> const lam = rad_lambda_grown[mfi].array();
        Array4<Real> const Er = rad_energy_padded[mfi].array();
        Array4<Real> const kap = kpr[mfi].array();
        Array4<Real> const limiter_filt_x = limiter_filt_x_fab.array();
#if AMREX_SPACEDIM >= 2
        Array4<Real> const limiter_filt_y = limiter_filt_y_fab.array();
#endif
#if AMREX_SPACEDIM == 3
        Array4<Real> const limiter_filt_z = limiter_filt_z_fab.array();
#endif

        for (int g = 0; g < Radiation::nGroups; ++g) {

            amrex::Loop(limiter_box, [=] (int i, int j, int k) noexcept
            {
                lam(i,j,k,g) = -1.0e50_rt;


                if (Er(i,j,k,g) == -1.0_rt) {
                    return;
                }

                Real r1 = 0.0;
                Real r2 = 0.0;
                Real r3 = 0.0;

                if (Er(i-1,j,k,g) == -1.0_rt) {
                    r1 = (Er(i+1,j,k,g) - Er(i,j,k,g)) / (dx[0]);
                }
                else if (Er(i+1,j,k,g) == -1.0_rt) {
                    r1 = (Er(i,j,k,g) - Er(i-1,j,k,g)) / (dx[0]);
                }
                else {
                    r1 = (Er(i+1,j,k,g) - Er(i-1,j,k,g)) / (2.e0_rt*dx[0]);
                }

#if AMREX_SPACEDIM >= 2
                if (Er(i,j-1,k,g) == -1.0_rt) {
                    r2 = (Er(i,j+1,k,g) - Er(i,j,k,g)) / (dx[1]);
                }
                else if (Er(i,j+1,k,g) == -1.0_rt) {
                    r2 = (Er(i,j,k,g) - Er(i,j-1,k,g)) / (dx[1]);
                }
                else {
                    r2 = (Er(i,j+1,k,g) - Er(i,j-1,k,g)) / (2.e0_rt*dx[1]);
                }
#endif

#if AMREX_SPACEDIM == 3
                if (Er(i,j,k-1,g) == -1.0_rt) {
                    r3 = (Er(i,j,k+1,g) - Er(i,j,k,g)) / dx[2];
                }
                else if (Er(i,j,k+1,g) == -1.0_rt) {
                    r3 = (Er(i,j,k,g) - Er(i,j,k-1,g)) / dx[2];
                }
                else {
                    r3 = (Er(i,j,k+1,g) - Er(i,j,k-1,g)) / (2.e0_rt*dx[2]);
                }
#endif

                Real r = std::sqrt(r1 * r1 + r2 * r2 + r3 * r3);
                r = r / (kap(i,j,k,g) * std::max(Er(i,j,k,g), 1.e-50_rt));

                lam(i,j,k,g) = LimitedDiffusionLambda(r);
            });


            int limiter_ilo = limiter_box.loVect3d()[0];
            int limiter_ihi = limiter_box.hiVect3d()[0];
            int limiter_jlo = limiter_box.loVect3d()[1];
            int limiter_jhi = limiter_box.hiVect3d()[1];
            int limiter_klo = limiter_box.loVect3d()[2];
            int limiter_khi = limiter_box.hiVect3d()[2];

            int region_ilo = bx.loVect3d()[0];
            int region_ihi = bx.hiVect3d()[0];
            int region_jlo = bx.loVect3d()[1];
            int region_jhi = bx.hiVect3d()[1];
            int region_klo = bx.loVect3d()[2];
            int region_khi = bx.hiVect3d()[2];

            if (filter_limiter_T > 0) {

                amrex::Loop(bx, [=] (int i, int j, int k) noexcept
                {
                    limiter_filt_x(i,j,k) = -1.0e-50_rt;
#if AMREX_SPACEDIM >= 2
                    limiter_filt_y(i,j,k) = -1.0e-50_rt;
#endif
#if AMREX_SPACEDIM == 3
                    limiter_filt_z(i,j,k) = -1.0e-50_rt;
#endif
                });


                int dir = 0;

                amrex::Loop(bx, [=] (int i, int j, int k) noexcept
                {
                    if (filter_limiter_T == 1) {
                        limiter_filt_x(i,j,k) = apply_filter<1>(Er, lam, dir, limiter_filter_strength, i, j, k, g);
                    }
                    else if (filter_limiter_T == 2) {
                        limiter_filt_x(i,j,k) = apply_filter<2>(Er, lam, dir, limiter_filter_strength, i, j, k, g);
                    }
                    else if (filter_limiter_T == 3) {
                        limiter_filt_x(i,j,k) = apply_filter<3>(Er, lam, dir, limiter_filter_strength, i, j, k, g);
                    }
                    else if (filter_limiter_T == 4) {
                        limiter_filt_x(i,j,k) = apply_filter<4>(Er, lam, dir, limiter_filter_strength, i, j, k, g);
                    }
                });


#if AMREX_SPACEDIM >= 2
                dir = 1;

                amrex::Loop(bx, [=] (int i, int j, int k) noexcept
                {
                    if (filter_limiter_T == 1) {
                        limiter_filt_y(i,j,k) = apply_filter<1>(Er, limiter_filt_x, dir, limiter_filter_strength, i, j, k, g);
                    }
                    else if (filter_limiter_T == 2) {
                        limiter_filt_y(i,j,k) = apply_filter<2>(Er, limiter_filt_x, dir, limiter_filter_strength, i, j, k, g);
                    }
                    else if (filter_limiter_T == 3) {
                        limiter_filt_y(i,j,k) = apply_filter<3>(Er, limiter_filt_x, dir, limiter_filter_strength, i, j, k, g);
                    }
                    else if (filter_limiter_T == 4) {
                        limiter_filt_y(i,j,k) = apply_filter<4>(Er, limiter_filt_x, dir, limiter_filter_strength, i, j, k, g);
                    }
                });
#endif


#if AMREX_SPACEDIM == 3
                dir = 2;

                amrex::Loop(bx, [=] (int i, int j, int k) noexcept
                {
                    if (filter_limiter_T == 1) {
                        limiter_filt_z(i,j,k) = apply_filter<1>(Er, limiter_filt_y, dir, limiter_filter_strength, i, j, k, g);
                    }
                    else if (filter_limiter_T == 2) {
                        limiter_filt_z(i,j,k) = apply_filter<2>(Er, limiter_filt_y, dir, limiter_filter_strength, i, j, k, g);
                    }
                    else if (filter_limiter_T == 3) {
                        limiter_filt_z(i,j,k) = apply_filter<3>(Er, limiter_filt_y, dir, limiter_filter_strength, i, j, k, g);
                    }
                    else if (filter_limiter_T == 4) {
                        limiter_filt_z(i,j,k) = apply_filter<4>(Er, limiter_filt_y, dir, limiter_filter_strength, i, j, k, g);
                    }
                });
#endif


                amrex::Loop(bx, [=] (int i, int j, int k) noexcept
                {
#if AMREX_SPACEDIM == 1
                    lam(i,j,k,g) = limiter_filt_x(i,j,k);
#elif AMREX_SPACEDIM == 2
                    lam(i,j,k,g) = limiter_filt_y(i,j,k);
#else
                    lam(i,j,k,g) = limiter_filt_z(i,j,k);
#endif
                });
            }


            amrex::Loop(limiter_box, [=] (int i, int j, int k) noexcept
            {

                if (Er(i,j,k,g) != -1.0_rt) {
                    return;
                }


                int ic = i;
                int jc = j;
                int kc = k;

                if (i < region_ilo) {
                    ic = region_ilo;
                }
                else if (i > region_ihi) {
                    ic = region_ihi;
                }

                if (j < region_jlo) {
                    jc = region_jlo;
                }
                else if (j > region_jhi) {
                    jc = region_jhi;
                }

                if (k < region_klo) {
                    kc = region_klo;
                }
                else if (k > region_khi) {
                    kc = region_khi;
                }

                lam(i,j,k,g) = lam(ic,jc,kc,g);
            });
        }
    }

    if (filter_limiter_T) {
        rad_lambda_grown.FillBoundary(parent->Geom(level).periodicity());
    }
  }
}


void Radiation::estimate_rad_gamma_pressure(const FArrayBox& state, const FArrayBox& Er,
                                FArrayBox& gPr, const Real*dx, const Box& box)
{
    auto rad_gamma_pressure_arr = gPr.array();
    auto rad_eden_arr = Er.array();

    if (radiation::limiter == 0) {

        amrex::ParallelFor(gPr.box(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            rad_gamma_pressure_arr(i,j,k) = 0.e0_rt;

            for (int g = 0; g < NGROUPS; ++g) {
                rad_gamma_pressure_arr(i,j,k) = rad_gamma_pressure_arr(i,j,k) + 4.e0_rt / 9.e0_rt * rad_eden_arr(i,j,k,g);
            }
        });

        Gpu::synchronize();

    }
    else {

        FArrayBox kappa_r(gPr.box(), nGroups);

        auto kappa_ross_arr = kappa_r.array();

        if (groups_enabled) {
            MultigroupDiffusion_compute_rosseland(kappa_r, state);
        }
        else {
            compute_rosseland_opacity(kappa_r, state);
        }


        int im = 1, ip = 1;
        Real xm = 2.0_rt, xp = 2.0_rt;

        if (!(gPr.box().loVect()[0] - 1 >= box.loVect()[0])) {
            im = 0;
            xm = 1.0_rt;
        }

        if (!(gPr.box().hiVect()[0] + 1 <= box.hiVect()[0])) {
            ip = 0;
            xp = 1.0_rt;
        }

#if AMREX_SPACEDIM >= 2
        int jm = 1, jp = 1;
        Real ym = 2.0_rt, yp = 2.0_rt;

        if (!(gPr.box().loVect()[1] - 1 >= box.hiVect()[1])) {
            jm = 0;
            ym = 1.0_rt;
        }

        if (!(gPr.box().hiVect()[1] + 1 <= box.hiVect()[1])) {
            jp = 0;
            yp = 1.0_rt;
        }
#endif

#if AMREX_SPACEDIM == 3
        int km = 1, kp = 1;
        Real zm = 2.0_rt, zp = 2.0_rt;

        if (!(gPr.box().loVect()[2] - 1 >= box.hiVect()[2])) {
            km = 0;
            zm = 1.0_rt;
        }

        if (!(gPr.box().hiVect()[2] + 1 <= box.hiVect()[2])) {
            kp = 0;
            zp = 1.0_rt;
        }
#endif

        const Real dx0 = dx[0];
#if AMREX_SPACEDIM >= 2
        const Real dx1 = dx[1];
#endif
#if AMREX_SPACEDIM == 3
        const Real dx2 = dx[2];
#endif

        amrex::ParallelFor(gPr.box(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            rad_gamma_pressure_arr(i,j,k) = 0.0_rt;


            bool i_inside = true, j_inside = true, k_inside = true;
            bool x_at_lo = false, y_at_lo = false, z_at_lo = false;
            bool x_at_hi = false, y_at_hi = false, z_at_hi = false;

            if (i == lbound(rad_gamma_pressure_arr).x) {
                i_inside = false;
                x_at_lo = true;
            }
            if (i == ubound(rad_gamma_pressure_arr).x) {
                i_inside = false;
                x_at_hi = true;
            }

#if AMREX_SPACEDIM >= 2
            if (j == lbound(rad_gamma_pressure_arr).y) {
                j_inside = false;
                y_at_lo = true;
            }
            if (j == ubound(rad_gamma_pressure_arr).y) {
                j_inside = false;
                y_at_hi = true;
            }
#endif

#if AMREX_SPACEDIM == 3
            if (k == lbound(rad_gamma_pressure_arr).z) {
                k_inside = false;
                z_at_lo = true;
            }
            if (k == ubound(rad_gamma_pressure_arr).z) {
                k_inside = false;
                z_at_hi = true;
            }
#endif

            for (int g = 0; g < NGROUPS; ++g) {

                Real gE1 = 0.0_rt;
                Real gE2 = 0.0_rt;
                Real gE3 = 0.0_rt;

                if (i_inside && j_inside && k_inside) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (2.0_rt * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-1,k,g)) / (2.0_rt * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-1,g)) / (2.0_rt * dx2);
#endif
                }

                else if (x_at_lo && y_at_lo && z_at_lo) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-im,j,k,g)) / (xm * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-jm,k,g)) / (ym * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-km,g)) / (zm * dx2);
#endif
                }

                else if (i_inside && y_at_lo && z_at_lo) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (2.0_rt * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-jm,k,g)) / (ym * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-km,g)) / (zm * dx2);
#endif
                }

                else if (x_at_hi && y_at_lo && z_at_lo) {
                    gE1 = (rad_eden_arr(i+ip,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (xp * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-jm,k,g)) / (ym * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-km,g)) / (zm * dx2);
#endif
                }

                else if (x_at_lo && j_inside && z_at_lo) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-im,j,k,g)) / (xm * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-1,k,g)) / (2.0_rt * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-km,g)) / (zm * dx2);
#endif
                }

                else if (i_inside && j_inside && z_at_lo) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (2.0_rt * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-1,k,g)) / (2.0_rt * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-km,g)) / (zm * dx2);
#endif
                }

                else if (x_at_hi && j_inside && z_at_lo) {
                    gE1 = (rad_eden_arr(i+ip,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (xp * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-1,k,g)) / (2.0_rt * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-km,g)) / (zm * dx2);
#endif
                }

                else if (x_at_lo && y_at_hi && z_at_lo) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-im,j,k,g)) / (xm * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+jp,k,g) - rad_eden_arr(i,j-1,k,g)) / (yp * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-km,g)) / (zm * dx2);
#endif
                }

                else if (i_inside && y_at_hi && z_at_lo) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (2.0_rt * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+jp,k,g) - rad_eden_arr(i,j-1,k,g)) / (yp * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-km,g)) / (zm * dx2);
#endif
                }

                else if (x_at_hi && y_at_hi && z_at_lo) {
                    gE1 = (rad_eden_arr(i+ip,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (xp * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+jp,k,g) - rad_eden_arr(i,j-1,k,g)) / (yp * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-km,g)) / (zm * dx2);
#endif
                }

                else if (x_at_lo && y_at_lo && k_inside) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-im,j,k,g)) / (xm * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-jm,k,g)) / (ym * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-1,g)) / (2.0_rt * dx2);
#endif
                }

                else if (i_inside && y_at_lo && k_inside) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (2.0_rt * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-jm,k,g)) / (ym * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-1,g)) / (2.0_rt * dx2);
#endif
                }

                else if (x_at_hi && y_at_lo && k_inside) {
                    gE1 = (rad_eden_arr(i+ip,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (xp * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-jm,k,g)) / (ym * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-1,g)) / (2.0_rt * dx2);
#endif
                }

                else if (x_at_lo && j_inside && k_inside) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-im,j,k,g)) / (xm * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-1,k,g)) / (2.0_rt * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-1,g)) / (2.0_rt * dx2);
#endif
                }

                else if (x_at_hi && j_inside && k_inside) {
                    gE1 = (rad_eden_arr(i+ip,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (xp * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-1,k,g)) / (2.0_rt * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-1,g)) / (2.0_rt * dx2);
#endif
                }

                else if (x_at_lo && y_at_hi && k_inside) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-im,j,k,g)) / (xm * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+jp,k,g) - rad_eden_arr(i,j-1,k,g)) / (yp * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-1,g)) / (2.0_rt * dx2);
#endif
                }

                else if (i_inside && y_at_hi && k_inside) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (2.0_rt * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+jp,k,g) - rad_eden_arr(i,j-1,k,g)) / (yp * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-1,g)) / (2.0_rt * dx2);
#endif
                }

                else if (x_at_hi && y_at_hi && k_inside) {
                    gE1 = (rad_eden_arr(i+ip,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (xp * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+jp,k,g) - rad_eden_arr(i,j-1,k,g)) / (yp * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+1,g) - rad_eden_arr(i,j,k-1,g)) / (2.0_rt * dx2);
#endif
                }

                else if (x_at_lo && y_at_lo && z_at_hi) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-im,j,k,g)) / (xm * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-jm,k,g)) / (ym * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+kp,g) - rad_eden_arr(i,j,k-1,g)) / (zp * dx2);
#endif
                }

                else if (i_inside && y_at_lo && z_at_hi) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (2.0_rt * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-jm,k,g)) / (ym * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+kp,g) - rad_eden_arr(i,j,k-1,g)) / (zp * dx2);
#endif
                }

                else if (x_at_hi && y_at_lo && z_at_hi) {
                    gE1 = (rad_eden_arr(i+ip,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (xp * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-jm,k,g)) / (ym * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+kp,g) - rad_eden_arr(i,j,k-1,g)) / (zp * dx2);
#endif
                }

                else if (x_at_lo && j_inside && z_at_hi) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-im,j,k,g)) / (xm * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-1,k,g)) / (2.0_rt * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+kp,g) - rad_eden_arr(i,j,k-1,g)) / (zp * dx2);
#endif
                }

                else if (i_inside && j_inside && z_at_hi) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (2.0_rt * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-1,k,g)) / (2.0_rt * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+kp,g) - rad_eden_arr(i,j,k-1,g)) / (zp * dx2);
#endif
                }

                else if (x_at_hi && j_inside && z_at_hi) {
                    gE1 = (rad_eden_arr(i+ip,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (xp * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+1,k,g) - rad_eden_arr(i,j-1,k,g)) / (2.0_rt * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+kp,g) - rad_eden_arr(i,j,k-1,g)) / (zp * dx2);
#endif
                }

                else if (x_at_lo && y_at_hi && z_at_hi) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-im,j,k,g)) / (xm * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+jp,k,g) - rad_eden_arr(i,j-1,k,g)) / (yp * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+kp,g) - rad_eden_arr(i,j,k-1,g)) / (zp * dx2);
#endif
                }

                else if (i_inside && y_at_hi && z_at_hi) {
                    gE1 = (rad_eden_arr(i+1,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (2.0_rt * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+jp,k,g) - rad_eden_arr(i,j-1,k,g)) / (yp * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+kp,g) - rad_eden_arr(i,j,k-1,g)) / (zp * dx2);
#endif
                }

                else if (x_at_hi && y_at_hi && z_at_hi) {
                    gE1 = (rad_eden_arr(i+ip,j,k,g) - rad_eden_arr(i-1,j,k,g)) / (xp * dx0);
#if AMREX_SPACEDIM >= 2
                    gE2 = (rad_eden_arr(i,j+jp,k,g) - rad_eden_arr(i,j-1,k,g)) / (yp * dx1);
#endif
#if AMREX_SPACEDIM == 3
                    gE3 = (rad_eden_arr(i,j,k+kp,g) - rad_eden_arr(i,j,k-1,g)) / (zp * dx2);
#endif
                }

                Real gE = std::sqrt(gE1 * gE1 + gE2 * gE2 + gE3 * gE3);

                Real r = gE / (kappa_ross_arr(i,j,k,g) * amrex::max(rad_eden_arr(i,j,k,g), 1.e-50_rt));
                Real lam = LimitedDiffusionLambda(r);

                Real radiation_gamma;
                if (radiation::moving_frame == 1) {
                    Real f = Eddington_factor(lam);
                    radiation_gamma = (3.0_rt - f) / 2.0_rt;
                }
                else {
                    radiation_gamma = lam + 1.0_rt;
                }

                rad_gamma_pressure_arr(i,j,k) = rad_gamma_pressure_arr(i,j,k) + lam * radiation_gamma * rad_eden_arr(i,j,k,g);
            }
        });

        Gpu::synchronize();

    }
}


void Radiation::MultigroupDiffusion_compute_rosseland(FArrayBox& kappa_r, const FArrayBox& state)
{
  BL_PROFILE("Radiation::MultigroupDiffusion_compute_rosseland (FArrayBox)");

  const Box& kbox = kappa_r.box();

  auto state_arr = state.array();
  auto kpr = kappa_r.array();

  GpuArray<Real, NGROUPS> group_freq_loc;
  for (int g = 0; g < NGROUPS; ++g) {
      group_freq_loc[g] = freq_center[g];
  }

  amrex::ParallelFor(kbox,
  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
  {
      Real rho = state_arr(i,j,k,URHO);
      Real temp = state_arr(i,j,k,UTEMP);
      Real Ye;
      if (NumAux > 0) {
          Ye = state_arr(i,j,k,UFX);
      } else {
          Ye = 0.e0_rt;
      }

      Real kp, kr;
      bool need_planck_opac = false;
      bool need_ross_opac = true;

      for (int g = 0; g < NGROUPS; ++g) {
          opacity(kp, kr, rho, temp, Ye, group_freq_loc[g], need_planck_opac, need_ross_opac);
          kpr(i,j,k,g) = kr;
      }
  });
  Gpu::synchronize();
}


void Radiation::MultigroupDiffusion_compute_rosseland(MultiFab& kappa_r, const MultiFab& state)
{
    BL_PROFILE("Radiation::MultigroupDiffusion_compute_rosseland (MultiFab)");

    GpuArray<Real, NGROUPS> group_freq_loc;
    for (int g = 0; g < NGROUPS; ++g) {
        group_freq_loc[g] = freq_center[g];
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(kappa_r, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox();

        auto state_arr = state[mfi].array();
        auto kpr = kappa_r[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Real rho = state_arr(i,j,k,URHO);
            Real temp = state_arr(i,j,k,UTEMP);
            Real Ye;
            if (NumAux > 0) {
                Ye = state_arr(i,j,k,UFX);
            } else {
                Ye = 0.e0_rt;
            }

            Real kp, kr;
            bool need_planck_opac = false;
            bool need_ross_opac = true;

            for (int g = 0; g < NGROUPS; ++g) {
                opacity(kp, kr, rho, temp, Ye, group_freq_loc[g], need_planck_opac, need_ross_opac);
                kpr(i,j,k,g) = kr;
            }
        });
    }
}

void Radiation::MultigroupDiffusion_compute_scattering(FArrayBox& kappa_s, const FArrayBox& state)
{
    BL_PROFILE("Radiation::MultigroupDiffusion_compute_scattering");

    const Box& kbox = kappa_s.box();

    auto kps = kappa_s.array();
    auto sta = state.array();

    const Real nu = freq_center[0];

    amrex::ParallelFor(kbox,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        Real rho = sta(i,j,k,URHO);
        Real temp = sta(i,j,k,UTEMP);
        Real Ye;
        if (NumAux > 0) {
            Ye = sta(i,j,k,UFX);
        }
        else {
            Ye = 0.e0_rt;
        }

        Real kp, kr;
        bool need_planck_opac = true;
        bool need_ross_opac = true;
        opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

        kps(i,j,k) = amrex::max(kr - kp, 0.e0_rt);
    });
    Gpu::synchronize();
}

void Radiation::bisect_temperature(MultiFab& rhoe_new, MultiFab& temp_next,
                              const MultiFab& gas_energy_star, const MultiFab& temp_iterate,
                              const MultiFab& S_new, const BoxArray& grids, int level)
{
  temp_next.plus(temp_iterate, 0, 1, 0);
  temp_next.mult(0.5, 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(rhoe_new,true); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      auto rhoe = rhoe_new[mfi].array();
      auto temp = temp_next[mfi].array();
      auto state = S_new[mfi].array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          Real rhoInv = 1.e0_rt / state(i,j,k,URHO);

          eos_re_t eos_state;
          eos_state.rho = state(i,j,k,URHO);
          eos_state.T   =  temp(i,j,k);
          for (int n = 0; n < NumSpec; ++n) {
              eos_state.xn[n] = state(i,j,k,UFS+n) * rhoInv;
          }
#if NAUX_NET > 0
          for (int n = 0; n < NumAux; ++n) {
              eos_state.aux[n] = state(i,j,k,UFX+n) * rhoInv;
          }
#endif

          eos(eos_input_rt, eos_state);

          rhoe(i,j,k) = eos_state.rho * eos_state.e;
      });
  }
}


void Radiation::rhs_to_rad_energy(MultiFab& rhs, Real dt, int level)
{
    auto geomdata = parent->Geom(level).data();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter ri(rhs, TilingIfNotGPU()); ri.isValid(); ++ri)
    {
        const Box& bx = ri.tilebox();

        auto rhs_arr = rhs[ri].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Real r, s;
            cell_center_geometry_factors(i, j, k, geomdata, r, s);

            rhs_arr(i,j,k) *= dt / r;
        });
    }
}
