#include <AMReX_LO_BCTYPES.H>
#include <AMReX_ParmParse.H>
#include <Radiation.H>
#include <RadTransportSolver.H>
#include <radiation_helpers.H>
#include <primitive_filter_cell.H>

#include <opacity.H>

#include <format>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <filter.H>

#include <planck_integrals.H>

#include <user_emissivity.H>

using namespace amrex;

void Radiation::compute_rosseland_opacity(MultiFab& kappa_r, const MultiFab& state)
{
  BL_PROFILE("Radiation::compute_rosseland_opacity (MultiFab)");

  const int rad_group_idx = 0;
  const Real nu = freq_center[rad_group_idx];

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(kappa_r, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& kbox = mfi.growntilebox();

      auto state_arr = state[mfi].array();
      auto kpr = kappa_r[mfi].array();

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

          opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

          kpr(i,j,k,rad_group_idx) = kr;
      });
  }
}

void Radiation::compute_rosseland_opacity(FArrayBox& kappa_r, const FArrayBox& state)
{
  BL_PROFILE("Radiation::compute_rosseland_opacity (FArrayBox)");

  const Box& kbox = kappa_r.box();

  const int rad_group_idx = 0;
  const Real nu = freq_center[rad_group_idx];

  auto state_arr = state.array();
  auto kpr = kappa_r.array();

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

      opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

      kpr(i,j,k,rad_group_idx) = kr;
  });
  Gpu::synchronize();
}

void Radiation::derive_rosseland_opacity(MultiFab& kappa_r,
                              AmrLevel* hydrosim,
                              int rad_group_idx)
{
  BL_PROFILE("Radiation::derive_rosseland_opacity");

  BL_ASSERT(kappa_r.nGrow() == 1);

  if(Radiation::nGroups == 0) {
      kappa_r.setVal(0.);
      return;
  }

  Real time = hydrosim->get_state_data(State_Type).curTime();
  MultiFab& S_new = hydrosim->get_new_data(State_Type);
  int nstate = S_new.nComp();

  FillPatchIterator fpi(*hydrosim, S_new, 1, time, State_Type, 0, nstate);
  MultiFab& state = fpi.get_mf();

  const Real nu = freq_center[rad_group_idx];

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
      FArrayBox fluid_rho_eint;
      for(MFIter mfi(kappa_r, TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
          const Box& bx = mfi.growntilebox();
          fluid_rho_eint.resize(bx, 1);
          Elixir density_energy_hold = fluid_rho_eint.elixir();

          auto density_energy_arr = fluid_rho_eint.array();
          auto state_arr = state[mfi].array();
          auto kpr = kappa_r[mfi].array();

          amrex::ParallelFor(bx,
          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
          {

              if (state_arr(i,j,k,UEINT) <= 0.e0_rt)
              {
                  density_energy_arr(i,j,k) = small_temp;
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

                  density_energy_arr(i,j,k) = eos_state.T;

                  state_arr(i,j,k,UTEMP) = density_energy_arr(i,j,k);
              }

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

              opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

              kpr(i,j,k,rad_group_idx) = kr;
          });
      }
  }
}

void Radiation::refresh_rosseland_from_temp(MultiFab& kappa_r,
                                           MultiFab& temp,
                                           MultiFab& state,
                                           const Geometry& geom,
                                           int rad_group_idx)
{
  BL_PROFILE("refresh_rosseland_from_temp");

  BL_ASSERT(kappa_r.nGrow() == 1);
  BL_ASSERT(temp.nGrow()    == 0);
  BL_ASSERT(kappa_r.nComp() == Radiation::nGroups);

  const Real nu = freq_center[rad_group_idx];

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      auto state_arr = state[mfi].array();
      auto temp_field_arr = temp[mfi].array();
      auto kpr = kappa_r[mfi].array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          state_arr(i,j,k,UTEMP) = temp_field_arr(i,j,k);

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

          opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

          kpr(i,j,k,rad_group_idx) = kr;
      });
  }

  kappa_r.FillBoundary(geom.periodicity());
}

void Radiation::derive_planck_and_temp(MultiFab& fkp,
                                    MultiFab& temp,
                                    MultiFab& state,
                                    int rad_group_idx, Real delta_t)
{
    BL_PROFILE("Radiation::derive_planck_and_temp");

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        auto scaled_planck_arr = fkp[mfi].array(rad_group_idx);
        auto temp_field_arr = temp[mfi].array();
        auto state_arr = state[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            if (temp_field_arr(i,j,k) <= 0.e0_rt)
            {
                temp_field_arr(i,j,k) = small_temp;
            }
            else
            {
                Real rhoInv = 1.e0_rt / state_arr(i,j,k,URHO);

                eos_re_t eos_state;
                eos_state.rho = state_arr(i,j,k,URHO);
                eos_state.T   = state_arr(i,j,k,UTEMP);
                eos_state.e   = temp_field_arr(i,j,k) * rhoInv;
                for (int n = 0; n < NumSpec; ++n) {
                    eos_state.xn[n] = state_arr(i,j,k,UFS+n) * rhoInv;
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; ++n) {
                    eos_state.aux[n] = state_arr(i,j,k,UFX+n) * rhoInv;
                }
#endif

                eos(eos_input_re, eos_state);

                temp_field_arr(i,j,k) = eos_state.T;
            }
        });

        const Real nu = freq_center[rad_group_idx];

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Real rho = state_arr(i,j,k,URHO);
            Real temp = state_arr(i,j,k,UTEMP);
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

            scaled_planck_arr(i,j,k) = kp;
        });

        int ncomp = temp[mfi].nComp();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            const Real min_temperature = 1.e-10_rt;

            for (int n = 0; n < ncomp; ++n) {
                if (temp_field_arr(i,j,k,n) < min_temperature) {
                    temp_field_arr(i,j,k,n) = min_temperature;
                }
            }
        });
    }
}

void Radiation::computeFluxLimiter(int level,
                            FaceMultiFabs& lambda,
                            int limiter_comp)
{
    BL_PROFILE("Radiation:computeFluxLimiter");

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        for (MFIter mfi(lambda[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();

            auto limiter_arr = lambda[idim][mfi].array(limiter_comp);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                limiter_arr(i,j,k) = LimitedDiffusionLambda(limiter_arr(i,j,k));
            });
        }
    }
}

static AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real transverse_difference (Real em, Real e0, Real ep)
{
    Real d = ep - em;

    if (em == -1.e0_rt) {
        d = 2.e0_rt * (ep - e0);
    }
    else if (ep == -1.e0_rt) {
        d = 2.e0_rt * (e0 - em);
    }

    return d;
}

void Radiation::evaluateGradientRatio(int level,
                               Array<MultiFab, AMREX_SPACEDIM>& R,
                               MultiFab& kappa_r, int kcomp,
                               MultiFab& Er, int rad_group_idx,
                               int nGrow_rad_energy, int GradComp)
{
  BL_PROFILE("Radiation::evaluateGradientRatio");
  BL_ASSERT(kappa_r.nGrow() == 1);

  MultiFab er_periodic_buffer;
  if (nGrow_rad_energy == 0) {
    if (radiation::limiter > 0) {
      const BoxArray& grids = parent->boxArray(level);
      const DistributionMapping& dmap = parent->DistributionMap(level);
      er_periodic_buffer.define(grids,dmap,1,1);
      er_periodic_buffer.setVal(-1.0);
      MultiFab::Copy(er_periodic_buffer, Er, rad_group_idx, 0, 1, 0);

      er_periodic_buffer.FillBoundary(parent->Geom(level).periodicity());
    }
  }

  MultiFab& RadEnergyGrown = (nGrow_rad_energy==0) ? er_periodic_buffer : Er;

  auto dx = parent->Geom(level).CellSizeArray();

  GpuArray<Real, 3> dxInv = {0.0, 0.0, 0.0};
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      dxInv[d] = 1.e0_rt / dx[d];
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

      for (MFIter mfi(R[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

          const Box& nbx = mfi.tilebox();

          auto scaled_grad_arr = R[idim][mfi].array(GradComp);

          if (radiation::limiter == 0) {

              R[idim][mfi].setVal<RunOn::Device>(0.0, GradComp);

          }
          else {

              int cross_term_flag = 0;

              if (radiation::limiter == 1) {
                  cross_term_flag = 0;
              } else if (radiation::limiter == 2) {
                  cross_term_flag = 1;
              } else {
                  amrex::Abort("Unknown limiter");
              }

              auto opacity_arr = kappa_r[mfi].array(kcomp);
              auto rad_eden_arr = RadEnergyGrown[mfi].array();

              amrex::ParallelFor(nbx,
              [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
              {
                  Real dal, dar, dbl, dbr;

                  const Real tiny = 1.e-50_rt;

                  if (idim == 0)
                  {
                      if (cross_term_flag == 1)
                      {
#if (AMREX_SPACEDIM >= 2)
                          dal = transverse_difference(rad_eden_arr(i-1,j-1,k), rad_eden_arr(i-1,j  ,k), rad_eden_arr(i-1,j+1,k));
                          dar = transverse_difference(rad_eden_arr(i  ,j-1,k), rad_eden_arr(i  ,j  ,k), rad_eden_arr(i  ,j+1,k));
#else
                          dal = 0.e0_rt;
                          dar = 0.e0_rt;
#endif

#if (AMREX_SPACEDIM == 3)
                          dbl = transverse_difference(rad_eden_arr(i-1,j,k-1), rad_eden_arr(i-1,j,k  ), rad_eden_arr(i-1,j,k+1));
                          dbr = transverse_difference(rad_eden_arr(i  ,j,k-1), rad_eden_arr(i  ,j,k  ), rad_eden_arr(i  ,j,k+1));
#else
                          dbl = 0.e0_rt;
                          dbr = 0.e0_rt;
#endif

                      }
                      else
                      {
                          dal = 0.e0_rt;
                          dar = 0.e0_rt;
                          dbl = 0.e0_rt;
                          dbr = 0.e0_rt;
                      }

                      Real rg;

                      if (rad_eden_arr(i-1,j,k) == -1.e0_rt)
                      {
                          rg = std::pow((rad_eden_arr(i+1,j,k) - rad_eden_arr(i,j,k)) * dxInv[0], 2) +
                               std::pow(0.5_rt * dar * dxInv[1], 2) +
                               std::pow(0.5_rt * dbr * dxInv[2], 2);
                      }
                      else if (rad_eden_arr(i,j,k) == -1.e0_rt)
                      {
                          rg = std::pow((rad_eden_arr(i-1,j,k) - rad_eden_arr(i-2,j,k)) * dxInv[0], 2) +
                               std::pow(0.5_rt * dal * dxInv[1], 2) +
                               std::pow(0.5_rt * dbl * dxInv[2], 2);
                      }
                      else
                      {
                          rg = std::pow((rad_eden_arr(i,j,k) - rad_eden_arr(i-1,j,k)) * dxInv[0], 2) +
                               std::pow((1.0_rt / 4.0_rt) * (dal + dar) * dxInv[1], 2) +
                               std::pow((1.0_rt / 4.0_rt) * (dbl + dbr) * dxInv[2], 2);
                      }

                      Real kap = face_opacity_average(opacity_arr(i-1,j,k), opacity_arr(i,j,k), dx[0], -1);
                      scaled_grad_arr(i,j,k) = std::sqrt(rg) / (kap * amrex::max(rad_eden_arr(i-1,j,k), rad_eden_arr(i,j,k), tiny));

                  }
                  else if (idim == 1)
                  {
                      if (cross_term_flag == 1)
                      {
                          dal = transverse_difference(rad_eden_arr(i-1,j-1,k  ), rad_eden_arr(i  ,j-1,k  ), rad_eden_arr(i+1,j-1,k  ));
                          dar = transverse_difference(rad_eden_arr(i-1,j  ,k  ), rad_eden_arr(i  ,j  ,k  ), rad_eden_arr(i+1,j  ,k  ));

#if (AMREX_SPACEDIM == 3)
                          dbl = transverse_difference(rad_eden_arr(i  ,j-1,k-1), rad_eden_arr(i  ,j-1,k  ), rad_eden_arr(i  ,j-1,k+1));
                          dbr = transverse_difference(rad_eden_arr(i  ,j  ,k-1), rad_eden_arr(i  ,j  ,k  ), rad_eden_arr(i  ,j  ,k+1));
#else
                          dbl = 0.e0_rt;
                          dbr = 0.e0_rt;
#endif
                      }
                      else
                      {
                          dal = 0.e0_rt;
                          dar = 0.e0_rt;
                          dbl = 0.e0_rt;
                          dbr = 0.e0_rt;
                      }

                      Real rg;

                      if (rad_eden_arr(i,j-1,k) == -1.e0_rt)
                      {
                          rg = std::pow((rad_eden_arr(i,j+1,k) - rad_eden_arr(i,j,k)) * dxInv[1], 2) +
                               std::pow(0.5_rt * dar * dxInv[0], 2) +
                               std::pow(0.5_rt * dbr * dxInv[2], 2);
                      }
                      else if (rad_eden_arr(i,j,k) == -1.e0_rt)
                      {
                          rg = std::pow((rad_eden_arr(i,j-1,k) - rad_eden_arr(i,j-2,k)) * dxInv[1], 2) +
                               std::pow(0.5_rt * dal * dxInv[0], 2) +
                               std::pow(0.5_rt * dbl * dxInv[2], 2);
                      }
                      else
                      {
                          rg = std::pow((rad_eden_arr(i,j,k) - rad_eden_arr(i,j-1,k)) * dxInv[1], 2) +
                               std::pow((1.0_rt / 4.0_rt) * (dal + dar) * dxInv[0], 2) +
                               std::pow((1.0_rt / 4.0_rt) * (dbl + dbr) * dxInv[2], 2);
                      }

                      Real kap = face_opacity_average(opacity_arr(i,j-1,k), opacity_arr(i,j,k), dx[1], -1);
                      scaled_grad_arr(i,j,k) = std::sqrt(rg) / (kap * amrex::max(rad_eden_arr(i,j-1,k), rad_eden_arr(i,j,k), tiny));
                  }
                  else
                  {
                      if (cross_term_flag == 1)
                      {
                          dal = transverse_difference(rad_eden_arr(i-1,j  ,k-1), rad_eden_arr(i  ,j  ,k-1), rad_eden_arr(i+1,j  ,k-1));
                          dar = transverse_difference(rad_eden_arr(i-1,j  ,k  ), rad_eden_arr(i  ,j  ,k  ), rad_eden_arr(i+1,j  ,k  ));

                          dbl = transverse_difference(rad_eden_arr(i  ,j-1,k-1), rad_eden_arr(i  ,j  ,k-1), rad_eden_arr(i  ,j+1,k-1));
                          dbr = transverse_difference(rad_eden_arr(i  ,j-1,k  ), rad_eden_arr(i  ,j  ,k  ), rad_eden_arr(i  ,j+1,k  ));
                      }
                      else
                      {
                          dal = 0.e0_rt;
                          dar = 0.e0_rt;
                          dbl = 0.e0_rt;
                          dbr = 0.e0_rt;
                      }

                      Real rg;

                      if (rad_eden_arr(i,j,k-1) == -1.e0_rt)
                      {
                          rg = std::pow((rad_eden_arr(i,j,k+1) - rad_eden_arr(i,j,k)) * dxInv[2], 2) +
                               std::pow(0.5_rt * dar * dxInv[0], 2) +
                               std::pow(0.5_rt * dbr * dxInv[1], 2);
                      }
                      else if (rad_eden_arr(i,j,k) == -1.e0_rt)
                      {
                          rg = std::pow((rad_eden_arr(i,j,k-1) - rad_eden_arr(i,j,k-2)) * dxInv[2], 2) +
                               std::pow(0.5_rt * dal * dxInv[0], 2) +
                               std::pow(0.5_rt * dbl * dxInv[1], 2);
                      }
                      else
                      {
                          rg = std::pow((rad_eden_arr(i,j,k) - rad_eden_arr(i,j,k-1)) * dxInv[2], 2) +
                               std::pow((1.0_rt / 4.0_rt) * (dal + dar) * dxInv[0], 2) +
                               std::pow((1.0_rt / 4.0_rt) * (dbl + dbr) * dxInv[1], 2);
                      }

                      Real kap = face_opacity_average(opacity_arr(i,j,k-1), opacity_arr(i,j,k), dx[2], -1);
                      scaled_grad_arr(i,j,k) = std::sqrt(rg) / (kap * amrex::max(rad_eden_arr(i,j,k-1), rad_eden_arr(i,j,k), tiny));
                  }
              });
          }
      }
  }
}

void Radiation::refresh_diffusion_coef(MultiFab& dcf, MultiFab& coupling_inverse, MultiFab& kp, MultiFab& kr,
                           const Geometry& geom)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(dcf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto diff_factor_arr = dcf[mfi].array();
        auto coupling_inverse_arr = coupling_inverse[mfi].array();
        auto planck_opacity_arr = kp[mfi].array();
        auto rosseland_opacity_arr = kr[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            diff_factor_arr(i,j,k) = 2.e0_rt * coupling_inverse_arr(i,j,k) * (planck_opacity_arr(i,j,k) / rosseland_opacity_arr(i,j,k));
        });
    }

    dcf.FillBoundary(geom.periodicity());
}

void Radiation::derive_rosseland_velocity_coef(MultiFab& kappa_r, MultiFab& v, MultiFab& dcf,
                                    Real delta_t, Real c,
                                    AmrLevel* hydrosim, int rad_group_idx)
{
    BL_ASSERT(kappa_r.nGrow() == 1);
    BL_ASSERT(      v.nGrow() == 1);
    BL_ASSERT(    dcf.nGrow() == 1);

    int nstate = hydrosim->get_new_data(State_Type).nComp();
    Real time = hydrosim->get_state_data(State_Type).curTime();

    FillPatchIterator rad_fill_iter(*hydrosim, kappa_r, 1, time, Radiation_Type, 0, 1);
    MultiFab& Er = rad_fill_iter.get_mf();

    FillPatchIterator state_fill_iter(*hydrosim, kappa_r, 1, time, State_Type, 0, nstate);
    MultiFab& S = state_fill_iter.get_mf();

    Real local_stefan_const = sigma;
    Real c_loc = c;
    Real local_temp_step = dT;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        FArrayBox temp, c_v, kp, kp2;
        for (MFIter mfi(kappa_r,true); mfi.isValid(); ++mfi)
        {
            const Box& reg = mfi.growntilebox();

            temp.resize(reg);
            Elixir temperature_hold = temp.elixir();

            kp.resize(reg);
            Elixir planck_opacity_hold = kp.elixir();

            kp2.resize(reg);
            Elixir planck_opacity_shift_hold = kp2.elixir();

            c_v.resize(reg);
            Elixir heat_capacity_hold = c_v.elixir();

            auto temp_field_arr = temp.array();
            auto cons_state_arr = S[mfi].array();
            auto gas_vel_arr = v[mfi].array();
            auto rad_energy_arr = Er[mfi].array();
            auto rosseland_opacity_arr = kappa_r[mfi].array();
            auto planck_opacity_arr = kp.array();
            auto planck_opacity_shift_arr = kp2.array();
            auto heat_capacity_arr = c_v.array();
            auto diff_factor_arr = dcf[mfi].array();

            amrex::ParallelFor(reg,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {

                if (cons_state_arr(i,j,k,UEINT) <= 0.e0_rt)
                {
                    temp_field_arr(i,j,k) = small_temp;
                }
                else
                {
                    Real rhoInv = 1.e0_rt / cons_state_arr(i,j,k,URHO);

                    eos_re_t eos_state;
                    eos_state.rho = cons_state_arr(i,j,k,URHO);
                    eos_state.T   = cons_state_arr(i,j,k,UTEMP);
                    eos_state.e   = cons_state_arr(i,j,k,UEINT) * rhoInv;
                    for (int n = 0; n < NumSpec; ++n) {
                        eos_state.xn[n] = cons_state_arr(i,j,k,UFS+n) * rhoInv;
                    }
#if NAUX_NET > 0
                    for (int n = 0; n < NumAux; ++n) {
                        eos_state.aux[n] = cons_state_arr(i,j,k,UFX+n) * rhoInv;
                    }
#endif

                    eos(eos_input_re, eos_state);

                    temp_field_arr(i,j,k) = eos_state.T;
                }
            });
            Gpu::synchronize();

            get_heat_capacity(c_v, temp, S[mfi], reg);

            S[mfi].copy<RunOn::Device>(temp,reg,0,reg,UTEMP,1);

            const Real nu = freq_center[rad_group_idx];

            amrex::ParallelFor(reg,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                Real rho = cons_state_arr(i,j,k,URHO);
                Real temp = cons_state_arr(i,j,k,UTEMP);
                Real Ye;
                if (NumAux > 0)  {
                    Ye = cons_state_arr(i,j,k,UFX);
                } else {
                    Ye = 0.e0_rt;
                }

                Real kp, kr;
                bool need_planck_opac = true;
                bool need_ross_opac = true;
                opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

                planck_opacity_arr(i,j,k) = kp;
                rosseland_opacity_arr(i,j,k) = kr;

                temp += local_temp_step;

                opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

                planck_opacity_shift_arr(i,j,k) = kp;
            });

            amrex::ParallelFor(reg,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                const Real fac0 = 4.e0_rt * local_stefan_const * delta_t / local_temp_step;
                const Real fac2 = c_loc * delta_t / local_temp_step;

                gas_vel_arr(i,j,k,0) = cons_state_arr(i,j,k,UMX) / cons_state_arr(i,j,k,URHO);
#if AMREX_SPACEDIM >= 2
                gas_vel_arr(i,j,k,1) = cons_state_arr(i,j,k,UMY) / cons_state_arr(i,j,k,URHO);
#endif
#if AMREX_SPACEDIM == 3
                gas_vel_arr(i,j,k,2) = cons_state_arr(i,j,k,UMZ) / cons_state_arr(i,j,k,URHO);
#endif

                Real alpha = fac0 * (planck_opacity_shift_arr(i,j,k) * std::pow(temp_field_arr(i,j,k) + local_temp_step, 4) -
                                     planck_opacity_arr(i,j,k) * std::pow(temp_field_arr(i,j,k), 4)) -
                             fac2 * (planck_opacity_shift_arr(i,j,k) - planck_opacity_arr(i,j,k)) * rad_energy_arr(i,j,k);

                Real frc = cons_state_arr(i,j,k,URHO) * heat_capacity_arr(i,j,k) + 1.0e-50_rt;
                Real coupling_inverse = frc / (alpha + frc);

                diff_factor_arr(i,j,k) = 2.e0_rt * coupling_inverse * (planck_opacity_arr(i,j,k) / rosseland_opacity_arr(i,j,k));
            });
        }
    }
}

void Radiation::filter_primitives(int level, MultiFab& State)
{
  HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
  const Geometry& geom = parent->Geom(level);
  auto geomdata = geom.data();

  int ngrow = filter_prim_temperature;
  int ncomp = State.nComp();
  Real time = hydrosim->get_state_data(Radiation_Type).curTime();

  FillPatchIterator fpi(*hydrosim,State,ngrow,time,State_Type,0,ncomp);
  MultiFab& S_fp = fpi.get_mf();

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(State,true); mfi.isValid(); ++mfi)
  {
      const Box& bx = mfi.tilebox();

      auto filtered_state_arr = S_fp[mfi].array();
      auto source_state_arr = State[mfi].array();

      int T = filter_prim_temperature;
      int S = filter_primitive_S;

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          primitive_filter_cell(i, j, k,
                    filtered_state_arr, source_state_arr,
                    T, S,
                    geomdata, time);
      });
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
