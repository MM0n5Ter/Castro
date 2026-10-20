#include <AMReX_REAL.H>

#include <Derive.H>
#include <HydroSim.H>

#ifdef DIFFUSION
#include <conduction_coeffs.H>
#endif

using namespace amrex;

#ifdef __cplusplus
extern "C"
{
#endif

    void derive_pressure(const Box& bx, FArrayBox& derfab, int  , int  ,
                    const FArrayBox& datfab, const Geometry&  ,
                    Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real rhoInv = 1.0_rt / dat(i,j,k,URHO);

        eos_rep_t eos_state;
        eos_state.rho  = dat(i,j,k,URHO);
        eos_state.T = dat(i,j,k,UTEMP);
        eos_state.e = dat(i,j,k,UEINT) * rhoInv;
        for (int n = 0; n < NumSpec; n++) {
          eos_state.xn[n] = dat(i,j,k,UFS+n) * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; n++) {
          eos_state.aux[n] = dat(i,j,k,UFX+n) * rhoInv;
        }
#endif

        eos(eos_input_re, eos_state);

        der(i,j,k,0) = eos_state.p;
      });
    }

    void derive_internal_from_total(const Box& bx, FArrayBox& derfab, int  , int  ,
                     const FArrayBox& datfab, const Geometry&  ,
                     Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real rhoInv = 1.0_rt/dat(i,j,k,URHO);
        Real ux = dat(i,j,k,UMX)*rhoInv;
        Real uy = dat(i,j,k,UMY)*rhoInv;
        Real uz = dat(i,j,k,UMZ)*rhoInv;

        der(i,j,k,0) = dat(i,j,k,UEDEN)*rhoInv -
          0.5_rt * (ux*ux + uy*uy + uz*uz);
      });
    }

    void derive_internal_from_stored(const Box& bx, FArrayBox& derfab, int  , int  ,
                     const FArrayBox& datfab, const Geometry&  ,
                     Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        der(i,j,k,0) = dat(i,j,k,UEINT) / dat(i,j,k,URHO);
      });
    }

    void derive_log_density(const Box& bx, FArrayBox& derfab, int  , int  ,
                      const FArrayBox& datfab, const Geometry&  ,
                      Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        der(i,j,k,0) = std::log10(dat(i,j,k,0));
      });
    }

    void derive_u_plus_c(const Box& bx, FArrayBox& derfab, int  , int  ,
                      const FArrayBox& datfab, const Geometry&  ,
                      Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real rhoInv = 1.0_rt / dat(i,j,k,URHO);

        eos_rep_t eos_state;
        eos_state.rho  = dat(i,j,k,URHO);
        eos_state.T = dat(i,j,k,UTEMP);
        eos_state.e = dat(i,j,k,UEINT) * rhoInv;
        for (int n = 0; n < NumSpec; n++) {
          eos_state.xn[n] = dat(i,j,k,UFS+n) * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; n++) {
          eos_state.aux[n] = dat(i,j,k,UFX+n) * rhoInv;
        }
#endif

        eos(eos_input_re, eos_state);

        der(i,j,k,0) = dat(i,j,k,UMX) / dat(i,j,k,URHO) + eos_state.cs;

      });
    }

    void derive_u_minus_c(const Box& bx, FArrayBox& derfab, int  , int  ,
                       const FArrayBox& datfab, const Geometry&  ,
                       Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real rhoInv = 1.0_rt / dat(i,j,k,URHO);

        eos_rep_t eos_state;
        eos_state.rho  = dat(i,j,k,URHO);
        eos_state.T = dat(i,j,k,UTEMP);
        eos_state.e = dat(i,j,k,UEINT) * rhoInv;
        for (int n = 0; n < NumSpec; n++) {
          eos_state.xn[n] = dat(i,j,k,UFS+n) * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; n++) {
          eos_state.aux[n] = dat(i,j,k,UFX+n) * rhoInv;
        }
#endif

        eos(eos_input_re, eos_state);

        der(i,j,k,0) = dat(i,j,k,UMX) / dat(i,j,k,URHO) - eos_state.cs;

      });
    }

    void derive_sound_speed(const Box& bx, FArrayBox& derfab, int  , int  ,
                          const FArrayBox& datfab, const Geometry&  ,
                          Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real rhoInv = 1.0_rt / dat(i,j,k,URHO);

        eos_rep_t eos_state;
        eos_state.rho  = dat(i,j,k,URHO);
        eos_state.T = dat(i,j,k,UTEMP);
        eos_state.e = dat(i,j,k,UEINT) * rhoInv;
        for (int n = 0; n < NumSpec; n++) {
          eos_state.xn[n] = dat(i,j,k,UFS+n) * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; n++) {
          eos_state.aux[n] = dat(i,j,k,UFX+n) * rhoInv;
        }
#endif

        eos(eos_input_re, eos_state);

        der(i,j,k,0) = eos_state.cs;

      });
    }

    void derive_adiabatic_index(const Box& bx, FArrayBox& derfab, int  , int  ,
                      const FArrayBox& datfab, const Geometry&  ,
                      Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real rhoInv = 1.0_rt / dat(i,j,k,URHO);

        eos_rep_t eos_state;
        eos_state.rho  = dat(i,j,k,URHO);
        eos_state.T = dat(i,j,k,UTEMP);
        eos_state.e = dat(i,j,k,UEINT) * rhoInv;
        for (int n = 0; n < NumSpec; n++) {
          eos_state.xn[n] = dat(i,j,k,UFS+n) * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; n++) {
          eos_state.aux[n] = dat(i,j,k,UFX+n) * rhoInv;
        }
#endif

        eos(eos_input_re, eos_state);

        der(i,j,k,0) = eos_state.gam1;

      });
    }

    void derive_mach_ratio(const Box& bx, FArrayBox& derfab, int  , int  ,
                          const FArrayBox& datfab, const Geometry&  ,
                          Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real rhoInv = 1.0_rt / dat(i,j,k,URHO);

        eos_rep_t eos_state;
        eos_state.rho  = dat(i,j,k,URHO);
        eos_state.T = dat(i,j,k,UTEMP);
        eos_state.e = dat(i,j,k,UEINT) * rhoInv;
        for (int n = 0; n < NumSpec; n++) {
          eos_state.xn[n] = dat(i,j,k,UFS+n) * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; n++) {
          eos_state.aux[n] = dat(i,j,k,UFX+n) * rhoInv;
        }
#endif

        eos(eos_input_re, eos_state);

        der(i,j,k,0) = std::sqrt(dat(i,j,k,UMX)*dat(i,j,k,UMX) +
                                 dat(i,j,k,UMY)*dat(i,j,k,UMY) +
                                 dat(i,j,k,UMZ)*dat(i,j,k,UMZ)) /
          dat(i,j,k,URHO) / eos_state.cs;

      });
    }

    void derive_entropy(const Box& bx, FArrayBox& derfab, int  , int  ,
                       const FArrayBox& datfab, const Geometry&  ,
                       Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real rhoInv = 1.0_rt / dat(i,j,k,URHO);

        eos_t eos_state;
        eos_state.rho  = dat(i,j,k,URHO);
        eos_state.T = dat(i,j,k,UTEMP);
        eos_state.e = dat(i,j,k,UEINT) * rhoInv;
        for (int n = 0; n < NumSpec; n++) {
          eos_state.xn[n] = dat(i,j,k,UFS+n) * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; n++) {
          eos_state.aux[n] = dat(i,j,k,UFX+n) * rhoInv;
        }
#endif

        eos(eos_input_re, eos_state);

        der(i,j,k,0) = eos_state.s;
      });
    }

#ifdef DIFFUSION
    void derive_conductivity(const Box& bx, FArrayBox& derfab, int  , int  ,
                    const FArrayBox& datfab, const Geometry&  ,
                    Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      fill_cell_conductivity(bx, dat, der);

    }

    void derive_diffusivity(const Box& bx, FArrayBox& derfab, int  , int  ,
                    const FArrayBox& datfab, const Geometry&  ,
                    Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      fill_conduction_coeff(bx, dat, der);

    }

    void derive_diffusion_term(const Box& bx, FArrayBox& derfab, int  , int  ,
                        const FArrayBox& datfab, const Geometry& geom,
                        Real  , const int*  , int  )
    {

      const Box& obx = amrex::grow(bx, 1);

      FArrayBox cell_coeff;
      cell_coeff.resize(obx, 1, The_Async_Arena());
      Array4<Real> const cell_conductivity_arr = cell_coeff.array();

      auto const dat = datfab.array();
      auto const der = derfab.array();

      fill_cell_conductivity(obx, dat, cell_conductivity_arr);

      auto dx = geom.CellSizeArray();
      auto problo = geom.ProbLoArray();
      const int coord_type = geom.Coord();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real cond_flux_xhi = 0.5_rt * (cell_conductivity_arr(i+1,j,k) + cell_conductivity_arr(i,j,k)) *
          (dat(i+1,j,k,UTEMP) - dat(i,j,k,UTEMP)) / dx[0];

        Real cond_flux_xlo = 0.5_rt * (cell_conductivity_arr(i,j,k) + cell_conductivity_arr(i-1,j,k)) *
          (dat(i,j,k,UTEMP) - dat(i-1,j,k,UTEMP)) / dx[0];

#if AMREX_SPACEDIM >= 2
        Real cond_flux_yhi = 0.5_rt * (cell_conductivity_arr(i,j+1,k) + cell_conductivity_arr(i,j,k)) *
          (dat(i,j+1,k,UTEMP) - dat(i,j,k,UTEMP)) / dx[1];

        Real cond_flux_ylo = 0.5_rt * (cell_conductivity_arr(i,j,k) + cell_conductivity_arr(i,j-1,k)) *
          (dat(i,j,k,UTEMP) - dat(i,j-1,k,UTEMP)) / dx[1];
#endif

#if AMREX_SPACEDIM == 3
        Real cond_flux_zhi = 0.5_rt * (cell_conductivity_arr(i,j,k+1) + cell_conductivity_arr(i,j,k)) *
          (dat(i,j,k+1,UTEMP) - dat(i,j,k,UTEMP)) / dx[2];

        Real cond_flux_zlo = 0.5_rt * (cell_conductivity_arr(i,j,k) + cell_conductivity_arr(i,j,k-1)) *
          (dat(i,j,k,UTEMP) - dat(i,j,k-1,UTEMP)) / dx[2];
#endif

        if (coord_type == 0) {
          der(i,j,k,0) = (cond_flux_xhi - cond_flux_xlo)/dx[0];
#if AMREX_SPACEDIM >= 2
          der(i,j,k,0) += (cond_flux_yhi - cond_flux_ylo)/dx[1];
#endif
#if AMREX_SPACEDIM == 3
          der(i,j,k,0) += (cond_flux_zhi - cond_flux_zlo)/dx[2];
#endif

        } else if (coord_type == 1) {
          Real rm = static_cast<Real>(i)*dx[0] + problo[0];
          Real r = (static_cast<Real>(i) + 0.5_rt)*dx[0] + problo[0];
          Real rp = (static_cast<Real>(i) + 1.0_rt)*dx[0] + problo[0];

          der(i,j,k,0) = (rp*cond_flux_xhi - rm*cond_flux_xlo)/(r*dx[0]);
#if AMREX_SPACEDIM == 2
          der(i,j,k,0) += (cond_flux_yhi - cond_flux_ylo)/dx[1];
#endif

        } else if (coord_type == 2) {
          Real rm = static_cast<Real>(i)*dx[0] + problo[0];
          Real r = (static_cast<Real>(i) + 0.5_rt)*dx[0] + problo[0];
          Real rp = (static_cast<Real>(i) + 1.0_rt)*dx[0] + problo[0];

          der(i,j,k,0) = (rp*rp*cond_flux_xhi - rm*rm*cond_flux_xlo)/(r*r*dx[0]);
#if AMREX_SPACEDIM == 2
          Real sin_lo = std::sin(static_cast<Real>(j)*dx[1] + problo[1]);
          Real sinc = std::sin((static_cast<Real>(j) + 0.5_rt)*dx[1] + problo[1]);
          Real sinp = std::sin((static_cast<Real>(j) + 1.0_rt)*dx[1] + problo[1]);

          der(i,j,k,0) += (sinp*cond_flux_yhi - sin_lo*cond_flux_ylo)/(r*r*sinc*dx[1]);
#endif
        }
      });
    }
#endif

#ifdef REACTIONS
    void derive_timescale_ratio(const Box& bx, FArrayBox& derfab, int  , int  ,
                             const FArrayBox& datfab, const Geometry& geom,
                             Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      auto dx = geom.CellSizeArray();

      Real dd = 0.0_rt;
#if AMREX_SPACEDIM == 1
      dd = dx[0];
#elif AMREX_SPACEDIM == 2
      dd = amrex::min(dx[0], dx[1]);
#else
      dd = amrex::min(dx[0], dx[1], dx[2]);
#endif

      int energy_gen_index = datfab.nComp()-1;

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real enuc = std::abs(dat(i,j,k,energy_gen_index)) / dat(i,j,k,URHO);

        if (enuc > 1.e-100_rt) {

          Real rhoInv = 1.0_rt / dat(i,j,k,URHO);

          eos_rep_t eos_state;
          eos_state.rho  = dat(i,j,k,URHO);
          eos_state.T = dat(i,j,k,UTEMP);
          eos_state.e = dat(i,j,k,UEINT) * rhoInv;
          for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n] = dat(i,j,k,UFS+n) * rhoInv;
          }
#if NAUX_NET > 0
          for (int n = 0; n < NumAux; n++) {
            eos_state.aux[n] = dat(i,j,k,UFX+n) * rhoInv;
          }
#endif

          eos(eos_input_re, eos_state);

          Real t_e = eos_state.e / enuc;
          Real t_s = dd / eos_state.cs;

          der(i,j,k,0) = t_s/t_e;

        } else {
          der(i,j,k,0) = 0.0_rt;
        }

      });
    }

    void derive_energy_generation(const Box& bx, FArrayBox& derfab, int  , int  ,
                    const FArrayBox& datfab, const Geometry&  ,
                    Real  , const int*  , int  )
    {
      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
          Real enuc = dat(i,j,k,1) / dat(i,j,k,0);

          der(i,j,k,0) = enuc;
      });
    }
#endif

  void derive_kinetic_energy (const Box& bx, FArrayBox& derfab, int  , int  ,
                     const FArrayBox& datfab, const Geometry&  ,
                     Real  , const int*  , int  )
  {

    auto const dat = datfab.array();
    auto const kineng = derfab.array();

    amrex::ParallelFor(bx,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                       {
                         kineng(i,j,k,0) = 0.5_rt / dat(i,j,k,0) * ( dat(i,j,k,1)*dat(i,j,k,1) +
                                                                     dat(i,j,k,2)*dat(i,j,k,2) +
                                                                     dat(i,j,k,3)*dat(i,j,k,3) );
                       });

  }

  void derive_noop(Real*  , const int*  , const int*  , const int*  ,
                  const Real*  , const int*  , const int*  , const int*  ,
                  const int*  , const int*  ,
                  const int*  , const int*  ,
                  const Real*  , const Real*  ,
                  const Real*  , const Real*  , const int*  ,
                  const int*  , const int*  )
  {

  }

  void derive_mass_fraction(const Box& bx, FArrayBox& derfab, int  , int  ,
                  const FArrayBox& datfab, const Geometry&  ,
                  Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
             der(i,j,k,0) = dat(i,j,k,1) / dat(i,j,k,0);
      });
    }

  void derive_elec_frac(const Box& bx, FArrayBox& derfab, int  , int  ,
                const FArrayBox& datfab, const Geometry&  ,
                Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real sum = 0.0_rt;
        Real xn;
        for (int n = 0; n < NumSpec; n++) {
          xn = dat(i,j,k,1+n) / dat(i,j,k,0);
          sum += xn * zion[n] / aion[n];
        }
        der(i,j,k,0) = sum;
      });
    }

  void derive_atomic_mass(const Box& bx, FArrayBox& derfab, int  , int  ,
                  const FArrayBox& datfab, const Geometry&  ,
                  Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real sum = 0.0_rt;
        Real xn;
        for (int n = 0; n < NumSpec; n++) {
          xn = dat(i,j,k,1+n)/dat(i,j,k,0);
          sum += xn/aion[n];
        }
        der(i,j,k,0) = 1.0_rt / sum;
      });
    }

  void derive_state_error(const Box& bx, FArrayBox& derfab, int  , int  ,
                   const FArrayBox& datfab, const Geometry&  ,
                   Real  , const int*  , int  )
  {

    auto const dat = datfab.array();
    auto const der = derfab.array();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

      der(i,j,k,0) = dat(i,j,k,0);

      der(i,j,k,1) = dat(i,j,k,1);

      der(i,j,k,2) = dat(i,j,k,2) / dat(i,j,k,0);

    });

  }

#ifdef __cplusplus
}
#endif
