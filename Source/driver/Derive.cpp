#include <AMReX_REAL.H>

#include <Derive.H>
#include <HydroSim.H>

#ifdef DIFFUSION
#include <conduction_util.H>
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

    void derive_velocity_component(const Box& bx, FArrayBox& derfab, int  , int  ,
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


    void derive_speed(const Box& bx, FArrayBox& derfab, int  , int  ,
                      const FArrayBox& datfab, const Geometry&  ,
                      Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real inv_density = 1.0_rt/dat(i,j,k,0);

        der(i,j,k,0) = std::sqrt( (dat(i,j,k,1) * dat(i,j,k,1) +
                                   dat(i,j,k,2) * dat(i,j,k,2) +
                                   dat(i,j,k,3) * dat(i,j,k,3)) ) * inv_density;
      });
    }


    void derive_gravity_magnitude(const Box& bx, FArrayBox& derfab, int  , int  ,
                       const FArrayBox& datfab, const Geometry&  ,
                       Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        der(i,j,k,0) = std::sqrt(dat(i,j,k,0)*dat(i,j,k,0) +
                                 dat(i,j,k,1)*dat(i,j,k,1) +
                                 dat(i,j,k,2)*dat(i,j,k,2));

      });
    }

    void derive_radial_velocity(const Box& bx, FArrayBox& derfab, int  , int  ,
                         const FArrayBox& datfab, const Geometry& geom,
                         Real  , const int*  , int  )
    {


      auto const dat = datfab.array();
      auto const der = derfab.array();

      auto dx = geom.CellSizeArray();
      auto problo = geom.ProbLoArray();
      auto geomdata = geom.data();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        GpuArray<Real, 3> loc = {0.0};
        loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];
#if AMREX_SPACEDIM >= 2
        loc[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
#endif
#if AMREX_SPACEDIM == 3
        loc[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
#endif

        if (plane_parallel_domain) {
#if AMREX_SPACEDIM == 2
          der(i,j,k,0) = dat(i,j,k,1)/dat(i,j,k,0);
#elif AMREX_SPACEDIM == 3

          Real r = std::sqrt(loc[0]*loc[0] + loc[1]*loc[1]);
          der(i,j,k,0) = (dat(i,j,k,1)*loc[0] + dat(i,j,k,2)*loc[1]) / (dat(i,j,k,0)*r);
#endif
        } else {
          Real r = distance(geomdata, loc);

          der(i,j,k,0) = (dat(i,j,k,1)*loc[0] +
                          dat(i,j,k,2)*loc[1] +
                          dat(i,j,k,3)*loc[2]) / ( dat(i,j,k,0)*r );
        }

      });
    }


    void derive_circular_speed(const Box& bx, FArrayBox& derfab, int  , int  ,
                       const FArrayBox& datfab, const Geometry& geom,
                       Real  , const int*  , int  )
    {


      auto const dat = datfab.array();
      auto const der = derfab.array();

      auto dx = geom.CellSizeArray();
      auto problo = geom.ProbLoArray();
      auto geomdata = geom.data();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        GpuArray<Real, 3> loc = {0.0};
        loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];
#if AMREX_SPACEDIM >= 2
        loc[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
#endif
#if AMREX_SPACEDIM == 3
        loc[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
#endif

        if (plane_parallel_domain) {
#if AMREX_SPACEDIM == 2
          der(i,j,k,0) = dat(i,j,k,3)/dat(i,j,k,0);
#elif AMREX_SPACEDIM == 3

          Real r = std::sqrt(loc[0]*loc[0] + loc[1]*loc[1]);
          der(i,j,k,0) = (-dat(i,j,k,1)*loc[1] + dat(i,j,k,2)*loc[0]) / (dat(i,j,k,0)*r);
#endif
        } else {
          Real r = distance(geomdata, loc);

          Real speed_sq = (dat(i,j,k,1)*dat(i,j,k,1) +
                        dat(i,j,k,2)*dat(i,j,k,2) +
                        dat(i,j,k,3)*dat(i,j,k,3))/(dat(i,j,k,0)*dat(i,j,k,0));

          Real vr = (dat(i,j,k,1)*loc[0] +
                     dat(i,j,k,2)*loc[1] +
                     dat(i,j,k,3)*loc[2]) / ( dat(i,j,k,0)*r );

          der(i,j,k,0) = std::sqrt(amrex::max(speed_sq - vr*vr, 0.0_rt));
        }

      });
    }


  void derive_momentum_magnitude(const Box& bx, FArrayBox& derfab, int  , int  ,
                    const FArrayBox& datfab, const Geometry&  ,
                    Real  , const int*  , int  )
  {

    auto const dat = datfab.array();
    auto const der = derfab.array();

    amrex::ParallelFor(bx,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                       {

                         der(i,j,k,0) = std::sqrt(dat(i,j,k,0)*dat(i,j,k,0) +
                                                  dat(i,j,k,1)*dat(i,j,k,1) +
                                                  dat(i,j,k,2)*dat(i,j,k,2));

                       });
  }

  void derive_ang_momentum_x (const Box& bx, FArrayBox& derfab, int  , int  ,
                      const FArrayBox& datfab, const Geometry& geom,
                      Real  , const int*  , int  )
  {

    int idir = 0;
    auto dx     = geom.CellSizeArray();
    auto problo = geom.ProbLoArray();

    auto const dat = datfab.array();
    auto const L = derfab.array();

    amrex::ParallelFor(bx,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                       {
                         Real loc[3];

                         loc[0] = problo[0] + (0.5_rt + i) * dx[0];

#if AMREX_SPACEDIM >= 2
                         loc[1] = problo[1] + (0.5_rt + j) * dx[1];
#else
                         loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
                         loc[2] = problo[2] + (0.5_rt + k) * dx[2];
#else
                         loc[2] = 0.0_rt;
#endif

                         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                           loc[dir] -= problem::center[dir];
                         }

                         if (idir == 0) {
                           L(i,j,k,0) = loc[1] * dat(i,j,k,3) - loc[2] * dat(i,j,k,2);
                         }
                         else if (idir == 1) {
                           L(i,j,k,0) = loc[2] * dat(i,j,k,1) - loc[0] * dat(i,j,k,3);
                         }
                         else {
                           L(i,j,k,0) = loc[0] * dat(i,j,k,2) - loc[1] * dat(i,j,k,1);
                         }

                       });

  }

  void derive_ang_momentum_y (const Box& bx, FArrayBox& derfab, int  , int  ,
                      const FArrayBox& datfab, const Geometry& geom,
                      Real  , const int*  , int  )
  {

    int idir = 1;
    auto dx     = geom.CellSizeArray();
    auto problo = geom.ProbLoArray();

    auto const dat = datfab.array();
    auto const L = derfab.array();

    amrex::ParallelFor(bx,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                       {
                         Real loc[3];

                         loc[0] = problo[0] + (0.5_rt + i) * dx[0];

#if AMREX_SPACEDIM >= 2
                         loc[1] = problo[1] + (0.5_rt + j) * dx[1];
#else
                         loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
                         loc[2] = problo[2] + (0.5_rt + k) * dx[2];
#else
                         loc[2] = 0.0_rt;
#endif
                         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                           loc[dir] -= problem::center[dir];
                         }

                         if (idir == 0) {
                           L(i,j,k,0) = loc[1] * dat(i,j,k,3) - loc[2] * dat(i,j,k,2);
                         }
                         else if (idir == 1) {
                           L(i,j,k,0) = loc[2] * dat(i,j,k,1) - loc[0] * dat(i,j,k,3);
                         }
                         else {
                           L(i,j,k,0) = loc[0] * dat(i,j,k,2) - loc[1] * dat(i,j,k,1);
                         }

                       });

  }

  void derive_ang_momentum_z (const Box& bx, FArrayBox& derfab, int  , int  ,
                      const FArrayBox& datfab, const Geometry& geom,
                      Real  , const int*  , int  )
  {

    int idir = 2;
    auto dx     = geom.CellSizeArray();
    auto problo = geom.ProbLoArray();

    auto const dat = datfab.array();
    auto const L = derfab.array();

    amrex::ParallelFor(bx,
                       [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                       {
                         Real loc[3];

                         loc[0] = problo[0] + (0.5_rt + i) * dx[0];

#if AMREX_SPACEDIM >= 2
                         loc[1] = problo[1] + (0.5_rt + j) * dx[1];
#else
                         loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
                         loc[2] = problo[2] + (0.5_rt + k) * dx[2];
#else
                         loc[2] = 0.0_rt;
#endif

                         for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                           loc[dir] -= problem::center[dir];
                         }

                         if (idir == 0) {
                           L(i,j,k,0) = loc[1] * dat(i,j,k,3) - loc[2] * dat(i,j,k,2);
                         }
                         else if (idir == 1) {
                           L(i,j,k,0) = loc[2] * dat(i,j,k,1) - loc[0] * dat(i,j,k,3);
                         }
                         else {
                           L(i,j,k,0) = loc[0] * dat(i,j,k,2) - loc[1] * dat(i,j,k,1);
                         }

                       });

  }

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

  void derive_vorticity(const Box& bx, FArrayBox& derfab, int  , int  ,
                     const FArrayBox& datfab, const Geometry& geom,
                     Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      auto dx = geom.CellSizeArray();

      const int coord_type = geom.Coord();

#if AMREX_SPACEDIM == 2
      auto problo = geom.ProbLoArray();
#endif

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        if (coord_type == 0) {

          Real vx = 0.5_rt * (dat(i+1,j,k,2) / dat(i+1,j,k,0) -
                              dat(i-1,j,k,2) / dat(i-1,j,k,0)) / dx[0];
          Real wx = 0.5_rt * (dat(i+1,j,k,3) / dat(i+1,j,k,0) -
                              dat(i-1,j,k,3) / dat(i-1,j,k,0)) / dx[0];

          Real uy = 0.0;
          Real wy = 0.0;

#if AMREX_SPACEDIM >= 2
          uy = 0.5_rt * (dat(i,j+1,k,1) / dat(i,j+1,k,0) -
                         dat(i,j-1,k,1) / dat(i,j-1,k,0)) / dx[1];
          wy = 0.5_rt * (dat(i,j+1,k,3) / dat(i,j+1,k,0) -
                         dat(i,j-1,k,3) / dat(i,j-1,k,0)) / dx[1];
#endif

          Real uz = 0.0;
          Real vz = 0.0;

#if AMREX_SPACEDIM == 3
          uz = 0.5_rt * (dat(i,j,k+1,1) / dat(i,j,k+1,0) -
                         dat(i,j,k-1,1) / dat(i,j,k-1,0)) / dx[2];
          vz = 0.5_rt * (dat(i,j,k+1,2) / dat(i,j,k+1,0) -
                         dat(i,j,k-1,2) / dat(i,j,k-1,0)) / dx[2];
#endif

         Real v1 = wy - vz;
         Real v2 = uz - wx;
         Real v3 = vx - uy;
         der(i,j,k,0) = std::sqrt(v1*v1 + v2*v2 + v3*v3);

        } else if (coord_type == 1) {
#if AMREX_SPACEDIM == 2

          Real r = (static_cast<Real>(i) + 0.5_rt)*dx[0] + problo[0];
          Real rm1 = (static_cast<Real>(i) - 0.5_rt)*dx[0] + problo[0];
          Real rp1 = (static_cast<Real>(i) + 1.5_rt)*dx[0] + problo[0];

          Real rad_vel_dz = 0.5_rt * (dat(i,j+1,k,1) / dat(i,j+1,k,0) -
                                dat(i,j-1,k,1) / dat(i,j-1,k,0)) / dx[1];

          Real phi_vel_dz = 0.5_rt * (dat(i,j+1,k,3) / dat(i,j+1,k,0) -
                                  dat(i,j-1,k,3) / dat(i,j-1,k,0)) / dx[1];

          Real ang_mom_dr = 0.5_rt * (rp1 * dat(i+1,j,k,3) / dat(i+1,j,k,0) -
                                   rm1 * dat(i-1,j,k,3) / dat(i-1,j,k,0)) / dx[0];

          Real z_vel_dr = 0.5_rt * (dat(i+1,j,k,2) / dat(i+1,j,k,0) -
                                dat(i-1,j,k,2) / dat(i-1,j,k,0)) / dx[0];

          der(i,j,k,0) = std::sqrt(phi_vel_dz*phi_vel_dz +
                                   (rad_vel_dz - z_vel_dr)*(rad_vel_dz - z_vel_dr) +
                                   (ang_mom_dr/r)*(ang_mom_dr/r));
#else
          der(i,j,k,0) = 0.0_rt;
#endif

        } else if (coord_type == 2) {
#if AMREX_SPACEDIM == 2

          Real rc  = (static_cast<Real>(i) + 0.5_rt)*dx[0] + problo[0];
          Real rm1 = (static_cast<Real>(i) - 0.5_rt)*dx[0] + problo[0];
          Real rp1 = (static_cast<Real>(i) + 1.5_rt)*dx[0] + problo[0];

          Real sinc  = std::sin((static_cast<Real>(j) + 0.5_rt)*dx[1] + problo[1]);
          Real sin_prev_ctr = std::sin((static_cast<Real>(j) - 0.5_rt)*dx[1] + problo[1]);
          Real sin_next_ctr = std::sin((static_cast<Real>(j) + 1.5_rt)*dx[1] + problo[1]);

          Real sin_vphi_dtheta = 0.5_rt * (sin_next_ctr * dat(i,j+1,k,3) / dat(i,j+1,k,0) -
                                         sin_prev_ctr * dat(i,j-1,k,3) / dat(i,j-1,k,0)) / dx[1];

          Real ang_mom_dr = 0.5_rt * (rp1 * dat(i+1,j,k,3) / dat(i+1,j,k,0) -
                                   rm1 * dat(i-1,j,k,3) / dat(i-1,j,k,0)) / dx[0];

          Real theta_mom_dr = 0.5_rt * (rp1 * dat(i+1,j,k,2) / dat(i+1,j,k,0) -
                                     rm1 * dat(i-1,j,k,2) / dat(i-1,j,k,0)) / dx[0];

          Real rad_vel_dtheta = 0.5_rt * (dat(i,j+1,k,1) / dat(i,j+1,k,0) -
                                    dat(i,j-1,k,1) / dat(i,j-1,k,0)) / dx[1];

          der(i,j,k,0) = std::sqrt((sin_vphi_dtheta/sinc)*(sin_vphi_dtheta/sinc) +
                                   (theta_mom_dr - rad_vel_dtheta)*(theta_mom_dr - rad_vel_dtheta) +
                                   (ang_mom_dr*ang_mom_dr)) / rc;
#else
          der(i,j,k,0) = 0.0;
#endif
        }
      });
    }

  void derive_flow_divergence(const Box& bx, FArrayBox& derfab, int  , int  ,
                         const FArrayBox& datfab, const Geometry& geom,
                         Real  , const int*  , int  )
    {

      auto const dat = datfab.array();
      auto const der = derfab.array();

      auto dx = geom.CellSizeArray();

      auto problo = geom.ProbLoArray();

      const int coord_type = geom.Coord();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        Real uhi = dat(i+1,j,k,1) / dat(i+1,j,k,0);
        Real ulo = dat(i-1,j,k,1) / dat(i-1,j,k,0);

#if AMREX_SPACEDIM >= 2
        Real vhi = dat(i,j+dg1,k,2) / dat(i,j+dg1,k,0);
        Real vlo = dat(i,j-dg1,k,2) / dat(i,j-dg1,k,0);
#endif

#if AMREX_SPACEDIM == 3
        Real whi = dat(i,j,k+dg2,3) / dat(i,j,k+dg2,0);
        Real wlo = dat(i,j,k-dg2,3) / dat(i,j,k-dg2,0);
#endif

        if (coord_type == 0) {

          der(i,j,k,0) = 0.5_rt * (uhi - ulo) / dx[0];
#if AMREX_SPACEDIM >= 2
          der(i,j,k,0) += 0.5_rt * (vhi - vlo) / dx[1];
#endif
#if AMREX_SPACEDIM == 3
          der(i,j,k,0) += 0.5_rt * (whi - wlo) / dx[2];
#endif
        } else if (coord_type == 1) {

          Real r = (static_cast<Real>(i) + 0.5_rt)*dx[0] + problo[0];
          Real rm1 = (static_cast<Real>(i) - 0.5_rt)*dx[0] + problo[0];
          Real rp1 = (static_cast<Real>(i) + 1.5_rt)*dx[0] + problo[0];

          der(i,j,k,0) = 0.5_rt * (rp1*uhi - rm1*ulo) / (r*dx[0]);
#if AMREX_SPACEDIM >= 2
          der(i,j,k,0) += 0.5_rt * (vhi - vlo) / dx[1];
#endif

        } else if (coord_type == 2) {

          Real r = (static_cast<Real>(i) + 0.5_rt)*dx[0] + problo[0];
          Real rm1 = (static_cast<Real>(i) - 0.5_rt)*dx[0] + problo[0];
          Real rp1 = (static_cast<Real>(i) + 1.5_rt)*dx[0] + problo[0];

          der(i,j,k,0) = 0.5_rt * (rp1*rp1*uhi - rm1*rm1*ulo) / (r*r*dx[0]);
#if AMREX_SPACEDIM == 2
          Real sinc = std::sin((static_cast<Real>(j) + 0.5_rt)*dx[1] + problo[1]);
          Real sin_prev_ctr = std::sin((static_cast<Real>(j) - 0.5_rt)*dx[1] + problo[1]);
          Real sin_next_ctr = std::sin((static_cast<Real>(j) + 1.5_rt)*dx[1] + problo[1]);

          der(i,j,k,0) += 0.5_rt * (vhi*sin_next_ctr - vlo*sin_prev_ctr) / (r*sinc*dx[1]);
#endif
        }

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

#ifdef MHD
  void derive_cell_bfield_x(const Box& bx, FArrayBox& derfab, int  , int  ,
                     const FArrayBox& datfab, const Geometry&  ,
                     Real  , const int*  , int  )
  {

    auto const dat = datfab.array();
    auto const der = derfab.array();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      der(i,j,k,0) = 0.5_rt * (dat(i,j,k,0) + dat(i+1,j,k,0));
    });

  }

  void derive_cell_bfield_y(const Box& bx, FArrayBox& derfab, int  , int  ,
                     const FArrayBox& datfab, const Geometry&  ,
                     Real  , const int*  , int  )
  {

    auto const dat = datfab.array();
    auto const der = derfab.array();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      der(i,j,k,0) = 0.5_rt * (dat(i,j,k,0) + dat(i,j+1,k,0));
    });

  }

  void derive_cell_bfield_z(const Box& bx, FArrayBox& derfab, int  , int  ,
                     const FArrayBox& datfab, const Geometry&  ,
                     Real  , const int*  , int  )
  {

    auto const dat = datfab.array();
    auto const der = derfab.array();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      der(i,j,k,0) = 0.5_rt * (dat(i,j,k,0) + dat(i,j,k+1,0));
    });

  }

  void derive_b_divergence(const Box& bx, FArrayBox& derfab, int  , int  ,
                  const FArrayBox& datfab, const Geometry& geom,
                  Real  , const int*  , int  )
  {

    auto const dat = datfab.array();
    auto const der = derfab.array();

    auto dx = geom.CellSizeArray();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      Real dBx = dat(i+1,j,k,0) - dat(i,j,k,0);
      der(i,j,k,0) = dBx / dx[0];

#if AMREX_SPACEDIM >= 2
      Real dBy = dat(i,j+1,k,1) - dat(i,j,k,1);
      der(i,j,k,0) += dBy / dx[1];
#endif

#if AMREX_SPACEDIM == 3
      Real dBz = dat(i,j,k+1,2) - dat(i,j,k,2);
      der(i,j,k,0) += dBz / dx[2];
#endif
    });
  }

#endif

#ifdef __cplusplus
}
#endif
