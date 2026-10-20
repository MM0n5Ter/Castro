#include <AMReX_REAL.H>

#include <Derive.H>
#include <HydroSim.H>

#ifdef DIFFUSION
#include <conduction_coeffs.H>
#endif

using namespace amrex;

namespace {

#ifdef MHD
template <int Dir>
void face_to_cell_average(const Box& bx, FArrayBox& derfab, const FArrayBox& datfab)
{
  auto const dat = datfab.array();
  auto const der = derfab.array();

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    der(i,j,k,0) = 0.5_rt * (dat(i,j,k,0) + dat(i+(Dir==0),j+(Dir==1),k+(Dir==2),0));
  });
}
#endif

void angular_momentum_component(const Box& bx, FArrayBox& derfab, const FArrayBox& datfab,
                                const Geometry& geom, int idir)
{
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

                         constexpr int lever[3] = {1, 2, 0};
                         constexpr int mom[3] = {2, 0, 1};

                         L(i,j,k,0) = loc[lever[idir]] * dat(i,j,k,1+mom[idir]) - loc[mom[idir]] * dat(i,j,k,1+lever[idir]);

                       });
}

void vector_magnitude(const Box& bx, FArrayBox& derfab, const FArrayBox& datfab)
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

}

#ifdef __cplusplus
extern "C"
{
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

  void derive_momentum_magnitude(const Box& bx, FArrayBox& derfab, int  , int  ,
                    const FArrayBox& datfab, const Geometry&  ,
                    Real  , const int*  , int  )
  {

    vector_magnitude(bx, derfab, datfab);
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

  void derive_ang_momentum_x (const Box& bx, FArrayBox& derfab, int  , int  ,
                      const FArrayBox& datfab, const Geometry& geom,
                      Real  , const int*  , int  )
  {
    angular_momentum_component(bx, derfab, datfab, geom, 0);
  }

  void derive_ang_momentum_y (const Box& bx, FArrayBox& derfab, int  , int  ,
                      const FArrayBox& datfab, const Geometry& geom,
                      Real  , const int*  , int  )
  {
    angular_momentum_component(bx, derfab, datfab, geom, 1);
  }

  void derive_ang_momentum_z (const Box& bx, FArrayBox& derfab, int  , int  ,
                      const FArrayBox& datfab, const Geometry& geom,
                      Real  , const int*  , int  )
  {
    angular_momentum_component(bx, derfab, datfab, geom, 2);
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

    void derive_gravity_magnitude(const Box& bx, FArrayBox& derfab, int  , int  ,
                       const FArrayBox& datfab, const Geometry&  ,
                       Real  , const int*  , int  )
    {

      vector_magnitude(bx, derfab, datfab);
    }

#ifdef MHD
  void derive_cell_bfield_x(const Box& bx, FArrayBox& derfab, int  , int  ,
                     const FArrayBox& datfab, const Geometry&  ,
                     Real  , const int*  , int  )
  {
    face_to_cell_average<0>(bx, derfab, datfab);
  }

  void derive_cell_bfield_y(const Box& bx, FArrayBox& derfab, int  , int  ,
                     const FArrayBox& datfab, const Geometry&  ,
                     Real  , const int*  , int  )
  {
    face_to_cell_average<1>(bx, derfab, datfab);
  }

  void derive_cell_bfield_z(const Box& bx, FArrayBox& derfab, int  , int  ,
                     const FArrayBox& datfab, const Geometry&  ,
                     Real  , const int*  , int  )
  {
    face_to_cell_average<2>(bx, derfab, datfab);
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
