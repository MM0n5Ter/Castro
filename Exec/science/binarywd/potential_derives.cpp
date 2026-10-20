#include <AMReX_REAL.H>

#include <Derive.H>
#include <HydroSim.H>
#include <fundamental_constants.H>
#include <prob_parameters.H>
#include <binary_setup.H>
#ifdef ROTATION
#include <Rotation.H>
#endif

using namespace amrex;

void ca_derprimarymask(const Box& bx, FArrayBox& derfab, int  , int  ,
                       const FArrayBox& datfab, const Geometry& geom,
                       Real  , const int*  , int  )
{
    Array4<Real const> const rho = datfab.array();
    Array4<Real> const mask = derfab.array();

    const auto geomdata = geom.data();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        const bool is_primary = true;
        mask(i,j,k) = stellar_mask(i, j, k, geomdata, rho(i,j,k), is_primary);
    });
}

void ca_dersecondarymask(const Box& bx, FArrayBox& derfab, int  , int  ,
                         const FArrayBox& datfab, const Geometry& geom,
                         Real  , const int*  , int  )
{
    Array4<Real const> const rho = datfab.array();
    Array4<Real> const mask = derfab.array();

    const auto geomdata = geom.data();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        bool is_primary = false;
        mask(i,j,k) = stellar_mask(i, j, k, geomdata, rho(i,j,k), is_primary);
    });
}

void ca_derrhophiGrav(const Box& bx, FArrayBox& derfab, int  , int  ,
                      const FArrayBox& datfab, const Geometry& geom,
                      Real  , const int*  , int  )
{

    amrex::ignore_unused(geom);

    auto const dat = datfab.array();
    auto const der = derfab.array();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        der(i,j,k,0) = dat(i,j,k,0) * dat(i,j,k,1);
    });
}

void ca_derrhophiRot(const Box& bx, FArrayBox& derfab, int  , int  ,
                     const FArrayBox& datfab, const Geometry& geom,
                     Real  , const int*  , int  )
{
    auto const dat = datfab.array();
    auto const der = derfab.array();

    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
    const auto coord = geom.Coord();
    const auto geomdata = geom.data();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {

        GpuArray<Real, 3> loc;
        loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];

#if AMREX_SPACEDIM >= 2
        loc[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
#else
        loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
        loc[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
#else
        loc[2] = 0.0_rt;
#endif

        auto omega = get_angular_velocity(geomdata, j);

        der(i,j,k,0) = dat(i,j,k,0) * centrifugal_potential(loc, omega, coord);
    });
}

void ca_derphieff(const Box& bx, FArrayBox& derfab, int  , int  ,
                  const FArrayBox& datfab, const Geometry& geom,
                  Real  , const int*  , int  )
{

    auto const dat = datfab.array();
    auto const der = derfab.array();

    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
    const auto coord = geom.Coord();
    const auto geomdata = geom.data();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {

        GpuArray<Real, 3> loc;
        loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];

#if AMREX_SPACEDIM >= 2
        loc[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
#else
        loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
        loc[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
#else
        loc[2] = 0.0_rt;
#endif
        auto omega = get_angular_velocity(geomdata, j);

        der(i,j,k,0) = dat(i,j,k,0) + centrifugal_potential(loc, omega, coord);
    });
}

void ca_derphieffpm_p(const Box& bx, FArrayBox& derfab, int  , int  ,
                      const FArrayBox& datfab, const Geometry& geom,
                      Real  , const int*  , int  )
{

    amrex::ignore_unused(datfab);

    auto const der = derfab.array();

    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
    const auto coord = geom.Coord();
    const auto geomdata = geom.data();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        der(i,j,k,0) = 0.0_rt;


        if (problem::mass_P <= 0.0_rt) return;

        GpuArray<Real, 3> loc;
        loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];

#if AMREX_SPACEDIM >= 2
        loc[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1];
#else
        loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
        loc[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2];
#else
        loc[2] = 0.0_rt;
#endif

        Real r = std::sqrt((loc[0] - problem::com_P[0]) * (loc[0] - problem::com_P[0]) +
                           (loc[1] - problem::com_P[1]) * (loc[1] - problem::com_P[1]) +
                           (loc[2] - problem::com_P[2]) * (loc[2] - problem::com_P[2]));

        for (int iloc = 0; iloc < AMREX_SPACEDIM; ++iloc) {
            loc[iloc] -= problem::center[iloc];
        }

        auto omega = get_angular_velocity(geomdata, j);

        der(i,j,k,0) = -C::Gconst * problem::mass_P / r + centrifugal_potential(loc, omega, coord);
    });
}

void ca_derphieffpm_s(const Box& bx, FArrayBox& derfab, int  , int  ,
                      const FArrayBox& datfab, const Geometry& geom,
                      Real  , const int*  , int  )
{

    amrex::ignore_unused(datfab);

    auto const der = derfab.array();

    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
    const auto coord = geom.Coord();
    const auto geomdata = geom.data();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        der(i,j,k,0) = 0.0_rt;


        if (problem::mass_S <= 0.0_rt) return;

        GpuArray<Real, 3> loc;
        loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];

#if AMREX_SPACEDIM >= 2
        loc[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1];
#else
        loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
        loc[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2];
#else
        loc[2] = 0.0_rt;
#endif

        Real r = std::sqrt((loc[0] - problem::com_S[0]) * (loc[0] - problem::com_S[0]) +
                           (loc[1] - problem::com_S[1]) * (loc[1] - problem::com_S[1]) +
                           (loc[2] - problem::com_S[2]) * (loc[2] - problem::com_S[2]));

        for (int iloc = 0; iloc < AMREX_SPACEDIM; ++iloc) {
            loc[iloc] -= problem::center[iloc];
        }

        auto omega = get_angular_velocity(geomdata, j);

        der(i,j,k,0) = -C::Gconst * problem::mass_S / r + centrifugal_potential(loc, omega, coord);
    });
}
