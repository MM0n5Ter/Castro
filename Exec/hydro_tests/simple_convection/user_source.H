#ifndef user_source_H
#define user_source_H

#include <prob_parameters.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_source (int i, int j, int k,
                     GeometryData const& geomdata,
                     Array4<const Real> const& state,
                     Array4<Real> const& src,
                     const Real dt, const Real time)
{

    for (int n = 0; n < NUM_SOURCE; ++n) {
        src(i,j,k,n) = 0.0_rt;
    }

    const Real* problo = geomdata.ProbLo();
    const Real* dx = geomdata.CellSize();

    amrex::Real yy = problo[1] + (static_cast<amrex::Real>(j) + 0.5_rt) * dx[1];
    amrex::Real dist = std::abs(yy - problem::y_height);

    auto rho = state(i,j,k,URHO);

    src(i,j,k,UEDEN) = rho * problem::e_rate * std::exp(-std::pow(dist / problem::thickness, 2.0));
    src(i,j,k,UEINT) = src(i,j,k,UEDEN);

}

#endif
