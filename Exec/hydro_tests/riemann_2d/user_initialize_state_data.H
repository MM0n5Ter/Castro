#ifndef user_initialize_state_data_H
#define user_initialize_state_data_H

#include <prob_parameters.H>

#include <eos.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_initialize_state_data(int i, int j, int k,
                                   Array4<Real> const& state,
                                   const GeometryData& geomdata)
{
    int coord_type = geomdata.Coord();

    const Real* dx = geomdata.CellSize();

    const Real* problo = geomdata.ProbLo();

#if AMREX_SPACEDIM != 2
    amrex::Abort("The Riemman Problem considered is 2D.");

#endif


#if AMREX_SPACEDIM == 2
    Real xx = problo[0] + dx[0] * (static_cast<Real>(i) + 0.5_rt);
    Real yy = problo[1] + dx[1] * (static_cast<Real>(j) + 0.5_rt);
#endif

    state(i, j, k, UMZ) = 0.0;

    if (xx < problem::center[0] && yy < problem::center[1])
    {
        state(i, j, k, URHO)   = problem::rho_3;

        state(i, j, k, UMX)    = state(i, j, k, URHO) * problem::ux_3;
        state(i, j, k, UMY)    = state(i, j, k, URHO) * problem::uy_3;
        state(i, j, k, UEINT)  = problem::p_3 / (eos_rp::eos_gamma - 1.0_rt);
        state(i, j, k, UEDEN)  = problem::p_3 / (eos_rp::eos_gamma - 1.0_rt)
                                     + 0.5 * problem::rho_3 * problem::ux_3 * problem::ux_3
                                     + 0.5 * problem::rho_3 * problem::uy_3 * problem::uy_3;

    } else if (xx > problem::center[0] && yy < problem::center[1])
    {
        state(i, j, k, URHO)  = problem::rho_4;

        state(i, j, k, UMX)   = state(i, j, k, URHO) * problem::ux_4;
        state(i, j, k, UMY)   = state(i, j, k, URHO) * problem::uy_4;
        state(i, j, k, UEINT) = problem::p_4 / (eos_rp::eos_gamma - 1.0_rt);

        state(i, j, k, UEDEN) = problem::p_4 / (eos_rp::eos_gamma - 1.0_rt)
                                     + 0.5 * problem::rho_4 * problem::ux_4 * problem::ux_4
                                     + 0.5 * problem::rho_4 * problem::uy_4 * problem::uy_4;

    } else if (xx < problem::center[0] && yy > problem::center[1])
    {
        state(i, j, k, URHO)  = problem::rho_2;

        state(i, j, k, UMX)   = state(i, j, k, URHO) * problem::ux_2;
        state(i, j, k, UMY)   = state(i, j, k, URHO) * problem::uy_2;
        state(i, j, k, UEINT) = problem::p_2 / (eos_rp::eos_gamma - 1.0_rt);

        state(i, j, k, UEDEN) = problem::p_2 / (eos_rp::eos_gamma - 1.0_rt)
                                     + 0.5 * problem::rho_2 * problem::ux_2 * problem::ux_2
                                     + 0.5 * problem::rho_2 * problem::uy_2 * problem::uy_2;
    } else if (xx > problem::center[0] & yy > problem::center[1])
    {
        state(i, j, k, URHO)  = problem::rho_1;

        state(i, j, k, UMX)   = state(i, j, k, URHO) * problem::ux_1;
        state(i, j, k, UMY)   = state(i, j, k, URHO) * problem::uy_1;
        state(i, j, k, UEINT) = problem::p_1 / (eos_rp::eos_gamma - 1.0_rt);

        state(i, j, k, UEDEN) = problem::p_1 / (eos_rp::eos_gamma - 1.0_rt)
                                     + 0.5 * problem::rho_1 * problem::ux_1 * problem::ux_1
                                     + 0.5 * problem::rho_1 * problem::uy_1 * problem::uy_1;
    }

    state(i, j, k,UFS) = state(i,j,k,URHO);
}

#endif
