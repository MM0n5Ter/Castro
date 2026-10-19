#ifndef user_initialize_state_data_H
#define user_initialize_state_data_H

#include <prob_parameters.H>
#include <eos.H>
#include <HydroSim_math.H>
#include <Rotation.H>
#include <HydroSim_helpers.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_initialize_state_data (int i, int j, int k,
                                    Array4<Real> const& state,
                                    const GeometryData& geomdata)
{

    const Real* problo = geomdata.ProbLo();
    const Real* probhi = geomdata.ProbHi();

#ifdef ROTATION
    auto omega = get_angular_velocity(geomdata, j);
#else
    Real omega[3] = {0.0_rt, 0.0_rt, 2.0_rt * M_PI};
#endif


    GpuArray<Real, 3> loc;
    position(i, j, k, geomdata, loc);

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        loc[dir] -= problem::center[dir];
    }

    Real R = std::sqrt(loc[0] * loc[0] + loc[1] * loc[1]);
    Real Z = loc[2];

    Real dist = std::sqrt(loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2]);


    Real rho_s = std::pow(C::Gconst * point_mass /
                          ((1.0_rt + polytrope_index) * K_const * problem::density_maximum_radius), polytrope_index);

    Real fac = problem::density_maximum_radius / dist -
        0.5_rt * problem::density_maximum_radius * problem::density_maximum_radius / (R * R) -
        0.5_rt * problem::density_maximum_radius * problem::density_maximum_radius / (problem::inner_radius * problem::outer_radius);

    Real rho;
    if (fac > 0.0_rt) {

        rho = rho_s * std::pow(fac, polytrope_index);

        if (rho < problem::background_density) {
            rho = problem::background_density;
        }

    } else {

        rho = problem::background_density;

    }

    GpuArray<Real, 3> vel = {0.0};
    if (rho > problem::background_density && do_frame_spin == 0) {
        cross_product(omega, loc, vel);
    }

    eos_t eos_state;
    eos_state.rho = rho;
    eos_state.T = 1.0;
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n]  = 1.0_rt / static_cast<Real>(NumSpec);
    }

    eos(eos_input_rt, eos_state);

    state(i,j,k,URHO) = rho;
    state(i,j,k,UTEMP) = eos_state.T;
    state(i,j,k,UEINT) = rho * eos_state.e;

    state(i,j,k,UMX) = rho * vel[0];
    state(i,j,k,UMY) = rho * vel[1];
    state(i,j,k,UMZ) = rho * vel[2];

    state(i,j,k,UEDEN) = state(i,j,k,UEINT) + 0.5_rt * rho * (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);

    for (int n = 0; n < NumSpec; n++) {
        state(i,j,k,UFS+n) = state(i,j,k,URHO) * (1.0_rt / static_cast<Real>(NumSpec));
    }
}

#endif
