#ifndef user_source_H
#define user_source_H

#include <prob_parameters.H>
#ifdef HYBRID_MOMENTUM
#include <hybrid.H>
#endif
#ifdef ROTATION
#include <Rotation.H>
#endif
#include <binarywd_util.H>
#include <binarywd_util.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_source (int i, int j, int k,
                     GeometryData const& geomdata,
                     Array4<const Real> const& state,
                     Array4<Real> const& src,
                     const Real dt, const Real time)
{
    using namespace problem;


    if (problem::problem == 1 && relaxation_damping_factor > 0.0_rt) {


        const Real dynamical_timescale = amrex::min(t_ff_P, t_ff_S);


        const Real relaxation_damping_timescale = relaxation_damping_factor * dynamical_timescale;


        const auto damping_factor = -(1.0_rt - 1.0_rt / (1.0_rt + dt / relaxation_damping_timescale)) / dt;

        GpuArray<Real, 3> loc;
        GpuArray<Real, 3> mom;
#ifdef HYBRID_MOMENTUM
        GpuArray<Real, 3> hybrid_vector;
#endif
        GpuArray<Real, 3> Sr;

        const auto rhoInv = 1.0_rt / state(i,j,k,URHO);
        position(i, j, k, geomdata, loc);

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            loc[dir] -= center[dir];
            mom[dir] = state(i,j,k,UMX+dir);
        }

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            Sr[dir] = mom[dir] * damping_factor;
            src(i,j,k,UMX+dir) += Sr[dir];
        }

#ifdef HYBRID_MOMENTUM
        cartesian_to_hybrid(loc, Sr, hybrid_vector);
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            src(i,j,k,UMR+dir) += hybrid_vector[dir];
        }
#endif

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            src(i,j,k,UEDEN) += rhoInv * mom[dir] * Sr[dir];
        }
    }


    bool do_radial_damping = false;

    if (problem::problem == 1 && problem::radial_damping_velocity_factor > 0.0_rt && problem::radial_damping_is_done != 1) {
        do_radial_damping = true;
    }

    if (problem::problem == 1 && problem::relaxation_damping_factor > 0.0_rt && problem::relaxation_is_done != 1) {
        do_radial_damping = false;
    }

    if (do_radial_damping) {
        GpuArray<Real, 3> acceleration = {0.0};


        Real a_curr = 0.0_rt;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            a_curr += (problem::com_P[dir] - problem::com_S[dir]) * (problem::com_P[dir] - problem::com_S[dir]);
        }
        a_curr = std::sqrt(a_curr);


        Real v_init = 2.0_rt * M_PI * problem::a / hydrosim::spin_period;

        if (stellar_mask(i, j, k, geomdata, state(i,j,k,URHO), true) > 0.0_rt) {
            GpuArray<Real, 3> v = {0.0_rt};
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                v[dir] = problem::vel_P[dir];
            }
            if (hydrosim::do_frame_spin == 1) {
                frame_to_inertial_velocity(i, j, k, geomdata, time, v);
            }

            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                acceleration[dir] = -0.5_rt * (problem::radial_damping_velocity_factor * v_init / a_curr) * v[dir];
            }
        }
        else if (stellar_mask(i, j, k, geomdata, state(i,j,k,URHO), false) > 0.0_rt) {
            GpuArray<Real, 3> v = {0.0_rt};
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                v[dir] = problem::vel_S[dir];
            }
            if (hydrosim::do_frame_spin == 1) {
                frame_to_inertial_velocity(i, j, k, geomdata, time, v);
            }

            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                acceleration[dir] = -0.5_rt * (problem::radial_damping_velocity_factor * v_init / a_curr) * v[dir];
            }
        }

        GpuArray<Real, 3> Sr = {0.0};
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            Sr[dir] = state(i,j,k,URHO) * acceleration[dir];
        }

#ifdef HYBRID_MOMENTUM
        GpuArray<Real, 3> loc;
        position(i, j, k, geomdata, loc);

        GpuArray<Real, 3> hybrid_Sr = {0.0};
        cartesian_to_hybrid(loc, Sr, hybrid_Sr);

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            src(i,j,k,UMR+dir) += hybrid_Sr[dir];
        }
#endif

        Real rhoInv = 1.0_rt / state(i,j,k,URHO);

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            src(i,j,k,UMX+dir) += Sr[dir];
            src(i,j,k,UEDEN) += rhoInv * state(i,j,k,UMX+dir) * Sr[dir];
        }
    }
}

#endif
