#ifndef HYDROSIM_SDC_UTIL_H
#define HYDROSIM_SDC_UTIL_H

#include <HydroSim.H>
#ifdef REACTIONS
#ifdef NEW_NETWORK_IMPLEMENTATION
#include <rhs.H>
#endif
#include <sdc_react_util.H>
#include <sdc_newton_solve.H>
#include <sdc_vode_solve.H>
#include <sdc_powell_solve.H>
#endif

namespace sdc_solvers {
    enum solver : std::uint8_t {
        newton = 1,
        vode = 2,
        vode_and_newton = 3,
        hybrid_powell = 4
    };
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
normalize_species_sdc(const int i, const int j, const int k,
                      Array4<Real> const& u) {

    GpuArray<Real, NumSpec> rho_xn;

    Real rho_xn_sum = 0.0_rt;

    for (int n = 0; n < NumSpec; ++n) {
        rho_xn[n] = u(i,j,k,USPEC+n);
        rho_xn[n] = amrex::Clamp(rho_xn[n], network_rp::small_x * u(i,j,k,URHO), u(i,j,k,URHO));
        rho_xn_sum += rho_xn[n];
    }

    for (int n = 0; n < NumSpec; ++n) {
        rho_xn[n] *= u(i,j,k,URHO) / rho_xn_sum;
        u(i,j,k,USPEC+n) = rho_xn[n];
    }
}

#ifdef REACTIONS

// The zone data of an update from time node m to node m+1: the state
// at node m, the state at node m+1 (it holds the previous iterate on
// input and the update on output) and the source term C.  The
// second-order update also uses the advective and reactive terms at
// node m to build its first guess.

struct NodeStep {
    Array4<const Real> state_m;
    Array4<Real> state_m1;
    Array4<const Real> C;
    Array4<const Real> A_m;
    Array4<const Real> R_m_old;
};

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
sdc_solve(GpuArray<Real, NUM_STATE> const& U_old,
          GpuArray<Real, NUM_STATE> const& C,
          const Real dt_m,
          const int sdc_iteration,
          GpuArray<Real, NUM_STATE>& U_new) {

    if (sdc_solver == sdc_solvers::newton) {
        // We are going to assume we already have a good guess
        // for the solve in U_new and just pass the solve onto
        // the main Newton solve
        const auto newton_result = newton::sdc_newton_subdivide(U_old, C, dt_m, sdc_iteration, U_new);

        // failing?
        if (newton_result.ierr != newton_status::success) {
            amrex::Abort("Newton subcycling failed in sdc_solve");
        }
    } else if (sdc_solver == sdc_solvers::vode) {
        // Use VODE to do the solution
        sdc_vode_solve(U_old, C, dt_m, sdc_iteration, U_new);

    } else if (sdc_solver == sdc_solvers::vode_and_newton) {
        // If it is the first iteration, we will use VODE to
        // predict the solution. Otherwise, we will use Newton.
        if (sdc_iteration == 0) {
            sdc_vode_solve(U_old, C, dt_m, sdc_iteration, U_new);
        }

        // Now U_new is the update that VODE predicts, so we
        // will use that as the initial guess to the Newton solve
        const auto newton_result = newton::sdc_newton_subdivide(U_old, C, dt_m, sdc_iteration, U_new);

        // Failing?
        if (newton_result.ierr != newton_status::success) {
            amrex::Abort("Newton failure in sdc_solve");
        }

    } else if (sdc_solver == sdc_solvers::hybrid_powell) {
        powell::sdc_powell_solve(U_old, C, dt_m, sdc_iteration, U_new);

    } else {
        amrex::Abort("invalid sdc_solver value");
    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
sdc_solve(const int i, const int j, const int k,
          Array4<const Real> const& U_old,
          Array4<const Real> const& C,
          const Real dt_m,
          const int sdc_iteration,
          Array4<Real> const& U_new) {
    // wrapper for the zone-by-zone version

    GpuArray<Real, NUM_STATE> U_old_zone;
    GpuArray<Real, NUM_STATE> U_new_zone;
    GpuArray<Real, NUM_STATE> C_zone;

    for (int n = 0; n < NUM_STATE; ++n) {
        U_old_zone[n] = U_old(i,j,k,n);
        U_new_zone[n] = U_new(i,j,k,n);
        C_zone[n] = C(i,j,k,n);
    }

    sdc_solve(U_old_zone, C_zone, dt_m, sdc_iteration, U_new_zone);

    for (int n = 0; n < NUM_STATE; ++n) {
        U_new(i,j,k,n) = U_new_zone[n];
    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
sdc_update_o2(const int i, const int j, const int k,
              NodeStep const& step,
              const Real dt_m,
              const int sdc_iteration, const int m_start) {
    // update node m to node m+1 via advection -- this is a second-order accurate update

    amrex::ignore_unused(m_start);

    // Here, dt_m is the timestep between time-nodes m and m+1

    GpuArray<Real, NUM_STATE> U_old;
    GpuArray<Real, NUM_STATE> U_new;
    GpuArray<Real, NUM_STATE> R_full;
    GpuArray<Real, NUM_STATE> C_zone;

    for (int n = 0; n < NUM_STATE; ++n) {
        U_old[n] = step.state_m(i,j,k,n);
        C_zone[n] = step.C(i,j,k,n);
    }

    // Only burn if we are within the temperature and density
    // limits for burning
    if (!okay_to_burn(U_old)) {
        for (int n = 0; n < NUM_STATE; ++n) {
            R_full[n] = 0.0_rt;
        }
    } else {

        // This is the full state -- this will be updated as we
        // solve the nonlinear system.  We want to start with a
        // good initial guess.  For later iterations, we should
        // begin with the result from the previous iteration.  For
        // the first iteration, let's try to extrapolate forward
        // in time.
        if (sdc_iteration == 0) {
            for (int n = 0; n < NUM_STATE; ++n) {
                U_new[n] = U_old[n] + dt_m * step.A_m(i,j,k,n) + dt_m * step.R_m_old(i,j,k,n);
            }
        } else {
            for (int n = 0; n < NUM_STATE; ++n) {
                U_new[n] = step.state_m1(i,j,k,n);
            }
        }

        sdc_solve(U_old, C_zone, dt_m, sdc_iteration, U_new);

        // we solved our system to some tolerance, but let's be sure
        // we are conservative by reevaluating the reactions and
        // doing the full step update
        burn_t burn_state;

        copy_cons_to_burn_type(U_new, burn_state);
        R_full = single_zone_react_source(burn_state);
    }

    for (int n = 0; n < NUM_STATE; ++n) {
        U_new[n] = U_old[n] + dt_m * R_full[n] + dt_m * C_zone[n];

        // copy back to node m+1
        step.state_m1(i,j,k,n) = U_new[n];
    }
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
sdc_update_centers_o4(const int i, const int j, const int k,
                      NodeStep const& step,
                      const Real dt_m,
                      const int sdc_iteration) {
    // Update node m to node m+1 on cell-centers.  This is an implicit
    // solve because of reactions.  dt_m is the timestep between m and
    // m+1

    // We come in with state_m1 being a guess for the updated solution
    if (okay_to_burn(i, j, k, step.state_m)) {
        sdc_solve(i, j, k, step.state_m, step.C, dt_m, sdc_iteration, step.state_m1);
    } else {
        // no reactions, so it is a straightforward update
        for (int n = 0; n < NUM_STATE; ++n) {
            step.state_m1(i,j,k,n) = step.state_m(i,j,k,n) + dt_m * step.C(i,j,k,n);
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
GpuArray<Real, NUM_STATE>
instantaneous_react(const int i, const int j, const int k,
                    Array4<const Real> const& state) {

    // convert from cons to prim -- show this be here or in C++-land?
    // or should I do things like we do in burn_state and convert it manually?
    // (in that case, I am not sure if I can assume UTEMP is defined)

    GpuArray<Real, NUM_STATE> R_source;

    if (okay_to_burn(i, j, k, state)) {
        R_source = single_zone_react_source(i, j, k, state);
    } else {
        for (int n = 0; n < NUM_STATE; ++n) {
            R_source[n] = 0.0_rt;
        }
    }

    return R_source;
}

#endif

#endif
