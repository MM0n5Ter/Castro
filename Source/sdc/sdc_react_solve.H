#ifndef SDC_REACT_SOLVE_H
#define SDC_REACT_SOLVE_H

#include <HydroSim.H>
#ifdef REACTIONS
#ifdef NEW_NETWORK_IMPLEMENTATION
#include <rhs.H>
#endif
#include <burn_rhs.H>
#include <implicit_burn_solvers.H>
#include <implicit_burn_solvers.H>
#include <implicit_burn_solvers.H>
#endif

namespace deferred_solvers {
    enum solver : std::uint8_t {
        newton = 1,
        vode = 2,
        stiff_and_newton = 3,
        hybrid_dogleg = 4
    };
}

#ifdef REACTIONS
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
instant_reaction_rate(const int i, const int j, const int k,
                    Array4<const Real> const& state,
                    Array4<Real> const& React_source) {


    if (burning_permitted(i, j, k, state)) {
        zone_react_source(i, j, k, state, React_source);
    } else {
        for (int n = 0; n < NUM_STATE; ++n) {
            React_source(i, j, k, n) = 0.0_rt;
        }
    }
}
#endif

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
normalize_species_deferred(const int i, const int j, const int k,
                      Array4<Real> const& u) {

    GpuArray<Real, NumSpec> partial_densities;

    Real partial_density_sum = 0.0_rt;

    for (int n = 0; n < NumSpec; ++n) {
        partial_densities[n] = u(i,j,k,UFS+n);
        partial_densities[n] = amrex::Clamp(partial_densities[n], network_rp::small_x * u(i,j,k,URHO), u(i,j,k,URHO));
        partial_density_sum += partial_densities[n];
    }

    for (int n = 0; n < NumSpec; ++n) {
        partial_densities[n] *= u(i,j,k,URHO) / partial_density_sum;
        u(i,j,k,UFS+n) = partial_densities[n];
    }
}

#ifdef REACTIONS
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
deferred_solve(const Real substep_dt,
          GpuArray<Real, NUM_STATE> const& Conserved_old,
          GpuArray<Real, NUM_STATE>& cons_new,
          GpuArray<Real, NUM_STATE> const& C,
          const int correction_iter) {

    int ierr;
    Real err_out;

    switch (sdc_nonlin_method) {
    case deferred_solvers::newton:
        ierr = newton::deferred_newton_subdivide(substep_dt, Conserved_old, cons_new, C, correction_iter, err_out);

        if (ierr != newton_outcome::success) {
            amrex::Abort("Newton subcycling failed in deferred_solve");
        }
        break;

    case deferred_solvers::vode:
        deferred_stiff_solve(substep_dt, Conserved_old, cons_new, C, correction_iter);
        break;

    case deferred_solvers::stiff_and_newton:
        if (correction_iter == 0) {
            deferred_stiff_solve(substep_dt, Conserved_old, cons_new, C, correction_iter);
        }

        ierr = newton::deferred_newton_subdivide(substep_dt, Conserved_old, cons_new, C, correction_iter, err_out);

        if (ierr != newton_outcome::success) {
            amrex::Abort("Newton failure in deferred_solve");
        }
        break;

    case deferred_solvers::hybrid_dogleg:
        powell::deferred_dogleg_solve(substep_dt, Conserved_old, cons_new, C, correction_iter);
        break;

    default:
        amrex::Abort("invalid sdc_nonlin_method value");
    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
deferred_solve(const int i, const int j, const int k,
          const Real substep_dt,
          Array4<const Real> const& Conserved_old,
          Array4<Real> const& cons_new,
          Array4<const Real> const& C,
          const int correction_iter) {

    GpuArray<Real, NUM_STATE> State_old_zone;
    GpuArray<Real, NUM_STATE> State_new_zone;
    GpuArray<Real, NUM_STATE> Correction_zone;

    for (int n = 0; n < NUM_STATE; ++n) {
        State_old_zone[n] = Conserved_old(i,j,k,n);
        State_new_zone[n] = cons_new(i,j,k,n);
        Correction_zone[n] = C(i,j,k,n);
    }

    deferred_solve(substep_dt, State_old_zone, State_new_zone, Correction_zone, correction_iter);

    for (int n = 0; n < NUM_STATE; ++n) {
        cons_new(i,j,k,n) = State_new_zone[n];
    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
deferred_update_o2(const int i, const int j, const int k,
              Array4<const Real> const& k_m,
              Array4<Real> const& k_n,
              Array4<const Real> const& A_m,
              Array4<const Real> const& React_m_old,
              Array4<const Real> const& C,
              const Real substep_dt,
              const int correction_iter, const int m_start) {

    amrex::ignore_unused(m_start);


    GpuArray<Real, NUM_STATE> Conserved_old;
    GpuArray<Real, NUM_STATE> cons_new;
    GpuArray<Real, NUM_STATE> React_full;
    GpuArray<Real, NUM_STATE> Correction_zone;

    for (int n = 0; n < NUM_STATE; ++n) {
        Conserved_old[n] = k_m(i,j,k,n);
        Correction_zone[n] = C(i,j,k,n);
    }

    if (!burning_permitted(Conserved_old)) {
        for (int n = 0; n < NUM_STATE; ++n) {
            React_full[n] = 0.0_rt;
        }
    } else {

        if (correction_iter == 0) {
            for (int n = 0; n < NUM_STATE; ++n) {
                cons_new[n] = Conserved_old[n] + substep_dt * A_m(i,j,k,n) + substep_dt * React_m_old(i,j,k,n);
            }
        } else {
            for (int n = 0; n < NUM_STATE; ++n) {
                cons_new[n] = k_n(i,j,k,n);
            }
        }

        deferred_solve(substep_dt, Conserved_old, cons_new, Correction_zone, correction_iter);

        burn_t burn_state;

        fill_burn_type_from_conserved(cons_new, burn_state);
        zone_react_source(burn_state, React_full);
    }

    for (int n = 0; n < NUM_STATE; ++n) {
        cons_new[n] = Conserved_old[n] + substep_dt * React_full[n] + substep_dt * Correction_zone[n];

        k_n(i,j,k,n) = cons_new[n];
    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
deferred_update_centers_o4(const int i, const int j, const int k,
                      Array4<const Real> const& Conserved_old,
                      Array4<Real> const& cons_new,
                      Array4<const Real> const& C,
                      const Real substep_dt,
                      const int correction_iter) {

    if (burning_permitted(i, j, k, Conserved_old)) {
        deferred_solve(i, j, k, substep_dt, Conserved_old, cons_new, C, correction_iter);
    } else {
        for (int n = 0; n < NUM_STATE; ++n) {
            cons_new(i,j,k,n) = Conserved_old(i,j,k,n) + substep_dt * C(i,j,k,n);
        }
    }
}
#endif

#endif
