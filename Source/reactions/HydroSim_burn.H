#ifndef HYDROSIM_REACT_H
#define HYDROSIM_REACT_H

    step_result apply_old_reactions (Real time, Real dt);

    step_result apply_new_reactions (Real time, Real dt);

    int burn_over_state(amrex::MultiFab& state,
                    amrex::MultiFab& reactions,
                    amrex::Real time,
                    amrex::Real dt,
                    const int strang_half);

    int burn_over_state(amrex::Real time, amrex::Real dt);

    static bool zones_are_burnable(const amrex::MultiFab& State);

#endif
