#ifndef HYDROSIM_REACT_H
#define HYDROSIM_REACT_H

///
/// Perform any old-time reactions.
///
    step_result apply_old_reactions (Real time, Real dt);

///
/// Perform any new-time reactions.
///
    step_result apply_new_reactions (Real time, Real dt);

///
/// React state through a single reaction timestep, saving reaction source terms to ``reactions`` MultiFab. This is the CTU version.
///
/// @param state        Current state
/// @param reactions    MultiFab to save reaction sources to
/// @param time         current time
/// @param dt           reaction timestep
///
    int burn_over_state(amrex::MultiFab& state,
                    amrex::MultiFab& reactions,
                    amrex::Real time,
                    amrex::Real dt,
                    const int strang_half);

///
/// Simplified SDC version of burn_over_state. Reacts the current state through a single timestep.
///
/// @param time     current time
/// @param dt       timestep
///
    int burn_over_state(amrex::Real time, amrex::Real dt);

///
/// Are there any zones in ``State`` that can burn?
///
/// @param State    State MultiFab
///
    static bool zones_are_burnable(const amrex::MultiFab& State);

#endif
