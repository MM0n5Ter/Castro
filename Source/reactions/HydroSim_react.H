#ifndef HYDROSIM_REACT_H
#define HYDROSIM_REACT_H

///
/// Perform the reactions for one half of the split update.
///
/// @param time         current time
/// @param dt           timestep
/// @param strang_half  0 for the old-time half (before the advance),
///                     1 for the new-time half (after the advance)
///
    advance_status handle_split_reactions (Real time, Real dt, int strang_half);

///
/// React state through a single reaction timestep, saving reaction source terms to ``reactions`` MultiFab. This is the CTU version.
///
/// @param state        Current state
/// @param reactions    MultiFab to save reaction sources to
/// @param time         current time
/// @param dt           reaction timestep
///
    int react_state(amrex::MultiFab& state,
                    amrex::MultiFab& reactions,
                    amrex::Real time,
                    amrex::Real dt,
                    const int strang_half);

///
/// Simplified SDC version of react_state. Reacts the current state through a single timestep.
///
/// @param time     current time
/// @param dt       timestep
///
    int react_state(amrex::Real time, amrex::Real dt);

///
/// Are there any zones in ``State`` that can burn?
///
/// @param State    State MultiFab
///
    static bool valid_zones_to_burn(const amrex::MultiFab& State);

#endif
