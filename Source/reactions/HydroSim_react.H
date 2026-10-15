#ifndef HYDROSIM_REACT_H
#define HYDROSIM_REACT_H

///
/// Perform any old-time reactions.
///
    advance_status handle_old_reactions (Real time, Real dt);

///
/// Perform any new-time reactions.
///
    advance_status handle_new_reactions (Real time, Real dt);

///
/// The half of the timestep that a split burn covers.
///
    enum class BurnHalf : std::uint8_t {
        First = 0,
        Second
    };

///
/// React state through a single reaction timestep, saving reaction source terms to ``reactions`` MultiFab. This is the CTU version.
///
/// @param state        Current state
/// @param reactions    MultiFab to save reaction sources to
/// @param time         current time
/// @param dt           reaction timestep
/// @param half         half of the timestep this burn covers
///
    int react_state(amrex::MultiFab& state,
                    amrex::MultiFab& reactions,
                    amrex::Real time,
                    amrex::Real dt,
                    const BurnHalf half);

///
/// Simplified SDC version of react_state. Reacts the current state through a single timestep.
///
/// @param time     current time
/// @param dt       timestep
///
    int react_state(amrex::Real time, amrex::Real dt);

#endif
