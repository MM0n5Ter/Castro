#ifndef HYDROSIM_GRAVITY_H
#define HYDROSIM_GRAVITY_H

///
/// Construct gravitational field at old timestep
///
/// @param time             current time
///
    void assemble_old_gravity (amrex::Real time);


///
/// Whether the level Poisson solve on this level is corrected by the
/// difference between the composite and level solutions
///
    bool applies_composite_correction () const;


///
/// Construct gravitational field at new timestep
///
/// @param time             current time
///
    void assemble_new_gravity (amrex::Real time);

///
/// Construct gravitational source terms at old timestep
///
/// @param state    Old state
/// @param time     current time
/// @param dt       timestep
/// @param source   MultiFab to save sources to
///
    void assemble_old_gravity_source(amrex::MultiFab& state, amrex::Real time, amrex::Real dt, amrex::MultiFab& source);


///
/// Construct gravitational source terms at new timestep
///
/// @param state_old    Old state
/// @param state_new    New state
/// @param time         current time
/// @param dt           timestep
/// @param source       MultiFab to save sources to
///
    void assemble_new_gravity_source(amrex::MultiFab& state_old, amrex::MultiFab& state_new, amrex::Real time, amrex::Real dt, amrex::MultiFab& source);

#endif
