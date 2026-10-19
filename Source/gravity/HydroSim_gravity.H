#ifndef HYDROSIM_GRAVITY_H
#define HYDROSIM_GRAVITY_H

///
/// Construct gravitational field at old timestep
///
/// @param time             current time
///
    void assemble_old_gravity (amrex::Real time);


///
/// Construct gravitational field at new timestep
///
/// @param time             current time
///
    void assemble_new_gravity (amrex::Real time);

///
/// Construct gravitational source terms at old timestep
///
/// @param source   MultiFab to save sources to
/// @param state    Old state
/// @param time     current time
/// @param dt       timestep
///
    void assemble_old_gravity_source(amrex::MultiFab& source, amrex::MultiFab& state, amrex::Real time, amrex::Real dt);


///
/// Construct gravitational source terms at new timestep
///
/// @param source       MultiFab to save sources to
/// @param field_old    Old state
/// @param field_new    New state
/// @param time         current time
/// @param dt           timestep
///
    void assemble_new_gravity_source(amrex::MultiFab& source, amrex::MultiFab& field_old, amrex::MultiFab& field_new, amrex::Real time, amrex::Real dt);

#endif
