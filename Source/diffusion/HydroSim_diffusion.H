#ifndef HYDROSIM_DIFFUSION_H
#define HYDROSIM_DIFFUSION_H

///
/// Construct diffusion source at old time
///
/// @param state    Old state
/// @param time     current time
/// @param dt       timestep
/// @param source   MultiFab to save source terms to
///
    void assemble_old_diff_source(amrex::MultiFab& state, amrex::Real time, amrex::Real dt, amrex::MultiFab& source);

///
/// Construct diffusion source at new time
///
/// @param state_old    Old state
/// @param state_new    New state
/// @param time         current time
/// @param dt           timestep
/// @param source       MultiFab to save source terms to
///
    void assemble_new_diff_source(amrex::MultiFab& state_old, amrex::MultiFab& state_new, amrex::Real time, amrex::Real dt, amrex::MultiFab& source);


///
/// Get thermal conductivity diffusion term at given time
///
/// @param time     current time
/// @param state    Current state
/// @param DiffTerm MultiFab to save term to
///
void computeConductionTerm (amrex::Real time, amrex::MultiFab& state, amrex::MultiFab& DiffTerm);


///
/// Calculate temperature or enthalpty diffusion terms and add to ``ext_src`` (multiplied by ``mult_factor``).
///
/// @param state        Current state
/// @param t            Current time
/// @param DiffTerm     MultiFab to save diffusion sources to
/// @param ext_src      Source terms to add diffusion sources to
/// @param mult_factor  Real, factor to multiply diffusion
///                     sources by before adding to other sources.
///
void add_conduction_to_source (amrex::MultiFab& state, amrex::Real t,
                                   amrex::MultiFab& DiffTerm, amrex::MultiFab& ext_src,
                                   amrex::Real mult_factor = 1.0);


#endif
