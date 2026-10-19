#ifndef HYDROSIM_CONDUCTION_H
#define HYDROSIM_CONDUCTION_H

///
/// Construct diffusion source at old time
///
/// @param source   MultiFab to save source terms to
/// @param state    Old state
/// @param time     current time
/// @param dt       timestep
///
    void assemble_old_conduction_source(amrex::MultiFab& source, amrex::MultiFab& state, amrex::Real time, amrex::Real dt);

///
/// Construct diffusion source at new time
///
/// @param source       MultiFab to save source terms to
/// @param field_old    Old state
/// @param field_new    New state
/// @param time         current time
/// @param dt           timestep
///
    void assemble_new_conduction_source(amrex::MultiFab& source, amrex::MultiFab& field_old, amrex::MultiFab& field_new, amrex::Real time, amrex::Real dt);


///
/// Get thermal conductivity diffusion term at given time
///
/// @param time     current time
/// @param state    Current state
/// @param ConductionTerm MultiFab to save term to
///
void computeConductionTerm (amrex::Real time, amrex::MultiFab& state, amrex::MultiFab& ConductionTerm);


///
/// Calculate temperature or enthalpty diffusion terms and add to ``external_source`` (multiplied by ``term_weight``).
///
/// @param external_source      Source terms to add diffusion sources to
/// @param state        Current state
/// @param ConductionTerm     MultiFab to save diffusion sources to
/// @param t            Current time
/// @param term_weight  Real, factor to multiply diffusion
///                     sources by before adding to other sources.
///
void add_conduction_to_source (amrex::MultiFab& external_source, amrex::MultiFab& state,
                                   amrex::MultiFab& ConductionTerm, amrex::Real t,
                                   amrex::Real term_weight = 1.0);


#endif
