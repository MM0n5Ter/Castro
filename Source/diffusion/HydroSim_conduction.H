#ifndef HYDROSIM_CONDUCTION_H
#define HYDROSIM_CONDUCTION_H

    void assemble_old_conduction_source(amrex::MultiFab& source, amrex::MultiFab& state, amrex::Real time, amrex::Real dt);

    void assemble_new_conduction_source(amrex::MultiFab& source, amrex::MultiFab& field_old, amrex::MultiFab& field_new, amrex::Real time, amrex::Real dt);


void computeConductionTerm (amrex::Real time, amrex::MultiFab& state, amrex::MultiFab& ConductionTerm);


void add_conduction_to_source (amrex::MultiFab& external_source, amrex::MultiFab& state,
                                   amrex::MultiFab& ConductionTerm, amrex::Real t,
                                   amrex::Real term_weight = 1.0);


#endif
