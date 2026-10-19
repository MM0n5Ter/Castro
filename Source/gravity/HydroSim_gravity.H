#ifndef HYDROSIM_GRAVITY_H
#define HYDROSIM_GRAVITY_H

    void assemble_old_gravity (amrex::Real time);


    void assemble_new_gravity (amrex::Real time);

    void assemble_old_gravity_source(amrex::MultiFab& source, amrex::MultiFab& state, amrex::Real time, amrex::Real dt);


    void assemble_new_gravity_source(amrex::MultiFab& source, amrex::MultiFab& field_old, amrex::MultiFab& field_new, amrex::Real time, amrex::Real dt);

#endif
