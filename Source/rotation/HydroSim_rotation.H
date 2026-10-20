#ifndef HYDROSIM_ROTATION_H
#define HYDROSIM_ROTATION_H

    void assemble_old_rotation_source(amrex::MultiFab& source, amrex::MultiFab& state, amrex::Real time, amrex::Real dt);


    void assemble_new_rotation_source(amrex::MultiFab& source, amrex::MultiFab& field_old, amrex::MultiFab& field_new, amrex::Real time, amrex::Real dt);


    void fill_centrifugal_potential_field(amrex::MultiFab& phi, amrex::MultiFab& state, amrex::Real time);

    void
    rsrc(const Box& bx,
         Array4<Real const> const& uold,
         const Real dt,
         Array4<Real> const& source);

    struct face_mass_flux_views {
        Array4<Real const> x, y, z;
    };

    void
    apply_rotation_correction(const Box& bx,
             Array4<Real const> const& uold,
             Array4<Real const> const& unew,
             const face_mass_flux_views& fluxes,
             Array4<Real const> const& vol,
             const Real dt,
             Array4<Real> const& source);

    void
    fill_centrifugal_potential(const Box& bx,
                              Array4<Real> const& phi,
                              const Real time);


    void
    fill_scaled_centrifugal_potential(const Box& bx,
                        Array4<Real> const& psi,
                        const Real time);
#endif
