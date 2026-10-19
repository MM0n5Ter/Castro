#ifndef HYDROSIM_DERIVE_H
#define HYDROSIM_DERIVE_H

#include <AMReX_BLFort.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

#ifdef __cplusplus
extern "C"
{
#endif

  void derive_pressure
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_internal_from_total
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_internal_from_stored
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_log_density
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_u_plus_c
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_u_minus_c
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_sound_speed
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_adiabatic_index
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_mach_ratio
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_entropy
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

#ifdef DIFFUSION
  void derive_conductivity
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_diffusivity
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_diffusion_term
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);
#endif

  void derive_energy_generation
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_timescale_ratio
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_velocity_component
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_speed
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_gravity_magnitude
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_radial_velocity
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_circular_speed
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_momentum_magnitude
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_ang_momentum_x (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                      const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
                      amrex::Real time, const int* bcrec, int level);

  void derive_ang_momentum_y (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                      const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
                      amrex::Real time, const int* bcrec, int level);

  void derive_ang_momentum_z (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                      const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
                      amrex::Real time, const int* bcrec, int level);

  void derive_kinetic_energy (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
                     amrex::Real time, const int* bcrec, int level);

  void derive_noop
    (BL_FORT_FAB_ARG_3D(der),const int* nvar,
     const BL_FORT_FAB_ARG_3D(data),const int* ncomp,
     const int* lo, const int* hi,
     const int* domain_lo, const int* domain_hi,
     const amrex::Real* delta, const amrex::Real* xlo,
     const amrex::Real* time, const amrex::Real* dt, const int* bcrec,
     const int* level, const int* grid_no);

  void derive_mass_fraction
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_atomic_mass
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_elec_frac
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_vorticity
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_flow_divergence
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_state_error
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

#ifdef MHD
  void derive_cell_bfield_x
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_cell_bfield_y
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_cell_bfield_z
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_b_divergence
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

#endif

#ifdef __cplusplus
}
#endif

/* problem-specific includes */
#include <User_Derive.H>

#endif
