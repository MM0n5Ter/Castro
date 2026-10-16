#ifndef HYDROSIM_DERIVE_H
#define HYDROSIM_DERIVE_H

#include <AMReX_BLFort.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

namespace derived_fields
{

  void derive_pressure
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_eint_from_total
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_eint_from_rhoe
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_log_density
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_u_plus_cs
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_u_minus_cs
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_sound_speed
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_gamma1
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_mach
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

  void derive_conduction_term
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);
#endif

  void derive_energy_rate
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_burn_time_ratio
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_velocity
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_speed
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_grav_magnitude
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_radial_velocity
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_circular_velocity
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_momentum_magnitude
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_ang_mom_x (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                      const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
                      amrex::Real time, const int* bcrec, int level);

  void derive_ang_mom_y (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                      const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
                      amrex::Real time, const int* bcrec, int level);

  void derive_ang_mom_z (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                      const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
                      amrex::Real time, const int* bcrec, int level);

  void derive_kinetic_energy (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
                     amrex::Real time, const int* bcrec, int level);

  void derive_nothing
    (BL_FORT_FAB_ARG_3D(der),const int* nvar,
     const BL_FORT_FAB_ARG_3D(data),const int* ncomp,
     const int* lo, const int* hi,
     const int* domain_lo, const int* domain_hi,
     const amrex::Real* delta, const amrex::Real* xlo,
     const amrex::Real* time, const amrex::Real* dt, const int* bcrec,
     const int* level, const int* grid_no);

  void derive_species
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_mean_mass_number
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_electron_fraction
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_vorticity
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_div_velocity
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_state_copy
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

#ifdef MHD
  void derive_b_center_x
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_b_center_y
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_b_center_z
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void derive_div_b
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

#endif

} // namespace derived_fields

/* problem-specific includes */
#include <Problem_Derive.H>

#endif
