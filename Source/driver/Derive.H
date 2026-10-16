#ifndef HYDROSIM_DERIVE_H
#define HYDROSIM_DERIVE_H

#include <AMReX_BLFort.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

namespace derived_fields
{

  void hs_derpres
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_dereint1
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_dereint2
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_derlogden
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_deruplusc
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_deruminusc
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_dersoundspeed
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_dergamma1
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_dermachnumber
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_derentropy
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

#ifdef DIFFUSION
  void hs_dercond
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_derdiffcoeff
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_derdiffterm
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);
#endif

  void hs_derenuc
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_derenuctimescale
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_dervel
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_dermagvel
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_dermaggrav
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_derradialvel
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_dercircvel
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_dermagmom
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_derangmomx (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                      const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
                      amrex::Real time, const int* bcrec, int level);

  void hs_derangmomy (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                      const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
                      amrex::Real time, const int* bcrec, int level);

  void hs_derangmomz (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                      const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
                      amrex::Real time, const int* bcrec, int level);

  void hs_derkineng (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
                     amrex::Real time, const int* bcrec, int level);

  void hs_dernull
    (BL_FORT_FAB_ARG_3D(der),const int* nvar,
     const BL_FORT_FAB_ARG_3D(data),const int* ncomp,
     const int* lo, const int* hi,
     const int* domain_lo, const int* domain_hi,
     const amrex::Real* delta, const amrex::Real* xlo,
     const amrex::Real* time, const amrex::Real* dt, const int* bcrec,
     const int* level, const int* grid_no);

  void hs_derspec
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_derabar
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_derye
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_dermagvort
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_derdivu
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_derstate
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

#ifdef MHD
  void hs_dermagcenx
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_dermagceny
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_dermagcenz
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

  void hs_derdivb
    (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
     const amrex::FArrayBox& datfab, const amrex::Geometry& geom,
     amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);

#endif

} // namespace derived_fields

/* problem-specific includes */
#include <Problem_Derive.H>

#endif
