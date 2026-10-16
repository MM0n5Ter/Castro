#ifndef HYDROSIM_RADDERIVE_H
#define HYDROSIM_RADDERIVE_H

#include <AMReX_BLFort.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

namespace derived_fields
{

  void hs_derertot(const amrex::Box& bx, amrex::FArrayBox& derfab, int /*dcomp*/, int /*ncomp*/,
                   const amrex::FArrayBox& datfab, const amrex::Geometry& /*geomdata*/,
                   amrex::Real /*time*/, const int* /*bcrec*/, int /*level*/);


} // namespace derived_fields

#endif
