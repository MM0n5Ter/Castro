#ifndef HYDROSIM_RADIATIVEDERIVE_H
#define HYDROSIM_RADIATIVEDERIVE_H

#include <AMReX_BLFort.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

#ifdef __cplusplus
extern "C"
{
#endif

  void derive_total_radiative_energy(const amrex::Box& bx, amrex::FArrayBox& derfab, int  , int  ,
                   const amrex::FArrayBox& datfab, const amrex::Geometry&  ,
                   amrex::Real  , const int*  , int  );


#ifdef __cplusplus
}
#endif

#endif
