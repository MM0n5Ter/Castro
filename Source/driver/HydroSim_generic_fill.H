#ifndef HYDROSIM_GENERIC_FILL_H
#define HYDROSIM_GENERIC_FILL_H

#include <AMReX_BLFort.H>
#include <HydroSim.H>

#ifdef AMREX_USE_GPU
#include <AMReX_Arena.H>
#endif

struct HydroSimGenericFill
{
    AMREX_GPU_DEVICE
    void operator() (const IntVect&  , Array4<Real> const&  ,
                     const int  , const int  ,
                     GeometryData const&  , const Real  ,
                     const BCRec*  , const int  ,
                     const int  ) const
        {
        }
};

void hs_generic_fill(amrex::Box const& bx, amrex::FArrayBox& data,
                     const int dcomp, const int numcomp,
                     amrex::Geometry const& geom, const amrex::Real time,
                     const amrex::Vector<amrex::BCRec>& bcr, const int bcomp,
                     const int scomp);

#endif
