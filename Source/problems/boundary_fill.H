#ifndef BOUNDARY_FILL_H
#define BOUNDARY_FILL_H

#include <AMReX_BLFort.H>

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
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

void hs_statefill(amrex::Box const& bx, amrex::FArrayBox& data,
                  const int dcomp, const int numcomp,
                  amrex::Geometry const& geom, const amrex::Real time,
                  const amrex::Vector<amrex::BCRec>& bcr, const int bcomp,
                  const int scomp);

void
hydrostatic_fill(const amrex::Box& bx, amrex::Array4<amrex::Real> const& adv,
         amrex::Geometry const& geom, const amrex::Vector<amrex::BCRec>& bcr,
         const amrex::Real time);

void
ambient_region_fill(const amrex::Box& bx, amrex::Array4<amrex::Real> const& state,
             amrex::Geometry const& geom, const amrex::Vector<amrex::BCRec>& bcr);

void
ambient_density_fill(const amrex::Box& bx, amrex::Array4<amrex::Real> const& state,
                amrex::Geometry const& geom, const amrex::Vector<amrex::BCRec>& bcr);

#endif
