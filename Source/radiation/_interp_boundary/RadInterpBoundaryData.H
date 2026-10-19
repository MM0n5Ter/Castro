#ifndef HYDROSIM_RAD_INTERPBOUNDARYDATA_H
#define HYDROSIM_RAD_INTERPBOUNDARYDATA_H

#include <iostream>

#include <AMReX_BLassert.H>
#include <AMReX_BoxArray.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Orientation.H>
#include <AMReX_Mask.H>
#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>
#include <RadBoundaryCondition.H>
#include <RadBoundaryData.H>



class RadInterpBoundaryData : public RadBoundaryData
{
public:
    RadInterpBoundaryData() : RadBoundaryData() {};
    RadInterpBoundaryData(const amrex::BoxArray& _grids, const amrex::DistributionMapping& _dmap,
                       int _ncomp, const amrex::Geometry& geom);
    void setBoundaryConditions(const amrex::BCRec& phys_bc,
                       const amrex::Geometry& geom, int ratio);
    virtual void setBoundaryConditions(const amrex::BCRec& phys_bc,
                               const amrex::Geometry& geom, amrex::IntVect& ratio) = 0;
    void setBndryValues(const amrex::MultiFab& mf, int mf_start,
                        int bnd_start, int num_comp,
                        const amrex::BCRec& phys_bc);
    void setBndryValues(amrex::BndryRegister& crse, int c_start,
                        const amrex::MultiFab& fine, int f_start,
                        int bnd_start, int num_comp, amrex::IntVect& ratio,
                        const amrex::BCRec& phys_bc);
    void setBndryValues(amrex::Real bv);
};


#endif

