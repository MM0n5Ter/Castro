#ifndef HYDROSIM_BASEBOUNDARY_H
#define HYDROSIM_BASEBOUNDARY_H

#include <RadInterpBoundaryData.H>
#include <AMReX_BC_TYPES.H>

enum BoundaryMode { Zero_BC = 0, Nonzero_BC };

class BaseBoundary : public RadInterpBoundaryData
{

public:
    BaseBoundary(const amrex::BoxArray& _grids, const amrex::DistributionMapping& _dmap,
          int _ncomp, const amrex::Geometry& _geom) :
        RadInterpBoundaryData(_grids,_dmap,_ncomp,_geom) { }



  virtual void setBoundaryFluxConditions(const amrex::BCRec& bc,
                       const BoundaryMode bc_value_mode = Nonzero_BC) = 0;


  const amrex::Vector< std::unique_ptr<amrex::BaseFab<int> > >& boundaryTypes(const amrex::Orientation& _face) const {
    return bc_type_fabs[_face];
  }


  virtual int isMixedFace(const amrex::Orientation&  ) const {
    return 0;
  }

protected:

  amrex::Vector< std::unique_ptr<amrex::BaseFab<int> > > bc_type_fabs[2*AMREX_SPACEDIM];

};

class BaseBoundaryBuilder
{
public:
  virtual BaseBoundary* operator()(const amrex::BoxArray& _grids,
                              const amrex::DistributionMapping& _dmap,
                              int _ncomp,
                              const amrex::Geometry& _geom) const = 0;
};

#endif
