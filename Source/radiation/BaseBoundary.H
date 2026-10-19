#ifndef HYDROSIM_BASEBOUNDARY_H
#define HYDROSIM_BASEBOUNDARY_H

#include <RadInterpBoundaryData.H>
#include <AMReX_BC_TYPES.H>

enum BoundaryMode { Zero_BC = 0, Nonzero_BC };

///
/// @class BaseBoundary
/// @brief
///
class BaseBoundary : public RadInterpBoundaryData
{

///
/// @param _grids
/// @param _dmap
/// @param _ncomp
/// @param _geom
///
public:
    BaseBoundary(const amrex::BoxArray& _grids, const amrex::DistributionMapping& _dmap,
          int _ncomp, const amrex::Geometry& _geom) :
        RadInterpBoundaryData(_grids,_dmap,_ncomp,_geom) { }



///
/// @param bc
/// @param bc_value_mode
///
  virtual void setBoundaryFluxConditions(const amrex::BCRec& bc,
                       const BoundaryMode bc_value_mode = Nonzero_BC) = 0;


///
/// @param _face
///
  const amrex::Vector< std::unique_ptr<amrex::BaseFab<int> > >& boundaryTypes(const amrex::Orientation& _face) const {
    return bc_type_fabs[_face];
  }


///
/// @param _face
///
  virtual int isMixedFace(const amrex::Orientation& /* _face */) const {
    return 0;
  }

protected:

///
/// If used, these arrays must be built/deleted by a derived class:
///
  amrex::Vector< std::unique_ptr<amrex::BaseFab<int> > > bc_type_fabs[2*AMREX_SPACEDIM];

};

///
/// @class BaseBoundaryBuilder
/// @brief
///
class BaseBoundaryBuilder
{
public:
  virtual BaseBoundary* operator()(const amrex::BoxArray& _grids,
                              const amrex::DistributionMapping& _dmap,
                              int _ncomp,
                              const amrex::Geometry& _geom) const = 0;
};

#endif
