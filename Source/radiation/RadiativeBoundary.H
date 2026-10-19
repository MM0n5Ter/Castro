#ifndef HYDROSIM_RADIATIVEBOUNDARY_H
#define HYDROSIM_RADIATIVEBOUNDARY_H

#include <BaseBoundary.H>

///
/// @class RadiativeBoundary
/// @brief
///
class RadiativeBoundary : public BaseBoundary
{

///
/// @param _grids
/// @param _dmap
/// @param _geom
///
public:
  RadiativeBoundary(const amrex::BoxArray& _grids, const amrex::DistributionMapping& _dmap,
           const amrex::Geometry& _geom);

///
/// @param _grids
/// @param _dmap
/// @param _geom
/// @param bv
///
  RadiativeBoundary(const amrex::BoxArray& _grids, const amrex::DistributionMapping& _dmap,
           const amrex::Geometry& _geom, amrex::Real bv);


///
/// @param phys_bc
/// @param geom
/// @param ratio
///
  void setBoundaryConditions(const amrex::BCRec& phys_bc,
                     const amrex::Geometry& geom, amrex::IntVect& ratio) override;


///
/// @param bc
/// @param bc_value_mode
///
  void setBoundaryFluxConditions(const amrex::BCRec& bc,
                         const BoundaryMode bc_value_mode = Nonzero_BC) override;


///
/// @param _face
///
  int isMixedFace(const amrex::Orientation& _face) const override {
    return (bc_kind[_face] > 1) ? 1 : 0;
  }


///
/// @param bc
/// @param ratio
///
  void setHomogValues(const amrex::BCRec& bc, amrex::IntVect& ratio);


///
/// @param Time
///
  static void setTime(amrex::Real Time) {
    time = Time;
  }
///
/// @warning hidden state change, use carefully!
///
  static void enterCorrection() {
    correction = 1;
  }
  static void leaveCorrection() {
    correction = 0;
  }

protected:
  static void init();

///
/// @param bv
///
  static void init(amrex::Real bv);

  static int first;            ///< only set up bcval once
  static amrex::Vector<int> bc_kind;    ///< 0 = use constant bcval; 1 = use function
  static amrex::Vector<amrex::Real> bcval;    ///< constant bcval across each face
  static amrex::Real time;            ///< time value passed to bc function
  static int correction;       ///< ignore bcval when this is set
};

///
/// @class RadiativeBoundaryBuilder
/// @brief
///
class RadiativeBoundaryBuilder : public BaseBoundaryBuilder
{
public:

///
/// @param _grids
/// @param _dmap
/// @param _ncomp
/// @param _geom
///
  BaseBoundary* operator()(const amrex::BoxArray& _grids,
                      const amrex::DistributionMapping& _dmap,
                      int /* _ncomp */,
                      const amrex::Geometry& _geom) const override {
      return new RadiativeBoundary(_grids, _dmap, _geom);
  }
};

#endif
