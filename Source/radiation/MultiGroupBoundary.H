#ifndef HYDROSIM_MULTIGROUPBOUNDARY_H
#define HYDROSIM_MULTIGROUPBOUNDARY_H

#include <BaseBoundary.H>

///
/// @class MultiGroupBoundary
/// @brief
///
class MultiGroupBoundary : public BaseBoundary
{

///
/// @param _grids
/// @param _dmap
/// @param _num_groups
/// @param _geom
///
public:
  MultiGroupBoundary(const amrex::BoxArray& _grids,
             const amrex::DistributionMapping& _dmap,
             const int _num_groups,
             const amrex::Geometry& _geom);

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


///
/// @param _num_groups
///
protected:
  static void init(const int _num_groups);
  static int ngroups;

  static int first;            ///< only set up bcval once
  static amrex::Vector<int> bc_kind;    ///< 0 = use constant bcval; 1 = use function
  static amrex::Vector< amrex::Vector<amrex::Real> > bcval;
  static amrex::Real time;            ///< time value passed to bc function
  static int correction;       ///< ignore bcval when this is set
};

///
/// @class MultiGroupBoundaryBuilder
/// @brief
///
class MultiGroupBoundaryBuilder : public BaseBoundaryBuilder
{
public:
  BaseBoundary* operator()(const amrex::BoxArray& _grids,
                      const amrex::DistributionMapping& _dmap,
                      int _num_groups,
                      const amrex::Geometry& _geom) const override {

///
/// @param _grids
/// @param _dmap
/// @param _num_groups
/// @param _geom
///
      return new MultiGroupBoundary(_grids, _dmap, _num_groups, _geom);
  }
};

#endif
