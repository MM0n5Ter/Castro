#ifndef HYDROSIM_MULTIGROUPBOUNDARY_H
#define HYDROSIM_MULTIGROUPBOUNDARY_H

#include <BaseBoundary.H>

class MultiGroupBoundary : public BaseBoundary
{

public:
  MultiGroupBoundary(const amrex::BoxArray& _grids,
             const amrex::DistributionMapping& _dmap,
             const int _num_groups,
             const amrex::Geometry& _geom);

  void setBoundaryConditions(const amrex::BCRec& phys_bc,
                     const amrex::Geometry& geom, amrex::IntVect& ratio) override;


  void setBoundaryFluxConditions(const amrex::BCRec& bc,
                         const BoundaryMode bc_value_mode = Nonzero_BC) override;


   int isMixedFace(const amrex::Orientation& _face) const override {
       return (bc_kind[_face] > 1) ? 1 : 0;
  }


  void setHomogValues(const amrex::BCRec& bc, amrex::IntVect& ratio);


  static void setTime(amrex::Real Time) {
    time = Time;
  }

  static void enterCorrection() {
    correction = 1;
  }
  static void leaveCorrection() {
    correction = 0;
  }


protected:
  static void init(const int _num_groups);
  static int ngroups;

  static int first;
  static amrex::Vector<int> bc_kind;
  static amrex::Vector< amrex::Vector<amrex::Real> > bcval;
  static amrex::Real time;
  static int correction;
};

class MultiGroupBoundaryBuilder : public BaseBoundaryBuilder
{
public:
  BaseBoundary* operator()(const amrex::BoxArray& _grids,
                      const amrex::DistributionMapping& _dmap,
                      int _num_groups,
                      const amrex::Geometry& _geom) const override {

      return new MultiGroupBoundary(_grids, _dmap, _num_groups, _geom);
  }
};

#endif
