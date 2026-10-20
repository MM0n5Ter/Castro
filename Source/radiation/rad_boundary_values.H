#ifndef HYDROSIM_RAD_BOUNDARY_VALUES_H
#define HYDROSIM_RAD_BOUNDARY_VALUES_H

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

class RadiativeBoundary : public BaseBoundary
{

public:
  RadiativeBoundary(const amrex::BoxArray& _grids, const amrex::DistributionMapping& _dmap,
           const amrex::Geometry& _geom);

  RadiativeBoundary(const amrex::BoxArray& _grids, const amrex::DistributionMapping& _dmap,
           const amrex::Geometry& _geom, amrex::Real bv);

  void setBoundaryConditions(const amrex::BCRec& phys_bc,
                     const amrex::Geometry& geom, amrex::IntVect& ratio) override;

  void setBoundaryFluxConditions(const amrex::BCRec& bc,
                         const BoundaryMode bc_value_mode = Nonzero_BC) override;

  int isMixedFace(const amrex::Orientation& _face) const override {
    return (bc_kind[_face] > 1) ? 1 : 0;
  }

  void setHomogValues(const amrex::BCRec& bc, amrex::IntVect& ratio);

  static void setTime(amrex::Real Time);
  static void enterCorrection();
  static void leaveCorrection();

protected:
  static void init();

  static void init(amrex::Real bv);

  static int first;
  static amrex::Vector<int> bc_kind;
  static amrex::Vector<amrex::Real> bcval;
  static amrex::Real time;
  static int correction;
};

inline
void
RadiativeBoundary::setTime(amrex::Real Time)
{
  time = Time;
}

inline
void
RadiativeBoundary::enterCorrection()
{
  correction = 1;
}

inline
void
RadiativeBoundary::leaveCorrection()
{
  correction = 0;
}

class RadiativeBoundaryBuilder : public BaseBoundaryBuilder
{
public:

  BaseBoundary* operator()(const amrex::BoxArray& _grids,
                      const amrex::DistributionMapping& _dmap,
                      int  ,
                      const amrex::Geometry& _geom) const override {
      return new RadiativeBoundary(_grids, _dmap, _geom);
  }
};

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
