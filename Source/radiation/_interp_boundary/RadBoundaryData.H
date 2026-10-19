#ifndef HYDROSIM_RAD_BOUNDARYDATA_H
#define HYDROSIM_RAD_BOUNDARYDATA_H

#include <AMReX_BLassert.H>
#include <AMReX_BoxArray.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Orientation.H>
#include <AMReX_BndryRegister.H>
#include <AMReX_Mask.H>
#include <RadBoundaryCondition.H>
#include <AMReX_Geometry.H>

#include <iostream>

class GeometryStandIn
{
protected:
    const amrex::Geometry& m_g;

public:
    GeometryStandIn(const amrex::Geometry& g) : m_g( g ) {}
    operator amrex::Geometry() const
    {  return m_g; }
};


class RadBoundaryData : public amrex::BndryRegister
{
public:

    enum MaskVal { covered = 0, not_covered = 1, outside_domain = 2 };

protected:
    amrex::Vector<RadBoundaryCondition>   bcond[2*AMREX_SPACEDIM];
    amrex::Vector<amrex::Real>    bcloc[2*AMREX_SPACEDIM];
    amrex::Vector<std::unique_ptr<amrex::Mask> > masks[2*AMREX_SPACEDIM];
    amrex::Geometry geom;

public:
    RadBoundaryData() : amrex::BndryRegister() {};

    ~RadBoundaryData() = default;

    RadBoundaryData(const RadBoundaryData& src) = delete;
    RadBoundaryData& operator = (const RadBoundaryData& src) = delete;
    RadBoundaryData(const RadBoundaryData&& src) = delete;
    RadBoundaryData& operator = (const RadBoundaryData&& src) = delete;

    RadBoundaryData(const amrex::BoxArray& _grids, const amrex::DistributionMapping& _dmap,
                 int _ncomp, const GeometryStandIn& geom);

    void define(const amrex::BoxArray& _grids, const amrex::DistributionMapping& _dmap,
                int _ncomp, const GeometryStandIn& geom);

    friend std::ostream& operator << (std::ostream&, const RadBoundaryData&);

    const amrex::FabSet& bndryValues(const amrex::Orientation& _face) const
    {
        return bndry[_face];
    }
    const amrex::Vector<amrex::Real>& bndryLocs(const amrex::Orientation& _face) const
    {
        return bcloc[_face];
    }
    const amrex::Vector<RadBoundaryCondition>& bndryConds(const amrex::Orientation& _face) const
    {
        return bcond[_face];
    }
    const amrex::Mask& bndryMasks (const amrex::Orientation& _face, int i) const
    {
        return *masks[_face][i];
    }
    const amrex::Box& getDomain() const
    {
        return geom.Domain();
    }
    const amrex::Geometry& getGeom() const
    {
        return geom;
    }
    void setValue(const amrex::Orientation& _face, int _n, amrex::Real _val)
    {
        bndry[_face][_n].setVal<amrex::RunOn::Host>(_val);
    }
    void assignFaceMask(const amrex::Orientation& _face, int _n, int _val)
    {
        masks[_face][_n]->setVal<amrex::RunOn::Host>(_val);
    }
    void setBoundCond(const amrex::Orientation& _face, int _n, const RadBoundaryCondition& _bcn)
    {
        bcond[_face][_n] = _bcn;
    }
    void setBoundLoc(const amrex::Orientation& _face, int _n, amrex::Real _val)
    {
        bcloc[_face][_n] = _val;
    }
    const amrex::FabSet &operator[](const amrex::Orientation &_face) const {
      return amrex::BndryRegister::bndry[_face];
    }
    amrex::FabSet &operator[](const amrex::Orientation &_face) {
      return amrex::BndryRegister::bndry[_face];
    }


};

#endif
