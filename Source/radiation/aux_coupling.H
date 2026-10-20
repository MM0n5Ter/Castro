#ifndef HYDROSIM_AUX_COUPLING_H
#define HYDROSIM_AUX_COUPLING_H

#include <list>
#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <HYPRE_sstruct_ls.h>

class AuxVar {

 protected:

  class AuxLink {
   public:
    AuxLink() :
      other(nullptr)
    {}

    AuxLink(AuxVar* p, amrex::Real r) :
        val(r), other(p)
    {}

    AuxLink(int lev, const amrex::IntVect& iv, amrex::Real r) :
        val(r), index(iv), level(lev), other(nullptr)
    {}

    [[nodiscard]] bool same_destination(const AuxLink& c) const {
      return ((other != nullptr)
              ? (other == c.other)
              : (c.other == nullptr && level == c.level && index == c.index));
    }

    amrex::Real val;
    amrex::IntVect index;
    int level;
    AuxVar *other;
  };

 public:

  AuxVar() = default;


  void add_secondary_link(AuxVar* p) {

    if (p->secondary_pushed == 0) {
      a.emplace_back(p, 1.0);
      p->secondary_pushed = 1;
    }
  }


  bool empty() {
    return a.empty();
  }


  [[nodiscard]] bool secondary() const {
    return secondary_pushed;
  }

  void push(AuxVar* p, amrex::Real r) {
    a.emplace_back(p, r);
  }

  void push(int lev, const amrex::IntVect& iv, amrex::Real r) {
    a.emplace_back(lev, iv, r);
  }

  void collapse();


  int read_stencil_sites(amrex::Vector<int>& levels, amrex::Vector<amrex::IntVect>& cells);


  int read_stencil_values(amrex::Vector<amrex::Real>& values);

  void clear();

 protected:

  std::list<AuxLink> a;
  int secondary_pushed{};
};

class BoxAuxStorage {

 public:


  BoxAuxStorage(const amrex::Box& bx) : domain(bx) {
    auto numpts = domain.numPts();
    dptr = new AuxVar[numpts];
  }

  ~BoxAuxStorage() {
    delete[] dptr;
  }

  BoxAuxStorage(const BoxAuxStorage& src) = delete;
  BoxAuxStorage(const BoxAuxStorage&& src) = delete;

  BoxAuxStorage& operator= (const BoxAuxStorage& src) = delete;
  BoxAuxStorage& operator= (const BoxAuxStorage&& src) = delete;

  AuxVar& operator()(const amrex::IntVect& p) {
    BL_ASSERT(!(dptr == 0));
    BL_ASSERT(domain.contains(p));
    return dptr[domain.index(p)];
  }

  [[nodiscard]] const amrex::Box& box() const {
    return domain;
  }

 protected:
  amrex::Box domain;
  AuxVar *dptr;
};

class BoundaryAuxBase {

 public:


  BoundaryAuxBase (const amrex::DistributionMapping& dm)
      : distributionMap(dm) {}

  enum Location { INTERIOR, OUTSIDE_GRID, HALO };

  int firstOwnedIndex();

  int nextOwnedIndex(int i);

  bool isValid(int i) {
    return (i >= 0 && i < distributionMap.ProcessorMap().size());
  }

 protected:

  amrex::DistributionMapping        distributionMap;
};

class BoundaryAuxStore : public BoundaryAuxBase {

 public:


  BoundaryAuxStore(const amrex::BoxArray& _grids,
              const amrex::DistributionMapping& _dmap,
              Location loc);


  BoxAuxStorage & operator() (const amrex::Orientation ori, int i) {
    return *aux[ori][i];
  }

 protected:

  amrex::BoxArray            grids;
  amrex::Vector<std::unique_ptr<BoxAuxStorage> >   aux[2*AMREX_SPACEDIM];
};

class CoarseBoundaryAuxStore : public BoundaryAuxBase {

 public:


  CoarseBoundaryAuxStore(const amrex::BoxArray& _coarse_boxes,
                  const amrex::DistributionMapping& _coarse_dmap,
                  const amrex::BoxArray& _fine_boxes, Location loc);


  CoarseBoundaryAuxStore(const CoarseBoundaryAuxStore& other, Location loc);


  CoarseBoundaryAuxStore(const amrex::BoxArray& _coarse_boxes,
                  const amrex::BoxArray& _fine_boxes,
                  const CoarseBoundaryAuxStore& other, Location loc);


  void rebuild_links(Location loc);


  void makeFaceStorage(amrex::IntVect& rat, int ncomp = 1);


  void resetFaceStorage(amrex::IntVect& rat, int ncomp = 1);


  void storeFaceCoeffs(const amrex::Orientation ori,
                    amrex::MultiFab& src,
                    int srccomp,
                    int destcomp,
                    int numcomp = 1);



  int size (const amrex::Orientation ori, int i) const {
      return static_cast<int>(aux[ori][i].size());
  }

  BoxAuxStorage& operator() (const amrex::Orientation ori, int i, int j) {
    return *aux[ori][i][j];
  }


  amrex::Mask& mask (const amrex::Orientation ori, int i, int j) {
    return *msk[ori][i][j];
  }


  amrex::FArrayBox& faceFab (const amrex::Orientation ori, int i, int j) {
    return *face_fabs[ori][i][j];
  }

 protected:


  void link_secondary_entries(Location loc);

  amrex::BoxArray                    coarse_boxes, fgrids;
  amrex::Vector< amrex::Vector<std::unique_ptr<BoxAuxStorage> > >  aux[2*AMREX_SPACEDIM];
  amrex::Vector< amrex::Vector<std::unique_ptr<amrex::Mask> > >       msk[2*AMREX_SPACEDIM];
  amrex::Vector< amrex::Vector<int> >                          fine_map[2*AMREX_SPACEDIM];
  amrex::Vector< amrex::Vector<std::unique_ptr<amrex::FArrayBox> > >  face_fabs[2*AMREX_SPACEDIM];
};

#endif
