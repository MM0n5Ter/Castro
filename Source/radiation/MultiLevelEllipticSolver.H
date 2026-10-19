#ifndef HYDROSIM_MULTILEVELELLIPTICSOLVER_H
#define HYDROSIM_MULTILEVELELLIPTICSOLVER_H

#include <AlphaBetaSolver.H>

#include <list>
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


  void push(AuxVar* p, amrex::Real r) {
    a.emplace_back(p, r);
  }


  void push(int lev, const amrex::IntVect& iv, amrex::Real r) {
    a.emplace_back(lev, iv, r);
  }


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

class MultiLevelEllipticSolver {

 public:

  MultiLevelEllipticSolver(int _coarse_lvl, int _fine_lvl, int _solver_choice);

  virtual ~MultiLevelEllipticSolver();

  MultiLevelEllipticSolver(const MultiLevelEllipticSolver& src) = delete;
  MultiLevelEllipticSolver(const MultiLevelEllipticSolver&& src) = delete;

  MultiLevelEllipticSolver operator=(const MultiLevelEllipticSolver& src) = delete;
  MultiLevelEllipticSolver operator=(const MultiLevelEllipticSolver&& src) = delete;

  void attachLevel(int level,
                const amrex::Geometry& _geom,
                const amrex::BoxArray& _grids,
                const amrex::DistributionMapping& _dmap,
                amrex::IntVect         _coarse_ratio);

  [[nodiscard]] int crseLevel() const {
    return crse_level;
  }
  [[nodiscard]] int fineLevel() const {
    return fine_level;
  }


  void setBndry(int level, BaseBoundary& _bd, int _comp = 0) {
      bd[level] = &_bd;
      bc_component = _comp;
  }

  BaseBoundary& levelBoundaries(int level) {
    return *bd[level];
  }
  static amrex::Real& fluxScale() {
    return flux_scale;
  }


  static void faceGeometryFactors(amrex::Vector<amrex::Real>& r,
                            const amrex::Box& reg,
                            const amrex::Orientation& ori,
                            const amrex::Geometry& geom) {

    AlphaBetaSolver::faceGeometryFactors(r, reg, ori, geom);
  }


  void setVerbose(int v) {
    verbose = v;
  }

  void makeMatrixLayout();


  void setScalars(amrex::Real alpha, amrex::Real beta);

  [[nodiscard]] amrex::Real getAlphaScalar() const {
    return alpha;
  }
  [[nodiscard]] amrex::Real getBetaScalar() const {
    return beta;
  }


  void setMassCoefficients(int level, const amrex::MultiFab &a);

  void setFaceCoefficients(int level, const amrex::MultiFab &b, int dir);


  void setBcAlphaCoef(int level, const amrex::MultiFab &Spa);


  const amrex::MultiFab& setMassCoefficients(int level) {
    return *acoefs[level];
  }

  const amrex::MultiFab& setFaceCoefficients(int level, int dir) {
      return (*bcoefs[level])[dir];
  }

  void hmac (const amrex::Box& bx,
             amrex::Array4<amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
             amrex::Array4<amrex::Real const> const& a,
             amrex::Real alpha);

  void fillBoundaryCoeffs (const amrex::Box& bx,
             amrex::Array4<amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
             amrex::Array4<amrex::Real const> const& b,
             amrex::Real beta, const amrex::Real* dx, int n);

  void fillLevelMat (const amrex::Box& bx,
              amrex::Array4<amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
              int cdir, int bct, int bho, amrex::Real bcl,
              amrex::Array4<int const> const& mask,
              amrex::Array4<amrex::Real const> const& b,
              amrex::Real beta, const amrex::Real* dx);

  void fillLevelMat3 (const amrex::Box& bx,
               int side_is_low, int idir,
               amrex::Array4<amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
               int cdir, int bctype,
               amrex::Array4<int const> const& tf,
               int bho, amrex::Real bcl,
               amrex::Array4<int const> const& mask,
               amrex::Array4<amrex::Real const> const& b,
               amrex::Real beta, const amrex::Real* dx,
               const amrex::GeometryData& geomdata, amrex::Real c,
               amrex::Array4<amrex::Real const> const& spa);

  virtual void assembleMatrix();
  void commitMatrix();

  virtual void fillLevelVectors(int level,
                                amrex::MultiFab& dest,
                                int icomp,
                                amrex::MultiFab& rhs,
                                BoundaryMode bc_variant);

          void loadLevelSolution(int level,
                                amrex::MultiFab& dest,
                                int icomp);
  virtual void fillLevelRhs(int level,
                                amrex::MultiFab& rhs,
                                BoundaryMode bc_variant);

  void commitVectors();

  void configureSolver(amrex::Real _rel_tolerance, amrex::Real _abs_tolerance, int maxiter);
  void solve();

  void getSolution(int level, amrex::MultiFab& dest, int icomp);

  amrex::Real currentResidualNorm();

  void releaseSolver();


  void faceFluxAtBoundary(int level,
                    amrex::MultiFab* Flux,
                    amrex::MultiFab& Er,
                    int icomp,
                    BoundaryMode bc_variant);


  void getProduct(int level, amrex::MultiFab& prod);

 protected:

  int crse_level, fine_level, solver_choice;

  amrex::Vector<amrex::Geometry> geom;
  amrex::Vector<amrex::BoxArray> grids;
  amrex::Vector<amrex::DistributionMapping> dmap;
  amrex::Vector<amrex::IntVect> fine_ratio;
  amrex::Vector<BaseBoundary*> bd;
  int bc_component;

  amrex::Vector< amrex::Vector<amrex::BoxArray> > subgrids;

  amrex::Vector<std::unique_ptr<amrex::MultiFab> > acoefs;
  amrex::Vector<std::unique_ptr<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> > > bcoefs;
  amrex::Real alpha, beta;
  amrex::Real reltol, abstol;

  amrex::Vector<std::unique_ptr<amrex::MultiFab> > SPa;

  amrex::Vector<std::unique_ptr<BoundaryAuxStore> >    coarse_interp, normal_grad_store;
  amrex::Vector<std::unique_ptr<CoarseBoundaryAuxStore> > c_interp, c_extrap_deriv, c_connections;

  int verbose, log_threshold, bho, subgrid_mode;

  HYPRE_SStructGrid     hgrid;
  HYPRE_SStructStencil  stencil;
  HYPRE_SStructGraph    graph;
  HYPRE_SStructMatrix   A, A0;
  HYPRE_SStructVector   b;
  HYPRE_SStructVector   x;
  HYPRE_SStructSolver   solver_handle;
  HYPRE_SStructSolver   split_preconditioner;
  HYPRE_Solver          solver;
  HYPRE_Solver          precond;
  int                   ObjectType;

  static amrex::Real flux_scale;



  static void writeVectorBox(HYPRE_SStructVector x,
                                 int part,
                                 const amrex::Box& reg,
                                 const amrex::BoxArray& sgr,
                                 amrex::Real *vec);


  static void readVectorBox(HYPRE_SStructVector x,
                                 int part,
                                 const amrex::Box& reg,
                                 const amrex::BoxArray& sgr,
                                 amrex::FArrayBox& f, int fcomp);

#if (AMREX_SPACEDIM == 1)
  static int vl[2];
  static int vh[2];
#endif

  static int* cellIndex1(const amrex::IntVect& i) {
#if (AMREX_SPACEDIM == 1)
    vl[0] = i[0];
    return vl;
#else

    return (int*) i.getVect();
#endif
  }


  static int* cellIndex2(const amrex::IntVect& i) {
#if (AMREX_SPACEDIM == 1)
    vh[0] = i[0];
    return vh;
#else

    return (int*) i.getVect();
#endif
  }


  static int* loV(const amrex::Box& b) {
#if (AMREX_SPACEDIM == 1)
    vl[0] = b.smallEnd(0);
    return vl;
#else

    return (int*) b.loVect();
#endif
  }


  static int* hiV(const amrex::Box& b) {
#if (AMREX_SPACEDIM == 1)
    vh[0] = b.bigEnd(0);
    return vh;
#else

    return (int*) b.hiVect();
#endif
  }

};

#endif
