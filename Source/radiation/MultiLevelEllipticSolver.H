#ifndef HYDROSIM_MULTILEVELELLIPTICSOLVER_H
#define HYDROSIM_MULTILEVELELLIPTICSOLVER_H

#include <AlphaBetaSolver.H>

#include <list>
#include <HYPRE_sstruct_ls.h>

///
/// An AuxVar is an intermediate variable used in building up a
/// linear system, defined as a linear combination of other
/// variables which are either true independent degrees of freedom
/// or other AuxVars.
///
/// @class AuxVar
/// @brief
///
class AuxVar {

 protected:

  class AuxLink {
   public:
    AuxLink() :
      other(nullptr)
    {}

///
/// @param p
/// @param r
///
    AuxLink(AuxVar* p, amrex::Real r) :
        val(r), other(p)
    {}

///
/// @param lev
/// @param iv
/// @param r
///
    AuxLink(int lev, const amrex::IntVect& iv, amrex::Real r) :
        val(r), index(iv), level(lev), other(nullptr)
    {}

///
/// @param c
///
    [[nodiscard]] bool same_destination(const AuxLink& c) const {
      return ((other != nullptr)
              ? (other == c.other)
              : (c.other == nullptr && level == c.level && index == c.index));
    }

    amrex::Real val;
    amrex::IntVect index;
    int level; //, grid; // may not need grid
    AuxVar *other;
  };

 public:

  AuxVar() = default;


///
/// @param p
/// @param r
///
  void push(AuxVar* p, amrex::Real r) {
    a.emplace_back(p, r);
  }


///
/// @param lev
/// @param iv
/// @param r
///
  void push(int lev, const amrex::IntVect& iv, amrex::Real r) {
    a.emplace_back(lev, iv, r);
  }


///
/// @param p
///
  void add_secondary_link(AuxVar* p) {

///
/// @param p->secondary_pushed
///
    if (p->secondary_pushed == 0) { // don't count the same secondary twice
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


///
/// @param levels
/// @param cells
///
  int read_stencil_sites(amrex::Vector<int>& levels, amrex::Vector<amrex::IntVect>& cells);


///
/// @param values
///
  int read_stencil_values(amrex::Vector<amrex::Real>& values);

  void clear();

 protected:

  std::list<AuxLink> a;
  int secondary_pushed{};
};

///
/// BoxAuxStorage is like a FAB, except that an AuxVar is stored at every
/// grid point rather than a double.  It is a minimal container class
/// with much less functionality than even a BaseFab.
/// Could a BaseFab<AuxVar> be used instead? (AuxVar does not have arithmetic)
///
/// @class BoxAuxStorage
/// @brief
///
class BoxAuxStorage {

 public:


///
/// @param bx
///
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

///
/// @class BoundaryAuxBase
/// @brief
///
class BoundaryAuxBase {

 public:


///
/// @param dm
///
  BoundaryAuxBase (const amrex::DistributionMapping& dm)
      : distributionMap(dm) {}

///
/// Location type explanation ---
///  INTERIOR: cells along insides of grid faces, overlaps marked "secondary".
///  OUTSIDE_GRID: cells along outsides of grid faces, overlaps (from other
///            grids this time) marked "secondary".
///  HALO: cells along outsides of grid faces, overlaps not marked.
///  (Note that HALO type is located at cell centers, but the overlap
///  behavior more logically resembles that of face-centered data.  It
///  is also analogous to ghost cells, in that ghost cells from
///  different grids are distinct memory locations even if they have
///  the same cell index.)
///
  enum Location { INTERIOR, OUTSIDE_GRID, HALO };

///
/// This base class provides parallel iterator functionality,
/// since I'm too lazy to write true iterator classes:
///
  int firstOwnedIndex();

///
/// @param i
///
  int nextOwnedIndex(int i);

///
/// @param i
///
  bool isValid(int i) {
    return (i >= 0 && i < distributionMap.ProcessorMap().size());
  }

 protected:

  amrex::DistributionMapping        distributionMap;
};


///
/// @class BoundaryAuxStore

/// @brief BoundaryAuxStore is like a BndryRegister with AuxVars in it.  This too
/// is a rather minimal container class.  It provides its own
/// pseudo-iterator instead of building on something like FabSet.
///
class BoundaryAuxStore : public BoundaryAuxBase {

 public:


///
/// @param _grids
/// @param _dmap
/// @param loc
///
  BoundaryAuxStore(const amrex::BoxArray& _grids,
              const amrex::DistributionMapping& _dmap,
              Location loc);

  //~BoundaryAuxStore(); destructor not needed

  BoxAuxStorage & operator() (const amrex::Orientation ori, int i) {
    return *aux[ori][i];
  }

 protected:

  amrex::BoxArray            grids;
  amrex::Vector<std::unique_ptr<BoxAuxStorage> >   aux[2*AMREX_SPACEDIM];
};

///
/// @class CoarseBoundaryAuxStore
///
/// @brief CoarseBoundaryAuxStore is like a BoundaryAuxStore organized by coarse grid
/// rather than by fine grid.  Entries only exist where the coarse grid
/// is owned by the local processor.  Each entry has subarrays
/// containing information for all fine grid edges falling within
/// the local coarse grid.  Note that while coarse_boxes defines the
/// coarse grid locations and fgrids defines the fine grid locations,
/// both of these BoxArrays must be at the same level of refinement!
/// This class does not know the refinement ratio or problem domain.
///
/// Mask arrays are also built, analogous to those in BndryData.
///
class CoarseBoundaryAuxStore : public BoundaryAuxBase {

 public:


///
/// @param _coarse_boxes
/// @param _coarse_dmap
/// @param _fine_boxes
/// @param loc
///
  CoarseBoundaryAuxStore(const amrex::BoxArray& _coarse_boxes,
                  const amrex::DistributionMapping& _coarse_dmap,
                  const amrex::BoxArray& _fine_boxes, Location loc);


///
/// @param other
/// @param loc
///
  CoarseBoundaryAuxStore(const CoarseBoundaryAuxStore& other, Location loc);


///
/// @param _coarse_boxes
/// @param _fine_boxes
/// @param other
/// @param loc
///
  CoarseBoundaryAuxStore(const amrex::BoxArray& _coarse_boxes,
                  const amrex::BoxArray& _fine_boxes,
                  const CoarseBoundaryAuxStore& other, Location loc);


///
/// @param loc
///
  void rebuild_links(Location loc); ///< use for new discretization


///
/// @param rat
/// @param ncomp
///
  void makeFaceStorage(amrex::IntVect& rat, int ncomp = 1);


///
/// @param rat
/// @param ncomp
///
  void resetFaceStorage(amrex::IntVect& rat, int ncomp = 1); ///< use when ncomp changes


///
/// @param ori
/// @param src
/// @param srccomp
/// @param destcomp
/// @param numcomp
///
  void storeFaceCoeffs(const amrex::Orientation ori,
                    amrex::MultiFab& src,
                    int srccomp,
                    int destcomp,
                    int numcomp = 1);

  //~CoarseBoundaryAuxStore(); destructor not needed


///
/// @param ori
/// @param i
///
  int size (const amrex::Orientation ori, int i) const {
      return static_cast<int>(aux[ori][i].size());
  }

  BoxAuxStorage& operator() (const amrex::Orientation ori, int i, int j) {
    return *aux[ori][i][j];
  }


///
/// @param ori
/// @param i
/// @param j
///
  amrex::Mask& mask (const amrex::Orientation ori, int i, int j) {
    return *msk[ori][i][j];
  }


///
/// @param ori
/// @param i
/// @param j
///
  amrex::FArrayBox& faceFab (const amrex::Orientation ori, int i, int j) {
    return *face_fabs[ori][i][j];
  }

 protected:


///
/// @param loc
///
  void link_secondary_entries(Location loc);

  amrex::BoxArray                    coarse_boxes, fgrids;
  amrex::Vector< amrex::Vector<std::unique_ptr<BoxAuxStorage> > >  aux[2*AMREX_SPACEDIM];
  amrex::Vector< amrex::Vector<std::unique_ptr<amrex::Mask> > >       msk[2*AMREX_SPACEDIM];
  amrex::Vector< amrex::Vector<int> >                          fine_map[2*AMREX_SPACEDIM];
  amrex::Vector< amrex::Vector<std::unique_ptr<amrex::FArrayBox> > >  face_fabs[2*AMREX_SPACEDIM];
};

///
/// @class MultiLevelEllipticSolver
/// @brief
///
class MultiLevelEllipticSolver {

 public:

///
/// solver_choice should be set directly from the inputs file as
/// hmabec.solver_choice.  The value passed in to the constructor will
/// be used as a default if that variable does not occur in inputs.
///
/// @param _coarse_lvl
/// @param _fine_lvl
/// @param _solver_choice
///
  MultiLevelEllipticSolver(int _coarse_lvl, int _fine_lvl, int _solver_choice);

  virtual ~MultiLevelEllipticSolver();

  MultiLevelEllipticSolver(const MultiLevelEllipticSolver& src) = delete;
  MultiLevelEllipticSolver(const MultiLevelEllipticSolver&& src) = delete;

  MultiLevelEllipticSolver operator=(const MultiLevelEllipticSolver& src) = delete;
  MultiLevelEllipticSolver operator=(const MultiLevelEllipticSolver&& src) = delete;

///
/// @param level
/// @param _geom
/// @param _grids
/// @param _dmap
/// @param _coarse_ratio
///
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


///
/// @param level
/// @param _bd
/// @param _comp
///
  void setBndry(int level, BaseBoundary& _bd, int _comp = 0) {
      bd[level] = &_bd;
      bc_component = _comp;
  }

///
/// @param level
///
  BaseBoundary& levelBoundaries(int level) {
    return *bd[level];
  }
  static amrex::Real& fluxScale() {
    return flux_scale;
  }


///
/// @param r
/// @param reg
/// @param ori
/// @param geom
///
  static void faceGeometryFactors(amrex::Vector<amrex::Real>& r,
                            const amrex::Box& reg,
                            const amrex::Orientation& ori,
                            const amrex::Geometry& geom) {

///
/// @param r
/// @param reg
/// @param ori
/// @param geom
///
    AlphaBetaSolver::faceGeometryFactors(r, reg, ori, geom);
  }


///
/// @param v
///
  void setVerbose(int v) {
    verbose = v;
  }

///
/// makeMatrixLayout builds the graph structure for the matrix.
/// It can be called once attachLevel and setBndry have been called for
/// every level.  No numerical data is used---the only info used from
/// the boundary object are the masks at the coarse-fine interfaces.
/// If there is only one level then it is not necessary for the call
/// to setBndry to precede this one.
///
/// makeMatrixLayout also allocates the coefficient arrays and
/// initializes the vector objects as well as the matrix object, so
/// it must be called before any numerical data can be loaded into
/// those structures.
///
  void makeMatrixLayout();


///
/// @param alpha
/// @param beta
///
  void setScalars(amrex::Real alpha, amrex::Real beta);

  [[nodiscard]] amrex::Real getAlphaScalar() const {
    return alpha;
  }
  [[nodiscard]] amrex::Real getBetaScalar() const {
    return beta;
  }


///
/// @param level
/// @param &a
///
  void setMassCoefficients(int level, const amrex::MultiFab &a);

///
/// @param level
/// @param &b
/// @param dir
///
  void setFaceCoefficients(int level, const amrex::MultiFab &b, int dir);


///
/// @param level
/// @param &Spa
///
  void setBcAlphaCoef(int level, const amrex::MultiFab &Spa);


///
/// @param level
///
  const amrex::MultiFab& setMassCoefficients(int level) {
    return *acoefs[level];
  }

///
/// @param level
/// @param dir
///
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

  virtual void assembleMatrix(); ///< once all level coeffs and scalars have been set
  void commitMatrix(); ///< once matrix load completed in derived classes also

///
/// The argument bc_variant in the following methods formerly defaulted
/// to 1.  For greater type safety (to avoid confusion with icomp) it
/// is now an enum with no default.  The argument icomp is always a
/// component number for the independent variable, whether it is
/// called Er, vector, or dest.
///
/// Call fillLevelVectors if dest and rhs are available at the
/// same time, otherwise call the other two.  In the latter case
/// the rhs MultiFab will be altered with boundary condition info
/// if bc_variant is true (but the main reason to call the two separate
/// functions is if rhs is a temporary, so this shouldn't be a problem).
///
/// The B vector depends on not only rhs, but also on boundary condition
/// information and setFaceCoefficients, so these must be set before the B vector
/// can be initialized.  commitMatrix need not have been called yet.
///
  virtual void fillLevelVectors(int level,
                                amrex::MultiFab& dest,
                                int icomp,
                                amrex::MultiFab& rhs, // will not be altered
                                BoundaryMode bc_variant);

///
/// @param level
/// @param dest
/// @param icomp
///
          void loadLevelSolution(int level,
                                amrex::MultiFab& dest,
                                int icomp);
  virtual void fillLevelRhs(int level,
                                amrex::MultiFab& rhs, // will be altered
                                BoundaryMode bc_variant);

  void commitVectors(); ///< once all level values have been loaded

///
/// setup, solve and clear phases separated so that
/// multiple calls to solve can be made:
///
/// commitMatrix should be called before configureSolver.  The calls
/// to set and finalize the vectors can be put off until after
/// configureSolver if desired.  It should be ok to call configureSolver
/// once and then call solve multiple times with different
/// right hand sides.
///
/// @param _rel_tolerance
/// @param _abs_tolerance
/// @param maxiter
///
  void configureSolver(amrex::Real _rel_tolerance, amrex::Real _abs_tolerance, int maxiter);
  void solve();

///
/// @param level
/// @param dest
/// @param icomp
///
  void getSolution(int level, amrex::MultiFab& dest, int icomp);

///
/// This is the 2-norm of the complete rhs, including b.c. contributions
///
  amrex::Real currentResidualNorm();

  void releaseSolver();


///
/// @param level
/// @param Flux
/// @param Er
/// @param icomp
/// @param bc_variant
///
  void faceFluxAtBoundary(int level,
                    amrex::MultiFab* Flux,
                    amrex::MultiFab& Er,
                    int icomp,
                    BoundaryMode bc_variant);


///
/// @param level
/// @param prod
///
  void getProduct(int level, amrex::MultiFab& prod);

 protected:

  int crse_level, fine_level, solver_choice;

  amrex::Vector<amrex::Geometry> geom;
  amrex::Vector<amrex::BoxArray> grids;
  amrex::Vector<amrex::DistributionMapping> dmap;
  amrex::Vector<amrex::IntVect> fine_ratio;
  amrex::Vector<BaseBoundary*> bd;
  int bc_component; ///< component number used for bd

  amrex::Vector< amrex::Vector<amrex::BoxArray> > subgrids;

  amrex::Vector<std::unique_ptr<amrex::MultiFab> > acoefs;
  amrex::Vector<std::unique_ptr<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> > > bcoefs;
  amrex::Real alpha, beta;
  amrex::Real reltol, abstol;

  amrex::Vector<std::unique_ptr<amrex::MultiFab> > SPa; ///< LO_SANCHEZ_POMRANING alpha

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

  // static utility functions follow:


///
/// @param x
/// @param part
/// @param reg
/// @param sgr
/// @param *vec
///
  static void writeVectorBox(HYPRE_SStructVector x,
                                 int part,
                                 const amrex::Box& reg,
                                 const amrex::BoxArray& sgr,
                                 amrex::Real *vec);


///
/// @param x
/// @param part
/// @param reg
/// @param sgr
/// @param f
/// @param fcomp
///
  static void readVectorBox(HYPRE_SStructVector x,
                                 int part,
                                 const amrex::Box& reg,
                                 const amrex::BoxArray& sgr,
                                 amrex::FArrayBox& f, int fcomp);

#if (AMREX_SPACEDIM == 1)
  static int vl[2];
  static int vh[2];
#endif

///
/// The implicit assumption in the following is that vl[1], vh[1] are
/// never changed from 0:
///
/// @param i
///
  static int* cellIndex1(const amrex::IntVect& i) {
#if (AMREX_SPACEDIM == 1)
    vl[0] = i[0];
    return vl;
#else

///
/// @param int*
///
    return (int*) i.getVect();
#endif
  }


///
/// @param i
///
  static int* cellIndex2(const amrex::IntVect& i) {
#if (AMREX_SPACEDIM == 1)
    vh[0] = i[0];
    return vh;
#else

///
/// @param int*
///
    return (int*) i.getVect();
#endif
  }


///
/// @param b
///
  static int* loV(const amrex::Box& b) {
#if (AMREX_SPACEDIM == 1)
    vl[0] = b.smallEnd(0);
    return vl;
#else

///
/// @param int*
///
    return (int*) b.loVect();
#endif
  }


///
/// @param b
///
  static int* hiV(const amrex::Box& b) {
#if (AMREX_SPACEDIM == 1)
    vh[0] = b.bigEnd(0);
    return vh;
#else

///
/// @param int*
///
    return (int*) b.hiVect();
#endif
  }

};

#endif
