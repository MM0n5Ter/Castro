#ifndef HYDROSIM_HYPRESOLVERS_H
#define HYDROSIM_HYPRESOLVERS_H

#include <list>
#include <HYPRE_sstruct_ls.h>
#include <AMReX_Array.H>

#include <AMReX_MultiFab.H>

#include <rad_boundary_values.H>

#include <_hypre_utilities.h>

#include <HYPRE_struct_ls.h>
#include <aux_coupling.H>

class AlphaBetaSolver {

 public:

  AlphaBetaSolver(const amrex::BoxArray& grids,
            const amrex::DistributionMapping& dmap,
            const amrex::Geometry& geom,
            int solver_choice = 0);

  ~AlphaBetaSolver();

  AlphaBetaSolver(const AlphaBetaSolver& a) = delete;
  AlphaBetaSolver(const AlphaBetaSolver&& a) = delete;
  AlphaBetaSolver& operator= (const AlphaBetaSolver& a) = delete;
  AlphaBetaSolver& operator= (const AlphaBetaSolver&& a) = delete;

  const amrex::MultiFab& setMassCoefficients() {
    return *acoefs;
  }

  void setScalars(amrex::Real alpha, amrex::Real beta);

  const amrex::MultiFab& setFaceCoefficients(int dir) {
    return *bcoefs[dir];
  }
  void setBndry(const BaseBoundary& bd, int _comp = 0) {
    bdp = &bd;
    bc_component = _comp;
  }

  void setMassCoefficients(const amrex::MultiFab &a);

  void setFaceCoefficients(const amrex::MultiFab &b, int dir);

  void setBcAlphaCoef(const amrex::MultiFab &Spa);

  void setVerbose(int v) {
    verbose = v;
  }

  [[nodiscard]] amrex::Real getAlphaScalar() const {
    return alpha;
  }

  [[nodiscard]] amrex::Real getBetaScalar() const {
    return beta;
  }
  const BaseBoundary& getBoundaryData() {
    return *bdp;
  }
  static amrex::Real& fluxScale() {
    return flux_scale;
  }

  static void faceGeometryFactors(amrex::Vector<amrex::Real>& r,
                            const amrex::Box& reg,
                            const amrex::Orientation& ori,
                            const amrex::Geometry& geom);

  void faceFluxAtBoundary(amrex::MultiFab* Flux, amrex::MultiFab& Er, int icomp, BoundaryMode bc_variant);

  void fillMassMat (const amrex::Box& bx,
               amrex::Array4<amrex::Real const> const& a,
               amrex::Real alpha,
               amrex::Array4<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM+1>> const& mat);

  void fillFaceMat (const amrex::Box& bx,
               amrex::Array4<amrex::Real const> const& b,
               amrex::Real beta, const amrex::Real* dx,
               int idir,
               amrex::Array4<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM+1>> const& mat);

  void fillBcMat (const amrex::Box& bx,
              int cdir, int bct, amrex::Real bcl,
              amrex::Array4<int const> const& mask,
              amrex::Array4<amrex::Real const> const& b,
              amrex::Real beta, const amrex::Real* dx,
              amrex::Array4<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM+1>> const& mat);

  void fillBcMat3 (const amrex::Box& bx,
               int side_is_low, int idir,
               int cdir, int bctype,
               amrex::Array4<int const> const& tf,
               amrex::Real bcl,
               amrex::Array4<int const> const& mask,
               amrex::Array4<amrex::Real const> const& b,
               amrex::Real beta, const amrex::GeometryData& geomdata,
               amrex::Real c,
               amrex::Array4<amrex::Real const> const& spa,
               amrex::Array4<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM+1>> const& mat);

  void configureSolver(amrex::Real _rel_tolerance, amrex::Real _abs_tolerance, int maxiter);

  static void fillBcVec (const amrex::Box& bx,
                     amrex::Array4<amrex::Real> const& vec,
                     int cdir, int bct, int bho, amrex::Real bcl,
                     amrex::Array4<amrex::Real const> const& bcval,
                     amrex::Array4<int const> const& mask,
                     amrex::Array4<amrex::Real const> const& b,
                     amrex::Real beta, const amrex::Real* dx);

  static void fillBcVec3 (const amrex::Box& bx,
                      int side_is_low, int idir,
                      amrex::Array4<amrex::Real> const& vec,
                      int cdir, int bct,
                      amrex::Array4<int const> const& tf,
                      int bho, amrex::Real bcl,
                      amrex::Array4<amrex::Real const> const& bcval,
                      amrex::Array4<int const> const& mask,
                      amrex::Array4<amrex::Real const> const& b,
                      amrex::Real beta, const amrex::GeometryData& geomdata);

  void solve(amrex::MultiFab& dest, int icomp, amrex::MultiFab& rhs, BoundaryMode bc_variant);

  amrex::Real currentResidualNorm();

  void releaseSolver();

 protected:

  const amrex::Geometry& geom;

  std::unique_ptr<amrex::MultiFab> acoefs;
  std::unique_ptr<amrex::MultiFab> bcoefs[AMREX_SPACEDIM];
  amrex::Real alpha, beta;
  amrex::Real dx[AMREX_SPACEDIM];
  amrex::Real reltol, abstol;

  std::unique_ptr<amrex::MultiFab> SPa;

  const BaseBoundary *bdp;
  int bc_component;

  int solver_choice, verbose, log_threshold, pfmg_smoother_type, bho;

  HYPRE_StructGrid    hgrid;

  HYPRE_StructMatrix  A, A0;
  HYPRE_StructVector  b;
  HYPRE_StructVector  x;

  HYPRE_StructSolver  solver;
  HYPRE_StructSolver  precond;

  static amrex::Real flux_scale;
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

  const amrex::MultiFab& setMassCoefficients(int level) {
    return *acoefs[level];
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

  const amrex::MultiFab& setFaceCoefficients(int level, int dir) {
      return (*bcoefs[level])[dir];
  }

  void makeMatrixLayout();

  void setScalars(amrex::Real alpha, amrex::Real beta);

  void setBndry(int level, BaseBoundary& _bd, int _comp = 0) {
      bd[level] = &_bd;
      bc_component = _comp;
  }
  void setVerbose(int v) {
    verbose = v;
  }

  void setMassCoefficients(int level, const amrex::MultiFab &a);

  void setFaceCoefficients(int level, const amrex::MultiFab &b, int dir);

  void setBcAlphaCoef(int level, const amrex::MultiFab &Spa);

  [[nodiscard]] amrex::Real getAlphaScalar() const {
    return alpha;
  }

  [[nodiscard]] amrex::Real getBetaScalar() const {
    return beta;
  }

  void hmac (const amrex::Box& bx,
             amrex::Array4<amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
             amrex::Array4<amrex::Real const> const& a,
             amrex::Real alpha);

  void fillBoundaryCoeffs (const amrex::Box& bx,
             amrex::Array4<amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
             amrex::Array4<amrex::Real const> const& b,
             amrex::Real beta, const amrex::Real* dx, int n);

  struct FaceBcSpec {
    int cdir;
    int bct;
    int bho;
    amrex::Real bcl;
  };

  void fillLevelMat (const amrex::Box& bx,
              amrex::Array4<amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM + 1>> const& mat,
              const FaceBcSpec& bc,
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

  void addBoundaryRhs(int level, const amrex::MFIter& mfi, const amrex::Box& reg,
                      amrex::Array4<amrex::Real> const& vec);

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
                                 amrex::FArrayBox& f, int fcomp);

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

class ExtendedMultiLevelSolver : public MultiLevelEllipticSolver {

 public:

  ExtendedMultiLevelSolver(int _coarse_lvl, int _fine_lvl, int _solver_choice)
    : MultiLevelEllipticSolver(_coarse_lvl, _fine_lvl, _solver_choice),
      aux_mass_coefs(fine_level+1),
      cell_coefs( fine_level+1),
      gradient_coefs1(fine_level+1),
      gradient_coefs2(fine_level+1)
  {}

  ~ExtendedMultiLevelSolver() override;

  ExtendedMultiLevelSolver(const ExtendedMultiLevelSolver& src) = delete;
  ExtendedMultiLevelSolver(const ExtendedMultiLevelSolver&& src) = delete;

  ExtendedMultiLevelSolver& operator= (const ExtendedMultiLevelSolver& src) = delete;
  ExtendedMultiLevelSolver& operator= (const ExtendedMultiLevelSolver&& src) = delete;

  amrex::Real& auxMassScale();
  amrex::Real& advectionScale();
  amrex::Real& gradientScale1();
  amrex::Real& gradientScale2();

  void setAuxMassCoefficients(int level, const amrex::MultiFab &a2, int dir);

  void setAdvectionCoefficients( int level, const amrex::MultiFab &c,  int dir);

  void setGradientCoefficients1(int level, const amrex::MultiFab &d1, int dir);

  void setGradientCoefficients2(int level, const amrex::MultiFab &d2, int dir);

  const amrex::MultiFab& setAuxMassCoefficients(int level, int dir) {
      return (*aux_mass_coefs[level])[dir];
  }

  const amrex::MultiFab& setAdvectionCoefficients( int level, int dir) {
      return (*cell_coefs[level])[dir];
  }

  const amrex::MultiFab& setGradientCoefficients1(int level, int dir) {
      return (*gradient_coefs1[level])[dir];
  }

  const amrex::MultiFab& setGradientCoefficients2(int level, int dir) {
      return (*gradient_coefs2[level])[dir];
  }

  void fillLevelVectors(int level,
                        amrex::MultiFab& dest,
                        int icomp,
                        amrex::MultiFab& rhs,
                        BoundaryMode bc_variant) override;
  void fillLevelRhs(int level,
                        amrex::MultiFab& rhs,
                        BoundaryMode bc_variant) override;

  void assembleMatrix() override;

  void computeBoundaryTerm(int level,
                     amrex::MultiFab* diffusive_face_term,
                     amrex::MultiFab& Er,
                     int icomp);

 protected:

  amrex::Vector<std::unique_ptr<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> > > aux_mass_coefs;
  amrex::Vector<std::unique_ptr<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> > > cell_coefs;
  amrex::Vector<std::unique_ptr<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> > > gradient_coefs1;
  amrex::Vector<std::unique_ptr<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> > > gradient_coefs2;
  amrex::Real alpha2{}, gamma{}, delta1{}, delta2{};
};

inline
amrex::Real&
ExtendedMultiLevelSolver::auxMassScale()
{
  return alpha2;
}

inline
amrex::Real&
ExtendedMultiLevelSolver::advectionScale()
{
  return gamma;
}

inline
amrex::Real&
ExtendedMultiLevelSolver::gradientScale1()
{
  return delta1;
}

inline
amrex::Real&
ExtendedMultiLevelSolver::gradientScale2()
{
  return delta2;
}

#endif
