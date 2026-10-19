#ifndef HYDROSIM_ALPHABETASOLVER_H
#define HYDROSIM_ALPHABETASOLVER_H

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>

#include <BaseBoundary.H>

#include <_hypre_utilities.h>
#include <HYPRE_struct_ls.h>

///
/// @class AlphaBetaSolver
/// @brief
///
class AlphaBetaSolver {

 public:

///
/// solver_choice = 0 for SMG
/// solver_choice = 1 for PFMG
///
/// @param grids
/// @param dmap
/// @param geom
/// @param solver_choice
///
  AlphaBetaSolver(const amrex::BoxArray& grids,
            const amrex::DistributionMapping& dmap,
            const amrex::Geometry& geom,
            int solver_choice = 0);

  ~AlphaBetaSolver();

  AlphaBetaSolver(const AlphaBetaSolver& a) = delete;
  AlphaBetaSolver(const AlphaBetaSolver&& a) = delete;
  AlphaBetaSolver& operator= (const AlphaBetaSolver& a) = delete;
  AlphaBetaSolver& operator= (const AlphaBetaSolver&& a) = delete;

///
/// @param v
///
  void setVerbose(int v) {
    verbose = v;
  }


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
/// @param &a
///
  void setMassCoefficients(const amrex::MultiFab &a);

///
/// @param &b
/// @param dir
///
  void setFaceCoefficients(const amrex::MultiFab &b, int dir);


///
/// @param &Spa
///
  void setBcAlphaCoef(const amrex::MultiFab &Spa);

  const amrex::MultiFab& setMassCoefficients() {
    return *acoefs;
  }

///
/// @param dir
///
  const amrex::MultiFab& setFaceCoefficients(int dir) {
    return *bcoefs[dir];
  }


///
/// @param bd
/// @param _comp
///
  void setBndry(const BaseBoundary& bd, int _comp = 0) {
    bdp = &bd;
    bc_component = _comp;
  }
  const BaseBoundary& getBoundaryData() {
    return *bdp;
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
                            const amrex::Geometry& geom);

///
/// The argument bc_variant in the following methods formerly defaulted
/// to 1.  For greater type safety (to avoid confusion with icomp) it
/// is now an enum with no default.  The argument icomp is always a
/// component number for the independent variable, whether it is
/// called Er, vector, or dest.
///
/// @param Flux
/// @param Er
/// @param icomp
/// @param bc_variant
///
  void faceFluxAtBoundary(amrex::MultiFab* Flux, amrex::MultiFab& Er, int icomp, BoundaryMode bc_variant);

  void fillMassMat (const amrex::Box& bx,
               amrex::Array4<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM+1>> const& mat,
               amrex::Array4<amrex::Real const> const& a,
               amrex::Real alpha);

  void fillFaceMat (const amrex::Box& bx,
               amrex::Array4<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM+1>> const& mat,
               amrex::Array4<amrex::Real const> const& b,
               amrex::Real beta, const amrex::Real* dx,
               int idir);

  void fillBcMat (const amrex::Box& bx,
              amrex::Array4<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM+1>> const& mat,
              int cdir, int bct, amrex::Real bcl,
              amrex::Array4<int const> const& mask,
              amrex::Array4<amrex::Real const> const& b,
              amrex::Real beta, const amrex::Real* dx);

  void fillBcMat3 (const amrex::Box& bx,
               int side_is_low, int idir,
               amrex::Array4<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM+1>> const& mat,
               int cdir, int bctype,
               amrex::Array4<int const> const& tf,
               amrex::Real bcl,
               amrex::Array4<int const> const& mask,
               amrex::Array4<amrex::Real const> const& b,
               amrex::Real beta, const amrex::GeometryData& geomdata,
               amrex::Real c,
               amrex::Array4<amrex::Real const> const& spa);

///
/// Three steps separated so that multiple calls to solve can be made
///
/// @param _rel_tolerance
/// @param _abs_tolerance
/// @param maxiter
///
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

///
/// @param dest
/// @param icomp
/// @param rhs
/// @param bc_variant
///
  void solve(amrex::MultiFab& dest, int icomp, amrex::MultiFab& rhs, BoundaryMode bc_variant);

  ///
  /// This is the 2-norm of the complete rhs, including b.c. contributions
  ///
  amrex::Real currentResidualNorm();

  void releaseSolver();

 protected:

  const amrex::Geometry& geom;

  std::unique_ptr<amrex::MultiFab> acoefs;
  std::unique_ptr<amrex::MultiFab> bcoefs[AMREX_SPACEDIM];
  amrex::Real alpha, beta;
  amrex::Real dx[AMREX_SPACEDIM];
  amrex::Real reltol, abstol;

  std::unique_ptr<amrex::MultiFab> SPa; ///< LO_SANCHEZ_POMRANING alpha

  const BaseBoundary *bdp;
  int bc_component; ///< component number used for bdp

  int solver_choice, verbose, log_threshold, pfmg_smoother_type, bho;

  HYPRE_StructGrid    hgrid;
  //HYPRE_StructStencil stencil;

  HYPRE_StructMatrix  A, A0;
  HYPRE_StructVector  b;
  HYPRE_StructVector  x;

  HYPRE_StructSolver  solver;
  HYPRE_StructSolver  precond;

  static amrex::Real flux_scale;
};

#endif
