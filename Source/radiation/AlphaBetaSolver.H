#ifndef HYDROSIM_ALPHABETASOLVER_H
#define HYDROSIM_ALPHABETASOLVER_H

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>

#include <BaseBoundary.H>

#include <_hypre_utilities.h>
#include <HYPRE_struct_ls.h>

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

  void setVerbose(int v) {
    verbose = v;
  }


  void setScalars(amrex::Real alpha, amrex::Real beta);

  [[nodiscard]] amrex::Real getAlphaScalar() const {
    return alpha;
  }
  [[nodiscard]] amrex::Real getBetaScalar() const {
    return beta;
  }


  void setMassCoefficients(const amrex::MultiFab &a);

  void setFaceCoefficients(const amrex::MultiFab &b, int dir);


  void setBcAlphaCoef(const amrex::MultiFab &Spa);

  const amrex::MultiFab& setMassCoefficients() {
    return *acoefs;
  }

  const amrex::MultiFab& setFaceCoefficients(int dir) {
    return *bcoefs[dir];
  }


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


  static void faceGeometryFactors(amrex::Vector<amrex::Real>& r,
                            const amrex::Box& reg,
                            const amrex::Orientation& ori,
                            const amrex::Geometry& geom);

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

#endif
