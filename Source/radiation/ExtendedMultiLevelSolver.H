#ifndef HYDROSIM_EXTENDEDMULTILEVELSOLVER_H
#define HYDROSIM_EXTENDEDMULTILEVELSOLVER_H

#include <MultiLevelEllipticSolver.H>


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

  amrex::Real& auxMassScale() {
    return alpha2;
  }
  amrex::Real& advectionScale() {
    return gamma;
  }
  amrex::Real& gradientScale1() {
    return delta1;
  }
  amrex::Real& gradientScale2() {
    return delta2;
  }


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

#endif
