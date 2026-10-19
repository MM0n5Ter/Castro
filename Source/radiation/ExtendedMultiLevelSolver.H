#ifndef HYDROSIM_EXTENDEDMULTILEVELSOLVER_H
#define HYDROSIM_EXTENDEDMULTILEVELSOLVER_H

#include <MultiLevelEllipticSolver.H>


///
/// @class ExtendedMultiLevelSolver
/// @brief
///
class ExtendedMultiLevelSolver : public MultiLevelEllipticSolver {

 public:


///
/// @param _coarse_lvl
/// @param _fine_lvl
/// @param _solver_choice
///
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


///
/// @param level
/// @param &a2
/// @param dir
///
  void setAuxMassCoefficients(int level, const amrex::MultiFab &a2, int dir);

///
/// @param level
/// @param &c
/// @param dir
///
  void setAdvectionCoefficients( int level, const amrex::MultiFab &c,  int dir);

///
/// @param level
/// @param &d1
/// @param dir
///
  void setGradientCoefficients1(int level, const amrex::MultiFab &d1, int dir);

///
/// @param level
/// @param &d2
/// @param dir
///
  void setGradientCoefficients2(int level, const amrex::MultiFab &d2, int dir);


///
/// @param level
/// @param dir
///
  const amrex::MultiFab& setAuxMassCoefficients(int level, int dir) {
      return (*aux_mass_coefs[level])[dir];
  }

///
/// @param level
/// @param dir
///
  const amrex::MultiFab& setAdvectionCoefficients( int level, int dir) {
      return (*cell_coefs[level])[dir];
  }

///
/// @param level
/// @param dir
///
  const amrex::MultiFab& setGradientCoefficients1(int level, int dir) {
      return (*gradient_coefs1[level])[dir];
  }

///
/// @param level
/// @param dir
///
  const amrex::MultiFab& setGradientCoefficients2(int level, int dir) {
      return (*gradient_coefs2[level])[dir];
  }

  void fillLevelVectors(int level,
                        amrex::MultiFab& dest,
                        int icomp,
                        amrex::MultiFab& rhs, ///< will not be altered
                        BoundaryMode bc_variant) override;
  void fillLevelRhs(int level,
                        amrex::MultiFab& rhs, ///< will not be altered
                        BoundaryMode bc_variant) override;

  void assembleMatrix() override;  ///< once all level coeffs and scalars have been set


///
/// @param level
/// @param diffusive_face_term
/// @param Er
/// @param icomp
///
  void computeBoundaryTerm(int level,
                     amrex::MultiFab* diffusive_face_term,
                     amrex::MultiFab& Er,
                     int icomp);

 protected:

  amrex::Vector<std::unique_ptr<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> > > aux_mass_coefs; ///< face-based
  amrex::Vector<std::unique_ptr<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> > > cell_coefs;  ///< face-based, 2 component
  amrex::Vector<std::unique_ptr<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> > > gradient_coefs1; ///< cell-based but directional
  amrex::Vector<std::unique_ptr<amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> > > gradient_coefs2; ///< face-based
  amrex::Real alpha2{}, gamma{}, delta1{}, delta2{}; ///< multipliers for the above
};

#endif
