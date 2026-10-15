#ifndef DIFFUSION_H
#define DIFFUSION_H

#include <AMReX_AmrLevel.H>
#include <AMReX_MLLinOp.H>

#include <diffusion_params.H>
#include <MGutils.H>

///
/// @class Diffusion
/// @brief
///
class Diffusion : public LevelSolverBase {

public:

///
/// Edge-centred conductivity coefficients, one MultiFab per direction
///
  using CoefficientVector = amrex::Vector<std::unique_ptr<amrex::MultiFab> >;

///
/// @param Parent
/// @param _phys_bc
///
  Diffusion(amrex::Amr* Parent, amrex::BCRec* _phys_bc);
  ~Diffusion () override = default;

///
/// Remove copy/move constructors/assignment operators.
///
  Diffusion (const Diffusion&) = delete;
  Diffusion (Diffusion&&) = delete;
  Diffusion& operator= (const Diffusion&) = delete;
  Diffusion& operator= (Diffusion&&) = delete;

  void read_params ();


///
/// @param jobInfoFile
///
  static void output_job_info_params(std::ostream& jobInfoFile);


///
/// @param level
/// @param level_data
/// @param volume
/// @param area
///
  void register_level (int                   level,
                      amrex::AmrLevel*             level_data,
                      amrex::MultiFab&             volume,
                      amrex::MultiFab*             area);


///
/// @param level
/// @param Temperature
/// @param CrseTemp
/// @param DiffTerm
/// @param temp_cond_coef
///
  void applyop(int level,amrex::MultiFab& Temperature,amrex::MultiFab& CrseTemp,
               amrex::MultiFab& DiffTerm, CoefficientVector& temp_cond_coef);

  void set_solver_bc_types() override;

protected:

///
/// BoxArray at each level
///
  amrex::Vector<amrex::BoxArray> grids;


///
/// @param level
/// @param Temperature
/// @param CrseTemp
/// @param DiffTerm
/// @param temp_cond_coef
///
  void applyop_mlmg(int level,amrex::MultiFab& Temperature,amrex::MultiFab& CrseTemp,
                    amrex::MultiFab& DiffTerm, CoefficientVector& temp_cond_coef);

};
#endif
