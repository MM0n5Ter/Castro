#ifndef CONDUCTION_H
#define CONDUCTION_H

#include <AMReX_AmrLevel.H>
#include <AMReX_MLLinOp.H>

#include <diffusion_params.H>

///
/// @class Diffusion
/// @brief
///
class Diffusion {

public:


///
/// @param Parent
/// @param _boundary_rec
///
  Diffusion(amrex::Amr* Parent, amrex::BCRec* _boundary_rec);
  ~Diffusion () = default;

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
  static void report_run_settings(std::ostream& jobInfoFile);


///
/// @param level
/// @param amr_level_ptr
/// @param volume
/// @param area
///
  void attach_level (int                   level,
                      amrex::AmrLevel*             amr_level_ptr,
                      amrex::MultiFab&             volume,
                      amrex::MultiFab*             area);


///
/// @param level
/// @param Temperature
/// @param CoarseTemperature
/// @param ConductionTerm
/// @param conductivity_fields
///
  void apply_stencil(int level,amrex::MultiFab& Temperature,amrex::MultiFab& CoarseTemperature,
               amrex::MultiFab& ConductionTerm, amrex::Vector<std::unique_ptr<amrex::MultiFab> >& conductivity_fields);

  void build_solver_bc();

protected:

///
/// Pointers to amr,amrlevel.
///
  amrex::Amr*             parent;
  amrex::Vector<amrex::AmrLevel*> LevelTable;


///
/// BoxArray at each level
///
  amrex::Vector<amrex::BoxArray> grids;

///
/// Volume and area fractions.
///
  amrex::Vector<amrex::MultiFab*> volume;
  amrex::Vector<amrex::MultiFab*> area;

  amrex::BCRec*       phys_bc;

  std::array<amrex::MLLinOp::BCType,AMREX_SPACEDIM> mlmg_lobc;
  std::array<amrex::MLLinOp::BCType,AMREX_SPACEDIM> mlmg_hibc;


///
/// @param level
/// @param Temperature
/// @param CoarseTemperature
/// @param ConductionTerm
/// @param conductivity_fields
///
  void apply_stencil_multigrid(int level,amrex::MultiFab& Temperature,amrex::MultiFab& CoarseTemperature,
                    amrex::MultiFab& ConductionTerm, amrex::Vector<std::unique_ptr<amrex::MultiFab> >& conductivity_fields);

};
#endif
