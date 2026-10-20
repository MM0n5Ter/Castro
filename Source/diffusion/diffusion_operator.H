#ifndef CONDUCTION_H
#define CONDUCTION_H

#include <AMReX_AmrLevel.H>
#include <AMReX_MLLinOp.H>

#include <diffusion_params.H>

class Diffusion {

public:


  Diffusion(amrex::Amr* Parent, amrex::BCRec* _boundary_rec);
  Diffusion (const Diffusion&) = delete;

  Diffusion (Diffusion&&) = delete;
  Diffusion& operator= (const Diffusion&) = delete;
  Diffusion& operator= (Diffusion&&) = delete;
  ~Diffusion () = default;

  void read_params ();


  static void report_run_settings(std::ostream& jobInfoFile);


  void attach_level (int                   level,
                      amrex::AmrLevel*             amr_level_ptr,
                      amrex::MultiFab&             volume,
                      amrex::MultiFab*             area);


  void apply_stencil(int level,amrex::MultiFab& Temperature,amrex::MultiFab& CoarseTemperature,
               amrex::MultiFab& ConductionTerm, amrex::Vector<std::unique_ptr<amrex::MultiFab> >& conductivity_fields);

  void build_solver_bc();

protected:

  amrex::Amr*             parent;
  amrex::Vector<amrex::AmrLevel*> LevelTable;


  amrex::Vector<amrex::BoxArray> grids;

  amrex::Vector<amrex::MultiFab*> volume;
  amrex::Vector<amrex::MultiFab*> area;

  amrex::BCRec*       phys_bc;

  std::array<amrex::MLLinOp::BCType,AMREX_SPACEDIM> mlmg_lobc;
  std::array<amrex::MLLinOp::BCType,AMREX_SPACEDIM> mlmg_hibc;


  void apply_stencil_multigrid(int level,amrex::MultiFab& Temperature,amrex::MultiFab& CoarseTemperature,
                    amrex::MultiFab& ConductionTerm, amrex::Vector<std::unique_ptr<amrex::MultiFab> >& conductivity_fields);

};
#endif
