#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_MLLinOp.H>

#ifndef HYDROSIM_MGUTILS_H
#define HYDROSIM_MGUTILS_H

using namespace amrex;

void
apply_metric(const Box& bx,
             Array4<Real> const& rhs, const Box& rbx,
             Array4<Real> const& ecx, const Box& xbx,

#if AMREX_SPACEDIM >= 2
             Array4<Real> const& ecy, const Box& ybx,
#endif
             const GpuArray<Real, AMREX_SPACEDIM>& problo,
             const GpuArray<Real, AMREX_SPACEDIM>& dx,
             const int coord_type);

///
/// @class LevelSolverBase
/// @brief Level data and domain boundary types shared by the MLMG-based solvers
///
class LevelSolverBase {

public:

///
/// @param Parent           Parent ``Amr`` object
/// @param _phys_bc         ``BCRec`` object, physical boundary conditions
///
  LevelSolverBase(Amr* Parent, BCRec* _phys_bc);

  virtual ~LevelSolverBase () = default;

///
/// Make multigrid boundary conditions
///
  virtual void set_solver_bc_types () = 0;

protected:

///
/// Linear solver boundary types on one side of the domain, one per direction
///
  using DomainBCTypes = std::array<MLLinOp::BCType, AMREX_SPACEDIM>;

///
/// Keep pointers to the state, volume and area data of level ``level``
///
/// @param level        integer, level number
/// @param level_data   ``AmrLevel`` object containing state data at that level
/// @param _volume      MultiFab, volume elements of cells
/// @param _area        MultiFab, area elements of cells
///
  void store_level (int level, AmrLevel* level_data, MultiFab& _volume, MultiFab* _area);

///
/// Pointers to amr,amrlevel.
///
  Amr*             parent;
  Vector<AmrLevel*> LevelData;

///
/// Volume and area fractions.
///
  Vector<MultiFab*> volume;
  Vector<MultiFab*> area;

  BCRec*       phys_bc;

  DomainBCTypes mlmg_lobc;
  DomainBCTypes mlmg_hibc;

};

#endif
