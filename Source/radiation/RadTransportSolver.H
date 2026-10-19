#ifndef HYDROSIM_RADTRANSPORT_H
#define HYDROSIM_RADTRANSPORT_H

#include <AMReX_Amr.H>

#include <AMReX_FluxRegister.H>

#include <RadiativeBoundary.H>
#include <MultiGroupBoundary.H>

#include <AlphaBetaSolver.H>
#include <MultiLevelEllipticSolver.H>
#include <ExtendedMultiLevelSolver.H>

#include <radiation_solve_params.H>

///
/// @class RadTransportSolver
/// @brief
///
class RadTransportSolver {

 public:


///
/// @param Parent
///
  RadTransportSolver (amrex::Amr* Parent, int level,
            const amrex::BoxArray& grids,
            const amrex::DistributionMapping& dmap);

///
/// query runtime parameters
///
  static void read_params ();

///
/// @param level
///
  void levelSetup(int level);

///
/// @param bd
///
  void levelBoundary(RadiativeBoundary& bd);

///
/// update multigroup version
///
/// @param multigroup_boundary
/// @param comp
///
  void levelBoundary(MultiGroupBoundary& multigroup_boundary, const int comp);


///
/// @param level
/// @param acoefs
///
  void applyCellMetrics(int level, amrex::MultiFab& acoefs);


///
/// @param idim
/// @param geom
/// @param edge_region
/// @param r
/// @param s
///
  static void edgeGeomWeights(int idim, const amrex::Geometry& geom, const amrex::Box& edge_region,
                            amrex::Vector<amrex::Real>& r, amrex::Vector<amrex::Real>& s);


///
/// @param level
/// @param &a
///
  void setLevelDiagCoeffs(int level, const amrex::MultiFab &a);

///
/// @param level
/// @param &b
/// @param dir
///
  void setLevelFaceCoeffs(int level, const amrex::MultiFab &b, int dir);

///
/// @param level
/// @param &c
/// @param dir
///
  void setLevelCellCoeffs(int level, const amrex::MultiFab &c, int dir);


///
/// @param level
/// @param fkp
/// @param eta
/// @param coupling_inverse
/// @param c
/// @param delta_t
/// @param theta
///
  void levelDiagCoeffs(int level,
                    amrex::MultiFab& fkp, amrex::MultiFab& eta, amrex::MultiFab& coupling_inverse,
                    amrex::Real c, amrex::Real delta_t, amrex::Real theta);


///
/// @param level
/// @param amrex::Array<amrex::MultiFab
/// @param lambda
/// @param kappa_r
/// @param kcomp
/// @param c
/// @param limiter_comp
///
  void levelFaceCoeffs(int level,
                    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& lambda,
                    amrex::MultiFab& kappa_r, int kcomp, amrex::Real c, int limiter_comp=0);


///
/// @param level
/// @param rhs
/// @param temp
/// @param fkp
/// @param eta
/// @param coupling_inverse
/// @param matter_energy
/// @param state_energy
/// @param flux_div_old
/// @param rad_energy_old
/// @param Edot
/// @param delta_t
/// @param sigma
/// @param c
/// @param theta
/// @param fine_correction
/// @param scale
/// @param rad_group_idx
/// @param nu
/// @param dnu
///
  void levelAssembleRhs(int level, amrex::MultiFab& rhs,
                amrex::MultiFab& temp,
                amrex::MultiFab& fkp, amrex::MultiFab& eta, amrex::MultiFab& coupling_inverse,
                amrex::MultiFab& matter_energy, amrex::MultiFab& state_energy,
                amrex::MultiFab& flux_div_old, amrex::MultiFab& rad_energy_old, amrex::MultiFab& Edot,
                amrex::Real delta_t, amrex::Real sigma, amrex::Real c, amrex::Real theta,
                amrex::FluxRegister* fine_correction, amrex::Real scale = 1.0,
                int rad_group_idx = -1, amrex::Real nu = -1.0, amrex::Real dnu = -1.0);


///
/// @param level
/// @param Er
/// @param rad_group_idx
/// @param rhs
/// @param sync_residual_scale
///
  void levelSolveSystem(int level, amrex::MultiFab& Er, int rad_group_idx, amrex::MultiFab& rhs,
                  amrex::Real sync_residual_scale);


///
/// @param level
/// @param amrex::Array<amrex::MultiFab
/// @param Flux
/// @param Er
/// @param rad_group_idx
///
  void levelFaceFlux(int level,
                 amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Flux,
                 amrex::MultiFab& Er, int rad_group_idx);

///
/// @param level
/// @param flux_from_fine
/// @param flux_to_coarse
/// @param amrex::Array<amrex::MultiFab
/// @param Flux
/// @param rad_group_idx
///
  void levelCoarseFineFlux(int level,
                    amrex::FluxRegister* flux_from_fine, amrex::FluxRegister* flux_to_coarse,
                    const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Flux,
                    int rad_group_idx);

///
/// @param level
/// @param amrex::Array<amrex::MultiFab
/// @param Flux
/// @param flx
/// @param flux_comp_base
///
  void levelFaceToCellFlux(int level, const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Flux,
                             amrex::MultiFab& flx, int flux_comp_base);

///
/// @param level
/// @param diffusive_face_term
/// @param Er
/// @param rad_group_idx
///
  void levelDiffusionTerm(int level, amrex::MultiFab& diffusive_face_term, amrex::MultiFab& Er, int rad_group_idx);
  void levelRelease();


///
/// <MultigroupDiffusion routines>
///
/// @param bcoefs
/// @param idim
/// @param kappa_r
/// @param kcomp
/// @param lambda
/// @param limiter_comp
/// @param c
/// @param geom
///
  void computeFaceCoeffs(amrex::MultiFab& bcoefs, int idim,
                      amrex::MultiFab& kappa_r, int kcomp,
                      amrex::MultiFab& lambda, int limiter_comp,
                      amrex::Real c, const amrex::Geometry& geom);

///
/// @param level
/// @param kappa_p
/// @param delta_t
/// @param c
/// @param rad_group_idx
/// @param pseudo_transient_tau
///
  void levelDiagCoeffs(int level, amrex::MultiFab& kappa_p,
                    amrex::Real delta_t, amrex::Real c, int rad_group_idx, amrex::Real pseudo_transient_tau);

///
/// @param level
/// @param rhs
/// @param jg
/// @param groupTempCoef
/// @param muYg
/// @param couple_source
/// @param thermal_frac
/// @param rad_energy_step
/// @param gas_energy_step
/// @param rad_energy_star
/// @param gas_energy_star
/// @param delta_t
/// @param rad_group_idx
/// @param it
/// @param pseudo_transient_tau
///
  void levelAssembleRhs(int level, amrex::MultiFab& rhs, const amrex::MultiFab& jg,
                const amrex::MultiFab& groupTempCoef,
                const amrex::MultiFab& couple_source,
                const amrex::MultiFab& thermal_frac,
                const amrex::MultiFab& rad_energy_step, const amrex::MultiFab& gas_energy_step,
                const amrex::MultiFab& rad_energy_star, const amrex::MultiFab& gas_energy_star,
                amrex::Real delta_t, int rad_group_idx, int it, amrex::Real pseudo_transient_tau);

  void levelBoundaryScale(int level, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& lambda, int rad_group_idx,
                 int lo_bc[], int hi_bc[]);

///
/// </ MultigroupDiffusion routines>
///
/// @param level
/// @param amrex::Array<amrex::MultiFab
/// @param lambda
/// @param vel
/// @param dcf
///
  void levelDiffCoeffs(int level, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& lambda,
                    amrex::MultiFab& vel, amrex::MultiFab& dcf);

///
/// temporarily change multipliers for C and D coefficients
///
  amrex::Real cSaved, d1Saved, d2Saved;

///
/// @param cScale
/// @param d1Scale
/// @param d2Scale
///
  void setSolverScale(amrex::Real cScale, amrex::Real d1Scale=0.0, amrex::Real d2Scale=0.0);
  void restoreSolverScale();

protected:

    amrex::Amr* parent;

    std::unique_ptr<AlphaBetaSolver> hd;
    std::unique_ptr<MultiLevelEllipticSolver> hm;
    std::unique_ptr<ExtendedMultiLevelSolver> hem;


};

#endif
