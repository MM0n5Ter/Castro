#ifndef HYDROSIM_RADTRANSPORT_H
#define HYDROSIM_RADTRANSPORT_H

#include <AMReX_Amr.H>

#include <AMReX_FluxRegister.H>

#include <rad_boundary_values.H>
#include <rad_boundary_values.H>

#include <HypreSolvers.H>
#include <HypreSolvers.H>
#include <HypreSolvers.H>

#include <radiation_solve_params.H>

class RadTransportSolver {

 public:


  RadTransportSolver (amrex::Amr* Parent, int level,
            const amrex::BoxArray& grids,
            const amrex::DistributionMapping& dmap);

  static void read_params ();

  void levelSetup(int level);

  void levelBoundary(RadiativeBoundary& bd);

  void levelBoundary(MultiGroupBoundary& multigroup_boundary, const int comp);


  void applyCellMetrics(int level, amrex::MultiFab& acoefs);


  static void edgeGeomWeights(int idim, const amrex::Geometry& geom, const amrex::Box& edge_region,
                            amrex::Vector<amrex::Real>& r, amrex::Vector<amrex::Real>& s);


  void setLevelDiagCoeffs(int level, const amrex::MultiFab &a);

  void setLevelFaceCoeffs(int level, const amrex::MultiFab &b, int dir);

  void setLevelCellCoeffs(int level, const amrex::MultiFab &c, int dir);


  void levelDiagCoeffs(int level,
                    amrex::MultiFab& fkp, amrex::MultiFab& eta, amrex::MultiFab& coupling_inverse,
                    amrex::Real c, amrex::Real delta_t, amrex::Real theta);


  void levelFaceCoeffs(int level,
                    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& lambda,
                    amrex::MultiFab& kappa_r, int kcomp, amrex::Real c, int limiter_comp=0);


  void levelAssembleRhs(int level, amrex::MultiFab& rhs,
                amrex::MultiFab& temp,
                amrex::MultiFab& fkp, amrex::MultiFab& eta, amrex::MultiFab& coupling_inverse,
                amrex::MultiFab& matter_energy, amrex::MultiFab& state_energy,
                amrex::MultiFab& flux_div_old, amrex::MultiFab& rad_energy_old, amrex::MultiFab& Edot,
                amrex::Real delta_t, amrex::Real sigma, amrex::Real c, amrex::Real theta,
                amrex::FluxRegister* fine_correction, amrex::Real scale = 1.0,
                int rad_group_idx = -1, amrex::Real nu = -1.0, amrex::Real dnu = -1.0);


  void levelSolveSystem(int level, amrex::MultiFab& Er, int rad_group_idx, amrex::MultiFab& rhs,
                  amrex::Real sync_residual_scale);


  void levelFaceFlux(int level,
                 amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Flux,
                 amrex::MultiFab& Er, int rad_group_idx);

  void levelCoarseFineFlux(int level,
                    amrex::FluxRegister* flux_from_fine, amrex::FluxRegister* flux_to_coarse,
                    const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Flux,
                    int rad_group_idx);

  void levelFaceToCellFlux(int level, const amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Flux,
                             amrex::MultiFab& flx, int flux_comp_base);

  void levelDiffusionTerm(int level, amrex::MultiFab& diffusive_face_term, amrex::MultiFab& Er, int rad_group_idx);
  void levelRelease();


  void computeFaceCoeffs(amrex::MultiFab& bcoefs, int idim,
                      amrex::MultiFab& kappa_r, int kcomp,
                      amrex::MultiFab& lambda, int limiter_comp,
                      amrex::Real c, const amrex::Geometry& geom);

  void levelDiagCoeffs(int level, amrex::MultiFab& kappa_p,
                    amrex::Real delta_t, amrex::Real c, int rad_group_idx, amrex::Real pseudo_transient_tau);

  void levelAssembleRhs(int level, amrex::MultiFab& rhs, const amrex::MultiFab& jg,
                const amrex::MultiFab& groupTempCoef,
                const amrex::MultiFab& couple_source,
                const amrex::MultiFab& thermal_frac,
                const amrex::MultiFab& rad_energy_step, const amrex::MultiFab& gas_energy_step,
                const amrex::MultiFab& rad_energy_star, const amrex::MultiFab& gas_energy_star,
                amrex::Real delta_t, int rad_group_idx, int it, amrex::Real pseudo_transient_tau);

  void levelBoundaryScale(int level, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& lambda, int rad_group_idx,
                 int lo_bc[], int hi_bc[]);

  void levelDiffCoeffs(int level, amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& lambda,
                    amrex::MultiFab& vel, amrex::MultiFab& dcf);

  amrex::Real cSaved, d1Saved, d2Saved;

  void setSolverScale(amrex::Real cScale, amrex::Real d1Scale=0.0, amrex::Real d2Scale=0.0);
  void restoreSolverScale();

protected:

    amrex::Amr* parent;

    std::unique_ptr<AlphaBetaSolver> hd;
    std::unique_ptr<MultiLevelEllipticSolver> hm;
    std::unique_ptr<ExtendedMultiLevelSolver> hem;


};

#endif
