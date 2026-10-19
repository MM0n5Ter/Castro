#ifndef HYDROSIM_RADIATION_H
#define HYDROSIM_RADIATION_H

#include <AMReX_AmrLevel.H>

#include <HydroSim.H>
#include <RadComponents.H>
#include <RadiativeBoundary.H>
#include <MultiGroupBoundary.H>
#include <RadTransportSolver.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_Array.H>

#include <radiation_params.H>

///
/// @class Radiation
/// @brief
///
class Radiation {

public:

  enum Solver_Kind { UnsetSolver = -1,
                     OneGroupSolver = 0,
                     // ProtoMultiGroupSolver = 1,  // deleted
                     //  NeutrinoSolver = 2,  // deleted
                     //  SGFLDHybrid = 3,  // deleted
                     //  SGFLDImplicit = 4, // deleted
                     SingleGroupFLDSolver = 5,
                     MultiGroupFLDSolver = 6};

  static Solver_Kind SolverKind;

  static amrex::Real radiativeToEnergy, EnergyToRadFactor, fluxFromRad;

  static int groups_enabled;
  static int nGroups;

  static int rad_hydro_coupled;
  static int lorentz_energy_term;

  static int comp_limiter, icomp_kappa_planck, icomp_kappa_rosseland;
  static int comp_lab_energy, icomp_lab_flux, icomp_comoving_flux;
  static int num_plot_vars;   // extra plot variables
  static amrex::Vector<std::string> plot_var_names;

  static int filter_limiter_T, limiter_filter_strength;
  static int filter_primitive_interval, filter_prim_temperature, filter_primitive_S;

  static int accelerate;        ///< controls multigroup convergence acceleration

  int verbose;

  int maxiter;           ///< iteration limit for implicit update loop
  int min_outer_iter;
  int convergence_check_mode; ///< 0 (default): both 2 and 3,
                              ///< 1: rhoe,
                              ///< 2: residue of Eq. rhoe,
                              ///< 3: T
  amrex::Real relInnerTol, absInnerTolerance; ///< tolerance for inner iteration of J equation
  int maxInnerIter;           ///< iteration limit for inner iteration of J equation
  int minInnerIter;
  int skipAccelIfNoGain;   ///< Skip acceleration if it doesn't help
  int matter_update_mode; ///< 0: conservative  1: non-conservative  2: C and NC interwoven
                          ///< The last outer iteration is always conservative.
  int bisect_after;  ///< Bisection after bisect_after iterations
  amrex::Real dedT_safety; ///< Make dedT larger for safety in Newton iteration
  int inner_convergence_mode;
  amrex::Real delta_e_rat_dt_tol; ///< energy change tolerance for adjusting timestep
  amrex::Real delta_T_rat_dt_tol; ///< T change tolerance for adjusting timestep
  int planck_lag_iter;     ///< after this number of iterations, lag planck
  int rosseland_lag_iter;  ///< after this number of iterations, lag rosseland
  int opacity_lag_iter;
  int limiter_lag_iter;    ///< after this number of iterations, lag limiter
  int inner_limiter_update_iters; ///< This is for MultigroupDiffusion solver.
                            ///< Stop updating limiter after ? inner iterations
                            ///< 0 means lagging by one outer iteration
  amrex::Real dT;               ///< temperature step for derivative estimate
  amrex::Real underrelax_factor;         ///< factor controlling progressive underrelaxation
  int apply_sync;               ///< perform sync (if false zero out sync source)
  int do_stimulated_emission;

  static void load_static_settings();

///
/// Physical constants for use in multigroup, made static and public
/// so that other modules such as RadTransportSolver can get to them easily.
/// Values are set in the Radiation constructor.
///
  static amrex::Real mev_to_erg, clight, planckConst, boltzmannConst, Avogadro;

///
/// Values used internally for speed of light, Stefan-Boltzmann constant,
/// radiation constant.  (For contrived test problems, these may differ
/// from the proper physical values, so be careful if you use them!)
///
  static amrex::Real c, sigma, radConstant;

  static int active_group_id;
  static std::string group_label;

  static int hydro_only;

  amrex::Vector<std::unique_ptr<amrex::MultiFab> > plot_fields;


///
/// @param Parent
/// @param hydrosim
/// @param restart
///
  Radiation(amrex::Amr* Parent, class HydroSim* hydrosim, int restart = 0);

///
/// @param level
/// @param grids
/// @param dmap
///
  void regrid(int level, const amrex::BoxArray& grids,
              const amrex::DistributionMapping& dmap);


///
/// @param level
///
  void close(int level);


///
/// @param level
/// @param grids
/// @param dmap
/// @param dir
/// @param is
///
  void restart(int level, const amrex::BoxArray& grids,
               const amrex::DistributionMapping& dmap,
               const std::string& dir, std::istream& is);


///
/// @param level
/// @param dir
/// @param os
/// @param how
///
  void checkPoint(int level,
                  const std::string& dir,
                  std::ostream&  os,
                  amrex::VisMF::How     how);


///
/// access to group information

///
/// @param rad_group_idx
///
  void select_active_group(int rad_group_idx);

/// @param i
///
  amrex::Real group_frequency(int i) {
    return freq_center[i];
  }


///
/// @param i
///
  amrex::Real group_width(int i) {
    return freq_width[i];
  }



///
/// diagnostics for adjusting timestep
/// @param lev
///
  amrex::Real energyChangeRatioMax(int lev) {
    return delta_e_rat_level[lev];
  }


///
/// @param lev
///
  amrex::Real tempChangeRatioMax(int lev) {
    return delta_T_rat_level[lev];
  }

  [[nodiscard]] amrex::Real energyChangeTol() const {
    return delta_e_rat_dt_tol;
  }

  [[nodiscard]] amrex::Real tempChangeTol() const {
    return delta_T_rat_dt_tol;
  }

///
/// access to conservation flux register for energy diagnostics
///
/// @param level
///
  amrex::FluxRegister* conservativeRegister(int level) {
      return conservative_register_old[level].get();
  }



///
/// @param level
///
  void post_init(int level);


///
/// @param level
///
  void begin_timestep(int level);


///
/// @param level
/// @param ncycle
///
  void reset_flux_records(int level, int ncycle);


///
/// @param level
///
  void prepare_delayed_reflux(int level);


///
/// @param level
/// @param rhs
/// @param indx
///
  void apply_delayed_reflux(int level, amrex::MultiFab& rhs, int indx);


///
/// @param level
/// @param iteration
/// @param ncycle
///
  void advance_single_group(int level, int iteration, int ncycle);


///
/// @param level
/// @param iteration
/// @param ncycle
///
  void MultigroupDiffusion_implicit_update(int level, int iteration, int ncycle);


///
/// @param os
///
  void dump_group_table(ostream& os);


///
/// @param level
///
  void reflux(int level);


///
/// @param level
/// @param grids
/// @param &Sborder
/// @param &RadEnergyGrown
/// @param &rad_lambda_grown
///
  void build_flux_limiter(int level, const amrex::BoxArray& grids,
                       const amrex::MultiFab &Sborder,
                       const amrex::MultiFab &RadEnergyGrown,
                       amrex::MultiFab &rad_lambda_grown);


///
/// @param state
/// @param Er
/// @param gPr
/// @param dx
/// @param box
///
  void estimate_rad_gamma_pressure(const amrex::FArrayBox& state, const amrex::FArrayBox& Er,
                       amrex::FArrayBox& gPr, const amrex::Real* dx, const amrex::Box& box);


///
/// @param level
/// @param State
///
  void filter_primitives(int level, amrex::MultiFab& State);

  // internal utility functions:


///
/// @param exch
/// @param Er
/// @param fkp
/// @param rad_group_idx
///
  void evaluate_energy_transfer(amrex::MultiFab& exch, amrex::MultiFab& Er, amrex::MultiFab& fkp,
                        int rad_group_idx = -1);


///
/// @param eta
/// @param coupling_inverse
/// @param state
/// @param temp
/// @param fkp
/// @param Er
/// @param delta_t
/// @param c
/// @param underrelax_scale
/// @param freeze_planck
/// @param rad_group_idx
///
  void compute_coupling_factors(amrex::MultiFab& eta, amrex::MultiFab& coupling_inverse, amrex::MultiFab& state,
                   amrex::MultiFab& temp, amrex::MultiFab& fkp, amrex::MultiFab& Er,
                   amrex::Real delta_t, amrex::Real c, amrex::Real underrelax_scale, int freeze_planck,
                   int rad_group_idx = 0);


///
/// @param relative
/// @param absolute
/// @param state_energy_density
/// @param matter_energy_density
/// @param eta
/// @param coupling_inverse
/// @param flux_div_old
/// @param flux_div_new
/// @param exchange
/// @param delta_t
///
  void apply_implicit_energy(amrex::Real& relative, amrex::Real& absolute,
                              amrex::MultiFab& state_energy_density, amrex::MultiFab& matter_energy_density,
                              amrex::MultiFab& eta, amrex::MultiFab& coupling_inverse,
                              amrex::MultiFab& flux_div_old, amrex::MultiFab& flux_div_new,
                              amrex::MultiFab& exchange, amrex::Real delta_t);


///
/// @param relative
/// @param absolute
/// @param state_energy_density
/// @param matter_energy_density
/// @param eta
/// @param coupling_inverse
/// @param flux_div_old
/// @param flux_div_new
/// @param exchange
/// @param diffusive_face_term
/// @param delta_t
///
  void apply_implicit_energy(amrex::Real& relative, amrex::Real& absolute,
                              amrex::MultiFab& state_energy_density, amrex::MultiFab& matter_energy_density,
                              amrex::MultiFab& eta, amrex::MultiFab& coupling_inverse,
                              amrex::MultiFab& flux_div_old, amrex::MultiFab& flux_div_new,
                              amrex::MultiFab& exchange, amrex::MultiFab& diffusive_face_term,
                              amrex::Real delta_t);


///
/// @param relative
/// @param absolute
/// @param state_energy_density
/// @param matter_energy_density
/// @param eta
/// @param coupling_inverse
/// @param rad_energy_new
/// @param flux_div_old
/// @param flux_div_new
/// @param temp
/// @param fkp
/// @param state
/// @param delta_t
///
  void apply_nonconservative_correction(amrex::Real& relative, amrex::Real& absolute,
                                     amrex::MultiFab& state_energy_density, amrex::MultiFab& matter_energy_density,
                                     amrex::MultiFab& eta, amrex::MultiFab& coupling_inverse,
                                     amrex::MultiFab& rad_energy_new,
                                     amrex::MultiFab& flux_div_old, amrex::MultiFab& flux_div_new,
                                     amrex::MultiFab& temp, amrex::MultiFab& fkp,
                                     amrex::MultiFab& state, amrex::Real delta_t);


///
/// @param state
/// @param state_energy_density
///
  void sync_state_from_energy(amrex::MultiFab& state,  amrex::MultiFab& state_energy_density);


///
/// @param bd
/// @param Er
/// @param time
/// @param level
///
  void fetchBndryData(RadiativeBoundary& bd, amrex::MultiFab& Er, amrex::Real time, int level);

///
/// multigroup version
///
/// @param multigroup_boundary
/// @param Er
/// @param time
/// @param level
///
  void fetchMultigridBndry(MultiGroupBoundary& multigroup_boundary, amrex::MultiFab& Er, amrex::Real time, int level);

///
/// @param multigroup_boundary
/// @param Er
/// @param level
///
  void fetchMultigridBndryZeroRad(MultiGroupBoundary& multigroup_boundary, amrex::MultiFab& Er, int level);


///
/// @param bdry
/// @param level
/// @param time
///
  void fillBndryRegister(amrex::BndryRegister& bdry, int level, amrex::Real time);

  // Flux limiter functions, potentially for use by all update modules

///
/// Computes the scaled gradient for use in flux limiters & Eddington factors
///
/// @param level
/// @param amrex::Array<amrex::MultiFab
/// @param R
/// @param kappa_r
/// @param kcomp
/// @param Er
/// @param rad_group_idx
/// @param nGrow_rad_energy
/// @param GradComp
///
  void evaluateGradientRatio(int level,
                      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& R,
                      amrex::MultiFab& kappa_r, int kcomp,
                      amrex::MultiFab& Er, int rad_group_idx,
                      int nGrow_rad_energy=0, int GradComp=0);

///
/// On input, lambda should contain scaled gradient.
/// On output this will be overwritten with the flux limiter.
///
/// @param level
/// @param amrex::Array<amrex::MultiFab
/// @param lambda
/// @param limiter_comp
///
  void computeFluxLimiter(int level,
                   amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& lambda,
                   int limiter_comp=0);

///
/// Fab versions of conversion functions.
///

///
/// @param c_v
/// @param temp
/// @param state
/// @param reg
///
  void get_heat_capacity(amrex::FArrayBox& c_v, amrex::FArrayBox& temp,
               amrex::FArrayBox& state, const amrex::Box& reg);


///
/// MultiFab versions of conversion functions call Fab versions for eos data
///
/// @param fluid_rho_eint
/// @param state
///
  void get_density_energy(amrex::MultiFab& fluid_rho_eint, amrex::MultiFab& state);

///
/// Planck mean is only needed in the grid interiors, but must be
/// computed separately for each fluid (fkp.nVar() == N fluids).
/// temp contains fluid_rho_eint on input
///
/// @param fkp
/// @param temp
/// @param state
/// @param rad_group_idx
/// @param delta_t
///
  void derive_planck_and_temp(amrex::MultiFab& fkp, amrex::MultiFab& temp, amrex::MultiFab& state,
                           int rad_group_idx = 0, amrex::Real delta_t = 0.0);

///
/// Rosseland mean is needed on an extended region (nGrow == 1).
/// Fills patches with state information on extended region
///
/// @param kappa_r
/// @param hydro_level
/// @param rad_group_idx
///
  void derive_rosseland_opacity(amrex::MultiFab& kappa_r, amrex::AmrLevel* hydro_level, int rad_group_idx = 0);

///
/// For the interior opacities, temperature is assumed to be known (input)
///
/// Updates Rosseland mean in interior of level, leaves ghost values
/// at physical or coarse-fine interfaces untouched
///
/// @param kappa_r
/// @param temp
/// @param state
/// @param geom
/// @param rad_group_idx
///
  void refresh_rosseland_from_temp(amrex::MultiFab& kappa_r,
                                  amrex::MultiFab& temp, amrex::MultiFab& state,
                                  const amrex::Geometry& geom,
                                  int rad_group_idx = 0);

///
/// RadiativeHydro
///
/// @param kappa_r
/// @param v
/// @param dcf
/// @param delta_t
/// @param c
/// @param hydro_level
/// @param rad_group_idx
///
  void derive_rosseland_velocity_coef(amrex::MultiFab& kappa_r, amrex::MultiFab& v, amrex::MultiFab& dcf,
                           amrex::Real delta_t, amrex::Real c,
                           amrex::AmrLevel* hydro_level,
                           int rad_group_idx = 0);

///
/// multigroup methods
///
/// @param verbose
///
  void build_groups(int verbose);


///
/// @param f
/// @param indx
///
  void extendBorderValues(amrex::MultiFab& f, int indx);


///
/// @param kappa_r
/// @param state
///
  void compute_rosseland_opacity(amrex::MultiFab& kappa_r, const amrex::MultiFab& state);

///
/// @param kappa_r
/// @param state
///
  void compute_rosseland_opacity(amrex::FArrayBox& kappa_r, const amrex::FArrayBox& state);

///
/// @param dcf
/// @param coupling_inverse
/// @param kp
/// @param kr
/// @param geom
///
  void refresh_diffusion_coef(amrex::MultiFab& dcf, amrex::MultiFab& coupling_inverse, amrex::MultiFab& kp, amrex::MultiFab& kr,
                  const amrex::Geometry& geom);

///
/// <MultigroupDiffusion>
///
  int use_kappa_deriv;


///
/// <Shestakov-Bolstad>
///
  int use_WienForm;
  amrex::Real Wien_Temp_Limit;

///
/// </ Shestakov-Bolstad>
///

///
/// @param relative_inner_resid
/// @param abs_residual_in
/// @param rad_energy_error
/// @param rad_energy_new
/// @param rad_energy_iter
/// @param kappa_p
/// @param thermal_frac_z
/// @param temp_next
/// @param delta_t
///
  void check_rad_convergence(amrex::Real& relative_inner_resid, amrex::Real& abs_residual_in, amrex::Real& rad_energy_error,
                            const amrex::MultiFab& rad_energy_new, const amrex::MultiFab& rad_energy_iter,
                            const amrex::MultiFab& kappa_p,
                            const amrex::MultiFab& thermal_frac_z,
                            const amrex::MultiFab& temp_next,
                            amrex::Real delta_t);

///
/// @param rhoe_new
/// @param gas_energy_star
/// @param gas_energy_step
/// @param rad_energy_new
/// @param temp_next
/// @param temp_iterate
/// @param rho
/// @param kappa_p
/// @param jg
/// @param dedT
/// @param rel_gas_energy
/// @param abs_gas_energy
/// @param rel_resid
/// @param abs_resid
/// @param rel_temp
/// @param abs_temp
/// @param delta_t
///
  void check_matter_convergence(const amrex::MultiFab& rhoe_new, const amrex::MultiFab& gas_energy_star,
                              const amrex::MultiFab& gas_energy_step, const amrex::MultiFab& rad_energy_new,
                              const amrex::MultiFab& temp_next, const amrex::MultiFab& temp_iterate,
                              const amrex::MultiFab& rho,
                              const amrex::MultiFab& kappa_p, const amrex::MultiFab& jg,
                              const amrex::MultiFab& dedT,
                              amrex::Real& rel_gas_energy, amrex::Real& abs_gas_energy,
                              amrex::Real& rel_resid,   amrex::Real& abs_resid,
                              amrex::Real& rel_temp,    amrex::Real& abs_temp,
                              amrex::Real delta_t);

///
/// @param couple_source
/// @param kappa_p
/// @param rad_energy_iter
/// @param jg
///
  void build_coupling_term(amrex::MultiFab& couple_source,
                        const amrex::MultiFab& kappa_p, const amrex::MultiFab& rad_energy_iter,
                        const amrex::MultiFab& jg);

///
/// @param thermal_frac
/// @param thermal_frac_z
/// @param rest_frac
/// @param dEmitT
/// @param dOpacT
/// @param dedT
/// @param rad_energy_star
/// @param rho
/// @param delta_t
/// @param pseudo_transient_tau
///
  void compute_thermal_fraction(amrex::MultiFab& thermal_frac, amrex::MultiFab& thermal_frac_z,
                    amrex::MultiFab& rest_frac, amrex::MultiFab& dEmitT,
                    const amrex::MultiFab& dOpacT, const amrex::MultiFab& dedT,
                    const amrex::MultiFab& rad_energy_star, const amrex::MultiFab& rho,
                    amrex::Real delta_t, amrex::Real pseudo_transient_tau);

///
/// @param S_new
/// @param temp_next
/// @param temp_iterate
/// @param kappa_p
/// @param kappa_r
/// @param jg
/// @param dEmitT
/// @param dOpacT
/// @param dedT
/// @param level
/// @param it
/// @param ngrow
///
  void eos_fill_opacity_emission(const amrex::MultiFab& S_new,
                              const amrex::MultiFab& temp_next,
                              const amrex::MultiFab& temp_iterate,
                              amrex::MultiFab& kappa_p, amrex::MultiFab& kappa_r, amrex::MultiFab& jg,
                              amrex::MultiFab& dEmitT, amrex::MultiFab& dOpacT, amrex::MultiFab& dedT,
                              int level, int it, int ngrow);

///
/// @param rad_energy_new
/// @param rad_energy_iter
/// @param kappa_p
/// @param kappa_r
/// @param thermal_frac
/// @param rest_frac
/// @param group_weight_mf
/// @param lambda
/// @param solver
/// @param multigroup_boundary
/// @param grids
/// @param level
/// @param time
/// @param delta_t
/// @param pseudo_transient_tau
///
  void gray_correction(amrex::MultiFab& rad_energy_new, amrex::MultiFab& rad_energy_iter,
                  amrex::MultiFab& kappa_p, amrex::MultiFab& kappa_r,
                  amrex::MultiFab& thermal_frac, amrex::MultiFab& rest_frac,
                  amrex::MultiFab& group_weight_mf,
                  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& lambda,
                  RadTransportSolver* solver, MultiGroupBoundary& multigroup_boundary,
                  const amrex::BoxArray& grids, int level, amrex::Real time, amrex::Real delta_t, amrex::Real pseudo_transient_tau);

///
/// @param rad_energy_new
/// @param rad_energy_iter
/// @param kappa_p
/// @param thermal_frac
/// @param group_weight_mf
/// @param delta_t
/// @param pseudo_transient_tau
///
  void local_convergence_boost(amrex::MultiFab& rad_energy_new, const amrex::MultiFab& rad_energy_iter,
                   const amrex::MultiFab& kappa_p,
                   const amrex::MultiFab& thermal_frac,
                   const amrex::MultiFab& group_weight_mf,
                   amrex::Real delta_t, amrex::Real pseudo_transient_tau);

///
/// @param state
/// @param rhoe
/// @param temp
/// @param grids
/// @param energy_rel_chg
/// @param dT
/// @param level
///
  void update_state_energy(amrex::MultiFab& state, const amrex::MultiFab& rhoe,
                           const amrex::MultiFab& temp, const amrex::BoxArray& grids,
                           amrex::Real& energy_rel_chg, amrex::Real& dT, int level);

///
/// @param rhoe_new
/// @param temp_next
/// @param rad_energy_new
/// @param rad_energy_iter
/// @param gas_energy_star
/// @param gas_energy_step
/// @param thermal_frac
/// @param thermal_frac_z
/// @param rest_frac
/// @param couple_source
/// @param kappa_p
/// @param jg
/// @param group_weight_mf
/// @param S_new
/// @param level
/// @param delta_t
/// @param pseudo_transient_tau
/// @param it
/// @param energy_conserving_mode
///
  void advance_matter(amrex::MultiFab& rhoe_new, amrex::MultiFab& temp_next,
                     const amrex::MultiFab& rad_energy_new, const amrex::MultiFab& rad_energy_iter,
                     const amrex::MultiFab& gas_energy_star,
                     const amrex::MultiFab& gas_energy_step,
                     const amrex::MultiFab& thermal_frac, const amrex::MultiFab& thermal_frac_z,
                     const amrex::MultiFab& rest_frac,
                     const amrex::MultiFab& couple_source,
                     const amrex::MultiFab& kappa_p, const amrex::MultiFab& jg,
                     const amrex::MultiFab& group_weight_mf,
                     const amrex::MultiFab& S_new,
                     int level, amrex::Real delta_t,
                     amrex::Real pseudo_transient_tau, int it, bool energy_conserving_mode);

///
/// @param rhoe_new
/// @param temp_next
/// @param gas_energy_star
/// @param temp_iterate
/// @param S_new
/// @param grids
/// @param level
///
  void bisect_temperature(amrex::MultiFab& rhoe_new, amrex::MultiFab& temp_next,
                     const amrex::MultiFab& gas_energy_star, const amrex::MultiFab& temp_iterate,
                     const amrex::MultiFab& S_new, const amrex::BoxArray& grids, int level);

///
/// for the hyperbolic solver
///
/// @param kappa_r
/// @param state
///
  void MultigroupDiffusion_compute_rosseland(amrex::MultiFab& kappa_r, const amrex::MultiFab& state);

///
/// @param kappa_r
/// @param state
///
  void MultigroupDiffusion_compute_rosseland(amrex::FArrayBox& kappa_r, const amrex::FArrayBox& state);


///
/// @param kappa_s
/// @param state
///
  void MultigroupDiffusion_compute_scattering(amrex::FArrayBox& kappa_s, const amrex::FArrayBox& state);


///
/// @param rhs
/// @param dt
/// @param level
///
  void rhs_to_rad_energy(amrex::MultiFab& rhs, amrex::Real dt, int level);

///
/// </ MultigroupDiffusion>
///
/// @param level
/// @param amrex::Array<amrex::MultiFab
/// @param lambda
///
  void store_limiter_plotvar(int level, const amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& lambda);

///
/// @param level
/// @param Snew
/// @param comoving_energy
/// @param F
/// @param flux_comp_base
///
  void store_lab_energy_plotvar(int level, const amrex::MultiFab& Snew,
                              const amrex::MultiFab& comoving_energy, const amrex::MultiFab& F, int flux_comp_base);

///
/// @param level
/// @param Snew
/// @param amrex::Array<amrex::MultiFab
/// @param lambda
/// @param Er
/// @param F
/// @param flux_comp_base
/// @param frame_sign
///
  void store_flux_plotvar(int level, const amrex::MultiFab& Snew,
                            const amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& lambda,
                            const amrex::MultiFab& Er, const amrex::MultiFab& F,
                            int flux_comp_base, const amrex::Real frame_sign = 1.0);

///
/// @param level
/// @param Snew
/// @param amrex::Array<amrex::MultiFab
/// @param lambda
/// @param Er
/// @param F
/// @param flux_comp_base
///
  void store_comoving_flux(int level, const amrex::MultiFab& Snew,
                                const amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& lambda,
                                const amrex::MultiFab& Er, const amrex::MultiFab& F, int flux_comp_base);

  amrex::Vector<amrex::Real> xnu, freq_center, freq_width, log_group_center, log_group_widths;

protected:

  amrex::Amr* parent;
  amrex::BCRec rad_boundary;          ///< types defined in LO_BCTYPES.H, not BCTYPES.H
  amrex::Real reltol, abstol;   ///< tolerances for implicit update loop
  amrex::Vector<amrex::Real> delta_e_rat_level, delta_T_rat_level;


///
/// arrays of flux registers
///   flux_accum_reg[level] is a amrex::FluxRegister, which contains fluxes for all groups
///
  amrex::Vector <std::unique_ptr<amrex::FluxRegister> > flux_accum_reg;
  amrex::Vector <std::unique_ptr<amrex::FluxRegister> > conservative_register_old;
  amrex::Vector <std::unique_ptr<amrex::FluxRegister> > flux_candidate_regs;


///
/// for deferred sync
///
  amrex::Vector<amrex::Real> dt_previous;


///
/// divergence of flux
///
  amrex::Vector<std::unique_ptr<amrex::MultiFab> > flux_div;

  std::string group_unit_label;
  amrex::Real group_output_scale;

  amrex::IntVect spot;

};

#endif
