#ifndef HYDROSIM_RADIATION_H
#define HYDROSIM_RADIATION_H

#include <AMReX_AmrLevel.H>

#include <HydroSim.H>
#include <rad_boundary_values.H>
#include <rad_boundary_values.H>
#include <RadTransportSolver.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_Array.H>

#include <radiation_params.H>

class Radiation {

public:

  enum Solver_Kind { UnsetSolver = -1,
                     OneGroupSolver = 0,
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
  static int num_plot_vars;
  static amrex::Vector<std::string> plot_var_names;

  static int filter_limiter_T, limiter_filter_strength;
  static int filter_primitive_interval, filter_prim_temperature, filter_primitive_S;

  static int accelerate;

  int verbose;

  int maxiter;
  int min_outer_iter;
  int convergence_check_mode;
  amrex::Real relInnerTol, absInnerTolerance;
  int maxInnerIter;
  int minInnerIter;
  int skipAccelIfNoGain;
  int matter_update_mode;
  int bisect_after;
  amrex::Real dedT_safety;
  int inner_convergence_mode;
  amrex::Real delta_e_rat_dt_tol;
  amrex::Real delta_T_rat_dt_tol;
  int planck_lag_iter;
  int rosseland_lag_iter;
  int opacity_lag_iter;
  int limiter_lag_iter;
  int inner_limiter_update_iters;
  amrex::Real dT;
  amrex::Real underrelax_factor;
  int apply_sync;
  int do_stimulated_emission;

  static void load_static_settings();

  static amrex::Real mev_to_erg, clight, planckConst, boltzmannConst, Avogadro;

  static amrex::Real c, sigma, radConstant;

  static int active_group_id;
  static std::string group_label;

  static int hydro_only;

  amrex::Vector<std::unique_ptr<amrex::MultiFab> > plot_fields;


  Radiation(amrex::Amr* Parent, class HydroSim* hydrosim, int restart = 0);

  void regrid(int level, const amrex::BoxArray& grids,
              const amrex::DistributionMapping& dmap);


  void close(int level);


  void restart(int level, const amrex::BoxArray& grids,
               const amrex::DistributionMapping& dmap,
               const std::string& dir, std::istream& is);


  void checkPoint(int level,
                  const std::string& dir,
                  std::ostream&  os,
                  amrex::VisMF::How     how);



  void select_active_group(int rad_group_idx);

  amrex::Real group_frequency(int i) {
    return freq_center[i];
  }


  amrex::Real group_width(int i) {
    return freq_width[i];
  }



  amrex::FluxRegister* conservativeRegister(int level) {
      return conservative_register_old[level].get();
  }


  [[nodiscard]] amrex::Real energyChangeTol() const {
    return delta_e_rat_dt_tol;
  }

  [[nodiscard]] amrex::Real tempChangeTol() const {
    return delta_T_rat_dt_tol;
  }

  amrex::Real energyChangeRatioMax(int lev) {
    return delta_e_rat_level[lev];
  }

  amrex::Real tempChangeRatioMax(int lev) {
    return delta_T_rat_level[lev];
  }



  void post_init(int level);


  void begin_timestep(int level);


  void reset_flux_records(int level, int ncycle);


  void prepare_delayed_reflux(int level);


  void apply_delayed_reflux(int level, amrex::MultiFab& rhs, int indx);


  void advance_single_group(int level, int iteration, int ncycle);


  void MultigroupDiffusion_implicit_update(int level, int iteration, int ncycle);


  void dump_group_table(ostream& os);


  void reflux(int level);


  void build_flux_limiter(int level, const amrex::BoxArray& grids,
                       const amrex::MultiFab &Sborder,
                       const amrex::MultiFab &RadEnergyGrown,
                       amrex::MultiFab &rad_lambda_grown);


  void estimate_rad_gamma_pressure(const amrex::FArrayBox& state, const amrex::FArrayBox& Er,
                       amrex::FArrayBox& gPr, const amrex::Real* dx, const amrex::Box& box);


  void filter_primitives(int level, amrex::MultiFab& State);



  void evaluate_energy_transfer(amrex::MultiFab& exch, amrex::MultiFab& Er, amrex::MultiFab& fkp,
                        int rad_group_idx = -1);


  void compute_coupling_factors(amrex::MultiFab& eta, amrex::MultiFab& coupling_inverse, amrex::MultiFab& state,
                   amrex::MultiFab& temp, amrex::MultiFab& fkp, amrex::MultiFab& Er,
                   amrex::Real delta_t, amrex::Real c, amrex::Real underrelax_scale, int freeze_planck,
                   int rad_group_idx = 0);


  void apply_implicit_energy(amrex::Real& relative, amrex::Real& absolute,
                              amrex::MultiFab& state_energy_density, amrex::MultiFab& matter_energy_density,
                              amrex::MultiFab& eta, amrex::MultiFab& coupling_inverse,
                              amrex::MultiFab& flux_div_old, amrex::MultiFab& flux_div_new,
                              amrex::MultiFab& exchange, amrex::Real delta_t);


  void apply_implicit_energy(amrex::Real& relative, amrex::Real& absolute,
                              amrex::MultiFab& state_energy_density, amrex::MultiFab& matter_energy_density,
                              amrex::MultiFab& eta, amrex::MultiFab& coupling_inverse,
                              amrex::MultiFab& flux_div_old, amrex::MultiFab& flux_div_new,
                              amrex::MultiFab& exchange, amrex::MultiFab& diffusive_face_term,
                              amrex::Real delta_t);


  void apply_nonconservative_correction(amrex::Real& relative, amrex::Real& absolute,
                                     amrex::MultiFab& state_energy_density, amrex::MultiFab& matter_energy_density,
                                     amrex::MultiFab& eta, amrex::MultiFab& coupling_inverse,
                                     amrex::MultiFab& rad_energy_new,
                                     amrex::MultiFab& flux_div_old, amrex::MultiFab& flux_div_new,
                                     amrex::MultiFab& temp, amrex::MultiFab& fkp,
                                     amrex::MultiFab& state, amrex::Real delta_t);


  void sync_state_from_energy(amrex::MultiFab& state,  amrex::MultiFab& state_energy_density);


  void fetchBndryData(RadiativeBoundary& bd, amrex::MultiFab& Er, amrex::Real time, int level);

  void fetchMultigridBndry(MultiGroupBoundary& multigroup_boundary, amrex::MultiFab& Er, amrex::Real time, int level);

  void fetchMultigridBndryZeroRad(MultiGroupBoundary& multigroup_boundary, amrex::MultiFab& Er, int level);


  void fillBndryRegister(amrex::BndryRegister& bdry, int level, amrex::Real time);


  void evaluateGradientRatio(int level,
                      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& R,
                      amrex::MultiFab& kappa_r, int kcomp,
                      amrex::MultiFab& Er, int rad_group_idx,
                      int nGrow_rad_energy=0, int GradComp=0);

  void computeFluxLimiter(int level,
                   amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& lambda,
                   int limiter_comp=0);


  void get_heat_capacity(amrex::FArrayBox& c_v, amrex::FArrayBox& temp,
               amrex::FArrayBox& state, const amrex::Box& reg);


  void get_density_energy(amrex::MultiFab& fluid_rho_eint, amrex::MultiFab& state);

  void derive_planck_and_temp(amrex::MultiFab& fkp, amrex::MultiFab& temp, amrex::MultiFab& state,
                           int rad_group_idx = 0, amrex::Real delta_t = 0.0);

  void derive_rosseland_opacity(amrex::MultiFab& kappa_r, amrex::AmrLevel* hydro_level, int rad_group_idx = 0);

  void refresh_rosseland_from_temp(amrex::MultiFab& kappa_r,
                                  amrex::MultiFab& temp, amrex::MultiFab& state,
                                  const amrex::Geometry& geom,
                                  int rad_group_idx = 0);

  void derive_rosseland_velocity_coef(amrex::MultiFab& kappa_r, amrex::MultiFab& v, amrex::MultiFab& dcf,
                           amrex::Real delta_t, amrex::Real c,
                           amrex::AmrLevel* hydro_level,
                           int rad_group_idx = 0);

  void build_groups(int verbose);


  void extendBorderValues(amrex::MultiFab& f, int indx);


  void compute_rosseland_opacity(amrex::MultiFab& kappa_r, const amrex::MultiFab& state);

  void compute_rosseland_opacity(amrex::FArrayBox& kappa_r, const amrex::FArrayBox& state);

  void refresh_diffusion_coef(amrex::MultiFab& dcf, amrex::MultiFab& coupling_inverse, amrex::MultiFab& kp, amrex::MultiFab& kr,
                  const amrex::Geometry& geom);

  int use_kappa_deriv;


  int use_WienForm;
  amrex::Real Wien_Temp_Limit;


  void check_rad_convergence(amrex::Real& relative_inner_resid, amrex::Real& abs_residual_in, amrex::Real& rad_energy_error,
                            const amrex::MultiFab& rad_energy_new, const amrex::MultiFab& rad_energy_iter,
                            const amrex::MultiFab& kappa_p,
                            const amrex::MultiFab& thermal_frac_z,
                            const amrex::MultiFab& temp_next,
                            amrex::Real delta_t);

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

  void build_coupling_term(amrex::MultiFab& couple_source,
                        const amrex::MultiFab& kappa_p, const amrex::MultiFab& rad_energy_iter,
                        const amrex::MultiFab& jg);

  void compute_thermal_fraction(amrex::MultiFab& thermal_frac, amrex::MultiFab& thermal_frac_z,
                    amrex::MultiFab& rest_frac, amrex::MultiFab& dEmitT,
                    const amrex::MultiFab& dOpacT, const amrex::MultiFab& dedT,
                    const amrex::MultiFab& rad_energy_star, const amrex::MultiFab& rho,
                    amrex::Real delta_t, amrex::Real pseudo_transient_tau);

  void eos_fill_opacity_emission(const amrex::MultiFab& S_new,
                              const amrex::MultiFab& temp_next,
                              const amrex::MultiFab& temp_iterate,
                              amrex::MultiFab& kappa_p, amrex::MultiFab& kappa_r, amrex::MultiFab& jg,
                              amrex::MultiFab& dEmitT, amrex::MultiFab& dOpacT, amrex::MultiFab& dedT,
                              int level, int it, int ngrow);

  void gray_correction(amrex::MultiFab& rad_energy_new, amrex::MultiFab& rad_energy_iter,
                  amrex::MultiFab& kappa_p, amrex::MultiFab& kappa_r,
                  amrex::MultiFab& thermal_frac, amrex::MultiFab& rest_frac,
                  amrex::MultiFab& group_weight_mf,
                  amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& lambda,
                  RadTransportSolver* solver, MultiGroupBoundary& multigroup_boundary,
                  const amrex::BoxArray& grids, int level, amrex::Real time, amrex::Real delta_t, amrex::Real pseudo_transient_tau);

  void local_convergence_boost(amrex::MultiFab& rad_energy_new, const amrex::MultiFab& rad_energy_iter,
                   const amrex::MultiFab& kappa_p,
                   const amrex::MultiFab& thermal_frac,
                   const amrex::MultiFab& group_weight_mf,
                   amrex::Real delta_t, amrex::Real pseudo_transient_tau);

  void update_state_energy(amrex::MultiFab& state, const amrex::MultiFab& rhoe,
                           const amrex::MultiFab& temp, const amrex::BoxArray& grids,
                           amrex::Real& energy_rel_chg, amrex::Real& dT, int level);

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

  void bisect_temperature(amrex::MultiFab& rhoe_new, amrex::MultiFab& temp_next,
                     const amrex::MultiFab& gas_energy_star, const amrex::MultiFab& temp_iterate,
                     const amrex::MultiFab& S_new, const amrex::BoxArray& grids, int level);

  void MultigroupDiffusion_compute_rosseland(amrex::MultiFab& kappa_r, const amrex::MultiFab& state);

  void MultigroupDiffusion_compute_rosseland(amrex::FArrayBox& kappa_r, const amrex::FArrayBox& state);


  void MultigroupDiffusion_compute_scattering(amrex::FArrayBox& kappa_s, const amrex::FArrayBox& state);


  void rhs_to_rad_energy(amrex::MultiFab& rhs, amrex::Real dt, int level);

  void store_limiter_plotvar(int level, const amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& lambda);

  void store_lab_energy_plotvar(int level, const amrex::MultiFab& Snew,
                              const amrex::MultiFab& comoving_energy, const amrex::MultiFab& F, int flux_comp_base);

  void store_flux_plotvar(int level, const amrex::MultiFab& Snew,
                            const amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& lambda,
                            const amrex::MultiFab& Er, const amrex::MultiFab& F,
                            int flux_comp_base, const amrex::Real frame_sign = 1.0);

  void store_comoving_flux(int level, const amrex::MultiFab& Snew,
                                const amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& lambda,
                                const amrex::MultiFab& Er, const amrex::MultiFab& F, int flux_comp_base);

  amrex::Vector<amrex::Real> xnu, freq_center, freq_width, log_group_center, log_group_widths;

protected:

  amrex::Amr* parent;
  amrex::BCRec rad_boundary;
  amrex::Real reltol, abstol;
  amrex::Vector<amrex::Real> delta_e_rat_level, delta_T_rat_level;


  amrex::Vector <std::unique_ptr<amrex::FluxRegister> > flux_accum_reg;
  amrex::Vector <std::unique_ptr<amrex::FluxRegister> > conservative_register_old;
  amrex::Vector <std::unique_ptr<amrex::FluxRegister> > flux_candidate_regs;


  amrex::Vector<amrex::Real> dt_previous;


  amrex::Vector<std::unique_ptr<amrex::MultiFab> > flux_div;

  std::string group_unit_label;
  amrex::Real group_output_scale;

  amrex::IntVect spot;

};

enum RadComponentIds  { Rad  = 0 };

#endif
