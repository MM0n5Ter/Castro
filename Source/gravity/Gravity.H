#ifndef FIELD_SOLVER_H
#define FIELD_SOLVER_H

#include <AMReX_AmrLevel.H>
#include <AMReX_MLLinOp.H>

#include <gravity_params.H>

using RealVector = amrex::Gpu::ManagedVector<amrex::Real>;

using PotentialGradBCFunct = amrex::PhysBCFunctNoOp;

enum class PotentialTime { Old, New };

class Gravity {

public:

  Gravity(amrex::Amr* Parent, int _top_level_hint, amrex::BCRec* _boundary_rec, int _density_comp);

  ~Gravity ();

  Gravity (const Gravity&) = delete;
  Gravity (Gravity&&) = delete;
  Gravity& operator= (const Gravity&) = delete;
  Gravity& operator= (Gravity&&) = delete;

  void read_params ();

  static void report_run_settings(std::ostream& jobInfoFile);

  void set_grid_points_gravity(int numpts);

  void attach_level (int                   level,
                      amrex::AmrLevel*             amr_level_ptr,
                      amrex::MultiFab&             volume,
                      amrex::MultiFab*             area);

  static std::string get_solver_kind ();

  static int get_solve_level_cap();

  static int NoSync();

  static int CompositeFixEnabled();

  static int get_self_check_flag ();

  void set_mass_shift(amrex::Real time, bool span_all_levels=true) const;

  amrex::Vector<std::unique_ptr<amrex::MultiFab> >& get_potential_gradient_prev(int level);

  amrex::Vector<std::unique_ptr<amrex::MultiFab> >& get_potential_gradient_curr(int level);

  amrex::MultiFab* read_prev_grad_comp(int level,int comp);

  void accumulate_grad_phi_now(int level, amrex::Vector<std::unique_ptr<amrex::MultiFab> >& addend);

  void swapTimeLevels (int level);

  void update_max_density();

  void solve_potential_level (int               level,
                      amrex::MultiFab&         phi,
                      const amrex::Vector<amrex::MultiFab*>& phi_gradient,
                      PotentialTime epoch);

  void solve_correction_phi (int                        crse_level,
                            int                        fine_level,
                            const amrex::Vector<amrex::MultiFab*>&          rhs,
                            const amrex::Vector<amrex::MultiFab*>&          delta_phi,
                            const amrex::Vector<amrex::Vector<amrex::MultiFab*> >&  grad_correction);

  void synchronize_gravity_levels (int crse_level, int fine_level,
                     const amrex::Vector<amrex::MultiFab*>& drho, const amrex::Vector<amrex::MultiFab*>& dphi);

  void multilevel_potential_solve (int level, int finest_level);

  void run_multilevel_solve      (int level, int finest_level,
                                     const amrex::Vector<amrex::Vector<amrex::MultiFab*> >& phi_gradient,
                                     PotentialTime epoch);

  void build_composite_excess_grad(int level,
                                        amrex::MultiFab& comp_phi,
                                        const amrex::Vector<amrex::MultiFab*>& composite_grad_phi,
                                        amrex::MultiFab& composite_minus_level_phi,
                                        amrex::Vector<std::unique_ptr<amrex::MultiFab> >& composite_minus_level_grad_phi);

  void FetchCoarsePotential(int level,
                  amrex::MultiFab& coarse_potential,
                  amrex::Real time);

  void read_old_gravity_field(int level, amrex::MultiFab& gravity_field_out, amrex::Real time);

  void get_new_gravity_vector(int level, amrex::MultiFab& gravity_field_out, amrex::Real time);

  static void check_residual (const amrex::Box& bx,
                      amrex::Array4<amrex::Real> const& rhs,
                      const amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM>& edge_grad,
                      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
                      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& problo,
                      int coord_type);

  void verify_level_grad_phi_before(int level);

  void verify_level_grad_phi_now(int level);

  void check_composite_potential(int level);

  void average_fine_edges_to_coarse(int level, PotentialTime epoch);

  void build_radial_field(int level, amrex::Real time, RealVector& radial_field);

  void interpolate_monopole_field(int level, RealVector& radial_field, amrex::MultiFab& gravity_field_out) const;

  void tally_radial_profile(const amrex::Box& bx,
                           amrex::Array4<amrex::Real const> const& u,
                           RealVector& radial_shell_mass,
                           RealVector& radial_shell_vol,
#ifdef GR_GRAV
                           RealVector& radial_shell_pres,
#endif
                           int n1d, int level) const;

  void fill_multipole_boundaries(int crse_level, int fine_level, const amrex::Vector<amrex::MultiFab*>& Rhs, amrex::MultiFab& phi);

  void init_multipole_field() const;

#if (AMREX_SPACEDIM == 3)

  void fill_direct_sum_boundaries(int crse_level, int fine_level, const amrex::Vector<amrex::MultiFab*>& Rhs, amrex::MultiFab& phi);
#endif

  void build_solver_bc();

  amrex::Amr*             parent;
  amrex::Vector<amrex::AmrLevel*> LevelTable;

  amrex::Vector< amrex::Vector<std::unique_ptr<amrex::MultiFab> > > phi_gradient_new;
  amrex::Vector< amrex::Vector<std::unique_ptr<amrex::MultiFab> > > phi_gradient_old;

  const amrex::Vector<amrex::BoxArray>& grids;
  const amrex::Vector<amrex::DistributionMapping>& dmap;

  amrex::Vector<amrex::Real> abs_tol;

  amrex::Vector<amrex::Real> rel_tol;

  amrex::Vector<amrex::Real> level_solve_residual_norm;

  amrex::Real peak_rhs_value;

  amrex::Vector<amrex::MultiFab*> volume;
  amrex::Vector<amrex::MultiFab*> area;

  int Density;
  int allocated_top_level;

  amrex::BCRec*       phys_bc;

  std::array<amrex::MLLinOp::BCType,AMREX_SPACEDIM> mlmg_lobc;
  std::array<amrex::MLLinOp::BCType,AMREX_SPACEDIM> mlmg_hibc;

  int   level_point_count;

  static int   verify_solves_flag;
  static amrex::Real  mass_shift;
  amrex::Vector< RealVector > radial_field_old;
  amrex::Vector< RealVector > radial_field_new;
  amrex::Vector< RealVector > radial_shell_mass;
  amrex::Vector< RealVector > radial_shell_vol;
#ifdef GR_GRAV
  amrex::Vector< RealVector > radial_shell_pres;
#endif
#if (AMREX_SPACEDIM < 3)

  void applyCurvatureTerms(int level,amrex::MultiFab& Rhs, const amrex::Vector<amrex::MultiFab*>& coeffs) const;
#endif

    void include_point_source_field (int level, amrex::MultiFab& phi, amrex::MultiFab& gravity_field_out) const;

    amrex::Vector<std::unique_ptr<amrex::MultiFab> > fetch_poisson_rhs (int crse_level, int nlevs, PotentialTime epoch);

    void sanity_check (int level);

    struct SolveTolerances {
        amrex::Real rel;
        amrex::Real abs;
    };

    amrex::Real invoke_mlmg_solver (int crse_level, int fine_level,
                                        const amrex::Vector<amrex::MultiFab*>& phi,
                                        const amrex::Vector<const amrex::MultiFab*>& rhs,
                                        const amrex::Vector<std::array<amrex::MultiFab*,AMREX_SPACEDIM> >& phi_gradient,
                                        const amrex::Vector<amrex::MultiFab*>& res,
                                        const amrex::MultiFab* const crse_bcdata,
                                        const SolveTolerances& tol) const;

    amrex::Real solve_potential_mlmg (int crse_level, int fine_level,
                                     const amrex::Vector<amrex::MultiFab*>& phi,
                                     const amrex::Vector<amrex::MultiFab*>& rhs,
                                     const amrex::Vector<amrex::Vector<amrex::MultiFab*> >& phi_gradient,
                                     const amrex::Vector<amrex::MultiFab*>& res,
                                     amrex::Real time);

};

#endif
