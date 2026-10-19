#ifndef FIELD_SOLVER_H
#define FIELD_SOLVER_H

#include <AMReX_AmrLevel.H>
#include <AMReX_MLLinOp.H>

#include <gravity_params.H>

// This vector can be accessed on the GPU.
using RealVector = amrex::Gpu::ManagedVector<amrex::Real>;

///
/// Multipole gravity data
///
namespace multipole {
    const int max_moment_order = 30;

    extern AMREX_GPU_MANAGED amrex::Real volumeScale;
    extern AMREX_GPU_MANAGED amrex::Real parityScale;

    extern AMREX_GPU_MANAGED amrex::Real rmax;

    extern AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> doFoldedAddLo;
    extern AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> doFoldedAddHi;
    extern AMREX_GPU_MANAGED bool doFoldedAdd;

    extern AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> doMirrorLo;
    extern AMREX_GPU_MANAGED amrex::Array1D<bool, 0, 2> doMirrorHi;

    extern AMREX_GPU_MANAGED amrex::Array2D<amrex::Real, 0, max_moment_order, 0, max_moment_order> factorialTable;
    extern AMREX_GPU_MANAGED amrex::Array1D<amrex::Real, 0, max_moment_order> axis_parity_table;
    extern AMREX_GPU_MANAGED amrex::Array2D<amrex::Real, 0, max_moment_order, 0, max_moment_order> azimuth_parity_table;
}

///
/// @class Gravity
/// @brief
///
class Gravity {

public:

///
/// Constructor
///
/// @param Parent           Parent ``Amr`` object
/// @param _top_level_hint    finest level
/// @param _boundary_rec         ``BCRec`` object, physical boundary conditions
/// @param _density_comp         index of density component
///
  Gravity(amrex::Amr* Parent, int _top_level_hint, amrex::BCRec* _boundary_rec, int _density_comp);

///
/// Deconstructor
///
  ~Gravity ();

///
/// Remove copy/move constructors/assignment operators.
///
  Gravity (const Gravity&) = delete;
  Gravity (Gravity&&) = delete;
  Gravity& operator= (const Gravity&) = delete;
  Gravity& operator= (Gravity&&) = delete;

///
/// Read gravity-related parameters from parameter file
///
  void read_params ();


///
/// @param jobInfoFile      std::ostream object
///
  static void report_run_settings(std::ostream& jobInfoFile);


///
/// Set ``Gravity`` object's ``level_point_count`` variable.
///
/// @param numpts       number of points
///
  void set_grid_points_gravity(int numpts);

///
/// Setup gravity at level ``level``.
///
/// @param level        integer, level number
/// @param amr_level_ptr   ``AmrLevel`` object containing state data at that level
/// @param volume       MultiFab, volume elements of cells
/// @param area         MultiFab, area elements of cells
///
  void attach_level (int                   level,
                      amrex::AmrLevel*             amr_level_ptr,
                      amrex::MultiFab&             volume,
                      amrex::MultiFab*             area);

///
/// Returns ``gravity_model_name``
///
  static std::string get_solver_kind ();

///
/// Returns ``solve_level_cap``
///
  static int get_solve_level_cap();

///
/// Returns ``no_sync``
///
  static int NoSync();

///
/// Returns ``composite_phi_fixup``
///
  static int CompositeFixEnabled();

///
/// Returns ``verify_solves_flag``
///
  static int get_self_check_flag ();


///
/// Set the ``mass_shift``
///
/// @param time             Current time
/// @param span_all_levels      boolean, do we iterate over all levels or mask off fine grids?
///
  void set_mass_shift(amrex::Real time, bool span_all_levels=true) const;

///
/// Return ``phi_gradient_old`` at given level
///
/// @param level        level index
///
  amrex::Vector<std::unique_ptr<amrex::MultiFab> >& get_potential_gradient_prev(int level);

///
/// Return ``phi_gradient_new`` at given level
///
/// @param level        level index
///
  amrex::Vector<std::unique_ptr<amrex::MultiFab> >& get_potential_gradient_curr(int level);


///
/// Return given component of ``phi_gradient_old`` at given level
///
/// @param level        level index
/// @param comp         component index
///
  amrex::MultiFab* read_prev_grad_comp(int level,int comp);


///
/// Return ``phi_gradient_new`` at given level plus the given vector
///
/// @param level        level index
/// @param addend       Vector of MultiFabs to add to grad phi
///
  void accumulate_grad_phi_now(int level, amrex::Vector<std::unique_ptr<amrex::MultiFab> >& addend);


///
/// Swap ``phi_gradient_old`` with ``phi_gradient_new`` at given level at set new ``phi_gradient_new`` to 1.e50.
///
/// @param level        level index
///
  void swapTimeLevels (int level);

///
/// Calculate the maximum value of the RHS over all levels.
/// This should only be called at a synchronization point where
/// all HydroSim levels have valid new time data at the same simulation time.
/// The RHS we will use is the density multiplied by 4*pi*G and also
/// multiplied by the metric terms, just as it would be in a real solve.
///
  void update_max_density();

///
/// Solve Poisson's equation to find the gravitational potential
///
/// @param level        level index
/// @param phi          MultiFab to store gravitational potential in
/// @param phi_gradient     Vector of MultiFabs, \f$ \nabla \Phi \f$
/// @param use_new_state       do we use state data at current time (1) or old time (0)?
///
  void solve_potential_level (int               level,
                      amrex::MultiFab&         phi,
                      const amrex::Vector<amrex::MultiFab*>& phi_gradient,
                      int               use_new_state);

///
/// Find delta phi
///
/// @param crse_level       index of coarse level
/// @param fine_level       index of fine level
/// @param rhs              Vector of MultiFabs with right hand side source terms
/// @param delta_phi        Vector of MultiFabs delta phi will be saved to
/// @param grad_correction   Vector of MultiFabs, gradient of delta phi
///
  void solve_correction_phi (int                        crse_level,
                            int                        fine_level,
                            const amrex::Vector<amrex::MultiFab*>&          rhs,
                            const amrex::Vector<amrex::MultiFab*>&          delta_phi,
                            const amrex::Vector<amrex::Vector<amrex::MultiFab*> >&  grad_correction);

///
/// Sync gravity across levels
///
/// @param crse_level       index of coarse level
/// @param fine_level       index of fine level
/// @param drho
/// @param dphi
///
  void synchronize_gravity_levels (int crse_level, int fine_level,
                     const amrex::Vector<amrex::MultiFab*>& drho, const amrex::Vector<amrex::MultiFab*>& dphi);


///
/// Multilevel solve for new phi from base level to finest level
///
/// @param level                        Base level index
/// @param finest_level                 Fine level index
///
  void multilevel_potential_solve (int level, int finest_level);

///
/// Actually do the multilevel solve for new phi from base level to finest level
///
/// @param level            Base level index
/// @param finest_level     Fine level index
/// @param phi_gradient         gradient of phi
/// @param use_new_state           Should we use the new state (1) or previous state (0)?
///
  void run_multilevel_solve      (int level, int finest_level,
                                     const amrex::Vector<amrex::Vector<amrex::MultiFab*> >& phi_gradient,
                                     int use_new_state);


///
/// Compute the difference between level and composite solves
///
/// @param level                       level index
/// @param comp_phi                    MultiFab containing computed phi
/// @param composite_grad_phi                   Vector of MultiFabs containing computed grad phi
/// @param composite_minus_level_phi        MultiFab, computed minus level phi
/// @param composite_minus_level_grad_phi   Vector of MultiFabs, computed minus level grad phi
///
  void build_composite_excess_grad(int level,
                                        amrex::MultiFab& comp_phi,
                                        const amrex::Vector<amrex::MultiFab*>& composite_grad_phi,
                                        amrex::MultiFab& composite_minus_level_phi,
                                        amrex::Vector<std::unique_ptr<amrex::MultiFab> >& composite_minus_level_grad_phi);


///
/// Get coarse phi on level ``level``-1
///
/// @param level        level index of fine data
/// @param coarse_potential     MultiFab to contain coarse phi
/// @param time         Current time
///
  void FetchCoarsePotential(int level,
                  amrex::MultiFab& coarse_potential,
                  amrex::Real time);


///
/// Get old gravity vector
///
/// @param level        Level index
/// @param gravity_field_out  MultiFab containing gravity vector
/// @param time         Current time
///
  void read_old_gravity_field(int level, amrex::MultiFab& gravity_field_out, amrex::Real time);

///
/// Get new gravity vector
///
/// @param level        Level index
/// @param gravity_field_out  MultiFab containing gravity vector
/// @param time         Current time
///
  void get_new_gravity_vector(int level, amrex::MultiFab& gravity_field_out, amrex::Real time);

///
/// Test whether using the edge-based gradients
/// to compute Div(Grad(Phi)) satisfies Lap(phi) = RHS
///
/// @param bx          box
/// @param rhs         right-hand-side
/// @param ecx         gradients wrt x
/// @param ecy         gradients wrt y
/// @param ecz         gradients wrt z
/// @param coord_type  coordinate system

  static void check_residual (const amrex::Box& bx,
                      amrex::Array4<amrex::Real> const& rhs,
                      amrex::Array4<amrex::Real> const& ecx,
#if AMREX_SPACEDIM >= 2
                      amrex::Array4<amrex::Real> const& ecy,
#endif
#if AMREX_SPACEDIM == 3
                      amrex::Array4<amrex::Real> const& ecz,
#endif
                      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
                      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& problo,
                      int coord_type);

///
/// @param level
///
  void verify_level_grad_phi_before(int level);

///
/// @param level
///
  void verify_level_grad_phi_now(int level);

///
/// @param level
///
  void check_composite_potential(int level);


///
/// @param level    Level index
/// @param use_new_state   Use new state data (1) or old state data (0)?
///
  void average_fine_edges_to_coarse(int level, int use_new_state);


///
/// Make radial gravity
///
/// @param level        Level index
/// @param time         Current time
/// @param radial_field  Vector containing gravity in radial direction
///
  void build_radial_field(int level, amrex::Real time, RealVector& radial_field);

///
/// Interpolate monopole gravitational field in the radial direction
///
/// @param level        Level index
/// @param radial_field  Radial gravity
/// @param gravity_field_out  Gravity vector
///
  void interpolate_monopole_field(int level, RealVector& radial_field, amrex::MultiFab& gravity_field_out) const;

///
/// Integrate radially outward to find radial mass distribution
///
/// @param bx           Box
/// @param u            Density (RHS of Poisson's equation)
/// @param radial_shell_mass  Radially integrated mass
/// @param radial_shell_vol   Radially integrated volume
/// @param radial_shell_pres  Radially integrated pressure
/// @param n1d          Number of radial points in the domain
/// @param level        Level index
///
  void tally_radial_profile(const amrex::Box& bx,
                           amrex::Array4<amrex::Real const> const& u,
                           RealVector& radial_shell_mass,
                           RealVector& radial_shell_vol,
#ifdef GR_GRAV
                           RealVector& radial_shell_pres,
#endif
                           int n1d, int level) const;

///
/// Implement multipole boundary conditions
///
/// @param crse_level
/// @param fine_level
/// @param Rhs
/// @param phi
///
  void fill_multipole_boundaries(int crse_level, int fine_level, const amrex::Vector<amrex::MultiFab*>& Rhs, amrex::MultiFab& phi);

///
/// Initialize multipole gravity
///
  void init_multipole_field() const;

#if (AMREX_SPACEDIM == 3)

///
/// Compute and fill direct sum boundary conditions
///
/// @param crse_level   Index of coarse level
/// @param fine_level   Index of fine level
/// @param Rhs          Vector of MultiFabs, right hand side
/// @param phi          MultiFab, phi
///
  void fill_direct_sum_boundaries(int crse_level, int fine_level, const amrex::Vector<amrex::MultiFab*>& Rhs, amrex::MultiFab& phi);
#endif

///
/// Make multigrid boundary conditions
///
  void build_solver_bc();

///
/// Pointers to amr,amrlevel.
///
  amrex::Amr*             parent;
  amrex::Vector<amrex::AmrLevel*> LevelTable;

///
/// Pointers to phi_gradient at previous and current time
///
  amrex::Vector< amrex::Vector<std::unique_ptr<amrex::MultiFab> > > phi_gradient_new;
  amrex::Vector< amrex::Vector<std::unique_ptr<amrex::MultiFab> > > phi_gradient_old;


///
/// BoxArray at each level
///
  const amrex::Vector<amrex::BoxArray>& grids;
  const amrex::Vector<amrex::DistributionMapping>& dmap;

///
/// Absolute tolerance on each level
///
  amrex::Vector<amrex::Real> abs_tol;

///
/// Relative tolerance on each level
///
  amrex::Vector<amrex::Real> rel_tol;

///
/// Resnorm at each level
///
  amrex::Vector<amrex::Real> level_solve_residual_norm;

///
/// Maximum value of the RHS (used for obtaining absolute tolerances)
///
  amrex::Real peak_rhs_value;

///
/// Volume and area fractions.
///
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
  static int   stencil_type;

  static amrex::Real inside_domain_radius;

#if (AMREX_SPACEDIM < 3)

///
/// Modify Rhs and coeffs with the appropriate metric terms.
///
/// @param level    Level index
/// @param Rhs      MultiFab
/// @param coeffs   Vector of MultiFabs
///
  void applyCurvatureTerms(int level,amrex::MultiFab& Rhs, const amrex::Vector<amrex::MultiFab*>& coeffs) const;
#endif

///
/// @param level        Index of level
/// @param phi          Gravitational potential
/// @param gravity_field_out  Gravity vector
///
    void include_point_source_field (int level, amrex::MultiFab& phi, amrex::MultiFab& gravity_field_out) const;

///
/// Get the rhs
///
/// @param crse_level   Index of coarse level
/// @param nlevs        Number of levels
/// @param use_new_state       Use new (1) or old (0) state data
///
    amrex::Vector<std::unique_ptr<amrex::MultiFab> > fetch_poisson_rhs (int crse_level, int nlevs, int use_new_state);


///
/// This is a sanity check on whether we are trying to fill multipole boundary conditiosn
///  for grids at this level > 0 -- this case is not currently supported.
///  Here we shrink the domain at this level by 1 in any direction which is not symmetry or periodic,
///  then ask if the grids at this level are contained in the shrunken domain.  If not, then grids
///  at this level touch the domain boundary and we must abort.
///
/// @param level    Level index
///
    void sanity_check (int level);

///
/// Do multigrid solve
///
/// @param crse_level   Coarse level index
/// @param fine_level   Fine level index
/// @param phi          Gravitational potential
/// @param rhs          Right hand side
/// @param phi_gradient     Grad phi
/// @param res
/// @param crse_bcdata
/// @param rel_tolerance      Relative tolerance
/// @param abs_tolerance      Absolute tolerance
///
    amrex::Real invoke_mlmg_solver (int crse_level, int fine_level,
                                        const amrex::Vector<amrex::MultiFab*>& phi,
                                        const amrex::Vector<const amrex::MultiFab*>& rhs,
                                        const amrex::Vector<std::array<amrex::MultiFab*,AMREX_SPACEDIM> >& phi_gradient,
                                        const amrex::Vector<amrex::MultiFab*>& res,
                                        const amrex::MultiFab* const crse_bcdata,
                                        amrex::Real rel_tolerance, amrex::Real abs_tolerance) const;


///
/// Do multigrid solve to find phi
///
/// @param crse_level   Coarse level index
/// @param fine_level   Fine level index
/// @param phi          Gravitational potential
/// @param rhs          Right hand side source term
/// @param phi_gradient     Grad phi
/// @param res
/// @param time         Current time
///
    amrex::Real solve_potential_mlmg (int crse_level, int fine_level,
                                     const amrex::Vector<amrex::MultiFab*>& phi,
                                     const amrex::Vector<amrex::MultiFab*>& rhs,
                                     const amrex::Vector<amrex::Vector<amrex::MultiFab*> >& phi_gradient,
                                     const amrex::Vector<amrex::MultiFab*>& res,
                                     amrex::Real time);

};

///
/// @class PotentialGradBCFunct
/// @brief A physical boundary condition function for grad phi
///
using PotentialGradBCFunct = amrex::PhysBCFunctNoOp;


#endif
