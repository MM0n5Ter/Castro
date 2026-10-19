#include <cmath>
#include <limits>
#include <numbers>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <AMReX_ParmParse.H>
#include <Gravity.H>
#include <HydroSim.H>

#include <AMReX_FillPatchUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>

#include <hydrosim_limits.H>

#include <fundamental_constants.H>

#include <Multipole_kernels.H>
#include <SolverUtils.H>

using namespace amrex;

#ifdef AMREX_DEBUG
int Gravity::verify_solves_flag  = 1;
#else
int Gravity::verify_solves_flag  = 0;
#endif
Real Gravity::mass_shift    =  0.0;

// ************************************************************************************** //

// FourPiG is defined as 4 * pi * G, where G is the gravitational constant.

// In CGS, this constant is currently
//      Gconst   =  6.67428e-8           cm^3/g/s^2 , which results in
//      FourPiG =  83.8503442814844e-8  cm^3/g/s^2

// ************************************************************************************** //

const Real FourPiG = 4.0 * std::numbers::pi * C::Gconst;

///
/// Multipole gravity data
///
AMREX_GPU_MANAGED Real multipole::volumeScale;
AMREX_GPU_MANAGED Real multipole::parityScale;

AMREX_GPU_MANAGED Real multipole::rmax;

AMREX_GPU_MANAGED Array1D<bool, 0, 2> multipole::doFoldedAddLo;
AMREX_GPU_MANAGED Array1D<bool, 0, 2> multipole::doFoldedAddHi;
AMREX_GPU_MANAGED bool multipole::doFoldedAdd;

AMREX_GPU_MANAGED Array1D<bool, 0, 2> multipole::doMirrorLo;
AMREX_GPU_MANAGED Array1D<bool, 0, 2> multipole::doMirrorHi;

AMREX_GPU_MANAGED Array2D<Real, 0, multipole::max_moment_order, 0, multipole::max_moment_order> multipole::factorialTable;
AMREX_GPU_MANAGED Array1D<Real, 0, multipole::max_moment_order> multipole::axis_parity_table;
AMREX_GPU_MANAGED Array2D<Real, 0, multipole::max_moment_order, 0, multipole::max_moment_order> multipole::azimuth_parity_table;

Gravity::Gravity(Amr* Parent, int _top_level_hint, BCRec* _boundary_rec, int _density_comp)
  :
    parent(Parent),
    LevelTable(LEVEL_LIMIT),
    phi_gradient_new(LEVEL_LIMIT),
    phi_gradient_old(LEVEL_LIMIT),
    grids(Parent->boxArray()),
    dmap(Parent->DistributionMap()),
    abs_tol(LEVEL_LIMIT),
    rel_tol(LEVEL_LIMIT),
    level_solve_residual_norm(LEVEL_LIMIT),
    volume(LEVEL_LIMIT),
    area(LEVEL_LIMIT),
    phys_bc(_boundary_rec)
{

     amrex::ignore_unused(_top_level_hint);

     AMREX_ALWAYS_ASSERT(parent->maxLevel() < LEVEL_LIMIT);

     Density = _density_comp;
     read_params();
     allocated_top_level = -1;

     radial_field_old.resize(LEVEL_LIMIT);
     radial_field_new.resize(LEVEL_LIMIT);
     radial_shell_mass.resize(LEVEL_LIMIT);
     radial_shell_vol.resize(LEVEL_LIMIT);
#ifdef GR_GRAV
     radial_shell_pres.resize(LEVEL_LIMIT);
#endif

     if (gravity::gravity_model_name == "PoissonGrav") {
         build_solver_bc();
         init_multipole_field();
     }
     peak_rhs_value = 0.0;
     level_point_count = -1;
}

Gravity::~Gravity() = default;

void
Gravity::read_params ()
{
    static bool done = false;

    if (!done)
    {
        const Geometry& default_geom = DefaultGeometry();

        ParmParse pp("gravity");

        if ( (gravity::gravity_model_name != "ConstantGrav") &&
             (gravity::gravity_model_name != "PoissonGrav") &&
             (gravity::gravity_model_name != "MonopoleGrav") )
             {
                std::cout << "Sorry -- dont know this gravity type"  << std::endl;
                amrex::Abort("Options are ConstantGrav, PoissonGrav, or MonopoleGrav");
             }

#if (AMREX_SPACEDIM == 1)
        if (gravity::gravity_model_name == "PoissonGrav")
        {
          amrex::Abort(" gravity::gravity_model_name = PoissonGrav doesn't work well in 1-d -- please set gravity::gravity_model_name = MonopoleGrav");
        }
        else if (gravity::gravity_model_name == "MonopoleGrav" && !(default_geom.IsSPHERICAL()))
        {
          amrex::Abort("Only use MonopoleGrav in 1D spherical coordinates");
        }
        else if (gravity::gravity_model_name == "ConstantGrav" && default_geom.IsSPHERICAL())
        {
          amrex::Abort("Can't use constant gravity in 1D spherical coordinates");
        }

#elif (AMREX_SPACEDIM == 2)
        if (gravity::gravity_model_name == "MonopoleGrav" && default_geom.IsCartesian() )
        {
          amrex::Abort(" gravity::gravity_model_name = MonopoleGrav doesn't make sense in 2D Cartesian coordinates");
        }
#endif

        if (pp.contains("accel_from_phi_grad") && !gravity::accel_from_phi_grad && gravity::gravity_model_name == "PoissonGrav") {
            amrex::Print() << "Warning: gravity::gravity_model_name = PoissonGrav assumes accel_from_phi_grad is true" << std::endl;
        }

        int nlevs = parent->maxLevel() + 1;

        // Allow run-time input of solver tolerance. If the user
        // provides no value, set a reasonable default value on the
        // coarse level, and then increase it by ref_ratio**2 as the
        // levels get finer to account for the change in the absolute
        // scale of the Laplacian. If the user provides one value, use
        // that on the coarse level, and increase it the same way for
        // the fine levels. If the user provides more than one value,
        // we expect them to provide one for every level, and we do
        // not apply the ref_ratio effect.

        int abs_tol_count = pp.countval("abs_tol");

        if (abs_tol_count <= 1) {

            Real tol;

            if (abs_tol_count == 1) {

                pp.get("abs_tol", tol);

            } else {

                if (default_geom.IsCartesian()) {
                    tol = 1.e-11;
                } else {
                    tol = 1.e-10;
                }

            }

            abs_tol[0] = tol;

            // Account for the fact that on finer levels, the scale of the
            // Laplacian changes due to the zone size changing. We assume
            // dx == dy == dz, so it is fair to say that on each level the
            // tolerance should increase by the factor ref_ratio**2, since
            // in absolute terms the Laplacian increases by that ratio too.
            // The actual tolerance we'll send in is the effective tolerance
            // on the finest level that we solve for.

            for (int lev = 1; lev < nlevs; ++lev) {
                abs_tol[lev] = abs_tol[lev - 1] * amrex::Math::powi<2>(parent->refRatio(lev - 1)[0]);
            }

        } else if (abs_tol_count >= nlevs) {

            pp.getarr("abs_tol", abs_tol, 0, nlevs);

        } else {

            amrex::Abort("If you are providing multiple values for abs_tol, you must provide at least one value for every level up to amr.max_level.");

        }

        // For the relative tolerance, we can again accept a single
        // scalar (same for all levels) or one for all levels. The
        // default value is zero, so that we only use the absolute
        // tolerance.  The multigrid always chooses the looser of the
        // two criteria in determining whether the solve has
        // converged.

        // Note that the parameter rel_tol used to be known as ml_tol,
        // so if we detect that the user has set ml_tol but not
        // rel_tol, we'll accept that for specifying the relative
        // tolerance. ml_tol is now considered deprecated and will be
        // removed in a future release.

        std::string rel_tol_key = "rel_tol";

        if (pp.contains("ml_tol")) {

            amrex::Warning("The gravity parameter ml_tol has been renamed rel_tol. ml_tol is now deprecated.");

            if (!pp.contains("rel_tol")) {
                rel_tol_key = "ml_tol";
            }

        }

        int rel_tol_count = pp.countval(rel_tol_key);

        if (rel_tol_count <= 1) {

            Real tol;

            if (rel_tol_count == 1) {

                pp.get(rel_tol_key, tol);

            } else {

                tol = 0.0;

            }

            for (int lev = 0; lev < LEVEL_LIMIT; ++lev) {
                rel_tol[lev] = tol;
            }

        } else if (rel_tol_count >= nlevs) {

            pp.getarr(rel_tol_key, rel_tol, 0, nlevs);

        } else {

            amrex::Abort("If you are providing multiple values for rel_tol, you must provide at least one value for every level up to amr.max_level.");

        }

        done = true;
    }
}

void
Gravity::report_run_settings(std::ostream& jobInfoFile)
{
#include <gravity_job_info_tests.H>
}


void
Gravity::set_grid_points_gravity (int numpts)
{
  level_point_count = numpts;
}

void
Gravity::attach_level (int                   level,
                        AmrLevel*             amr_level_ptr,
                        MultiFab&             _cell_volume,
                        MultiFab*             _face_area)
{
    if (gravity::verbose > 1) {
        amrex::Print() << "Installing Gravity level " << level << '\n';
    }

    LevelTable[level] = amr_level_ptr;

    volume[level] = &_cell_volume;

    area[level] = _face_area;

    level_solve_residual_norm[level] = 0.0;

    const Geometry& geom = amr_level_ptr->Geom();

    if (gravity::gravity_model_name == "PoissonGrav") {

       const DistributionMapping& dm = amr_level_ptr->DistributionMap();

       phi_gradient_old[level].resize(AMREX_SPACEDIM);
       for (int n=0; n<AMREX_SPACEDIM; ++n) {
           phi_gradient_old[level][n] = std::make_unique<MultiFab>(amr_level_ptr->getEdgeBoxArray(n),dm,1,1);
       }
       phi_gradient_new[level].resize(AMREX_SPACEDIM);
       for (int n=0; n<AMREX_SPACEDIM; ++n) {
           phi_gradient_new[level][n] = std::make_unique<MultiFab>(amr_level_ptr->getEdgeBoxArray(n),dm,1,1);
       }

    } else if (gravity::gravity_model_name == "MonopoleGrav") {

        if (!geom.isAllPeriodic())
        {
           int n1d = gravity::radial_bin_ratio*level_point_count;

           radial_field_old[level].resize(n1d);
           radial_field_new[level].resize(n1d);
           radial_shell_mass[level].resize(n1d);
           radial_shell_vol[level].resize(n1d);
#ifdef GR_GRAV
           radial_shell_pres[level].resize(n1d);
#endif
        }

    }

    allocated_top_level = level;
}

std::string Gravity::get_solver_kind()
{
  return gravity::gravity_model_name;
}

int Gravity::get_solve_level_cap()
{
  return gravity::solve_level_cap;
}

int Gravity::NoSync()
{
  return gravity::no_sync;
}

int Gravity::CompositeFixEnabled()
{
  return gravity::composite_phi_fixup;
}

int Gravity::get_self_check_flag()
{
  return verify_solves_flag;
}

Vector<std::unique_ptr<MultiFab> >&
Gravity::get_potential_gradient_prev(int level)
{
  return phi_gradient_old[level];
}

MultiFab*
Gravity::read_prev_grad_comp(int level, int comp)
{
  return phi_gradient_old[level][comp].get();
}

Vector<std::unique_ptr<MultiFab> >&
Gravity::get_potential_gradient_curr(int level)
{
  return phi_gradient_new[level];
}

void
Gravity::accumulate_grad_phi_now(int level, Vector<std::unique_ptr<MultiFab> >& addend)
{
    for (int n = 0; n < AMREX_SPACEDIM; n++) {
        phi_gradient_new[level][n]->plus(*addend[n],0,1,0);
    }
}

void
Gravity::swapTimeLevels (int level)
{
    BL_PROFILE("Gravity::swapTimeLevels()");

    if (gravity::gravity_model_name == "PoissonGrav") {
        for (int n=0; n < AMREX_SPACEDIM; n++) {
            std::swap(phi_gradient_old[level][n], phi_gradient_new[level][n]);
            phi_gradient_new[level][n]->setVal(1.e50);
        }
    }
}

void
Gravity::solve_potential_level (int               level,
                        MultiFab&         phi,
                        const Vector<MultiFab*>& phi_gradient,
                        int               use_new_state)
{
    BL_PROFILE("Gravity::solve_potential_level()");

    if (gravity::verbose > 1) {
        amrex::Print() << " ... solve for phi at level " << level << std::endl;
    }

    const Real strt = ParallelDescriptor::second();

    if (use_new_state == 0) {
        sanity_check(level);
    }

    Real time;
    if (use_new_state == 1) {
      time = LevelTable[level]->get_state_data(Potential_Type).curTime();
    } else {
      time = LevelTable[level]->get_state_data(Potential_Type).prevTime();
    }

    // If we are below the solve_level_cap, do the Poisson solve.
    // Otherwise, interpolate using a fillpatch from solve_level_cap.

    if (level <= gravity::solve_level_cap) {

        Vector<MultiFab*> phi_ptrs(1, &phi);

        const auto& poisson_rhs = fetch_poisson_rhs(level, 1, use_new_state);

        Vector< Vector<MultiFab*> > phi_gradient_ptrs(1);
        phi_gradient_ptrs[0].resize(AMREX_SPACEDIM);
        for (int i = 0; i < AMREX_SPACEDIM ; i++) {
            phi_gradient_ptrs[0][i] = phi_gradient[i];
        }

        Vector<MultiFab*> empty_residuals;

        level_solve_residual_norm[level] = solve_potential_mlmg(level, level,
                                                          phi_ptrs,
                                                          amrex::GetVecOfPtrs(poisson_rhs),
                                                          phi_gradient_ptrs,
                                                          empty_residuals,
                                                          time);

    }
    else {

        LevelTable[level]->FillCoarsePatch(phi, 0, time, Potential_Type, 0, 1, 1);

    }

    if (gravity::verbose)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real end = ParallelDescriptor::second() - strt;
        amrex::Real level_real = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(end,IOProc);
        amrex::Print() << "Gravity::solve_potential_level() time = " << end << " on level "
                       << level_real << std::endl << std::endl;
#ifdef BL_LAZY
        });
#endif
    }
}

void
Gravity::synchronize_gravity_levels (int crse_level, int fine_level, const Vector<MultiFab*>& drho, const Vector<MultiFab*>& dphi)
{
    BL_PROFILE("Gravity::synchronize_gravity_levels()");

    // There is no need to do a synchronization if
    // we didn't solve on the fine levels.

    if (fine_level > gravity::solve_level_cap) {
        return;
    } else {
        fine_level = amrex::min(fine_level, gravity::solve_level_cap);
    }

    BL_ASSERT(parent->finestLevel()>crse_level);
    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
          std::cout << " ... synchronize_gravity_levels at crse_level " << crse_level << '\n';
          std::cout << " ...     up to finest_level     " << fine_level << '\n';
    }

    const Geometry& crse_geom = parent->Geom(crse_level);
    const Box& crse_domain = crse_geom.Domain();

    int nlevs = fine_level - crse_level + 1;

    // Construct delta(phi) and delta(phi_gradient). delta(phi)
    // needs a ghost zone for holding the boundary condition
    // in the same way that phi does.

    Vector<std::unique_ptr<MultiFab> > delta_phi(nlevs);

    for (int lev = crse_level; lev <= fine_level; ++lev) {
        delta_phi[lev - crse_level] = std::make_unique<MultiFab>(grids[lev], dmap[lev], 1, 1);
        delta_phi[lev - crse_level]->setVal(0.0);
    }

    Vector< Vector<std::unique_ptr<MultiFab> > > edge_phi_gradient(nlevs);

    for (int lev = crse_level; lev <= fine_level; ++lev) {
        edge_phi_gradient[lev - crse_level].resize(AMREX_SPACEDIM);

        const DistributionMapping& dm = LevelTable[lev]->DistributionMap();
        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            edge_phi_gradient[lev - crse_level][n] = std::make_unique<MultiFab>(LevelTable[lev]->getEdgeBoxArray(n), dm, 1, 0);
            edge_phi_gradient[lev - crse_level][n]->setVal(0.0);
        }
    }

    // Construct a container for the right-hand-side (4 * pi * G * drho + dphi).
    // dphi appears in the construction of the boundary conditions because it
    // indirectly represents a change in mass on the domain (the mass motion that
    // occurs on the fine grid, whose gravitational effects are now indirectly
    // being propagated to the coarse grid).

    // We will temporarily leave the RHS divided by (4 * pi * G) because that
    // is the form expected by the boundary condition routine.

    Vector<std::unique_ptr<MultiFab> > poisson_rhs(nlevs);

    for (int lev = crse_level; lev <= fine_level; ++lev) {
        poisson_rhs[lev - crse_level] = std::make_unique<MultiFab>(LevelTable[lev]->boxArray(), LevelTable[lev]->DistributionMap(), 1, 0);
        MultiFab::Copy(*poisson_rhs[lev - crse_level], *dphi[lev - crse_level], 0, 0, 1, 0);
        poisson_rhs[lev - crse_level]->mult(1.0 / FourPiG);
        MultiFab::Add(*poisson_rhs[lev - crse_level], *drho[lev - crse_level], 0, 0, 1, 0);
    }

    // Construct the boundary conditions for the Poisson solve.

    if (crse_level == 0 && !crse_geom.isAllPeriodic()) {

        if (gravity::verbose > 1) {
            amrex::Print() << " ... Making bc's for delta_phi at crse_level 0"  << std::endl;
        }

#if (AMREX_SPACEDIM == 3)
      if ( gravity::boundary_direct_sum )
          fill_direct_sum_boundaries(crse_level,fine_level,amrex::GetVecOfPtrs(poisson_rhs),*delta_phi[crse_level]);
      else {
          fill_multipole_boundaries(crse_level,fine_level,amrex::GetVecOfPtrs(poisson_rhs),*delta_phi[crse_level]);
      }
#elif (AMREX_SPACEDIM == 2)
      fill_multipole_boundaries(crse_level,fine_level,amrex::GetVecOfPtrs(poisson_rhs),*delta_phi[crse_level]);
#else
      fill_multipole_boundaries(crse_level,fine_level,amrex::GetVecOfPtrs(poisson_rhs),*delta_phi[crse_level]);
#endif

    }

    // Restore the factor of (4 * pi * G) for the Poisson solve.
    for (int lev = crse_level; lev <= fine_level; ++lev)
        poisson_rhs[lev - crse_level]->mult(FourPiG);

    // In the all-periodic case we enforce that the RHS sums to zero.
    // We only do this if we're periodic and the coarse level covers the whole domain.
    // In principle this could be true for level > 0, so we'll test on whether the number
    // of points on the level is equal to the number of points possible on the level.
    // Note that since we did the average-down, we can stick with the data on the coarse
    // level since the averaging down is conservative.

    if (crse_geom.isAllPeriodic() && (grids[crse_level].numPts() == crse_domain.numPts()))
    {

        // We assume that if we're fully periodic then we're going to be in Cartesian
        // coordinates, so to get the average value of the RHS we can divide the sum
        // of the RHS by the number of points. This correction should probably be
        // volume weighted if we somehow got here without being Cartesian.

        Real mean_offset = poisson_rhs[0]->sum() / static_cast<Real>(grids[crse_level].numPts());

        if (gravity::verbose > 1) {
            amrex::Print() << "WARNING: Adjusting RHS in synchronize_gravity_levels solve by " << mean_offset << '\n';
        }

        for (int lev = fine_level; lev >= crse_level; --lev) {
            poisson_rhs[lev-crse_level]->plus(-mean_offset, 0, 1, 0);
        }
    }

    // Do multi-level solve for delta_phi.

    solve_correction_phi(crse_level, fine_level,
                        amrex::GetVecOfPtrs(poisson_rhs),
                        amrex::GetVecOfPtrs(delta_phi),
                        amrex::GetVecOfVecOfPtrs(edge_phi_gradient));

    // In the all-periodic case we enforce that delta_phi averages to zero.

    if (crse_geom.isAllPeriodic() && (grids[crse_level].numPts() == crse_domain.numPts()) ) {

        Real mean_offset = delta_phi[0]->sum() / static_cast<Real>(grids[crse_level].numPts());

        for (int lev = crse_level; lev <= fine_level; ++lev) {
            delta_phi[lev - crse_level]->plus(-mean_offset, 0, 1, 1);
        }

    }

    // Add delta_phi to phi_new, and grad(delta_phi) to grad(delta_phi_curr) on each level.
    // Update the cell-centered gravity too.

    for (int lev = crse_level; lev <= fine_level; lev++) {

        LevelTable[lev]->get_new_data(Potential_Type).plus(*delta_phi[lev - crse_level], 0, 1, 0);

        for (int n = 0; n < AMREX_SPACEDIM; n++) {
            phi_gradient_new[lev][n]->plus(*edge_phi_gradient[lev - crse_level][n], 0, 1, 0);
        }

        get_new_gravity_vector(lev, LevelTable[lev]->get_new_data(Accel_Type),
                            LevelTable[lev]->get_state_data(State_Type).curTime());

    }

    int use_new_state = 1;

    for (int lev = fine_level-1; lev >= crse_level; --lev)
    {

        // Average phi_new from fine to coarse level

        const IntVect& ratio = parent->refRatio(lev);

        amrex::average_down(LevelTable[lev+1]->get_new_data(Potential_Type),
                             LevelTable[lev  ]->get_new_data(Potential_Type),
                             0, 1, ratio);

        // Average the edge-based phi_gradient from finer to coarser level

        average_fine_edges_to_coarse(lev, use_new_state);

        // Average down the gravitational acceleration too.

        amrex::average_down(LevelTable[lev+1]->get_new_data(Accel_Type),
                             LevelTable[lev  ]->get_new_data(Accel_Type),
                             0, 1, ratio);

    }

}

void
Gravity::FetchCoarsePotential(int level,
                    MultiFab& coarse_potential,
                    Real      time      )
{
    BL_PROFILE("Gravity::FetchCoarsePotential()");

    BL_ASSERT(level!=0);

    coarse_potential.clear();
    coarse_potential.define(grids[level-1], dmap[level-1], 1, 1); // BUT NOTE we don't trust phi's ghost cells.

    const Real t_old = LevelTable[level-1]->get_state_data(Potential_Type).prevTime();
    const Real t_new = LevelTable[level-1]->get_state_data(Potential_Type).curTime();
    Real alpha = (time - t_old) / (t_new - t_old);
    Real oneMinusAlpha = 1.0_rt - alpha;
    const Real threshold = 1.e-6_rt;

    MultiFab const& phi_new = LevelTable[level-1]->get_new_data(Potential_Type);

    if (std::abs(oneMinusAlpha) < threshold) {
        MultiFab::Copy(coarse_potential, phi_new, 0, 0, 1, 1);
    }
    else if (std::abs(alpha) < threshold) {
        // Note we only access the old time if it's actually needed, to guard against
        // scenarios where it may not be allocated yet, for example after a restart when
        // the old time was not dumped to the checkpoint.
        MultiFab const& phi_old = LevelTable[level-1]->get_old_data(Potential_Type);
        MultiFab::Copy(coarse_potential, phi_old, 0, 0, 1, 1);
    }
    else {
        MultiFab const& phi_old = LevelTable[level-1]->get_old_data(Potential_Type);
        MultiFab::LinComb(coarse_potential, alpha, phi_new, 0, oneMinusAlpha, phi_old, 0, 0, 1, 1);
    }

    const Geometry& geom = parent->Geom(level-1);
    coarse_potential.FillBoundary(geom.periodicity());
}

void
Gravity::multilevel_potential_solve (int level, int highest_level_in)
{
    BL_PROFILE("Gravity::multilevel_potential_solve()");

    if (gravity::verbose > 1) {
        amrex::Print() << "... multilevel solve for new phi at base level " << level << " to finest level " << highest_level_in << std::endl;
    }

    const Real strt = ParallelDescriptor::second();

    for (int lev = level; lev <= highest_level_in; lev++) {
       BL_ASSERT(phi_gradient_new[lev].size()==AMREX_SPACEDIM);
       for (int n=0; n<AMREX_SPACEDIM; ++n)
       {
           phi_gradient_new[lev][n] = std::make_unique<MultiFab>(LevelTable[lev]->getEdgeBoxArray(n),
                                                              LevelTable[lev]->DistributionMap(),1,1);
       }
    }

    int use_new_state = 1;
    run_multilevel_solve(level, highest_level_in, amrex::GetVecOfVecOfPtrs(phi_gradient_new), use_new_state);

    if (gravity::verbose)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        Real      end    = ParallelDescriptor::second() - strt;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(end,IOProc);
        amrex::Print() << "Gravity::multilevel_potential_solve() time = " << end << std::endl << std::endl;
#ifdef BL_LAZY
        });
#endif
    }
}

void
Gravity::run_multilevel_solve (int crse_level, int highest_level_in,
                                  const Vector<Vector<MultiFab*> >& phi_gradient,
                                  int use_new_state)
{
    BL_PROFILE("Gravity::run_multilevel_solve()");

    for (int ilev = crse_level; ilev <= highest_level_in ; ++ilev) {
        sanity_check(ilev);
    }

    int nlevels = highest_level_in - crse_level + 1;

    Vector<MultiFab*> phi_ptrs(nlevels);
    for (int ilev = 0; ilev < nlevels; ilev++)
    {
        int amr_lev = ilev + crse_level;
        if (use_new_state == 1) {
            phi_ptrs[ilev] = &LevelTable[amr_lev]->get_new_data(Potential_Type);
        } else {
            phi_ptrs[ilev] = &LevelTable[amr_lev]->get_old_data(Potential_Type);
        }
    }

    const auto& poisson_rhs = fetch_poisson_rhs(crse_level, nlevels, use_new_state);

    Vector<Vector<MultiFab*> > phi_gradient_ptrs(nlevels);
    for (int ilev = 0; ilev < nlevels; ilev++)
    {
        int amr_lev = ilev + crse_level;
        phi_gradient_ptrs[ilev] = phi_gradient[amr_lev];
    }

    Real time;
    if (use_new_state == 1) {
        time = LevelTable[crse_level]->get_state_data(Potential_Type).curTime();
    } else {
        time = LevelTable[crse_level]->get_state_data(Potential_Type).prevTime();
    }

    int fine_level = amrex::min(highest_level_in, gravity::solve_level_cap);

    if (fine_level >= crse_level) {

        Vector<MultiFab*> empty_residuals;
        solve_potential_mlmg(crse_level, fine_level,
                            phi_ptrs, amrex::GetVecOfPtrs(poisson_rhs), phi_gradient_ptrs, empty_residuals,
                            time);

        // Average phi from fine to coarse level
        for (int amr_lev = fine_level; amr_lev > crse_level; amr_lev--)
        {
            const IntVect& ratio = parent->refRatio(amr_lev-1);
            if (use_new_state == 1)
            {
                amrex::average_down(LevelTable[amr_lev  ]->get_new_data(Potential_Type),
                                    LevelTable[amr_lev-1]->get_new_data(Potential_Type),
                                    0, 1, ratio);
            }
            else if (use_new_state == 0)
            {
                amrex::average_down(LevelTable[amr_lev  ]->get_old_data(Potential_Type),
                                    LevelTable[amr_lev-1]->get_old_data(Potential_Type),
                                    0, 1, ratio);
            }
        }

        // Average phi_gradient from fine to coarse level
        for (int amr_lev = fine_level; amr_lev > crse_level; amr_lev--) {
            average_fine_edges_to_coarse(amr_lev-1,use_new_state);
        }

    }

    // For all levels on which we're not doing the solve, interpolate from
    // the coarsest level with correct data. Note that since FillCoarsePatch
    // fills from the coarse level just below it, we need to fill from the
    // lowest level upwards using successive interpolations.

    for (int amr_lev = gravity::solve_level_cap+1; amr_lev <= highest_level_in; amr_lev++) {

        // Interpolate the potential.

        if (use_new_state == 1) {

            MultiFab& phi = LevelTable[amr_lev]->get_new_data(Potential_Type);

            LevelTable[amr_lev]->FillCoarsePatch(phi,0,time,Potential_Type,0,1,1);

        }
        else {

            MultiFab& phi = LevelTable[amr_lev]->get_old_data(Potential_Type);

            LevelTable[amr_lev]->FillCoarsePatch(phi,0,time,Potential_Type,0,1,1);

        }

        // Interpolate the phi_gradient.

        // Instantiate a bare physical BC function for phi_gradient. It doesn't do anything
        // since the fine levels for Poisson gravity do not touch the physical boundary.

        PotentialGradBCFunct gradient_phys_bc;

        // We need to use a interpolater that works with data on faces.

        Interpolater* gradient_interp = &face_linear_interp;

        // For the BCs, we will use the Accel_Type BCs for convenience, but these will
        // not do anything because we do not fill on physical boundaries.

        const Vector<BCRec>& gradient_bcs = LevelTable[amr_lev]->get_desc_lst()[Accel_Type].getBCs();

        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            amrex::InterpFromCoarseLevel(*phi_gradient[amr_lev][n], time, *phi_gradient[amr_lev-1][n],
                                         0, 0, 1,
                                         parent->Geom(amr_lev-1), parent->Geom(amr_lev),
                                         gradient_phys_bc, 0, gradient_phys_bc, 0, parent->refRatio(amr_lev-1),
                                         gradient_interp, gradient_bcs, 0);
        }

    }

}

void
Gravity::read_old_gravity_field(int level, MultiFab& gravity_field_out, Real time)
{
    BL_PROFILE("Gravity::read_old_gravity_field()");

    int ng = gravity_field_out.nGrow();

    // Fill data from the level below if we're not doing a solve on this level.

    if (level > gravity::solve_level_cap) {

        LevelTable[level]->FillCoarsePatch(gravity_field_out,0,time,Accel_Type,0,3,ng);

        return;

    }

    // Note that gravity_field_out coming into this routine always has three components.
    // So we'll define a temporary MultiFab with AMREX_SPACEDIM dimensions.
    // Then at the end we'll copy in all AMREX_SPACEDIM dimensions from this into
    // the outgoing gravity_field_out, leaving any higher dimensions unchanged.

    MultiFab gravity_view(grids[level], dmap[level], AMREX_SPACEDIM, ng);
    gravity_view.setVal(0.0,ng);

    const Geometry& geom = parent->Geom(level);

    if (gravity::gravity_model_name == "ConstantGrav") {

        if (AMREX_SPACEDIM == 2 && geom.Coord() == 2) {
            // 2D spherical r-theta, we want g in the radial direction
            gravity_view.setVal(gravity::uniform_accel, 0, 1, ng);
        } else {
            // Set to constant value in the AMREX_SPACEDIM direction and zero in all others.
            gravity_view.setVal(gravity::uniform_accel, AMREX_SPACEDIM-1, 1, ng);
        }

    } else if (gravity::gravity_model_name == "MonopoleGrav") {

       const Real prev_time = LevelTable[level]->get_state_data(State_Type).prevTime();
       build_radial_field(level,prev_time,radial_field_old[level]);
       interpolate_monopole_field(level,radial_field_old[level],gravity_view);

    } else if (gravity::gravity_model_name == "PoissonGrav") {

       amrex::average_face_to_cellcenter(gravity_view, amrex::GetVecOfConstPtrs(phi_gradient_old[level]), geom);
       gravity_view.mult(-1.0, ng); // g = - grad(phi)

    } else {
       amrex::Abort("Unknown gravity_model_name in read_old_gravity_field");
    }

    // Do the copy to the output vector.

    for (int dir = 0; dir < 3; dir++) {
        if (dir < AMREX_SPACEDIM) {
            MultiFab::Copy(gravity_field_out, gravity_view, dir, dir, 1, ng);
        } else {
            gravity_field_out.setVal(0.,dir,1,ng);
        }
    }

#if (AMREX_SPACEDIM > 1)
    if (gravity::gravity_model_name != "ConstantGrav") {
        // Fill ghost cells
        AmrLevel* amrlev = &parent->getLevel(level) ;
        AmrLevel::FillPatch(*amrlev,gravity_field_out,ng,time,Accel_Type,0,AMREX_SPACEDIM);
    }
#endif

    auto* cs = dynamic_cast<HydroSim*>(&parent->getLevel(level));
    if (cs->using_central_mass()) {
        MultiFab& phi = LevelTable[level]->get_old_data(Potential_Type);
        include_point_source_field(level,phi,gravity_field_out);
    }
}

void
Gravity::get_new_gravity_vector(int level, MultiFab& gravity_field_out, Real time)
{
    BL_PROFILE("Gravity::get_new_gravity_vector()");

    int ng = gravity_field_out.nGrow();

    // Fill data from the level below if we're not doing a solve on this level.

    if (level > gravity::solve_level_cap) {

        LevelTable[level]->FillCoarsePatch(gravity_field_out,0,time,Accel_Type,0,3,ng);

        return;

    }

    // Note that gravity_field_out coming into this routine always has three components.
    // So we'll define a temporary MultiFab with AMREX_SPACEDIM dimensions.
    // Then at the end we'll copy in all AMREX_SPACEDIM dimensions from this into
    // the outgoing gravity_field_out, leaving any higher dimensions unchanged.

    MultiFab gravity_view(grids[level],dmap[level],AMREX_SPACEDIM,ng);
    gravity_view.setVal(0.0,ng);
    const Geometry& geom = parent->Geom(level);

    if (gravity::gravity_model_name == "ConstantGrav") {

        if (AMREX_SPACEDIM == 2 && geom.Coord() == 2) {
            // 2D spherical r-theta, we want g in the radial direction
            gravity_view.setVal(gravity::uniform_accel, 0, 1, ng);
        } else {
            // Set to constant value in the AMREX_SPACEDIM direction
            gravity_view.setVal(gravity::uniform_accel, AMREX_SPACEDIM-1, 1, ng);
        }

    } else if (gravity::gravity_model_name == "MonopoleGrav") {

        // We always fill radial_field_new (at every level)
        const Real cur_time = LevelTable[level]->get_state_data(State_Type).curTime();
        build_radial_field(level,cur_time,radial_field_new[level]);
        interpolate_monopole_field(level,radial_field_new[level],gravity_view);

    } else if (gravity::gravity_model_name == "PoissonGrav") {

        amrex::average_face_to_cellcenter(gravity_view, amrex::GetVecOfConstPtrs(phi_gradient_new[level]), geom);
        gravity_view.mult(-1.0, ng); // g = - grad(phi)

    } else {
       amrex::Abort("Unknown gravity_model_name in get_new_gravity_vector");
    }

    // Do the copy to the output vector.

    for (int dir = 0; dir < 3; dir++) {
        if (dir < AMREX_SPACEDIM) {
            MultiFab::Copy(gravity_field_out, gravity_view, dir, dir, 1, ng);
        } else {
            gravity_field_out.setVal(0.,dir,1,ng);
        }
    }

#if (AMREX_SPACEDIM > 1)
    if (gravity::gravity_model_name != "ConstantGrav" && ng>0) {
        // Fill ghost cells
        AmrLevel* amrlev = &parent->getLevel(level) ;
        AmrLevel::FillPatch(*amrlev,gravity_field_out,ng,time,Accel_Type,0,AMREX_SPACEDIM);
    }
#endif

    auto* cs = dynamic_cast<HydroSim*>(&parent->getLevel(level));
    if (cs->using_central_mass()) {
        MultiFab& phi = LevelTable[level]->get_new_data(Potential_Type);
        include_point_source_field(level,phi,gravity_field_out);
    }
}

void
Gravity::check_residual (const Box& bx,
                        Array4<Real> const& poisson_rhs,
                        Array4<Real> const& ecx,
#if AMREX_SPACEDIM >= 2
                        Array4<Real> const& ecy,
#endif
#if AMREX_SPACEDIM == 3
                        Array4<Real> const& ecz,
#endif
                        const GpuArray<Real, AMREX_SPACEDIM>& dx,
                        const GpuArray<Real, AMREX_SPACEDIM>& problo,
                        int coord_type)
{
    // Test whether using the edge-based gradients
    // to compute Div(Grad(Phi)) satisfies Lap(phi) = RHS
    // Fill the RHS array with the residual

    AMREX_ALWAYS_ASSERT(coord_type >= 0 && coord_type <= 2);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // Cartesian
        if (coord_type == 0) {

            Real laplacian_val = (ecx(i+1,j,k) - ecx(i,j,k)) / dx[0];
#if AMREX_SPACEDIM >= 2
            laplacian_val += (ecy(i,j+1,k) - ecy(i,j,k)) / dx[1];
#endif
#if AMREX_SPACEDIM == 3
            laplacian_val += (ecz(i,j,k+1) - ecz(i,j,k)) / dx[2];
#endif

            poisson_rhs(i,j,k) -= laplacian_val;

        // r-z
        } else if (coord_type == 1) {

            Real rlo  = problo[0] + static_cast<Real>(i) * dx[0];
            Real rhi  = rlo + dx[0];
            Real rcen = 0.5_rt * (rlo + rhi);

            Real laplacian_val = (rhi * ecx(i+1,j,k) - rlo * ecx(i,j,k)) / (rcen * dx[0]);
#if AMREX_SPACEDIM >= 2
            laplacian_val += (ecy(i,j+1,k) - ecy(i,j,k)) / dx[1];
#endif

            poisson_rhs(i,j,k) -= laplacian_val;

        // spherical
        } else if (coord_type == 2) {

            Real rlo  = problo[0] + static_cast<Real>(i) * dx[0];
            Real rhi  = rlo + dx[0];
            Real rcen = 0.5_rt * (rlo + rhi);

            Real laplacian_val = (rhi * rhi * ecx(i+1,j,k) - rlo * rlo * ecx(i,j,k)) / (rcen * rcen * dx[0]);
            poisson_rhs(i,j,k) -= laplacian_val;

        }
    });
}

void
Gravity::verify_level_grad_phi_before(int level)
{
    BL_PROFILE("Gravity::verify_level_grad_phi_before()");

    // Fill the RHS for the solve
    const MultiFab& S_old = LevelTable[level]->get_old_data(State_Type);
    MultiFab Rhs(grids[level],dmap[level],1,0);
    MultiFab::Copy(Rhs,S_old, URHO,0,1,0);

    const Geometry& geom = parent->Geom(level);

    // This is a correction for fully periodic domains only
    if ( geom.isAllPeriodic() )
    {
       if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor() && mass_shift != 0.0) {
          std::cout << " ... subtracting average density from RHS at level ... "
                    << level << " " << mass_shift << std::endl;
       }
       Rhs.plus(-mass_shift,0,1,0);
    }

    Rhs.mult(FourPiG);

    if (gravity::verbose > 1) {
       Real rhs_max_abs = Rhs.norm0();
       amrex::Print() << "... verify_level_grad_phi_before at level " << level << std::endl;
       amrex::Print() << "       norm of RHS             " << rhs_max_abs << std::endl;
    }

    auto dx     = parent->Geom(level).CellSizeArray();
    auto problo = parent->Geom(level).ProbLoArray();
    const int coord_type = geom.Coord();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(Rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        check_residual(bx,
                      Rhs.array(mfi),
                      (*phi_gradient_old[level][0]).array(mfi),
#if AMREX_SPACEDIM >= 2
                      (*phi_gradient_old[level][1]).array(mfi),
#endif
#if AMREX_SPACEDIM == 3
                      (*phi_gradient_old[level][2]).array(mfi),
#endif
                      dx, problo, coord_type);
    }

    if (gravity::verbose > 1) {
       Real resnorm = Rhs.norm0();
       amrex::Print() << "       norm of residual        " << resnorm << std::endl;
    }
}

void
Gravity::verify_level_grad_phi_now(int level)
{
    BL_PROFILE("Gravity::verify_level_grad_phi_now()");

    // Fill the RHS for the solve
    const MultiFab& S_new = LevelTable[level]->get_new_data(State_Type);
    MultiFab Rhs(grids[level],dmap[level],1,0);
    MultiFab::Copy(Rhs,S_new, URHO, 0,1,0);

    const Geometry& geom = parent->Geom(level);

    // This is a correction for fully periodic domains only
    if ( geom.isAllPeriodic() )
    {
       if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor() && mass_shift != 0.0) {
           std::cout << " ... subtracting average density from RHS in solve ... " << mass_shift << std::endl;
       }
       Rhs.plus(-mass_shift,0,1,0);
    }

    Rhs.mult(FourPiG);

    if (gravity::verbose > 1) {
       Real rhs_max_abs = Rhs.norm0();
       if (ParallelDescriptor::IOProcessor()) {
          std::cout << "... verify_level_grad_phi_now at level " << level << std::endl;
          std::cout << "       norm of RHS             " << rhs_max_abs << std::endl;
        }
    }

    auto dx     = geom.CellSizeArray();
    auto problo = geom.ProbLoArray();
    const int coord_type = geom.Coord();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(Rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        check_residual(bx,
                      Rhs.array(mfi),
                      (*phi_gradient_new[level][0]).array(mfi),
#if AMREX_SPACEDIM >= 2
                      (*phi_gradient_new[level][1]).array(mfi),
#endif
#if AMREX_SPACEDIM == 3
                      (*phi_gradient_new[level][2]).array(mfi),
#endif
                      dx, problo, coord_type);
    }

    if (gravity::verbose > 1) {
       Real resnorm = Rhs.norm0();
       amrex::Print() << "       norm of residual        " << resnorm << std::endl;
    }
}

void
Gravity::build_composite_excess_grad(int level,
                                          MultiFab& comp_phi,
                                          const Vector<MultiFab*>& composite_grad_phi,
                                          MultiFab& composite_minus_level_phi,
                                          Vector<std::unique_ptr<MultiFab> >& composite_minus_level_grad_phi)
{
    BL_PROFILE("Gravity::build_composite_excess_grad()");

    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
        std::cout << "\n";
        std::cout << "... compute difference between level and composite solves at level " << level << "\n";
        std::cout << "\n";
    }

    composite_minus_level_phi.define(LevelTable[level]->boxArray(),
                                LevelTable[level]->DistributionMap(),
                                1, 0);

    MultiFab::Copy(composite_minus_level_phi, comp_phi, 0, 0, 1, 0);
    composite_minus_level_phi.minus(parent->getLevel(level).get_old_data(Potential_Type), 0, 1, 0);

    composite_minus_level_grad_phi.resize(AMREX_SPACEDIM);
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        composite_minus_level_grad_phi[n] = std::make_unique<MultiFab>(LevelTable[level]->getEdgeBoxArray(n),
                                                                  LevelTable[level]->DistributionMap(), 1, 0);
        MultiFab::Copy(*composite_minus_level_grad_phi[n], *composite_grad_phi[n], 0, 0, 1, 0);
        composite_minus_level_grad_phi[n]->minus(*phi_gradient_old[level][n], 0, 1, 0);
    }

}

void
Gravity::average_fine_edges_to_coarse(int level, int use_new_state)
{
    BL_PROFILE("Gravity::average_fine_edges_to_coarse()");

    // NOTE: this is called with level == the coarser of the two levels involved
    if (level == parent->finestLevel()) {
        return;
    }

    //
    // Coarsen() the fine stuff on processors owning the fine data.
    //
    BoxArray coarse_grad_fine_BA(grids[level+1].size());

    IntVect fine_ratio = parent->refRatio(level);

    for (int i = 0; i < coarse_grad_fine_BA.size(); ++i) {
        coarse_grad_fine_BA.set(i,amrex::coarsen(grids[level+1][i],fine_ratio));
    }

    Vector<std::unique_ptr<MultiFab> > coarse_grad_fine(AMREX_SPACEDIM);
    for (int n=0; n<AMREX_SPACEDIM; ++n)
    {
        BoxArray eba = coarse_grad_fine_BA;
        eba.surroundingNodes(n);
        coarse_grad_fine[n] = std::make_unique<MultiFab>(eba,dmap[level+1],1,0);
    }

    auto& phi_gradient = (use_new_state) ? phi_gradient_new : phi_gradient_old;

    amrex::average_down_faces(amrex::GetVecOfConstPtrs(phi_gradient[level+1]),
                               amrex::GetVecOfPtrs(coarse_grad_fine),
                               fine_ratio);

    const Geometry& cgeom = parent->Geom(level);

    for (int n = 0; n < AMREX_SPACEDIM; ++n)
    {
        phi_gradient[level][n]->ParallelCopy(*coarse_grad_fine[n], cgeom.periodicity());
    }
}

void
Gravity::check_composite_potential (int crse_level)
{
    BL_PROFILE("Gravity::check_composite_potential()");

    // the lhas only really been tested with crse_level = 0
    AMREX_ALWAYS_ASSERT(crse_level == 0);

    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
        std::cout << "   " << '\n';
        std::cout << "... check_composite_potential at base level " << crse_level << '\n';
    }

    int highest_level_here = parent->finestLevel();
    int nlevels = highest_level_here - crse_level + 1;

    Vector<std::unique_ptr<MultiFab> > phi(nlevels);
    Vector<std::unique_ptr<MultiFab> > poisson_rhs(nlevels);
    Vector<std::unique_ptr<MultiFab> > res(nlevels);
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        int amr_lev = crse_level + ilev;

        phi[ilev] = std::make_unique<MultiFab>(grids[amr_lev],dmap[amr_lev],1,1);
        MultiFab::Copy(*phi[ilev],
                       LevelTable[amr_lev]->get_new_data(Potential_Type),
                       0,0,1,1);

        poisson_rhs[ilev] = std::make_unique<MultiFab>(grids[amr_lev],dmap[amr_lev],1,1);
        MultiFab::Copy(*poisson_rhs[ilev],
                       LevelTable[amr_lev]->get_new_data(State_Type),
                       URHO, 0,1,0);

        res[ilev] = std::make_unique<MultiFab>(grids[amr_lev],dmap[amr_lev],1,0);
        res[ilev]->setVal(0.);
    }

    Real time = LevelTable[crse_level]->get_state_data(Potential_Type).curTime();

    Vector< Vector<MultiFab*> > phi_gradient_none;
    solve_potential_mlmg(crse_level, highest_level_here,
                        amrex::GetVecOfPtrs(phi),
                        amrex::GetVecOfPtrs(poisson_rhs),
                        phi_gradient_none,
                        amrex::GetVecOfPtrs(res),
                        time);

    // Average residual from fine to coarse level before printing the norm
    for (int amr_lev = highest_level_here-1; amr_lev >= crse_level; --amr_lev)
    {
        const IntVect& ratio = parent->refRatio(amr_lev);
        int ilev = amr_lev - crse_level;
        amrex::average_down(*res[ilev+1], *res[ilev],
                             0, 1, ratio);
    }

    for (int amr_lev = crse_level; amr_lev <= highest_level_here; ++amr_lev) {
        int ilev = amr_lev - crse_level;
        Real resnorm = res[ilev]->norm0();
        amrex::Print() << "      ... norm of composite residual at level "
                       << amr_lev << "  " << resnorm << '\n';
    }
    amrex::Print() << std::endl;
}

void
Gravity::interpolate_monopole_field(int level, RealVector& radial_field, MultiFab& gravity_field_out) const
{
    BL_PROFILE("Gravity::interpolate_monopole_field()");

    int n1d = static_cast<int>(radial_field.size());

    const Geometry& geom = parent->Geom(level);
    const auto dx = geom.CellSizeArray();
    const Real dr = dx[0] / static_cast<Real>(gravity::radial_bin_ratio);

    const auto problo = geom.ProbLoArray();
    const auto geomdata = geom.data();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(gravity_field_out, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox();

        auto gravity_view = gravity_field_out.array(mfi);
        const Real* const radial_accel_ptr = radial_field.dataPtr();

        // Note that we are interpolating onto the entire range of gravity_view,
        // including the ghost cells.

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            GpuArray<Real, 3> loc;

            loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];

#if AMREX_SPACEDIM >= 2
            loc[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
#else
            loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
            loc[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
#else
            loc[2] = 0.0_rt;
#endif

            Real r = distance(geomdata, loc);

            int index = static_cast<int>(r / dr);

            Real cen = (static_cast<Real>(index) + 0.5_rt) * dr;
            Real xi = r - cen;

            Real accel_mag;

            if (index == 0) {

                // Linear interpolation or extrapolation
                Real slope = (radial_accel_ptr[index+1] - radial_accel_ptr[index]) / dr;
                accel_mag = radial_accel_ptr[index] + slope * xi;

            } else if (index == n1d-1) {

                // Linear interpolation or extrapolation
                Real slope = (radial_accel_ptr[index] - radial_accel_ptr[index-1]) / dr;
                accel_mag = radial_accel_ptr[index] + slope * xi;

            } else if (index > n1d-1) {

#ifndef AMREX_USE_GPU
                if (level == 0) {
                    std::cout << "PUT_RADIAL_GRAV: INDEX TOO BIG " << index << " > " << n1d-1 << "\n";
                    std::cout << "AT (i,j,k) " << i << " " << j << " " << k << "\n";
                    std::cout << "R / DR IS " << r << " " << dr << "\n";
                    amrex::Abort("Error:: Gravity.cpp :: interpolate_monopole_field");
                } else {
                    // NOTE: we don't do anything to this point if it's outside the
                    //       radial grid and level > 0
                }
#endif

            } else {

                // Quadratic interpolation
                Real ghi = radial_accel_ptr[index+1];
                Real gmd = radial_accel_ptr[index  ];
                Real glo = radial_accel_ptr[index-1];
                accel_mag = ( ghi -   2.0_rt * gmd + glo) * xi * xi / (2.0_rt * dr * dr) +
                           ( ghi                  - glo) * xi      / (2.0_rt * dr     ) +
                           (-ghi + 26.e0_rt * gmd - glo) / 24.e0_rt;

                Real stencil_min = amrex::min(gmd, amrex::min(glo, ghi));
                Real stencil_max = amrex::max(gmd, amrex::max(glo, ghi));
                accel_mag = amrex::max(accel_mag, stencil_min);
                accel_mag = amrex::min(accel_mag, stencil_max);

            }

            if (index <= n1d-1) {

                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    gravity_view(i,j,k,n) = accel_mag * (loc[n] / r);
                }

            }

        });
    }
}

void
Gravity::tally_radial_profile(const Box& bx,
                             Array4<Real const> const& u,
                             RealVector& radial_mass_tile,
                             RealVector& radial_vol_tile,
#ifdef GR_GRAV
                             RealVector& radial_pres_tile,
#endif
                             int n1d, int level) const
{
    const Geometry& geom = parent->Geom(level);

    GpuArray<Real, 3> dx, problo;
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        dx[i] = geom.CellSizeArray()[i];
        problo[i] = geom.ProbLoArray()[i];
    }
    for (int i = AMREX_SPACEDIM; i < 3; ++i) {
        dx[i] = 0.0_rt;
        problo[i] = 0.0_rt;
    }

    Real dr = dx[0] / static_cast<Real>(gravity::radial_bin_ratio);
    Real drinv = 1.0_rt / dr;

    const int coord_type = geom.Coord();
    const auto geomdata = geom.data();

    AMREX_ALWAYS_ASSERT(coord_type >= 0 && coord_type <= 2);

    Real symmetry_mult = 1.0_rt;

    if (coord_type == 0) {

        if ((std::abs(problem::center[0] - problo[0]) < 1.e-2_rt * dx[0]) &&
            (std::abs(problem::center[1] - problo[1]) < 1.e-2_rt * dx[1]) &&
            (std::abs(problem::center[2] - problo[2]) < 1.e-2_rt * dx[2])) {

            symmetry_mult = 8.0_rt;

        }

    } else if (coord_type == 1) {

        if (std::abs(problem::center[1] - problo[1]) < 1.e-2_rt * dx[1]) {

            symmetry_mult = 2.0_rt;

        }

    }

    Real fac = static_cast<Real>(gravity::radial_bin_ratio);

    Real dx_frac = dx[0] / fac;
    Real subcell_dy = dx[1] / fac;
    Real subcell_dz = dx[2] / fac;

    Real* const radial_mass_bins_ptr = radial_mass_tile.dataPtr();
    Real* const radial_vol_bins_ptr = radial_vol_tile.dataPtr();
#ifdef GR_GRAV
    Real* const radial_pres_bins_ptr = radial_pres_tile.dataPtr();
#endif

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        GpuArray<Real, 3> loc;
        loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];
        Real lo_i = problo[0] + static_cast<Real>(i) * dx[0] - problem::center[0];

        loc[1]= problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
        Real lo_j = problo[1] + static_cast<Real>(j) * dx[1] - problem::center[1];

        loc[2]= problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
        Real lo_k = problo[2] + static_cast<Real>(k) * dx[2] - problem::center[2];

        Real r = distance(geomdata, loc);
        int index = static_cast<int>(r * drinv);

        // We may be coming in here with a masked out zone (in a zone on a coarse
        // level underlying a fine level). We don't want to be calling the EOS in
        // this case, so we'll skip these masked out zones (which will have rho
        // exactly equal to zero).

        if (u(i,j,k,URHO) == 0.0_rt) {
            return;
        }

#ifdef GR_GRAV
        Real rhoInv = 1.0_rt / u(i,j,k,URHO);

        eos_t eos_state;

        eos_state.rho = u(i,j,k,URHO);
        eos_state.e   = u(i,j,k,UEINT) * rhoInv;
        eos_state.T   = u(i,j,k,UTEMP);
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = u(i,j,k,UFS+n) * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; ++n) {
            eos_state.aux[n] = u(i,j,k,UFX+n) * rhoInv;
        }
#endif

        // Compute pressure from the EOS
        eos(eos_input_re, eos_state);
#endif

        if (index > n1d - 1) {

#ifndef AMREX_USE_GPU
            if (level == 0) {
                std::cout << "   " << "\n";
                std::cout << ">>> Error: Gravity_nd::ca_compute_radial_mass " << i << " " << j << " " << k << "\n";
                std::cout << ">>> ... index too big: " << index << " > " << n1d-1 << "\n";
                std::cout << ">>> ... at (i,j,k)   : " << i << " " << j << " " << k << "\n";
                amrex::Abort("Error:: Gravity_nd.F90 :: ca_compute_radial_mass");
            }
#endif

        } else {

            for (int kk = 0; kk <= dg2 * (gravity::radial_bin_ratio - 1); ++kk) {
                Real zz   = lo_k + (static_cast<Real>(kk) + 0.5_rt) * subcell_dz;
                Real subcell_z_sq = zz * zz;

                for (int jj = 0; jj <= dg1 * (gravity::radial_bin_ratio - 1); ++jj) {
                    Real yy   = lo_j + (static_cast<Real>(jj) + 0.5_rt) * subcell_dy;
                    Real subcell_y_sq = yy * yy;

                    for (int ii = 0; ii <= gravity::radial_bin_ratio - 1; ++ii) {
                        Real xx    = lo_i + (static_cast<Real>(ii) + 0.5_rt) * dx_frac;
                        Real subcell_x_sq  = xx * xx;

                        r     = std::sqrt(subcell_x_sq + subcell_y_sq + subcell_z_sq);
                        index = static_cast<int>(r * drinv);

                        Real vol_frac{};

                        if (coord_type == 0) {

                            vol_frac = symmetry_mult * dx_frac * subcell_dy * subcell_dz;

                        } else if (coord_type == 1) {

                            vol_frac = 2.0_rt * std::numbers::pi * dx_frac * subcell_dy * symmetry_mult * xx;

                        } else if (coord_type == 2) {

                            Real rlo = std::abs(lo_i + static_cast<Real>(ii  ) * dx_frac);
                            Real rhi = std::abs(lo_i + static_cast<Real>(ii+1) * dx_frac);
                            vol_frac = (4.0_rt / 3.0_rt) * std::numbers::pi * (rhi * rhi * rhi - rlo * rlo * rlo);

                        }

                        if (index <= n1d - 1) {
                            Gpu::Atomic::Add(&radial_mass_bins_ptr[index], vol_frac * u(i,j,k,URHO));
                            Gpu::Atomic::Add(&radial_vol_bins_ptr[index], vol_frac);
#ifdef GR_GRAV
                            Gpu::Atomic::Add(&radial_pres_bins_ptr[index], vol_frac * eos_state.p);
#endif
                        }

                    }
                }
            }

        }

    });
}

void
Gravity::init_multipole_field() const
{
    if (gravity::lnum < 0) {
        amrex::Abort("lnum negative");
    }

    if (gravity::lnum > multipole::max_moment_order) {
        amrex::Abort("lnum greater than max_moment_order");
    }

    int lo_bc[3];
    int hi_bc[3];

    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
      lo_bc[dir] = phys_bc->lo(dir);
      hi_bc[dir] = phys_bc->hi(dir);
    }
    for (int dir = AMREX_SPACEDIM; dir < 3; dir++)
    {
      lo_bc[dir] = -1;
      hi_bc[dir] = -1;
    }

    const auto problo = parent->Geom(0).ProbLoArray();
    const auto probhi = parent->Geom(0).ProbHiArray();

    // If any of the boundaries are symmetric, we need to account for the mass that is assumed
    // to lie on the opposite side of the symmetric axis. If the center in any direction
    // coincides with the boundary, then we can simply double the mass as a result of that reflection.
    // Otherwise, we need to do a more general solve. We include a logical that is set to true
    // if any boundary is symmetric, so that we can avoid unnecessary function calls.

    multipole::volumeScale = 1.0_rt;
    multipole::parityScale = 1.0_rt;

    multipole::doFoldedAdd = false;

    for (int n = 0; n < 3; ++n) {
        multipole::doFoldedAddLo(n) = false;
        multipole::doFoldedAddHi(n) = false;

        multipole::doMirrorLo(n) = false;
        multipole::doMirrorHi(n) = false;
    }

    const Real wallDistTol = 1.0e-2_rt;

    for (int b = 0; b < AMREX_SPACEDIM; ++b) {

        if ((lo_bc[b] == amrex::PhysBCType::symmetry) && (parent->Geom(0).Coord() == 0)) {
            if (std::abs(problem::center[b] - problo[b]) < wallDistTol) {
                multipole::volumeScale *= 2.0_rt;
                multipole::doMirrorLo(b) = true;
            }
            else {
                multipole::doFoldedAddLo(b) = true;
                multipole::doFoldedAdd      = true;
            }
        }

        if ((hi_bc[b] == amrex::PhysBCType::symmetry) && (parent->Geom(0).Coord() == 0)) {
            if (std::abs(problem::center[b] - probhi[b]) < wallDistTol) {
                multipole::volumeScale *= 2.0_rt;
                multipole::doMirrorHi(b) = true;
            }
            else {
                multipole::doFoldedAddHi(b) = true;
                multipole::doFoldedAdd      = true;
            }
        }

    }

    // Compute pre-factors now to save computation time, for qC and qS

    for (int l = 0; l <= multipole::max_moment_order; ++l) {
        multipole::axis_parity_table(l) = 1.0_rt;
        for (int m = 0; m <= multipole::max_moment_order; ++m) {
            multipole::factorialTable(l,m) = 0.0_rt;
            multipole::azimuth_parity_table(l,m) = 1.0_rt;
        }
    }

    for (int l = 0; l <= multipole::max_moment_order; ++l) {

        // The odd l Legendre polynomials are odd in their argument, so
        // a symmetric reflection about the z axis leads to a total cancellation.

        multipole::axis_parity_table(l) = 1.0_rt;

        if (l % 2 != 0) {
            if (AMREX_SPACEDIM == 3 && (multipole::doMirrorLo(2) || multipole::doMirrorHi(2))) {  // NOLINT(bugprone-branch-clone)
                multipole::axis_parity_table(l) = 0.0_rt;
            }
            else if (AMREX_SPACEDIM == 2 && parent->Geom(0).Coord() == 1) {
                multipole::axis_parity_table(l) = 0.0_rt;
            }
            else if (AMREX_SPACEDIM == 1 && parent->Geom(0).Coord() == 2) {
                multipole::axis_parity_table(l) = 0.0_rt;
            }
        }

        // In 1D spherical, every term above l == 0 cancels out since the integration
        // of the Legendre polynomial from 0 to pi is zero.

        if (l > 0) {
            if (AMREX_SPACEDIM == 1 && parent->Geom(0).Coord() == 2) {
                multipole::axis_parity_table(l) = 0.0_rt;
            }
        }

        for (int m = 1; m <= l; ++m) {

            // The parity properties of the associated Legendre polynomials are:
            // P_l^m (-x) = (-1)^(l+m) P_l^m (x)
            // Therefore, a complete cancellation occurs if l+m is odd and
            // we are reflecting about the z axis.

            // Additionally, the cosine and sine terms flip sign when reflected
            // about the x or y axis, so if we have a reflection about x or y
            // then the terms have a complete cancellation.

            if (AMREX_SPACEDIM == 3) {
                multipole::azimuth_parity_table(l,m) = 1.0_rt;
            }
            else if (AMREX_SPACEDIM == 2 && parent->Geom(0).Coord() == 1) {  // NOLINT(bugprone-branch-clone)
                multipole::azimuth_parity_table(l,m) = 0.0_rt;
            }
            else if (AMREX_SPACEDIM == 1 && parent->Geom(0).Coord() == 2) {
                multipole::azimuth_parity_table(l,m) = 0.0_rt;
            }

            if ((l+m) % 2 != 0 && (multipole::doMirrorLo(2) || multipole::doMirrorHi(2))) {
                multipole::azimuth_parity_table(l,m) = 0.0_rt;
            }

            if (multipole::doMirrorLo(0) || multipole::doMirrorLo(1) || multipole::doMirrorHi(0) || multipole::doMirrorHi(1)) {
                multipole::azimuth_parity_table(l,m) = 0.0_rt;
            }

            multipole::factorialTable(l,m) = 2.0_rt * factorial(l-m) / factorial(l+m) * multipole::volumeScale;

        }

    }

    // Now let's take care of a safety issue. The multipole calculation involves taking powers of r^l,
    // which can overflow the floating point exponent limit if lnum is very large. Therefore,
    // we will normalize all distances to the maximum possible physical distance from the center,
    // which is the diagonal from the center to the edge of the box. Then r^l will always be
    // less than or equal to one. For large enough lnum, this may still result in roundoff
    // errors that don't make your answer any more precise, but at least it avoids
    // possible NaN issues from having numbers that are too large for double precision.
    // We will put the rmax factor back in at the end of ca_put_multipole_phi.

    Real maxWidth = probhi[0] - problo[0];
    if (AMREX_SPACEDIM >= 2) {
        maxWidth = amrex::max(maxWidth, probhi[1] - problo[1]);
    }
    if (AMREX_SPACEDIM == 3) {
        maxWidth = amrex::max(maxWidth, probhi[2] - problo[2]);
    }

    multipole::rmax = 0.5_rt * maxWidth * std::sqrt(static_cast<Real>(AMREX_SPACEDIM));  // NOLINT(modernize-use-std-numbers)
}

void
Gravity::fill_multipole_boundaries(int crse_level, int fine_level, const Vector<MultiFab*>& Rhs, MultiFab& phi)
{
    BL_PROFILE("Gravity::fill_multipole_boundaries()");

    // Multipole BCs only make sense to construct if we are starting from the coarse level.

    BL_ASSERT(crse_level == 0);

    BL_ASSERT(gravity::lnum >= 0);

    const Real strt = ParallelDescriptor::second();

#if (AMREX_SPACEDIM == 3)
    const int npts = level_point_count;
#else
    const int npts = 1;
#endif

    // Storage arrays for the multipole moments.
    // We will initialize them to zero, and then
    // sum up the results over grids.
    // Note that since Boxes are defined with
    // AMREX_SPACEDIM dimensions, we cannot presently
    // use this array to fill the interior of the
    // domain in 2D, since we can only have one
    // radial index for calculating the multipole moments.

    Box momentBox0( IntVect(AMREX_D_DECL(0, 0, 0)), IntVect(AMREX_D_DECL(gravity::lnum, 0,    npts-1)) );
    Box momentBoxC( IntVect(AMREX_D_DECL(0, 0, 0)), IntVect(AMREX_D_DECL(gravity::lnum, gravity::lnum, npts-1)) );
    Box momentBoxS( IntVect(AMREX_D_DECL(0, 0, 0)), IntVect(AMREX_D_DECL(gravity::lnum, gravity::lnum, npts-1)) );

    FArrayBox qL0(momentBox0);
    FArrayBox qLC(momentBoxC);
    FArrayBox qLS(momentBoxS);

    FArrayBox qU0(momentBox0);
    FArrayBox qUC(momentBoxC);
    FArrayBox qUS(momentBoxS);

    qL0.setVal<RunOn::Device>(0.0);
    qLC.setVal<RunOn::Device>(0.0);
    qLS.setVal<RunOn::Device>(0.0);
    qU0.setVal<RunOn::Device>(0.0);
    qUC.setVal<RunOn::Device>(0.0);
    qUS.setVal<RunOn::Device>(0.0);

    // This section needs to be generalized for computing
    // full multipole gravity, not just BCs. At present this
    // does nothing.

#if (AMREX_SPACEDIM == 3)
    int surface_only_mode = 1;
#else
    const int surface_only_mode = 1;
#endif

    // Use all available data in constructing the boundary conditions,
    // unless the user has indicated that a maximum level at which
    // to stop using the more accurate data.

    for (int lev = crse_level; lev <= fine_level; ++lev) {

        // Create a local copy of the RHS so that we can mask it.

        MultiFab source(Rhs[lev - crse_level]->boxArray(),
                        Rhs[lev - crse_level]->DistributionMap(), 1, 0);

        MultiFab::Copy(source, *Rhs[lev - crse_level], 0, 0, 1, 0);

        if (lev < fine_level) {
        auto *hydrosim_level = dynamic_cast<HydroSim*>(&(parent->getLevel(lev+1)));
        if (hydrosim_level != nullptr) {
        const MultiFab& mask = hydrosim_level->make_fine_mask();
        MultiFab::Multiply(source, mask, 0, 0, 1, 0);
        } else {
                amrex::Abort("unable to access mask");
            }
        }

        // Loop through the grids and compute the individual contributions
        // to the various moments. The multipole moment constructor
        // is coded to only add to the moment arrays, so it is safe
        // to directly hand the arrays to them.

        const auto dx = parent->Geom(lev).CellSizeArray();
        const auto problo = parent->Geom(lev).ProbLoArray();
        const auto probhi = parent->Geom(lev).ProbHiArray();
        int coord_type = parent->Geom(lev).Coord();

#ifdef _OPENMP
        int nthreads = omp_get_max_threads();
        Vector<std::unique_ptr<FArrayBox> > local_lowerMoment0(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_lowerMomentC(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_lowerMomentS(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_upperMoment0(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_upperMomentC(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_upperMomentS(nthreads);
        for (int i=0; i<nthreads; i++) {
            local_lowerMoment0[i].reset(new FArrayBox(momentBox0));
            local_lowerMomentC[i].reset(new FArrayBox(momentBoxC));
            local_lowerMomentS[i].reset(new FArrayBox(momentBoxS));
            local_upperMoment0[i].reset(new FArrayBox(momentBox0));
            local_upperMomentC[i].reset(new FArrayBox(momentBoxC));
            local_upperMomentS[i].reset(new FArrayBox(momentBoxS));
        }
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
            local_lowerMoment0[tid]->setVal<RunOn::Device>(0.0);
            local_lowerMomentC[tid]->setVal<RunOn::Device>(0.0);
            local_lowerMomentS[tid]->setVal<RunOn::Device>(0.0);
            local_upperMoment0[tid]->setVal<RunOn::Device>(0.0);
            local_upperMomentC[tid]->setVal<RunOn::Device>(0.0);
            local_upperMomentS[tid]->setVal<RunOn::Device>(0.0);
#endif
            for (MFIter mfi(source, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();

#ifdef _OPENMP
                auto lowerMoment0_arr = local_lowerMoment0[tid]->array();
                auto lowerMomentC_arr = local_lowerMomentC[tid]->array();
                auto lowerMomentS_arr = local_lowerMomentS[tid]->array();
                auto upperMoment0_arr = local_upperMoment0[tid]->array();
                auto upperMomentC_arr = local_upperMomentC[tid]->array();
                auto upperMomentS_arr = local_upperMomentS[tid]->array();
#else
                auto lowerMoment0_arr = qL0.array();
                auto lowerMomentC_arr = qLC.array();
                auto lowerMomentS_arr = qLS.array();
                auto upperMoment0_arr = qU0.array();
                auto upperMomentC_arr = qUC.array();
                auto upperMomentS_arr = qUS.array();
#endif

                auto rho = source[mfi].array();
                auto vol = (*volume[lev])[mfi].array();

                amrex::ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::Gpu::Handler const& handler) noexcept
                {
                    // If we're using this to construct boundary values, then only fill
                    // the outermost bin.

                    int nlo = 0;
                    if (surface_only_mode == 1) {
                        nlo = npts-1;
                    }

                    // Note that we don't currently support dx != dy != dz, so this is acceptable.

                    Real binScale = multipole::rmax / dx[0];

                    Real cutoff_cubed_inv = 1.0_rt / (multipole::rmax * multipole::rmax * multipole::rmax);

                    Real x = (problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0]) / multipole::rmax;

#if AMREX_SPACEDIM >= 2
                    Real y = (problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1]) / multipole::rmax;
#else
                    Real y = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
                    Real z = (problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2]) / multipole::rmax;
#else
                    Real z = 0.0_rt;
#endif

                    Real r = std::sqrt(x * x + y * y + z * z);

                    Real cosPolar{}, sampleAzimuth{};
                    int index{};

                    if (AMREX_SPACEDIM == 3) {
                        index = static_cast<int>(r * binScale);
                        cosPolar = z / r;
                        sampleAzimuth = std::atan2(y, x);
                    }
                    else if (AMREX_SPACEDIM == 2 && coord_type == 1) {
                        index = nlo; // We only do the boundary potential in 2D.
                        cosPolar = y / r;
                        sampleAzimuth = z;
                    }
                    else if (AMREX_SPACEDIM == 1 && coord_type == 2) {
                        index = nlo; // We only do the boundary potential in 1D.
                        cosPolar = 1.0_rt;
                        sampleAzimuth = 0.0_rt;
                    }

                    // Now, compute the multipole moments.

                    accumulate_moments(cosPolar, sampleAzimuth, r, rho(i,j,k), vol(i,j,k) * cutoff_cubed_inv,
                                  lowerMoment0_arr, lowerMomentC_arr, lowerMomentS_arr, upperMoment0_arr, upperMomentC_arr, upperMomentS_arr,
                                  npts, nlo, index, handler, true);

                    // Now add in contributions if we have any symmetric boundaries in 3D.
                    // The symmetric boundary in 2D axisymmetric is handled separately.

                    if (multipole::doFoldedAdd) {

                        accumulate_mirror_moments(x, y, z, problo, probhi,
                                                rho(i,j,k), vol(i,j,k) * cutoff_cubed_inv,
                                                lowerMoment0_arr, lowerMomentC_arr, lowerMomentS_arr, upperMoment0_arr, upperMomentC_arr, upperMomentS_arr,
                                                npts, nlo, index, handler);

                    }
                });
            }

#ifdef _OPENMP
            int np0 = momentBox0.numPts();
            int npC = momentBoxC.numPts();
            int npS = momentBoxS.numPts();
            Real* pL0 = qL0.dataPtr();
            Real* pLC = qLC.dataPtr();
            Real* pLS = qLS.dataPtr();
            Real* pU0 = qU0.dataPtr();
            Real* pUC = qUC.dataPtr();
            Real* pUS = qUS.dataPtr();
#pragma omp barrier
#pragma omp for nowait
            for (int i=0; i<np0; ++i)
            {
                for (int it=0; it<nthreads; it++) {
                    const Real* pp = local_lowerMoment0[it]->dataPtr();
                    pL0[i] += pp[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<npC; ++i)
            {
                for (int it=0; it<nthreads; it++) {
                    const Real* pp = local_lowerMomentC[it]->dataPtr();
                    pLC[i] += pp[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<npS; ++i)
            {
                for (int it=0; it<nthreads; it++) {
                    const Real* pp = local_lowerMomentS[it]->dataPtr();
                    pLS[i] += pp[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<np0; ++i)
            {
                for (int it=0; it<nthreads; it++) {
                  const Real* pp = local_upperMoment0[it]->dataPtr();
                  pU0[i] += pp[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<npC; ++i)
            {
                for (int it=0; it<nthreads; it++) {
                  const Real* pp = local_upperMomentC[it]->dataPtr();
                  pUC[i] += pp[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<npS; ++i)
            {
                for (int it=0; it<nthreads; it++) {
                    const Real* pp = local_upperMomentS[it]->dataPtr();
                    pUS[i] += pp[i];
                }
            }
#endif

        } // end OpenMP parallel loop

    } // end loop over levels

    // Now, do a global reduce over all processes.

    Real* lowerMoment0_ptr = qL0.dataPtr();
    Real* lowerMomentC_ptr = qLC.dataPtr();
    Real* lowerMomentS_ptr = qLS.dataPtr();

    // Create temporary pinned host containers in case we need them.

    FArrayBox lowerMoment0_host(The_Pinned_Arena());
    FArrayBox lowerMomentC_host(The_Pinned_Arena());
    FArrayBox lowerMomentS_host(The_Pinned_Arena());

    if (!ParallelDescriptor::UseGpuAwareMpi()) {
        if (The_Arena() == The_Managed_Arena()) {
            qL0.prefetchToHost();
            qLC.prefetchToHost();
            qLS.prefetchToHost();
        }
        else if (The_Arena() == The_Device_Arena()) {
            lowerMoment0_host.resize(momentBox0);
            lowerMomentC_host.resize(momentBoxC);
            lowerMomentS_host.resize(momentBoxS);

            lowerMoment0_host.copy<RunOn::Device>(qL0, momentBox0);
            lowerMomentC_host.copy<RunOn::Device>(qLC, momentBoxC);
            lowerMomentS_host.copy<RunOn::Device>(qLS, momentBoxS);

            lowerMoment0_ptr = lowerMoment0_host.dataPtr();
            lowerMomentC_ptr = lowerMomentC_host.dataPtr();
            lowerMomentS_ptr = lowerMomentS_host.dataPtr();
        }
    }

    Gpu::synchronize();

    ParallelDescriptor::ReduceRealSum(lowerMoment0_ptr, static_cast<int>(momentBox0.numPts()));
    ParallelDescriptor::ReduceRealSum(lowerMomentC_ptr, static_cast<int>(momentBoxC.numPts()));
    ParallelDescriptor::ReduceRealSum(lowerMomentS_ptr, static_cast<int>(momentBoxS.numPts()));

    if (!ParallelDescriptor::UseGpuAwareMpi()) {
        if (The_Arena() == The_Managed_Arena()) {
            qL0.prefetchToDevice();
            qLC.prefetchToDevice();
            qLS.prefetchToDevice();
        }
        else if (The_Arena() == The_Device_Arena()) {
            qL0.copy<RunOn::Device>(lowerMoment0_host, momentBox0);
            qLC.copy<RunOn::Device>(lowerMomentC_host, momentBoxC);
            qLS.copy<RunOn::Device>(lowerMomentS_host, momentBoxS);
        }
    }

    if (surface_only_mode != 1) {

        Real* upperMoment0_ptr = qU0.dataPtr();
        Real* upperMomentC_ptr = qUC.dataPtr();
        Real* upperMomentS_ptr = qUS.dataPtr();

        FArrayBox upperMoment0_host(The_Pinned_Arena());
        FArrayBox upperMomentC_host(The_Pinned_Arena());
        FArrayBox upperMomentS_host(The_Pinned_Arena());

        if (!ParallelDescriptor::UseGpuAwareMpi()) {
            if (The_Arena() == The_Managed_Arena()) {
                qU0.prefetchToHost();
                qUC.prefetchToHost();
                qUS.prefetchToHost();
            }
            else if (The_Arena() == The_Device_Arena()) {
                upperMoment0_host.resize(momentBox0);
                upperMomentC_host.resize(momentBoxC);
                upperMomentS_host.resize(momentBoxS);

                upperMoment0_host.copy<RunOn::Device>(qU0, momentBox0);
                upperMomentC_host.copy<RunOn::Device>(qUC, momentBoxC);
                upperMomentS_host.copy<RunOn::Device>(qUS, momentBoxS);

                upperMoment0_ptr = upperMoment0_host.dataPtr();
                upperMomentC_ptr = upperMomentC_host.dataPtr();
                upperMomentS_ptr = upperMomentS_host.dataPtr();
            }
        }

        Gpu::synchronize();

        ParallelDescriptor::ReduceRealSum(upperMoment0_ptr, static_cast<int>(momentBox0.numPts()));
        ParallelDescriptor::ReduceRealSum(upperMomentC_ptr, static_cast<int>(momentBoxC.numPts()));
        ParallelDescriptor::ReduceRealSum(upperMomentS_ptr, static_cast<int>(momentBoxS.numPts()));

        if (!ParallelDescriptor::UseGpuAwareMpi()) {
            if (The_Arena() == The_Managed_Arena()) {
                qU0.prefetchToDevice();
                qUC.prefetchToDevice();
                qUS.prefetchToDevice();
            }
            else if (The_Arena() == The_Device_Arena()) {
                qU0.copy<RunOn::Device>(upperMoment0_host, momentBox0);
                qUC.copy<RunOn::Device>(upperMomentC_host, momentBoxC);
                qUS.copy<RunOn::Device>(upperMomentS_host, momentBoxS);
            }
        }

    }

    // Finally, construct the boundary conditions using the
    // complete multipole moments, for all points on the
    // boundary that are held on this process.

    const Box& domain = parent->Geom(crse_level).Domain();
    const auto dx = parent->Geom(crse_level).CellSizeArray();
    const auto problo = parent->Geom(crse_level).ProbLoArray();
    int coord_type = parent->Geom(crse_level).Coord();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox();

        auto lowerMoment0_arr = qL0.array();
        auto lowerMomentC_arr = qLC.array();
        auto lowerMomentS_arr = qLS.array();
        auto phi_arr = phi[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const int* domlo = domain.loVect();
            const int* domhi = domain.hiVect();

            // If we're using this to construct boundary values, then only use
            // the outermost bin.

            int nlo = 0;
            if (surface_only_mode == 1) {
                nlo = npts-1;
            }

            Real cutoff_cubed = multipole::rmax * multipole::rmax * multipole::rmax;

            Real x;
            if (i > domhi[0]) {
                x = problo[0] + (static_cast<Real>(i  )         ) * dx[0] - problem::center[0];
            }
            else if (i < domlo[0]) {
                x = problo[0] + (static_cast<Real>(i+1)         ) * dx[0] - problem::center[0];
            }
            else {
                x = problo[0] + (static_cast<Real>(i  ) + 0.5_rt) * dx[0] - problem::center[0];
            }

            x = x / multipole::rmax;

#if AMREX_SPACEDIM >= 2
            Real y;
            if (j > domhi[1]) {
                y = problo[1] + (static_cast<Real>(j  )         ) * dx[1] - problem::center[1];
            }
            else if (j < domlo[1]) {
                y = problo[1] + (static_cast<Real>(j+1)         ) * dx[1] - problem::center[1];
            }
            else {
                y = problo[1] + (static_cast<Real>(j  ) + 0.5_rt) * dx[1] - problem::center[1];
            }
#else
            Real y = 0.0_rt;
#endif

            y = y / multipole::rmax;

#if AMREX_SPACEDIM == 3
            Real z;
            if (k > domhi[2]) {
                z = problo[2] + (static_cast<Real>(k  )         ) * dx[2] - problem::center[2];
            }
            else if (k < domlo[2]) {
                z = problo[2] + (static_cast<Real>(k+1)         ) * dx[2] - problem::center[2];
            }
            else {
                z = problo[2] + (static_cast<Real>(k  ) + 0.5_rt) * dx[2] - problem::center[2];
            }
#else
            Real z = 0.0;
#endif

            z = z / multipole::rmax;

            // Only adjust ghost zones here

            if (i < domlo[0] || i > domhi[0]
#if AMREX_SPACEDIM >= 2
                || j < domlo[1] || j > domhi[1]
#endif
#if AMREX_SPACEDIM >= 3
                || k < domlo[2] || k > domhi[2]
#endif
                ) {

                // There are some cases where r == 0. This might occur, for example,
                // when we have symmetric BCs and our corner is at one edge.
                // In this case, we'll set phi to zero for safety, to avoid NaN issues.
                // These cells should not be accessed anyway during the gravity solve.

                Real r = std::sqrt(x * x + y * y + z * z);

                if (r < 1.0e-12_rt) {
                    phi_arr(i,j,k) = 0.0_rt;
                    return;
                }

                Real cosPolar{}, sampleAzimuth{};
                if (AMREX_SPACEDIM == 3) {
                    cosPolar = z / r;
                    sampleAzimuth = std::atan2(y, x);
                }
                else if (AMREX_SPACEDIM == 2 && coord_type == 1) {
                    cosPolar = y / r;
                    sampleAzimuth = 0.0_rt;
                }

                phi_arr(i,j,k) = 0.0_rt;

                // Compute the potentials on the ghost cells.

                Real legendreValue, legendreValue1, legendreValue2;
                Real assocLegendreValue, assocLegendreValue1, assocLegendreValue2;

                for (int n = nlo; n <= npts - 1; ++n) {

                    for (int l = 0; l <= gravity::lnum; ++l) {

                        evaluateLegendreByDegree(l, legendreValue, legendreValue1, legendreValue2, cosPolar);

                        Real r_U = std::pow(r, -l-1);

                        // Make sure we undo the volume scaling here.

                        phi_arr(i,j,k) += lowerMoment0_arr(l,0,n) * legendreValue * r_U * cutoff_cubed;

                    }

                    for (int m = 1; m <= gravity::lnum; ++m) {
                        for (int l = 1; l <= gravity::lnum; ++l) {

                            if (m > l) {
                                continue;
                            }

                            evaluateAssociatedLegendre(l, m, assocLegendreValue, assocLegendreValue1, assocLegendreValue2, cosPolar);

                            Real r_U = std::pow(r, -l-1);

                            // Make sure we undo the volume scaling here.

                            phi_arr(i,j,k) += (lowerMomentC_arr(l,m,n) * std::cos(m * sampleAzimuth) + lowerMomentS_arr(l,m,n) * std::sin(m * sampleAzimuth)) *
                                              assocLegendreValue * r_U * cutoff_cubed;

                        }
                    }

                }

                phi_arr(i,j,k) = -C::Gconst * phi_arr(i,j,k) / multipole::rmax;
            }
        });
    }

    if (gravity::verbose)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        Real      end    = ParallelDescriptor::second() - strt;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(end,IOProc);
        amrex::Print() << "Gravity::fill_multipole_boundaries() time = " << end << std::endl << std::endl;
#ifdef BL_LAZY
        });
#endif
    }

}

#if (AMREX_SPACEDIM == 3)
void
Gravity::fill_direct_sum_boundaries(int crse_level, int fine_level, const Vector<MultiFab*>& Rhs, MultiFab& phi)
{
    BL_PROFILE("Gravity::fill_direct_sum_boundaries()");

    BL_ASSERT(crse_level==0);

    const Real strt = ParallelDescriptor::second();

    const Geometry& crse_geom = parent->Geom(crse_level);

    // Storage arrays for the BCs.

    const int* domlo = crse_geom.Domain().loVect();
    const int* domhi = crse_geom.Domain().hiVect();

    const int lowIndexXY[3] = {domlo[0]-1, domlo[1]-1, 0         };
    const int highIndexXY[3] = {domhi[0]+1, domhi[1]+1, 0         };

    const int lowIndexXZ[3] = {domlo[0]-1, 0         , domlo[2]-1};
    const int highIndexXZ[3] = {domhi[0]+1, 0         , domhi[2]+1};

    const int lowIndexYZ[3] = {0         , domlo[1]-1, domlo[2]-1};
    const int highIndexYZ[3] = {0         , domhi[1]+1, domhi[2]+1};

    const int bc_lo[3] = {domlo[0]-1, domlo[1]-1, domlo[2]-1};
    const int bc_hi[3] = {domhi[0]+1, domhi[1]+1, domhi[2]+1};

    GpuArray<Real, 3> boundary_spacing;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        boundary_spacing[n] = crse_geom.CellSizeArray()[n];
    }
    for (int n = AMREX_SPACEDIM; n < 3; ++n) {
        boundary_spacing[n] = 0.0_rt;
    }

    GpuArray<Real, 3> problo;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        problo[n] = crse_geom.ProbLoArray()[n];
    }
    for (int n = AMREX_SPACEDIM; n < 3; ++n) {
        problo[n] = 0.0_rt;
    }

    GpuArray<Real, 3> probhi;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        probhi[n] = crse_geom.ProbHiArray()[n];
    }
    for (int n = AMREX_SPACEDIM; n < 3; ++n) {
        probhi[n] = 0.0_rt;
    }

    IntVect lowCornerXY( lowIndexXY );
    IntVect highCornerXY  ( highIndexXY );
    IntVect lowCornerXZ( lowIndexXZ );
    IntVect highCornerXZ  ( highIndexXZ );
    IntVect lowCornerYZ( lowIndexYZ );
    IntVect highCornerYZ  ( highIndexYZ );

    Box faceBoxXY(lowCornerXY, highCornerXY);
    Box faceBoxXZ(lowCornerXZ, highCornerXZ);
    Box faceBoxYZ(lowCornerYZ, highCornerYZ);

    const long pointCountXY = faceBoxXY.numPts();
    const long pointCountXZ = faceBoxXZ.numPts();
    const long pointCountYZ = faceBoxYZ.numPts();

    FArrayBox boundaryXYLo(faceBoxXY);
    FArrayBox boundaryXYHi(faceBoxXY);
    FArrayBox boundaryXZLo(faceBoxXZ);
    FArrayBox boundaryXZHi(faceBoxXZ);
    FArrayBox boundaryYZLo(faceBoxYZ);
    FArrayBox boundaryYZHi(faceBoxYZ);

    boundaryXYLo.setVal<RunOn::Device>(0.0);
    boundaryXYHi.setVal<RunOn::Device>(0.0);
    boundaryXZLo.setVal<RunOn::Device>(0.0);
    boundaryXZHi.setVal<RunOn::Device>(0.0);
    boundaryYZLo.setVal<RunOn::Device>(0.0);
    boundaryYZHi.setVal<RunOn::Device>(0.0);

    // Loop through the grids and compute the individual contributions
    // to the BCs. The BC constructor is coded to only add to the
    // BCs, so it is safe to directly hand the arrays to them.

    int physbc_lo[3];
    int physbc_hi[3];

    for (int dir = 0; dir < 3; dir++) {
        physbc_lo[dir] = phys_bc->lo(dir);
        physbc_hi[dir] = phys_bc->hi(dir);
    }

    for (int lev = crse_level; lev <= fine_level; ++lev) {

        // Create a local copy of the RHS so that we can mask it.

        MultiFab source(Rhs[lev - crse_level]->boxArray(),
                        Rhs[lev - crse_level]->DistributionMap(),
                        1, 0);

        MultiFab::Copy(source, *Rhs[lev - crse_level], 0, 0, 1, 0);

        if (lev < fine_level) {
            const MultiFab& mask = dynamic_cast<HydroSim*>(&(parent->getLevel(lev+1)))->make_fine_mask();
            MultiFab::Multiply(source, mask, 0, 0, 1, 0);
        }

        const auto dx = parent->Geom(lev).CellSizeArray();

#ifdef _OPENMP
        int nthreads = omp_get_max_threads();
        Vector<std::unique_ptr<FArrayBox> > local_boundaryXYLo(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_boundaryXYHi(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_boundaryXZLo(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_boundaryXZHi(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_boundaryYZLo(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_boundaryYZHi(nthreads);
        for (int i=0; i<nthreads; i++) {
            local_boundaryXYLo[i].reset(new FArrayBox(faceBoxXY));
            local_boundaryXYHi[i].reset(new FArrayBox(faceBoxXY));
            local_boundaryXZLo[i].reset(new FArrayBox(faceBoxXZ));
            local_boundaryXZHi[i].reset(new FArrayBox(faceBoxXZ));
            local_boundaryYZLo[i].reset(new FArrayBox(faceBoxYZ));
            local_boundaryYZHi[i].reset(new FArrayBox(faceBoxYZ));
        }
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
            local_boundaryXYLo[tid]->setVal<RunOn::Gpu>(0.0);
            local_boundaryXYHi[tid]->setVal<RunOn::Gpu>(0.0);
            local_boundaryXZLo[tid]->setVal<RunOn::Gpu>(0.0);
            local_boundaryXZHi[tid]->setVal<RunOn::Gpu>(0.0);
            local_boundaryYZLo[tid]->setVal<RunOn::Gpu>(0.0);
            local_boundaryYZHi[tid]->setVal<RunOn::Gpu>(0.0);
#endif
            for (MFIter mfi(source, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box bx = mfi.tilebox();

                const auto rho = source[mfi].array();
                const auto vol = (*volume[lev])[mfi].array();

                // Determine if we need to add contributions from any symmetric boundaries.

                GpuArray<bool, 3> doFoldedAddLo {false};
                GpuArray<bool, 3> doFoldedAddHi {false};
                bool doFoldedAdd {false};

                for (int b = 0; b < 3; ++b) {
                    if (physbc_lo[b] == amrex::PhysBCType::symmetry) {
                        doFoldedAddLo[b] = true;
                        doFoldedAdd      = true;
                    }

                    if (physbc_hi[b] == amrex::PhysBCType::symmetry) {
                        doFoldedAddHi[b] = true;
                        doFoldedAdd      = true;
                    }
                }

#ifdef _OPENMP
                auto boundaryXYLo_arr = local_boundaryXYLo[tid]->array();
                auto boundaryXYHi_arr = local_boundaryXYHi[tid]->array();
                auto boundaryXZLo_arr = local_boundaryXZLo[tid]->array();
                auto boundaryXZHi_arr = local_boundaryXZHi[tid]->array();
                auto boundaryYZLo_arr = local_boundaryYZLo[tid]->array();
                auto boundaryYZHi_arr = local_boundaryYZHi[tid]->array();
#else
                auto boundaryXYLo_arr = boundaryXYLo.array();
                auto boundaryXYHi_arr = boundaryXYHi.array();
                auto boundaryXZLo_arr = boundaryXZLo.array();
                auto boundaryXZHi_arr = boundaryXZHi.array();
                auto boundaryYZLo_arr = boundaryYZLo.array();
                auto boundaryYZHi_arr = boundaryYZHi.array();
#endif

                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    GpuArray<Real, 3> loc, boundaryLoc;
                    loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];

#if AMREX_SPACEDIM >= 2
                    loc[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1];
#else
                    loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
                    loc[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2];
#else
                    loc[2] = 0.0_rt;
#endif

                    // Do xy interfaces first. Note that the boundary conditions
                    // on phi are expected to live directly on the interface.
                    // We also have to handle the domain corners correctly. We are
                    // assuming that bc_lo = domlo - 1 and bc_hi = domhi + 1, where
                    // domlo and domhi are the coarse domain extent.

                    for (int m = bc_lo[1]; m <= bc_hi[1]; ++m) {
                        if (m == bc_lo[1]) {
                            boundaryLoc[1] = problo[1];
                        }
                        else if (m == bc_hi[1]) {
                            boundaryLoc[1] = probhi[1];
                        }
                        else {
                            boundaryLoc[1] = problo[1] + (static_cast<Real>(m) + 0.5_rt) * boundary_spacing[1];
                        }
                        Real dy2 = (loc[1] - boundaryLoc[1]) * (loc[1] - boundaryLoc[1]);

                        for (int l = bc_lo[0]; l <= bc_hi[0]; ++l) {
                            if (l == bc_lo[0]) {
                                boundaryLoc[0] = problo[0];
                            }
                            else if (l == bc_hi[0]) {
                                boundaryLoc[0] = probhi[0];
                            }
                            else {
                                boundaryLoc[0] = problo[0] + (static_cast<Real>(l) + 0.5_rt) * boundary_spacing[0];
                            }
                            Real dx2 = (loc[0] - boundaryLoc[0]) * (loc[0] - boundaryLoc[0]);

                            boundaryLoc[2] = problo[2];
                            Real dz2 = (loc[2] - boundaryLoc[2]) * (loc[2] - boundaryLoc[2]);

                            Real r = std::sqrt(dx2 + dy2 + dz2);

                            Real dbc = -C::Gconst * rho(i,j,k) * vol(i,j,k) / r;

                            // Now, add any contributions from mass that is hidden behind
                            // a symmetric boundary.

                            if (doFoldedAdd) {

                                dbc += accumulate_mirror_contribution(loc, boundaryLoc, problo, probhi,
                                                                rho(i,j,k), vol(i,j,k),
                                                                doFoldedAddLo, doFoldedAddHi);

                            }

                            Gpu::Atomic::Add(&boundaryXYLo_arr(l,m,0), dbc);

                            boundaryLoc[2] = probhi[2];
                            dz2 = (loc[2] - boundaryLoc[2]) * (loc[2] - boundaryLoc[2]);

                            r = std::sqrt(dx2 + dy2 + dz2);

                            dbc = -C::Gconst * rho(i,j,k) * vol(i,j,k) / r;

                            if (doFoldedAdd) {

                                dbc += accumulate_mirror_contribution(loc, boundaryLoc, problo, probhi,
                                                                rho(i,j,k), vol(i,j,k),
                                                                doFoldedAddLo, doFoldedAddHi);

                            }

                            Gpu::Atomic::Add(&boundaryXYHi_arr(l,m,0), dbc);

                        }

                    }

                    // Now do xz interfaces.

                    for (int n = bc_lo[2]; n <= bc_hi[2]; ++n) {
                        if (n == bc_lo[2]) {
                            boundaryLoc[2] = problo[2];
                        }
                        else if (n == bc_hi[2]) {
                            boundaryLoc[2] = probhi[2];
                        }
                        else {
                            boundaryLoc[2] = problo[2] + (static_cast<Real>(n) + 0.5_rt) * boundary_spacing[2];
                        }
                        Real dz2 = (loc[2] - boundaryLoc[2]) * (loc[2] - boundaryLoc[2]);

                        for (int l = bc_lo[0]; l <= bc_hi[0]; ++l) {
                            if (l == bc_lo[0]) {
                                boundaryLoc[0] = problo[0];
                            }
                            else if (l == bc_hi[0]) {
                                boundaryLoc[0] = probhi[0];
                            }
                            else {
                                boundaryLoc[0] = problo[0] + (static_cast<Real>(l) + 0.5_rt) * boundary_spacing[0];
                            }
                            Real dx2 = (loc[0] - boundaryLoc[0]) * (loc[0] - boundaryLoc[0]);

                            boundaryLoc[1] = problo[1];
                            Real dy2 = (loc[1] - boundaryLoc[1]) * (loc[1] - boundaryLoc[1]);

                            Real r = std::sqrt(dx2 + dy2 + dz2);

                            Real dbc = -C::Gconst * rho(i,j,k) * vol(i,j,k) / r;

                            if (doFoldedAdd) {

                                dbc += accumulate_mirror_contribution(loc, boundaryLoc, problo, probhi,
                                                                rho(i,j,k), vol(i,j,k),
                                                                doFoldedAddLo, doFoldedAddHi);

                            }

                            Gpu::Atomic::Add(&boundaryXZLo_arr(l,0,n), dbc);

                            boundaryLoc[1] = probhi[1];
                            dy2 = (loc[1] - boundaryLoc[1]) * (loc[1] - boundaryLoc[1]);

                            r = std::sqrt(dx2 + dy2 + dz2);

                            dbc = -C::Gconst * rho(i,j,k) * vol(i,j,k) / r;

                            if (doFoldedAdd) {

                                dbc += accumulate_mirror_contribution(loc, boundaryLoc, problo, probhi,
                                                                rho(i,j,k), vol(i,j,k),
                                                                doFoldedAddLo, doFoldedAddHi);

                            }

                            Gpu::Atomic::Add(&boundaryXZHi_arr(l,0,n), dbc);

                        }

                    }

                    // Finally, do yz interfaces.

                    for (int n = bc_lo[2]; n <= bc_hi[2]; ++n) {
                        if (n == bc_lo[2]) {
                            boundaryLoc[2] = problo[2];
                        }
                        else if (n == bc_hi[2]) {
                            boundaryLoc[2] = probhi[2];
                        }
                        else {
                            boundaryLoc[2] = problo[2] + (static_cast<Real>(n) + 0.5_rt) * boundary_spacing[2];
                        }
                        Real dz2 = (loc[2] - boundaryLoc[2]) * (loc[2] - boundaryLoc[2]);

                        for (int m = bc_lo[1]; m <= bc_hi[1]; ++m) {
                            if (m == bc_lo[1]) {
                                boundaryLoc[1] = problo[1];
                            }
                            else if (m == bc_hi[1]) {
                                boundaryLoc[1] = probhi[1];
                            }
                            else {
                                boundaryLoc[1] = problo[1] + (static_cast<Real>(m) + 0.5_rt) * boundary_spacing[1];
                            }
                            Real dy2 = (loc[1] - boundaryLoc[1]) * (loc[1] - boundaryLoc[1]);

                            boundaryLoc[0] = problo[0];
                            Real dx2 = (loc[0] - boundaryLoc[0]) * (loc[0] - boundaryLoc[0]);

                            Real r = std::sqrt(dx2 + dy2 + dz2);

                            Real dbc = -C::Gconst * rho(i,j,k) * vol(i,j,k) / r;

                            if (doFoldedAdd) {

                                dbc += accumulate_mirror_contribution(loc, boundaryLoc, problo, probhi,
                                                                rho(i,j,k), vol(i,j,k),
                                                                doFoldedAddLo, doFoldedAddHi);

                            }

                            Gpu::Atomic::Add(&boundaryYZLo_arr(0,m,n), dbc);

                            boundaryLoc[0] = probhi[0];
                            dx2 = (loc[0] - boundaryLoc[0]) * (loc[0] - boundaryLoc[0]);

                            r = std::sqrt(dx2 + dy2 + dz2);

                            dbc = -C::Gconst * rho(i,j,k) * vol(i,j,k) / r;

                            if (doFoldedAdd) {

                                dbc += accumulate_mirror_contribution(loc, boundaryLoc, problo, probhi,
                                                                rho(i,j,k), vol(i,j,k),
                                                                doFoldedAddLo, doFoldedAddHi);

                            }

                            Gpu::Atomic::Add(&boundaryYZHi_arr(0,m,n), dbc);

                        }

                    }

                });

            }

#ifdef _OPENMP
            Real* boundaryXYLo_data = boundaryXYLo.dataPtr();
            Real* boundaryXYHi_data = boundaryXYHi.dataPtr();
            Real* boundaryXZLo_data = boundaryXZLo.dataPtr();
            Real* boundaryXZHi_data = boundaryXZHi.dataPtr();
            Real* boundaryYZLo_data = boundaryYZLo.dataPtr();
            Real* boundaryYZHi_data = boundaryYZHi.dataPtr();
#pragma omp barrier
#pragma omp for nowait
            for (int i=0; i<pointCountXY; i++) {
                for (int it=0; it<nthreads; it++) {
                    const Real* pl = local_boundaryXYLo[it]->dataPtr();
                    const Real* ph = local_boundaryXYHi[it]->dataPtr();
                    boundaryXYLo_data[i] += pl[i];
                    boundaryXYHi_data[i] += ph[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<pointCountXZ; i++) {
                for (int it=0; it<nthreads; it++) {
                    const Real* pl = local_boundaryXZLo[it]->dataPtr();
                    const Real* ph = local_boundaryXZHi[it]->dataPtr();
                    boundaryXZLo_data[i] += pl[i];
                    boundaryXZHi_data[i] += ph[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<pointCountYZ; i++) {
                for (int it=0; it<nthreads; it++) {
                    const Real* pl = local_boundaryYZLo[it]->dataPtr();
                    const Real* ph = local_boundaryYZHi[it]->dataPtr();
                    boundaryYZLo_data[i] += pl[i];
                    boundaryYZHi_data[i] += ph[i];
                }
            }
#endif
        }

    } // end loop over levels

    // because the number of elements in mpi_reduce is int
    BL_ASSERT(pointCountXY <= std::numeric_limits<int>::max());
    BL_ASSERT(pointCountXZ <= std::numeric_limits<int>::max());
    BL_ASSERT(pointCountYZ <= std::numeric_limits<int>::max());

    ParallelDescriptor::ReduceRealSum(boundaryXYLo.dataPtr(), static_cast<int>(pointCountXY));
    ParallelDescriptor::ReduceRealSum(boundaryXYHi.dataPtr(), static_cast<int>(pointCountXY));
    ParallelDescriptor::ReduceRealSum(boundaryXZLo.dataPtr(), static_cast<int>(pointCountXZ));
    ParallelDescriptor::ReduceRealSum(boundaryXZHi.dataPtr(), static_cast<int>(pointCountXZ));
    ParallelDescriptor::ReduceRealSum(boundaryYZLo.dataPtr(), static_cast<int>(pointCountYZ));
    ParallelDescriptor::ReduceRealSum(boundaryYZHi.dataPtr(), static_cast<int>(pointCountYZ));

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx= mfi.growntilebox();

        auto p = phi[mfi].array();

        auto boundaryXYLo_arr = boundaryXYLo.array();
        auto boundaryXYHi_arr = boundaryXYHi.array();
        auto boundaryXZLo_arr = boundaryXZLo.array();
        auto boundaryXZHi_arr = boundaryXZHi.array();
        auto boundaryYZLo_arr = boundaryYZLo.array();
        auto boundaryYZHi_arr = boundaryYZHi.array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (i == bc_lo[0]) {
                p(i,j,k) = boundaryYZLo_arr(0,j,k);
            }

            if (i == bc_hi[0]) {
                p(i,j,k) = boundaryYZHi_arr(0,j,k);
            }

            if (j == bc_lo[1]) {
                p(i,j,k) = boundaryXZLo_arr(i,0,k);
            }

            if (j == bc_hi[1]) {
                p(i,j,k) = boundaryXZHi_arr(i,0,k);
            }

            if (k == bc_lo[2]) {
                p(i,j,k) = boundaryXYLo_arr(i,j,0);
            }

            if (k == bc_hi[2]) {
                p(i,j,k) = boundaryXYHi_arr(i,j,0);
            }
        });
    }

    if (gravity::verbose)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        Real      end    = ParallelDescriptor::second() - strt;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(end,IOProc);
        amrex::Print() << "Gravity::fill_direct_sum_boundaries() time = " << end << std::endl << std::endl;
#ifdef BL_LAZY
        });
#endif
    }

}
#endif

#if (AMREX_SPACEDIM < 3)
void
Gravity::applyCurvatureTerms(int level, MultiFab& Rhs, const Vector<MultiFab*>& coeffs) const
{
    BL_PROFILE("Gravity::applyCurvatureTerms()");

    const auto dx = parent->Geom(level).CellSizeArray();
    const auto problo = parent->Geom(level).ProbLoArray();
    const int coord_type = parent->Geom(level).Coord();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(Rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(1);
        const Box& xbx = mfi.nodaltilebox(0);
#if AMREX_SPACEDIM >= 2
        const Box& ybx = mfi.nodaltilebox(1);
#endif

        // Modify Rhs and coeffs with the appropriate metric terms.
        apply_curvilinear_weights(bx,
                     Rhs.array(mfi), (Rhs[mfi]).box(),
                     (*coeffs[0]).array(mfi), xbx,
#if AMREX_SPACEDIM >= 2
                     (*coeffs[1]).array(mfi), ybx,
#endif
                     problo, dx, coord_type);
    }
}

#endif

void
Gravity::build_solver_bc ()
{
    const Geometry& geom = parent->Geom(0);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (geom.isPeriodic(idim)) {
            mlmg_lobc[idim] = MLLinOp::BCType::Periodic;
            mlmg_hibc[idim] = MLLinOp::BCType::Periodic;
        } else {
            if (phys_bc->lo(idim) == amrex::PhysBCType::symmetry) {
                mlmg_lobc[idim] = MLLinOp::BCType::Neumann;
            } else {
                mlmg_lobc[idim] = MLLinOp::BCType::Dirichlet;
            }
            if (phys_bc->hi(idim) == amrex::PhysBCType::symmetry) {
                mlmg_hibc[idim] = MLLinOp::BCType::Neumann;
            } else {
                mlmg_hibc[idim] = MLLinOp::BCType::Dirichlet;
            }
        }
    }

    // Set Neumann bc at r=0.
    if (geom.IsSPHERICAL() || geom.IsRZ() ) {
        mlmg_lobc[0] = MLLinOp::BCType::Neumann;
    }
}

void
Gravity::set_mass_shift (Real time, bool span_all_levels) const
{
    BL_PROFILE("Gravity::set_mass_shift()");

    const Geometry& geom = parent->Geom(0);

    if (!geom.isAllPeriodic()) {
        mass_shift = 0.0;
    }
    else {
        Real mass_shift_old = mass_shift;
        mass_shift = 0.0;

        if (span_all_levels)
        {
            for (int lev = 0; lev <= parent->finestLevel(); lev++) {
                auto* cs = dynamic_cast<HydroSim*>(&parent->getLevel(lev));
                if (cs != nullptr) {
                    mass_shift += cs->volWeightedTotal("density", time);
                } else {
                    amrex::Abort("unable to access volWeightedTotal");
                }
            }
        }
        else
        {
            auto* cs = dynamic_cast<HydroSim*>(&parent->getLevel(0));
            if (cs != nullptr) {
                mass_shift = cs->volWeightedTotal("density", time, false, false);  // do not mask off fine grids
            } else {
                amrex::Abort("unable to access volWeightedTotal");
            }
        }

        mass_shift = mass_shift / geom.ProbSize();
        if (gravity::verbose > 1) {
            amrex::Print() << "Defining average density to be " << mass_shift << std::endl;
        }

        Real diff = std::abs(mass_shift - mass_shift_old);
        Real eps = 1.e-10 * std::abs(mass_shift_old);
        if (diff > eps && mass_shift_old > 0)
        {
            if (ParallelDescriptor::IOProcessor())
            {
                std::cout << " ... new vs old mass_shift " << mass_shift << " " << mass_shift_old
                          << " ... diff is " << diff <<  std::endl;
                std::cout << " ... Gravity::set_mass_shift -- total mass has changed!" << std::endl;;
            }
        }
    }
}

void
Gravity::include_point_source_field (int level, MultiFab& phi, MultiFab& gravity_field_out) const
{
    BL_PROFILE("Gravity::include_point_source_field()");

    const auto dx     = parent->Geom(level).CellSizeArray();
    const auto problo = parent->Geom(level).ProbLoArray();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(gravity_field_out, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox();

        Array4<Real> const grav_vec_arr = gravity_field_out.array(mfi);
        Array4<Real> const phi_arr = phi.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Compute radial gravity due to a point mass at center[:].

            Real x = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];
#if AMREX_SPACEDIM >= 2
            Real y = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
#else
            Real y = 0.0;
#endif
#if AMREX_SPACEDIM == 3
            Real z = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
#else
            Real z = 0.0;
#endif


            if(hydrosim::point_mass_is_shifted == 1)
            {
                Real offset_radius = hydrosim::point_mass_shift;

                if(AMREX_SPACEDIM == 1)
                {
                    x += offset_radius;
                }
                else if(AMREX_SPACEDIM ==2)
                {
                    y += offset_radius;
                }
                else if(AMREX_SPACEDIM == 3)
                {
                    z += offset_radius;
                }

            }

            Real rsq = x * x + y * y + z * z;
            Real radial_point_accel = -C::Gconst * hydrosim::point_mass / rsq;

            Real inv_dist = 1.e0_rt / std::sqrt(rsq);

            // Note that gravity_view may have more ghost zones than
            // phi, so we need to check that we're doing
            // valid indexing here.

            if (phi_arr.contains(i,j,k)) {
                phi_arr(i,j,k) -= C::Gconst * hydrosim::point_mass * inv_dist;
            }

            grav_vec_arr(i,j,k,0) += radial_point_accel * (x * inv_dist);
            grav_vec_arr(i,j,k,1) += radial_point_accel * (y * inv_dist);
            grav_vec_arr(i,j,k,2) += radial_point_accel * (z * inv_dist);
        });
    }
}

void
Gravity::build_radial_field(int level, Real time, RealVector& radial_field)
{
    BL_PROFILE("Gravity::build_radial_field()");

    const Real strt = ParallelDescriptor::second();

    // This is just here in case we need to debug ...
    int diag_print_on = 0;

    Real total_over_levels = 0.;

    for (int lev = 0; lev <= level; lev++)
    {
        const Real t_old = LevelTable[lev]->get_state_data(State_Type).prevTime();
        const Real t_new = LevelTable[lev]->get_state_data(State_Type).curTime();
        const Real eps   = (t_new - t_old) * 1.e-6;

        // Create MultiFab with NUM_STATE components and no ghost cells
        MultiFab S(grids[lev],dmap[lev],NUM_STATE,0);

        if ( eps == 0.0 ) {  // NOLINT(bugprone-branch-clone,-warnings-as-errors)
            // Old and new time are identical; this should only happen if
            // dt is smaller than roundoff compared to the current time,
            // in which case we're probably in trouble anyway,
            // but we will still handle it gracefully here.
            MultiFab::Copy(S, LevelTable[lev]->get_new_data(State_Type), 0, 0, NUM_STATE, 0);
        }
        else if ( std::abs(time-t_old) < eps)
        {
            MultiFab::Copy(S, LevelTable[lev]->get_old_data(State_Type), 0, 0, NUM_STATE, 0);
        }
        else if ( std::abs(time-t_new) < eps)
        {
            MultiFab::Copy(S, LevelTable[lev]->get_new_data(State_Type), 0, 0, NUM_STATE, 0);
        }
        else if (time > t_old && time < t_new)
        {
            Real alpha   = (time - t_old)/(t_new - t_old);
            Real oneMinusAlpha = 1.0 - alpha;

            MultiFab::Copy(S, LevelTable[lev]->get_old_data(State_Type), 0, 0, NUM_STATE, 0);
            S.mult(oneMinusAlpha);

            MultiFab S_new(grids[lev],dmap[lev],NUM_STATE,0);
            MultiFab::Copy(S_new, LevelTable[lev]->get_new_data(State_Type), 0, 0, NUM_STATE, 0);
            S_new.mult(alpha);

            S.plus(S_new,0,NUM_STATE,0);
        }
        else
        {
            std::cout << " Level / Time in build_radial_field is: " << lev << " " << time  << std::endl;
            std::cout << " but old / new time      are: " << t_old << " " << t_new << std::endl;
            amrex::Abort("Problem in Gravity::build_radial_field");
        }

        if (lev < level)
        {
            auto* fine_level = dynamic_cast<HydroSim*>(&(parent->getLevel(lev+1)));
        if (fine_level != nullptr) {
        const MultiFab& mask = fine_level->make_fine_mask();
        for (int n = 0; n < NUM_STATE; ++n) {
            MultiFab::Multiply(S, mask, 0, n, 1, 0);
        }
        } else {
                amrex::Abort("unable to create mask");
            }
        }

        int n1d = static_cast<int>(radial_shell_mass[lev].size());

#ifdef GR_GRAV
        Real* const cur_pres_bins = radial_shell_pres[lev].dataPtr();
#endif
        Real* const cur_vol_bins = radial_shell_vol[lev].dataPtr();
        Real* const cur_mass_bins = radial_shell_mass[lev].dataPtr();

        amrex::ParallelFor(n1d,
        [=] AMREX_GPU_DEVICE (int i) noexcept
        {
#ifdef GR_GRAV
            cur_pres_bins[i] = 0.;
#endif
            cur_vol_bins[i] = 0.;
            cur_mass_bins[i] = 0.;
        });

#ifdef _OPENMP
        int nthreads = omp_get_max_threads();
#ifdef GR_GRAV
        Vector< RealVector > local_radial_pres(nthreads);
#endif
        Vector< RealVector > local_radial_mass(nthreads);
        Vector< RealVector > local_radial_vol (nthreads);
        for (int i=0; i<nthreads; i++) {
#ifdef GR_GRAV
            local_radial_pres[i].resize(n1d,0.0);
#endif
            local_radial_mass[i].resize(n1d,0.0);
            local_radial_vol [i].resize(n1d,0.0);
        }
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#endif
            for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                FArrayBox& fab = S[mfi];

                tally_radial_profile(bx,
                                    fab.array(),
#ifdef _OPENMP
                                    local_radial_mass[tid],
                                    local_radial_vol[tid],
#ifdef GR_GRAV
                                    local_radial_pres[tid],
#endif
#else
                                    radial_shell_mass[lev],
                                    radial_shell_vol[lev],
#ifdef GR_GRAV
                                    radial_shell_pres[lev],
#endif
#endif
                                    n1d, lev);
            }

#ifdef _OPENMP
#pragma omp barrier
#pragma omp for
            for (int i=0; i<n1d; i++) {
                for (int it=0; it<nthreads; it++) {
#ifdef GR_GRAV
                    radial_shell_pres[lev][i] += local_radial_pres[it][i];
#endif
                    radial_shell_mass[lev][i] += local_radial_mass[it][i];
                    radial_shell_vol [lev][i] += local_radial_vol [it][i];
                }
            }
#endif
        }

        if (!ParallelDescriptor::UseGpuAwareMpi()) {
            Gpu::prefetchToHost(radial_shell_mass[lev].begin(), radial_shell_mass[lev].end());
            Gpu::prefetchToHost(radial_shell_vol[lev].begin(), radial_shell_vol[lev].end());
#ifdef GR_GRAV
            Gpu::prefetchToHost(radial_shell_pres[lev].begin(), radial_shell_pres[lev].end());
#endif
        }

        ParallelDescriptor::ReduceRealSum(radial_shell_mass[lev].dataPtr() ,n1d);
        ParallelDescriptor::ReduceRealSum(radial_shell_vol[lev].dataPtr()  ,n1d);
#ifdef GR_GRAV
        ParallelDescriptor::ReduceRealSum(radial_shell_pres[lev].dataPtr()  ,n1d);
#endif

        if (!ParallelDescriptor::UseGpuAwareMpi()) {
            Gpu::prefetchToDevice(radial_shell_mass[lev].begin(), radial_shell_mass[lev].end());
            Gpu::prefetchToDevice(radial_shell_vol[lev].begin(), radial_shell_vol[lev].end());
#ifdef GR_GRAV
            Gpu::prefetchToDevice(radial_shell_pres[lev].begin(), radial_shell_pres[lev].end());
#endif
        }

        if (diag_print_on > 0)
        {
            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            reduce_op.eval(n1d, reduce_data,
            [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
            {
                return {cur_mass_bins[i]};
            });

            ReduceTuple hv = reduce_data.value();
            Real sum = amrex::get<0>(hv);

            total_over_levels += sum;
        }
    }

    if (diag_print_on > 0) {
        amrex::Print() << "Gravity::build_radial_field: Sum of mass over all levels " << total_over_levels << std::endl;
    }

    int n1d = static_cast<int>(radial_shell_mass[level].size());
    RealVector radial_mass_total(n1d,0);

    const Real* const target_mass_bins = radial_shell_mass[level].dataPtr();
    Real* const mass_total = radial_mass_total.dataPtr();

    // First add the contribution from this level
    amrex::ParallelFor(n1d,
    [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        mass_total[i] = target_mass_bins[i];
    });

    // Now add the contribution from coarser levels
    if (level > 0)
    {
        int ratio = parent->refRatio(level-1)[0];
        for (int lev = level-1; lev >= 0; lev--)
        {
            if (lev < level-1) {
                ratio *= parent->refRatio(lev)[0];
            }

            const Real* const cur_mass_bins = radial_shell_mass[lev].dataPtr();

            amrex::ParallelFor(n1d/ratio,
            [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                for (int n = 0; n < ratio; n++)
                {
                    mass_total[ratio*i+n] += 1./double(ratio) * cur_mass_bins[i];
                }
            });
        }
    }

    if (diag_print_on > 0 && ParallelDescriptor::IOProcessor())
    {
        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        reduce_op.eval(n1d, reduce_data,
        [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
        {
            return {mass_total[i]};
        });

        ReduceTuple hv = reduce_data.value();
        Real added_total = amrex::get<0>(hv);

        std::cout << "Gravity::build_radial_field: Sum of combined mass " << added_total << std::endl;
    }

    const Geometry& geom = parent->Geom(level);
    const Real* dx = geom.CellSize();
    Real dr        = dx[0] / static_cast<Real>(gravity::radial_bin_ratio);

    RealVector radial_vol_total(n1d,0);
    RealVector radial_density_total(n1d,0);

    Real* const vol_total = radial_vol_total.dataPtr();
    Real* const density_total = radial_density_total.dataPtr();

    const Real* const level_vol = radial_shell_vol[level].dataPtr();

    // First add the contribution from this level
    amrex::ParallelFor(n1d,
    [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        vol_total[i] = level_vol[i];
    });

    // Now add the contribution from coarser levels
    if (level > 0)
    {
        int ratio = parent->refRatio(level-1)[0];
        for (int lev = level-1; lev >= 0; lev--)
        {
            if (lev < level-1) {
                ratio *= parent->refRatio(lev)[0];
            }

            const Real* cur_vol_bins = radial_shell_vol[lev].dataPtr();

            amrex::ParallelFor(n1d/ratio,
            [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                for (int n = 0; n < ratio; n++)
                {
                    vol_total[ratio*i+n]  += 1./double(ratio) * cur_vol_bins[i];
                }
            });
        }
    }

    amrex::ParallelFor(n1d,
    [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        density_total[i] = mass_total[i];
        if (vol_total[i] > 0.) {
            density_total[i] /= vol_total[i];
        }
    });

#ifdef GR_GRAV
    RealVector radial_pres_total(n1d,0);

    Real* const pres_total = radial_pres_total.dataPtr();
    Real* const target_pres_bins = radial_shell_pres[level].dataPtr();

    // First add the contribution from this level
    amrex::ParallelFor(n1d,
    [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        pres_total[i] = target_pres_bins[i];
    });

    // Now add the contribution from coarser levels
    if (level > 0)
    {
        int ratio = parent->refRatio(level-1)[0];
        for (int lev = level-1; lev >= 0; lev--)
        {
            if (lev < level-1) ratio *= parent->refRatio(lev)[0];

            const Real* cur_pres_bins = radial_shell_pres[lev].dataPtr();

            amrex::ParallelFor(n1d/ratio,
            [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                for (int n = 0; n < ratio; n++) {
                    pres_total[ratio*i+n] += 1./double(ratio) * cur_pres_bins[i];
                }
            });
        }
    }

    amrex::ParallelFor(n1d,
    [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        if (vol_total[i] > 0.) {
            pres_total[i] /= vol_total[i];
        }
    });
#endif

    // Integrate radially outward to define the gravity

    const Real* const mass = radial_mass_total.dataPtr();
#ifdef GR_GRAV
    Real* const den = radial_density_total.dataPtr();
    Real* const pres = radial_pres_total.dataPtr();
#endif
    Real* const gravity_view = radial_field.dataPtr();

    // At a given radius r corresponding to an index i, the gravity is
    // g(r) = -G * M(r) / r**2. The enclosed mass can be computed via
    // an inclusive prefix sum, which yields the same result one would
    // obtain with a serial calculation from 0 to i but with the ability
    // to be parallelized. The first lambda returns the mass at radial
    // zone index i (which includes the upper shell of zone i-1 and the
    // lower shell of zone i) and the second lambda accepts the current
    // enclosed mass as an argument and computes the gravity at the
    // corresponding radius.

    Scan::PrefixSum<Real> (n1d,
        [=] AMREX_GPU_DEVICE (int i) -> Real
        {
            Real dM = 0.0;

            if (i > 0) {
                // The mass at (i-1) is distributed into an upper and lower shell; the
                // contribution to the mass at zone center i is from the upper shell.
                Real rlo = (static_cast<Real>(i-1)         ) * dr;
                Real rc  = (static_cast<Real>(i-1) + 0.5_rt) * dr;
                Real rhi = (static_cast<Real>(i-1) + 1.0_rt) * dr;

                Real shell_volume = (4.0_rt / 3.0_rt * std::numbers::pi) * dr * (rhi * rhi * rhi - rc  * rc  * rc);
                Real zone_volume  = (4.0_rt / 3.0_rt * std::numbers::pi) * dr * (rhi * rhi * rhi - rlo * rlo * rlo);
                dM = dM + (shell_volume / zone_volume) * mass[i-1];
            }

            Real rlo = (static_cast<Real>(i)         ) * dr;
            Real rc  = (static_cast<Real>(i) + 0.5_rt) * dr;
            Real rhi = (static_cast<Real>(i) + 1.0_rt) * dr;

            // The mass at (i) is distributed into an upper and lower shell; the
            // contribution to the mass at zone center i is from the lower shell.
            Real shell_volume = (4.0_rt / 3.0_rt * std::numbers::pi) * dr * (rc  * rc  * rc  - rlo * rlo * rlo);
            Real zone_volume  = (4.0_rt / 3.0_rt * std::numbers::pi) * dr * (rhi * rhi * rhi - rlo * rlo * rlo);
            dM = dM + (shell_volume / zone_volume) * mass[i];

            return dM;
        },
        [=] AMREX_GPU_DEVICE (int i, Real const& running_mass)
        {
            Real rc = (static_cast<Real>(i) + 0.5_rt) * dr;

            gravity_view[i] = -C::Gconst * running_mass / (rc * rc);

#ifdef GR_GRAV
            // Tolman-Oppenheimer-Volkoff (TOV) post-Newtonian correction

            if (den[i] > 0.0_rt) {
                Real ga = (1.0_rt + pres[i] / (den[i] * C::c_light * C::c_light));
                Real gb = (1.0_rt + (4.0_rt * std::numbers::pi) * rc * rc * rc * pres[i] / (running_mass * C::c_light * C::c_light));
                Real gc = 1.0_rt / (1.0_rt - 2.0_rt * C::Gconst * running_mass / (rc * C::c_light * C::c_light));

                gravity_view[i] = gravity_view[i] * ga * gb * gc;
            }
#endif
        },
        Scan::Type::inclusive);

    if (gravity::verbose)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        Real      end    = ParallelDescriptor::second() - strt;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(end,IOProc);
        amrex::Print() << "Gravity::build_radial_field() time = " << end << std::endl << std::endl;
#ifdef BL_LAZY
        });
#endif
    }
}

Vector<std::unique_ptr<MultiFab> >
Gravity::fetch_poisson_rhs (int crse_level, int nlevs, int use_new_state)
{
    Vector<std::unique_ptr<MultiFab> > poisson_rhs(nlevs);

    for (int ilev = 0; ilev < nlevs; ++ilev)
    {
        int amr_lev = ilev + crse_level;
        poisson_rhs[ilev] = std::make_unique<MultiFab>(grids[amr_lev],dmap[amr_lev],1,0);
        MultiFab& state = (use_new_state == 1) ?
            LevelTable[amr_lev]->get_new_data(State_Type) :
            LevelTable[amr_lev]->get_old_data(State_Type);
        MultiFab::Copy(*poisson_rhs[ilev], state, URHO, 0,1,0);
    }
    return poisson_rhs;
}

void
Gravity::sanity_check (int level)
{
    // This is a sanity check on whether we are trying to fill multipole boundary conditiosn
    //  for grids at this level > 0 -- this case is not currently supported.
    //  Here we shrink the domain at this level by 1 in any direction which is not symmetry or periodic,
    //  then ask if the grids at this level are contained in the shrunken domain.  If not, then grids
    //  at this level touch the domain boundary and we must abort.

    const Geometry& geom = parent->Geom(level);

    if (level > 0  && !geom.isAllPeriodic())
    {
        Box inset_domain(geom.Domain());
        for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
        {
            if (!geom.isPeriodic(dir))
            {
                if (phys_bc->lo(dir) != amrex::PhysBCType::symmetry) {
                    inset_domain.growLo(dir,-1);
                }
                if (phys_bc->hi(dir) != amrex::PhysBCType::symmetry) {
                    inset_domain.growHi(dir,-1);
                }
            }
        }
        if (!inset_domain.contains(grids[level].minimalBox())) {
            amrex::Error("Oops -- don't know how to set boundary conditions for grids at this level that touch the domain boundary!");
        }
    }
}

void
Gravity::update_max_density()
{
    BL_PROFILE("Gravity::update_max_density()");

    // Calculate the maximum value of the RHS over all levels.
    // This should only be called at a synchronization point where
    // all HydroSim levels have valid new time data at the same simulation time.
    // The RHS we will use is the density multiplied by 4*pi*G and also
    // multiplied by the metric terms, just as it would be in a real solve.

    int crse_level = 0;
    int nlevs = parent->finestLevel() + 1;
    int use_new_state = 1;

    const auto& poisson_rhs = fetch_poisson_rhs(crse_level, nlevs, use_new_state);

    const Geometry& level0_geom = parent->Geom(0);

#if (AMREX_SPACEDIM == 3)
    if ( level0_geom.isAllPeriodic() )
    {
        for (int lev = 0; lev < nlevs; ++lev) {
            poisson_rhs[lev]->plus(-mass_shift,0,1,0);
        }
    }
#endif

    for (int lev = 0; lev < nlevs; ++lev)
    {
        poisson_rhs[lev]->mult(FourPiG);
    }

#if (AMREX_SPACEDIM < 3)
    if (level0_geom.IsSPHERICAL() || level0_geom.IsRZ() )
    {
        Vector<Vector<std::unique_ptr<MultiFab> > > coeffs(nlevs);

        for (int lev = 0; lev < nlevs; ++lev) {

            // We need to include this bit about the coefficients because
            // it's required by applyCurvatureTerms.

            coeffs[lev].resize(AMREX_SPACEDIM);

            for (int i = 0; i < AMREX_SPACEDIM ; i++) {
                coeffs[lev][i] = std::make_unique<MultiFab>(amrex::convert(grids[lev],
                                                                           IntVect::TheDimensionVector(i)),
                                                            dmap[lev], 1, 0);

                coeffs[lev][i]->setVal(1.0);
            }

            applyCurvatureTerms(lev, *poisson_rhs[lev], amrex::GetVecOfPtrs(coeffs[lev]));
        }
    }
#endif

    peak_rhs_value = 0.0;

    for (int lev = 0; lev < nlevs; ++lev)
        peak_rhs_value = std::max(peak_rhs_value, poisson_rhs[lev]->max(0));

}

Real
Gravity::solve_potential_mlmg (int crse_level, int fine_level,
                              const Vector<MultiFab*>& phi,
                              const Vector<MultiFab*>& poisson_rhs,
                              const Vector<Vector<MultiFab*> >& phi_gradient,
                              const Vector<MultiFab*>& res,
                              Real time)
{
    BL_PROFILE("Gravity::solve_potential_mlmg()");

    int nlevs = fine_level-crse_level+1;

    if (crse_level == 0 && !(parent->Geom(0).isAllPeriodic()))
    {
        if (gravity::verbose > 1) {
            amrex::Print() << " ... Making bc's for phi at level 0\n";
        }

#if (AMREX_SPACEDIM == 3)
        if ( gravity::boundary_direct_sum ) {
            fill_direct_sum_boundaries(crse_level, fine_level, poisson_rhs, *phi[0]);
        } else {
            fill_multipole_boundaries(crse_level, fine_level, poisson_rhs, *phi[0]);
        }
#elif (AMREX_SPACEDIM == 2)
        fill_multipole_boundaries(crse_level, fine_level, poisson_rhs, *phi[0]);
#else
        fill_multipole_boundaries(crse_level, fine_level, poisson_rhs, *phi[0]);
#endif
    }

    for (int ilev = 0; ilev < nlevs; ++ilev)
    {
        poisson_rhs[ilev]->mult(FourPiG);
    }

    MultiFab CoarsePotential;
    const MultiFab* crse_bcdata = nullptr;
    if (crse_level > 0)
    {
        FetchCoarsePotential(crse_level, CoarsePotential, time);
        crse_bcdata = &CoarsePotential;
    }

    Real rel_tolerance = rel_tol[fine_level];

    // The absolute tolerance is determined by the error tolerance
    // chosen by the user (tol) multiplied by the maximum value of
    // the RHS (4 * pi * G * rho). If we're doing periodic BCs, we
    // subtract off the mass_shift corresponding to the average
    // density on the domain. This will automatically be zero for
    // non-periodic BCs. And this also accounts for the metric
    // terms that are applied in non-Cartesian coordinates.

    Real abs_tolerance = abs_tol[fine_level] * peak_rhs_value;

    Vector<const MultiFab*> crhs{poisson_rhs.begin(), poisson_rhs.end()};
    Vector<std::array<MultiFab*,AMREX_SPACEDIM> > gp;
    for (const auto& x : phi_gradient) {
        gp.push_back({AMREX_D_DECL(x[0],x[1],x[2])});
    }

    return invoke_mlmg_solver(crse_level, fine_level, phi, crhs, gp, res,
                                  crse_bcdata, rel_tolerance, abs_tolerance);
}

void
Gravity::solve_correction_phi(int crse_level, int fine_level,
                             const Vector<MultiFab*>& poisson_rhs,
                             const Vector<MultiFab*>& delta_phi,
                             const Vector<Vector<MultiFab*> >&  grad_correction)
{
    BL_PROFILE("Gravity::solve_correction_phi");

    BL_ASSERT(grad_correction.size() == fine_level - crse_level + 1);
    BL_ASSERT(delta_phi.size() == fine_level - crse_level + 1);

    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
      std::cout << "... solving for delta_phi at crse_level = " << crse_level << std::endl;
      std::cout << "...                    up to fine_level = " << fine_level << std::endl;
    }

    Vector<const MultiFab*> crhs{poisson_rhs.begin(), poisson_rhs.end()};
    Vector<std::array<MultiFab*,AMREX_SPACEDIM> > gp;
    for (const auto& x : grad_correction) {
        gp.push_back({AMREX_D_DECL(x[0],x[1],x[2])});
    }

    Real rel_tolerance = 0.0;
    Real abs_tolerance = *(std::max_element(level_solve_residual_norm.begin() + crse_level,
                                      level_solve_residual_norm.begin() + fine_level+1));

    invoke_mlmg_solver(crse_level, fine_level, delta_phi, crhs, gp, {},
                           nullptr, rel_tolerance, abs_tolerance);
}

Real
Gravity::invoke_mlmg_solver (int crse_level, int fine_level,
                                 const amrex::Vector<amrex::MultiFab*>& phi,
                                 const amrex::Vector<const amrex::MultiFab*>& poisson_rhs,
                                 const amrex::Vector<std::array<amrex::MultiFab*,AMREX_SPACEDIM> >& phi_gradient,
                                 const amrex::Vector<amrex::MultiFab*>& res,
                                 const amrex::MultiFab* const crse_bcdata,
                                 amrex::Real rel_tolerance, amrex::Real abs_tolerance) const
{
    BL_PROFILE("Gravity::invoke_mlmg_solver()");

    Real last_residual_norm = -1.0;

    int nlevs = fine_level-crse_level+1;

    Vector<Geometry> gmv;
    Vector<BoxArray> bav;
    Vector<DistributionMapping> dmv;
    for (int ilev = 0; ilev < nlevs; ++ilev)
    {
        gmv.push_back(parent->Geom(ilev+crse_level));
        bav.push_back(poisson_rhs[ilev]->boxArray());
        dmv.push_back(poisson_rhs[ilev]->DistributionMap());
    }

    LPInfo info;
    info.setAgglomeration(gravity::mlmg_grid_merge);
    info.setConsolidation(gravity::mlmg_proc_gather);

    MLPoisson mlpoisson(gmv, bav, dmv, info);

    // BC
    mlpoisson.setDomainBC(mlmg_lobc, mlmg_hibc);
    if (mlpoisson.needsCoarseDataForBC())
    {
        mlpoisson.setCoarseFineBC(crse_bcdata, parent->refRatio(crse_level-1)[0]);
    }

    for (int ilev = 0; ilev < nlevs; ++ilev)
    {
        mlpoisson.setLevelBC(ilev, phi[ilev]);
    }

    MLMG mlmg(mlpoisson);
    mlmg.setVerbose(gravity::verbose - 1); // With normal verbosity we don't want MLMG information
    if (crse_level == 0) {
        mlmg.setMaxFmgIter(gravity::mlmg_fmg_cycles);
    } else {
        mlmg.setMaxFmgIter(0); // Vcycle
    }

    AMREX_ALWAYS_ASSERT( !phi_gradient.empty() or !res.empty() );
    AMREX_ALWAYS_ASSERT(  phi_gradient.empty() or  res.empty() );

    if (!phi_gradient.empty())
    {
        if (!gmv[0].isAllPeriodic()) {
            mlmg.setConvergenceNormType(MLMGNormType::bnorm);
        }

        mlmg.setNSolve(gravity::mlmg_coarse_nsolve);
        last_residual_norm = mlmg.solve(phi, poisson_rhs, rel_tolerance, abs_tolerance);

        mlmg.getGradSolution(phi_gradient);
    }
    else if (!res.empty())
    {
        mlmg.compResidual(res, phi, poisson_rhs);
    }

    return last_residual_norm;
}
