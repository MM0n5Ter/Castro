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

            for (int lev = 1; lev < nlevs; ++lev) {
                abs_tol[lev] = abs_tol[lev - 1] * amrex::Math::powi<2>(parent->refRatio(lev - 1)[0]);
            }

        } else if (abs_tol_count >= nlevs) {

            pp.getarr("abs_tol", abs_tol, 0, nlevs);

        } else {

            amrex::Abort("If you are providing multiple values for abs_tol, you must provide at least one value for every level up to amr.max_level.");

        }

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
                mass_shift = cs->volWeightedTotal("density", time, false, false);
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
Gravity::sanity_check (int level)
{

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
