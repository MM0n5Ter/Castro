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

#include <multipole_moments.H>
#include <SolverUtils.H>

using namespace amrex;

const Real FourPiG = 4.0 * std::numbers::pi * C::Gconst;

static void fill_coarse_boundary_phi (Gravity& grav, int crse_level, int fine_level,
                                      const Vector<MultiFab*>& rhs, MultiFab& phi)
{
#if (AMREX_SPACEDIM == 3)
    if ( gravity::boundary_direct_sum ) {
        grav.fill_direct_sum_boundaries(crse_level, fine_level, rhs, phi);
    } else {
        grav.fill_multipole_boundaries(crse_level, fine_level, rhs, phi);
    }
#elif (AMREX_SPACEDIM == 2)
    grav.fill_multipole_boundaries(crse_level, fine_level, rhs, phi);
#else
    grav.fill_multipole_boundaries(crse_level, fine_level, rhs, phi);
#endif
}

static MultiFab& potential_field_at (AmrLevel* lev, PotentialTime epoch)
{
    if (epoch == PotentialTime::New) {
        return lev->get_new_data(Potential_Type);
    }
    return lev->get_old_data(Potential_Type);
}

static Real potential_time_at (AmrLevel* lev, PotentialTime epoch)
{
    if (epoch == PotentialTime::New) {
        return lev->get_state_data(Potential_Type).curTime();
    }
    return lev->get_state_data(Potential_Type).prevTime();
}

void
Gravity::solve_potential_level (int               level,
                        MultiFab&         phi,
                        const Vector<MultiFab*>& phi_gradient,
                        PotentialTime epoch)
{
    BL_PROFILE("Gravity::solve_potential_level()");

    if (gravity::verbose > 1) {
        amrex::Print() << " ... solve for phi at level " << level << std::endl;
    }

    const Real strt = ParallelDescriptor::second();

    if (epoch == PotentialTime::Old) {
        sanity_check(level);
    }

    Real time = potential_time_at(LevelTable[level], epoch);


    if (level <= gravity::solve_level_cap) {

        Vector<MultiFab*> phi_ptrs(1, &phi);

        const auto& poisson_rhs = fetch_poisson_rhs(level, 1, epoch);

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

    const PotentialTime epoch = PotentialTime::New;
    run_multilevel_solve(level, highest_level_in, amrex::GetVecOfVecOfPtrs(phi_gradient_new), epoch);

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
                                  PotentialTime epoch)
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
        phi_ptrs[ilev] = &potential_field_at(LevelTable[amr_lev], epoch);
    }

    const auto& poisson_rhs = fetch_poisson_rhs(crse_level, nlevels, epoch);

    Vector<Vector<MultiFab*> > phi_gradient_ptrs(nlevels);
    for (int ilev = 0; ilev < nlevels; ilev++)
    {
        int amr_lev = ilev + crse_level;
        phi_gradient_ptrs[ilev] = phi_gradient[amr_lev];
    }

    Real time = potential_time_at(LevelTable[crse_level], epoch);

    int fine_level = amrex::min(highest_level_in, gravity::solve_level_cap);

    if (fine_level >= crse_level) {

        Vector<MultiFab*> empty_residuals;
        solve_potential_mlmg(crse_level, fine_level,
                            phi_ptrs, amrex::GetVecOfPtrs(poisson_rhs), phi_gradient_ptrs, empty_residuals,
                            time);

        for (int amr_lev = fine_level; amr_lev > crse_level; amr_lev--)
        {
            const IntVect& ratio = parent->refRatio(amr_lev-1);
            amrex::average_down(potential_field_at(LevelTable[amr_lev  ], epoch),
                                potential_field_at(LevelTable[amr_lev-1], epoch),
                                0, 1, ratio);
        }

        for (int amr_lev = fine_level; amr_lev > crse_level; amr_lev--) {
            average_fine_edges_to_coarse(amr_lev-1,epoch);
        }

    }


    for (int amr_lev = gravity::solve_level_cap+1; amr_lev <= highest_level_in; amr_lev++) {


        MultiFab& phi = potential_field_at(LevelTable[amr_lev], epoch);

        LevelTable[amr_lev]->FillCoarsePatch(phi,0,time,Potential_Type,0,1,1);



        PotentialGradBCFunct gradient_phys_bc;


        Interpolater* gradient_interp = &face_linear_interp;


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

Real
Gravity::invoke_mlmg_solver (int crse_level, int fine_level,
                                 const amrex::Vector<amrex::MultiFab*>& phi,
                                 const amrex::Vector<const amrex::MultiFab*>& poisson_rhs,
                                 const amrex::Vector<std::array<amrex::MultiFab*,AMREX_SPACEDIM> >& phi_gradient,
                                 const amrex::Vector<amrex::MultiFab*>& res,
                                 const amrex::MultiFab* const crse_bcdata,
                                 const SolveTolerances& tol) const
{
    BL_PROFILE("Gravity::invoke_mlmg_solver()");

    Real last_residual_norm = -1.0;

    int nlevs = fine_level-crse_level+1;

    AMREX_ALWAYS_ASSERT(static_cast<int>(phi.size()) >= nlevs &&
                        static_cast<int>(poisson_rhs.size()) >= nlevs);

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
    mlmg.setVerbose(gravity::verbose - 1);
    if (crse_level == 0) {
        mlmg.setMaxFmgIter(gravity::mlmg_fmg_cycles);
    } else {
        mlmg.setMaxFmgIter(0);
    }

    AMREX_ALWAYS_ASSERT( !phi_gradient.empty() or !res.empty() );
    AMREX_ALWAYS_ASSERT(  phi_gradient.empty() or  res.empty() );

    if (!phi_gradient.empty())
    {
        if (!gmv[0].isAllPeriodic()) {
            mlmg.setConvergenceNormType(MLMGNormType::bnorm);
        }

        mlmg.setNSolve(gravity::mlmg_coarse_nsolve);
        last_residual_norm = mlmg.solve(phi, poisson_rhs, tol.rel, tol.abs);

        mlmg.getGradSolution(phi_gradient);
    }
    else if (!res.empty())
    {
        mlmg.compResidual(res, phi, poisson_rhs);
    }

    return last_residual_norm;
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

        fill_coarse_boundary_phi(*this, crse_level, fine_level, poisson_rhs, *phi[0]);
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


    Real abs_tolerance = abs_tol[fine_level] * peak_rhs_value;

    Vector<const MultiFab*> crhs{poisson_rhs.begin(), poisson_rhs.end()};
    Vector<std::array<MultiFab*,AMREX_SPACEDIM> > gp;
    for (const auto& x : phi_gradient) {
        gp.push_back({AMREX_D_DECL(x[0],x[1],x[2])});
    }

    return invoke_mlmg_solver(crse_level, fine_level, phi, crhs, gp, res,
                                  crse_bcdata, {rel_tolerance, abs_tolerance});
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
                           nullptr, {rel_tolerance, abs_tolerance});
}

Vector<std::unique_ptr<MultiFab> >
Gravity::fetch_poisson_rhs (int crse_level, int nlevs, PotentialTime epoch)
{
    Vector<std::unique_ptr<MultiFab> > poisson_rhs(nlevs);

    for (int ilev = 0; ilev < nlevs; ++ilev)
    {
        int amr_lev = ilev + crse_level;
        poisson_rhs[ilev] = std::make_unique<MultiFab>(grids[amr_lev],dmap[amr_lev],1,0);
        MultiFab& state = (epoch == PotentialTime::New) ?
            LevelTable[amr_lev]->get_new_data(State_Type) :
            LevelTable[amr_lev]->get_old_data(State_Type);
        MultiFab::Copy(*poisson_rhs[ilev], state, URHO, 0,1,0);
    }
    return poisson_rhs;
}

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

    if (geom.IsSPHERICAL() || geom.IsRZ() ) {
        mlmg_lobc[0] = MLLinOp::BCType::Neumann;
    }
}

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
Gravity::FetchCoarsePotential(int level,
                    MultiFab& coarse_potential,
                    Real      time      )
{
    BL_PROFILE("Gravity::FetchCoarsePotential()");

    BL_ASSERT(level!=0);

    coarse_potential.clear();
    coarse_potential.define(grids[level-1], dmap[level-1], 1, 1);

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
Gravity::synchronize_gravity_levels (int crse_level, int fine_level, const Vector<MultiFab*>& drho, const Vector<MultiFab*>& dphi)
{
    BL_PROFILE("Gravity::synchronize_gravity_levels()");


    if (fine_level > gravity::solve_level_cap) {
        return;
    }

    fine_level = amrex::min(fine_level, gravity::solve_level_cap);

    BL_ASSERT(parent->finestLevel()>crse_level);
    if (gravity::verbose > 1 && ParallelDescriptor::IOProcessor()) {
          std::cout << " ... synchronize_gravity_levels at crse_level " << crse_level << '\n';
          std::cout << " ...     up to finest_level     " << fine_level << '\n';
    }

    const Geometry& crse_geom = parent->Geom(crse_level);
    const Box& crse_domain = crse_geom.Domain();

    int nlevs = fine_level - crse_level + 1;


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



    Vector<std::unique_ptr<MultiFab> > poisson_rhs(nlevs);

    for (int lev = crse_level; lev <= fine_level; ++lev) {
        poisson_rhs[lev - crse_level] = std::make_unique<MultiFab>(LevelTable[lev]->boxArray(), LevelTable[lev]->DistributionMap(), 1, 0);
        MultiFab::Copy(*poisson_rhs[lev - crse_level], *dphi[lev - crse_level], 0, 0, 1, 0);
        poisson_rhs[lev - crse_level]->mult(1.0 / FourPiG);
        MultiFab::Add(*poisson_rhs[lev - crse_level], *drho[lev - crse_level], 0, 0, 1, 0);
    }


    if (crse_level == 0 && !crse_geom.isAllPeriodic()) {

        if (gravity::verbose > 1) {
            amrex::Print() << " ... Making bc's for delta_phi at crse_level 0"  << std::endl;
        }

      fill_coarse_boundary_phi(*this, crse_level, fine_level, amrex::GetVecOfPtrs(poisson_rhs), *delta_phi[crse_level]);

    }

    for (int lev = crse_level; lev <= fine_level; ++lev)
        poisson_rhs[lev - crse_level]->mult(FourPiG);


    if (crse_geom.isAllPeriodic() && (grids[crse_level].numPts() == crse_domain.numPts()))
    {


        Real mean_offset = poisson_rhs[0]->sum() / static_cast<Real>(grids[crse_level].numPts());

        if (gravity::verbose > 1) {
            amrex::Print() << "WARNING: Adjusting RHS in synchronize_gravity_levels solve by " << mean_offset << '\n';
        }

        for (int lev = fine_level; lev >= crse_level; --lev) {
            poisson_rhs[lev-crse_level]->plus(-mean_offset, 0, 1, 0);
        }
    }


    solve_correction_phi(crse_level, fine_level,
                        amrex::GetVecOfPtrs(poisson_rhs),
                        amrex::GetVecOfPtrs(delta_phi),
                        amrex::GetVecOfVecOfPtrs(edge_phi_gradient));


    if (crse_geom.isAllPeriodic() && (grids[crse_level].numPts() == crse_domain.numPts()) ) {

        Real mean_offset = delta_phi[0]->sum() / static_cast<Real>(grids[crse_level].numPts());

        for (int lev = crse_level; lev <= fine_level; ++lev) {
            delta_phi[lev - crse_level]->plus(-mean_offset, 0, 1, 1);
        }

    }


    for (int lev = crse_level; lev <= fine_level; lev++) {

        LevelTable[lev]->get_new_data(Potential_Type).plus(*delta_phi[lev - crse_level], 0, 1, 0);

        for (int n = 0; n < AMREX_SPACEDIM; n++) {
            phi_gradient_new[lev][n]->plus(*edge_phi_gradient[lev - crse_level][n], 0, 1, 0);
        }

        get_new_gravity_vector(lev, LevelTable[lev]->get_new_data(Accel_Type),
                            LevelTable[lev]->get_state_data(State_Type).curTime());

    }

    const PotentialTime epoch = PotentialTime::New;

    for (int lev = fine_level-1; lev >= crse_level; --lev)
    {


        const IntVect& ratio = parent->refRatio(lev);

        amrex::average_down(LevelTable[lev+1]->get_new_data(Potential_Type),
                             LevelTable[lev  ]->get_new_data(Potential_Type),
                             0, 1, ratio);


        average_fine_edges_to_coarse(lev, epoch);


        amrex::average_down(LevelTable[lev+1]->get_new_data(Accel_Type),
                             LevelTable[lev  ]->get_new_data(Accel_Type),
                             0, 1, ratio);

    }

}

void
Gravity::update_max_density()
{
    BL_PROFILE("Gravity::update_max_density()");


    int crse_level = 0;
    int nlevs = parent->finestLevel() + 1;
    const PotentialTime epoch = PotentialTime::New;

    const auto& poisson_rhs = fetch_poisson_rhs(crse_level, nlevs, epoch);

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

    peak_rhs_value = std::accumulate(poisson_rhs.begin(), poisson_rhs.begin() + nlevs, Real(0.0),
                                     [] (Real peak, const auto& rhs) { return std::max(peak, rhs->max(0)); });

}

static void set_uniform_gravity (MultiFab& gravity_view, const Geometry& geom, int ng)
{
    if (AMREX_SPACEDIM == 2 && geom.Coord() == 2) {
        gravity_view.setVal(gravity::uniform_accel, 0, 1, ng);
    } else {
        gravity_view.setVal(gravity::uniform_accel, AMREX_SPACEDIM-1, 1, ng);
    }
}

static void widen_to_three_components (MultiFab& out, const MultiFab& view, int ng)
{
    for (int dir = 0; dir < 3; dir++) {
        if (dir < AMREX_SPACEDIM) {
            MultiFab::Copy(out, view, dir, dir, 1, ng);
        } else {
            out.setVal(0.,dir,1,ng);
        }
    }
}

void
Gravity::read_old_gravity_field(int level, MultiFab& gravity_field_out, Real time)
{
    BL_PROFILE("Gravity::read_old_gravity_field()");

    int ng = gravity_field_out.nGrow();


    if (level > gravity::solve_level_cap) {

        LevelTable[level]->FillCoarsePatch(gravity_field_out,0,time,Accel_Type,0,3,ng);

        return;

    }


    MultiFab gravity_view(grids[level], dmap[level], AMREX_SPACEDIM, ng);
    gravity_view.setVal(0.0,ng);

    const Geometry& geom = parent->Geom(level);

    if (gravity::gravity_model_name == "ConstantGrav") {

        set_uniform_gravity(gravity_view, geom, ng);

    } else if (gravity::gravity_model_name == "MonopoleGrav") {

       const Real prev_time = LevelTable[level]->get_state_data(State_Type).prevTime();
       build_radial_field(level,prev_time,radial_field_old[level]);
       interpolate_monopole_field(level,radial_field_old[level],gravity_view);

    } else if (gravity::gravity_model_name == "PoissonGrav") {

       amrex::average_face_to_cellcenter(gravity_view, amrex::GetVecOfConstPtrs(phi_gradient_old[level]), geom);
       gravity_view.mult(-1.0, ng);

    } else {
       amrex::Abort("Unknown gravity_model_name in read_old_gravity_field");
    }


    widen_to_three_components(gravity_field_out, gravity_view, ng);

#if (AMREX_SPACEDIM > 1)
    if (gravity::gravity_model_name != "ConstantGrav") {
        AmrLevel* amrlev = &parent->getLevel(level) ;
        AmrLevel::FillPatch(*amrlev,gravity_field_out,ng,time,Accel_Type,0,AMREX_SPACEDIM);
    }
#endif

    auto* cs = dynamic_cast<HydroSim*>(&parent->getLevel(level));
    if (cs == nullptr) {
        amrex::Abort("unable to access HydroSim level");
    }
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


    if (level > gravity::solve_level_cap) {

        LevelTable[level]->FillCoarsePatch(gravity_field_out,0,time,Accel_Type,0,3,ng);

        return;

    }


    MultiFab gravity_view(grids[level],dmap[level],AMREX_SPACEDIM,ng);
    gravity_view.setVal(0.0,ng);
    const Geometry& geom = parent->Geom(level);

    if (gravity::gravity_model_name == "ConstantGrav") {

        set_uniform_gravity(gravity_view, geom, ng);

    } else if (gravity::gravity_model_name == "MonopoleGrav") {

        const Real cur_time = LevelTable[level]->get_state_data(State_Type).curTime();
        build_radial_field(level,cur_time,radial_field_new[level]);
        interpolate_monopole_field(level,radial_field_new[level],gravity_view);

    } else if (gravity::gravity_model_name == "PoissonGrav") {

        amrex::average_face_to_cellcenter(gravity_view, amrex::GetVecOfConstPtrs(phi_gradient_new[level]), geom);
        gravity_view.mult(-1.0, ng);

    } else {
       amrex::Abort("Unknown gravity_model_name in get_new_gravity_vector");
    }


    widen_to_three_components(gravity_field_out, gravity_view, ng);

#if (AMREX_SPACEDIM > 1)
    if (gravity::gravity_model_name != "ConstantGrav" && ng>0) {
        AmrLevel* amrlev = &parent->getLevel(level) ;
        AmrLevel::FillPatch(*amrlev,gravity_field_out,ng,time,Accel_Type,0,AMREX_SPACEDIM);
    }
#endif

    auto* cs = dynamic_cast<HydroSim*>(&parent->getLevel(level));
    if (cs == nullptr) {
        amrex::Abort("unable to access HydroSim level");
    }
    if (cs->using_central_mass()) {
        MultiFab& phi = LevelTable[level]->get_new_data(Potential_Type);
        include_point_source_field(level,phi,gravity_field_out);
    }
}

void
Gravity::check_residual (const Box& bx,
                        Array4<Real> const& poisson_rhs,
                        const GpuArray<Array4<Real const>, AMREX_SPACEDIM>& edge_grad,
                        const GpuArray<Real, AMREX_SPACEDIM>& dx,
                        const GpuArray<Real, AMREX_SPACEDIM>& problo,
                        int coord_type)
{

    AMREX_ALWAYS_ASSERT(coord_type >= 0 && coord_type <= 2);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (coord_type == 0) {

            Real laplacian_val = (edge_grad[0](i+1,j,k) - edge_grad[0](i,j,k)) / dx[0];
#if AMREX_SPACEDIM >= 2
            laplacian_val += (edge_grad[1](i,j+1,k) - edge_grad[1](i,j,k)) / dx[1];
#endif
#if AMREX_SPACEDIM == 3
            laplacian_val += (edge_grad[2](i,j,k+1) - edge_grad[2](i,j,k)) / dx[2];
#endif

            poisson_rhs(i,j,k) -= laplacian_val;

        } else if (coord_type == 1) {

            Real rlo  = problo[0] + static_cast<Real>(i) * dx[0];
            Real rhi  = rlo + dx[0];
            Real rcen = 0.5_rt * (rlo + rhi);

            Real laplacian_val = (rhi * edge_grad[0](i+1,j,k) - rlo * edge_grad[0](i,j,k)) / (rcen * dx[0]);
#if AMREX_SPACEDIM >= 2
            laplacian_val += (edge_grad[1](i,j+1,k) - edge_grad[1](i,j,k)) / dx[1];
#endif

            poisson_rhs(i,j,k) -= laplacian_val;

        } else if (coord_type == 2) {

            Real rlo  = problo[0] + static_cast<Real>(i) * dx[0];
            Real rhi  = rlo + dx[0];
            Real rcen = 0.5_rt * (rlo + rhi);

            Real laplacian_val = (rhi * rhi * edge_grad[0](i+1,j,k) - rlo * rlo * edge_grad[0](i,j,k)) / (rcen * rcen * dx[0]);
            poisson_rhs(i,j,k) -= laplacian_val;

        }
    });
}

void
Gravity::check_composite_potential (int crse_level)
{
    BL_PROFILE("Gravity::check_composite_potential()");

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
Gravity::average_fine_edges_to_coarse(int level, PotentialTime epoch)
{
    BL_PROFILE("Gravity::average_fine_edges_to_coarse()");

    if (level == parent->finestLevel()) {
        return;
    }

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

    auto& phi_gradient = (epoch == PotentialTime::New) ? phi_gradient_new : phi_gradient_old;

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
Gravity::verify_level_grad_phi_now(int level)
{
    BL_PROFILE("Gravity::verify_level_grad_phi_now()");

    const MultiFab& S_new = LevelTable[level]->get_new_data(State_Type);
    MultiFab Rhs(grids[level],dmap[level],1,0);
    MultiFab::Copy(Rhs,S_new, URHO, 0,1,0);

    const Geometry& geom = parent->Geom(level);

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
                      {AMREX_D_DECL((*phi_gradient_new[level][0]).const_array(mfi),
                                   (*phi_gradient_new[level][1]).const_array(mfi),
                                   (*phi_gradient_new[level][2]).const_array(mfi))},
                      dx, problo, coord_type);
    }

    if (gravity::verbose > 1) {
       Real resnorm = Rhs.norm0();
       amrex::Print() << "       norm of residual        " << resnorm << std::endl;
    }
}

void
Gravity::verify_level_grad_phi_before(int level)
{
    BL_PROFILE("Gravity::verify_level_grad_phi_before()");

    const MultiFab& S_old = LevelTable[level]->get_old_data(State_Type);
    MultiFab Rhs(grids[level],dmap[level],1,0);
    MultiFab::Copy(Rhs,S_old, URHO,0,1,0);

    const Geometry& geom = parent->Geom(level);

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
                      {AMREX_D_DECL((*phi_gradient_old[level][0]).const_array(mfi),
                                   (*phi_gradient_old[level][1]).const_array(mfi),
                                   (*phi_gradient_old[level][2]).const_array(mfi))},
                      dx, problo, coord_type);
    }

    if (gravity::verbose > 1) {
       Real resnorm = Rhs.norm0();
       amrex::Print() << "       norm of residual        " << resnorm << std::endl;
    }
}
