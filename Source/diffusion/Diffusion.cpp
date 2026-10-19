#include <AMReX_ParmParse.H>
#include <Diffusion.H>
#include <HydroSim.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <SolverUtils.H>

#include <hydrosim_limits.H>

using namespace amrex;

Diffusion::Diffusion(Amr* Parent, BCRec* _boundary_rec)
  :
    parent(Parent),
    LevelTable(LEVEL_LIMIT),
    grids(LEVEL_LIMIT),
    volume(LEVEL_LIMIT),
    area(LEVEL_LIMIT),
    phys_bc(_boundary_rec)
{
    AMREX_ALWAYS_ASSERT(parent->maxLevel() < LEVEL_LIMIT);

    build_solver_bc();
}


void
Diffusion::report_run_settings(std::ostream& jobInfoFile)
{
#include <diffusion_job_info_tests.H>
}


void
Diffusion::attach_level (int                   level,
                          AmrLevel*             amr_level_ptr,
                          MultiFab&             _cell_volume,
                          MultiFab*             _face_area)
{
    if (verbose) {
        amrex::Print() << "Installing Diffusion level " << level << '\n';
    }

    LevelTable[level] = amr_level_ptr;

    volume[level] = &_cell_volume;

    area[level] = _face_area;

    BoxArray ba(LevelTable[level]->boxArray());
    grids[level] = ba;
}

void
Diffusion::apply_stencil (int level, MultiFab& Temperature,
                    MultiFab& CoarseTemperature, MultiFab& ConductionTerm,
                    Vector<std::unique_ptr<MultiFab> >& conductivity_fields)
{
    apply_stencil_multigrid(level, Temperature, CoarseTemperature, ConductionTerm, conductivity_fields);
}


void
Diffusion::build_solver_bc ()
{
    const Geometry& geom = parent->Geom(0);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (geom.isPeriodic(idim)) {
            mlmg_lobc[idim] = MLLinOp::BCType::Periodic;
            mlmg_hibc[idim] = MLLinOp::BCType::Periodic;
        } else {
            if (phys_bc->lo(idim) == amrex::PhysBCType::inflow) {
                mlmg_lobc[idim] = MLLinOp::BCType::Dirichlet;
            } else {
                mlmg_lobc[idim] = MLLinOp::BCType::Neumann;
            }
            if (phys_bc->hi(idim) == amrex::PhysBCType::inflow) {
                mlmg_hibc[idim] = MLLinOp::BCType::Dirichlet;
            } else {
                mlmg_hibc[idim] = MLLinOp::BCType::Neumann;
            }
        }
    }

    const auto* problo = geom.ProbLo();
    if ((geom.IsSPHERICAL() || geom.IsRZ()) && problo[0] == 0.0_rt) {
        mlmg_lobc[0] = MLLinOp::BCType::Neumann;
    }

}

void
Diffusion::apply_stencil_multigrid (int level, MultiFab& Temperature,
                         MultiFab& CoarseTemperature, MultiFab& ConductionTerm,
                         Vector<std::unique_ptr<MultiFab> >& conductivity_fields)
{
    BL_PROFILE("Diffusion::apply_stencil_multigrid()");

    if (verbose && ParallelDescriptor::IOProcessor()) {
        std::cout << "   " << '\n';
        std::cout << "... compute diffusive term at level " << level << '\n';
    }

    const Geometry& geom = parent->Geom(level);
    const BoxArray& ba = Temperature.boxArray();
    const DistributionMapping& dm = Temperature.DistributionMap();

    LPInfo info;
    info.setMetricTerm(true);
    info.setMaxCoarseningLevel(0);
    info.setAgglomeration(false);
    info.setConsolidation(false);

    MLABecLaplacian mlabec({geom}, {ba}, {dm}, info);
    mlabec.setMaxOrder(diffusion::solver_max_order);

    mlabec.setDomainBC(mlmg_lobc, mlmg_hibc);

    if (level > 0) {
        const auto& rr = parent->refRatio(level-1);
        mlabec.setCoarseFineBC(&CoarseTemperature, rr[0]);
    }
    mlabec.setLevelBC(0, &Temperature);

    mlabec.setScalars(0.0, -1.0);
    mlabec.setBCoeffs(0, Array<MultiFab const*, AMREX_SPACEDIM>{AMREX_D_DECL(conductivity_fields[0].get(),
                                                                             conductivity_fields[1].get(),
                                                                             conductivity_fields[2].get())});

    MLMG mlmg(mlabec);
    mlmg.setVerbose(verbose);
    mlmg.apply({&ConductionTerm}, {&Temperature});
}
