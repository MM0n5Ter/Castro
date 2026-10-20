#ifndef WIN32
#include <unistd.h>
#endif

#include <iomanip>

#include <algorithm>
#include <cstdio>
#include <vector>
#include <iostream>
#include <string>
#include <ctime>
#include <memory>
#include <numbers>
#include <format>

#include <AMReX_Utility.H>
#include <AMReX_CONSTANTS.H>
#include <HydroSim.H>
#include <global.H>
#include <runtime_param_init.H>
#include <AMReX_VisMF.H>
#include <AMReX_TagBox.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_ParmParse.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef AMREX_PARTICLES
#include <AMReX_Particles.H>
#endif

#ifdef GRAVITY
#include <Gravity.H>
#endif

#ifdef DIFFUSION
#include <diffusion_operator.H>
#endif

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <extern_parameters.H>
#include <prob_parameters.H>

#include <ambient.H>
#include <hydrosim_limits.H>

#include <solver_constants.H>

using namespace amrex;

static void gather_register_faces (const FluxRegister& reg, int dir, MultiFab& dest)
{
    for (OrientationIter fi; fi.isValid(); ++fi) {
        const FabSet& fs = reg[fi()];
        if (fi().coordDir() == dir) {
            fs.copyTo(dest, 0, 0, 0, dest.nComp());
        }
    }
}

void
HydroSim::reflux (int crse_level, int fine_level, bool from_post_step)
{
    BL_PROFILE("HydroSim::reflux()");

    BL_ASSERT(fine_level > crse_level);

    const Real strt = ParallelDescriptor::second();

    const bool update_coarse_fluxes = refresh_sources_after_reflux || !from_post_step;
    const Real crse_dt = parent->dtLevel(crse_level);

#ifdef GRAVITY
    const bool sync_potential = do_gravity_flag && gravity->get_solver_kind() == "PoissonGrav" && gravity->NoSync() == 0 && from_post_step;

    int nlevs = fine_level - crse_level + 1;

    Vector<std::unique_ptr<MultiFab> > drho(nlevs);
    Vector<std::unique_ptr<MultiFab> > dphi(nlevs);

    if (sync_potential)  {

        for (int lev = crse_level; lev <= fine_level; ++lev) {

            const auto& amrlevel = getLevel(lev);
            const auto& ba = amrlevel.boxArray();
            const auto& dm = amrlevel.DistributionMap();

            drho[lev - crse_level] = std::make_unique<MultiFab>(ba, dm, 1, 0);
            dphi[lev - crse_level] = std::make_unique<MultiFab>(ba, dm, 1, 0);

            drho[lev - crse_level]->setVal(0.0);
            dphi[lev - crse_level]->setVal(0.0);

        }

    }
#endif

    FluxRegister* reg = nullptr;

    for (int lev = fine_level; lev > crse_level; --lev) {

        reg = &getLevel(lev).flux_reg;

        HydroSim& crse_lev = getLevel(lev-1);
#ifdef GRAVITY
        HydroSim& fine_lev = getLevel(lev);
#endif

        MultiFab& crse_state = crse_lev.get_new_data(State_Type);

        MultiFab ghosted_coarse_state(crse_state.boxArray(), crse_state.DistributionMap(), crse_state.nComp(), 1);

        crse_lev.fill_ghost_field(ghosted_coarse_state, crse_lev.state[State_Type].curTime(), 1);

        reg->ClearInternalBorders(crse_lev.geom);

        MultiFab scratch_fluxes[AMREX_SPACEDIM];

        for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

            scratch_fluxes[idir].define(crse_lev.fluxes[idir]->boxArray(),
                                     crse_lev.fluxes[idir]->DistributionMap(),
                                     crse_lev.fluxes[idir]->nComp(), crse_lev.fluxes[idir]->nGrow());

            scratch_fluxes[idir].setVal(0.0);

            gather_register_faces(*reg, idir, scratch_fluxes[idir]);

        }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        for (MFIter mfi(crse_state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();

            auto U = ghosted_coarse_state[mfi].array();
            auto V = crse_lev.volume[mfi].array();

            for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
                const Box& nbx = amrex::surroundingNodes(bx, idir);
                auto F = scratch_fluxes[idir][mfi].array();
#ifndef MHD
                auto A = crse_lev.area[idir][mfi].array();
                Real dt = crse_dt;

                bool scale_by_area_dt = false;
                crse_lev.clip_fluxes_at_density_floor(nbx, idir, U, V, F, A, dt, scale_by_area_dt);
#endif
                amrex::ParallelFor(nbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    bool clear_fluxes = false;

                    Real rho = U(i,j,k,URHO);
                    Real densityDelta = F(i,j,k,URHO) / V(i,j,k);
                    Real invDensityNew = 1.0_rt / (rho + densityDelta);

                    for (int n = 0; n < NumSpec; ++n) {
                        Real rhoX = U(i,j,k,UFS+n);
                        Real speciesDensityDelta = F(i,j,k,UFS+n) / V(i,j,k);
                        Real MassfracNew = (rhoX + AMREX_SPACEDIM * speciesDensityDelta) * invDensityNew;

                        if (MassfracNew < -hydrosim::massfrac_check_tolerance ||
                            MassfracNew > 1.0_rt + hydrosim::massfrac_check_tolerance) {
                            clear_fluxes = true;
                            break;
                        }
                    }

                    if (clear_fluxes) {
                        for (int n = 0; n < NUM_STATE; ++n) {
                            F(i,j,k,n) = 0.0;
                        }
                    }
                });
            }
        }

        for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

            for (OrientationIter fi; fi.isValid(); ++fi) {
                if (fi().coordDir() == idir) {
                    FabSet& fs = (*reg)[fi()];
                    fs.copyFrom(scratch_fluxes[idir], 0, 0, 0, scratch_fluxes[idir].nComp());
                }
            }

            if (update_coarse_fluxes) {

                MultiFab::Add(*crse_lev.fluxes[idir], scratch_fluxes[idir], 0, 0, crse_lev.fluxes[idir]->nComp(), 0);

                MultiFab::Add(*crse_lev.mass_face_flux[idir], scratch_fluxes[idir], URHO, 0, 1, 0);
            }

        }

        reg->Reflux(crse_state, crse_lev.volume, 1.0, 0, 0, NUM_STATE, crse_lev.geom);

#ifdef GRAVITY
        int ilev = lev - crse_level - 1;

        if (sync_potential) {
            reg->Reflux(*drho[ilev], crse_lev.volume, 1.0, 0, URHO, 1, crse_lev.geom);
            amrex::average_down(*drho[ilev + 1], *drho[ilev], 0, 1, getLevel(lev).crse_ratio);
        }
#endif

        reg->setVal(0.0);

#if (AMREX_SPACEDIM <= 2)
        if (!Geom().IsCartesian()) {

            reg = &getLevel(lev).pressure_flux_reg;

            MultiFab dr(crse_lev.grids, crse_lev.dmap, 1, 0);
            dr.setVal(crse_lev.geom.CellSize(0));

            reg->ClearInternalBorders(crse_lev.geom);

            reg->Reflux(crse_state, dr, 1.0, 0, UMX, 1, crse_lev.geom);

            if (update_coarse_fluxes) {

                MultiFab scratch_radial_fluxes(crse_lev.Pressure_radial.boxArray(),
                                    crse_lev.Pressure_radial.DistributionMap(),
                                    crse_lev.Pressure_radial.nComp(), crse_lev.Pressure_radial.nGrow());

                scratch_radial_fluxes.setVal(0.0);

                gather_register_faces(*reg, 0, scratch_radial_fluxes);

                MultiFab::Add(crse_lev.Pressure_radial, scratch_radial_fluxes, 0, 0, crse_lev.Pressure_radial.nComp(), 0);

            }

            reg->setVal(0.0);

        }
#endif

#ifdef RADIATION

        if (Radiation::rad_hydro_coupled) {

            reg = &getLevel(lev).rad_edge_reg;

            reg->ClearInternalBorders(crse_lev.geom);

            reg->Reflux(crse_lev.get_new_data(Radiation_Type), crse_lev.volume, 1.0, 0, 0, Radiation::nGroups, crse_lev.geom);

            if (update_coarse_fluxes) {

                for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

                    MultiFab scratch_fluxes(crse_lev.rad_face_flux[idir]->boxArray(),
                                         crse_lev.rad_face_flux[idir]->DistributionMap(),
                                         crse_lev.rad_face_flux[idir]->nComp(), crse_lev.rad_face_flux[idir]->nGrow());

                    scratch_fluxes.setVal(0.0);

                    gather_register_faces(*reg, idir, scratch_fluxes);

                    MultiFab::Add(*crse_lev.rad_face_flux[idir], scratch_fluxes, 0, 0, crse_lev.rad_face_flux[idir]->nComp(), 0);

                }

            }

            reg->setVal(0.0);

        }

#endif

#ifdef GRAVITY
        if (sync_potential)  {

            reg = &getLevel(lev).pot_flux_reg;

            for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                reg->CrseInit(*(gravity->get_potential_gradient_curr(lev-1)[i]), crse_lev.area[i], i, 0, 0, 1, -1.0);
                reg->FineAdd(*(gravity->get_potential_gradient_curr(lev)[i]), fine_lev.area[i], i, 0, 0, 1, 1.0);
            }

            reg->Reflux(*dphi[ilev], crse_lev.volume, 1.0, 0, 0, 1, crse_lev.geom);

            amrex::average_down(*dphi[ilev + 1], *dphi[ilev], 0, 1, getLevel(lev).crse_ratio);

            reg->setVal(0.0);

        }
#endif

    }

#ifdef GRAVITY
    if (sync_potential) {
      gravity->synchronize_gravity_levels(crse_level, fine_level, amrex::GetVecOfPtrs(drho), amrex::GetVecOfPtrs(dphi));
    }
#endif

    if (refresh_sources_after_reflux && from_post_step &&
        (time_scheme == CornerTrace ||
         time_scheme == SimplifiedIterCorrection)) {

        for (int lev = fine_level; lev >= crse_level; --lev) {

            MultiFab& S_old = getLevel(lev).get_old_data(State_Type);

            MultiFab& S_new = getLevel(lev).get_new_data(State_Type);
#ifdef MHD
            MultiFab& magfield_x_new = getLevel(lev).get_new_data(Bfield_Type_x);
            MultiFab& magfield_y_new = getLevel(lev).get_new_data(Bfield_Type_y);
            MultiFab& magfield_z_new = getLevel(lev).get_new_data(Bfield_Type_z);
#endif
            MultiFab& source = getLevel(lev).get_new_data(Forcing_Type);
            Real time = getLevel(lev).state[State_Type].curTime();
            Real dt_level_advance = getLevel(lev).dt_level_step;
            Real dt_expected = parent->dtLevel(lev);

            if (getLevel(lev).apply_source_terms()) {

                getLevel(lev).apply_source_term_to_state(S_new, source, -dt_level_advance, 0);
                getLevel(lev).clean_state(
#ifdef MHD
                                          magfield_x_new, magfield_y_new, magfield_z_new,
#endif
                                          S_new, time, 0);

            }

            if (allow_step_retry && dt_level_advance < dt_expected && getLevel(lev).retain_prev_field) {

                for (int k = 0; k < num_field_types; k++) {

                    if (getLevel(lev).prev_state[k]->hasOldData()) {

                        MultiFab& old = getLevel(lev).get_old_data(k);
                        MultiFab::Copy(getLevel(lev).prev_state[k]->newData(), old, 0, 0, old.nComp(), old.nGrow());
                        MultiFab::Copy(old, getLevel(lev).prev_state[k]->oldData(), 0, 0, old.nComp(), old.nGrow());

                        getLevel(lev).state[k].setTimeLevel(time, dt_level_advance, 0.0);
                        getLevel(lev).prev_state[k]->setTimeLevel(time, dt_expected, 0.0);

                    }

                }

            }

            if (getLevel(lev).apply_source_terms()) {
                bool apply_source_flag = true;
                getLevel(lev).run_new_sources(
#ifdef MHD
                                magfield_x_new, magfield_y_new, magfield_z_new,
#endif
                                source, S_old, S_new, time, dt_level_advance, apply_source_flag);
            }

            if (allow_step_retry && dt_level_advance < dt_expected && getLevel(lev).retain_prev_field) {

                for (int k = 0; k < num_field_types; k++) {

                    if (getLevel(lev).prev_state[k]->hasOldData()) {

                        MultiFab& old = getLevel(lev).get_old_data(k);
                        MultiFab::Copy(old, getLevel(lev).prev_state[k]->newData(), 0, 0, old.nComp(), old.nGrow());

                        getLevel(lev).state[k].setTimeLevel(time, dt_expected, 0.0);
                        getLevel(lev).prev_state[k]->setTimeLevel(time, dt_level_advance, 0.0);

                    }

                }

                if (lev == 0 || lev > level) {
                    amrex::FillNull(getLevel(lev).prev_state);
                    getLevel(lev).retain_prev_field = false;
                }

            }

        }

    }

    if (verbose)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real end = ParallelDescriptor::second() - strt;
        amrex::Real level_real = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(end,IOProc);
        if (ParallelDescriptor::IOProcessor()) {
            std::cout << "HydroSim::reflux() at level " << level_real
                      << " : time = " << end << std::endl;
        }
#ifdef BL_LAZY
        });
#endif
    }
}

void
HydroSim::FluxRegCoarseInit() {
    BL_PROFILE("HydroSim::FluxRegCoarseInit");

    if (level == parent->finestLevel()) {
      return;
    }

    HydroSim& fine_level = getLevel(level+1);

    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      fine_level.flux_reg.CrseInit(*fluxes[i], i, 0, 0, NUM_STATE, flux_coarse_scale);
    }

#if (AMREX_SPACEDIM <= 2)
    if (!Geom().IsCartesian()) {
      fine_level.pressure_flux_reg.CrseInit(Pressure_radial, 0, 0, 0, 1, pressure_coarse_scale);
    }
#endif

#ifdef RADIATION
    if (Radiation::rad_hydro_coupled) {
      for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        fine_level.rad_edge_reg.CrseInit(*rad_face_flux[i], i, 0, 0, Radiation::nGroups, flux_coarse_scale);
      }
    }
#endif

}

void
HydroSim::FluxRegFineAccum() {

    BL_PROFILE("HydroSim::FluxRegFineAccum()");

    if (level == 0) {
      return;
    }

    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      flux_reg.FineAdd(*fluxes[i], i, 0, 0, NUM_STATE, flux_refined_scale);
    }

#if (AMREX_SPACEDIM <= 2)
    if (!Geom().IsCartesian()) {
      getLevel(level).pressure_flux_reg.FineAdd(Pressure_radial, 0, 0, 0, 1, pressure_refined_scale);
    }
#endif

#ifdef RADIATION
    if (Radiation::rad_hydro_coupled) {
      for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        getLevel(level).rad_edge_reg.FineAdd(*rad_face_flux[i], i, 0, 0, Radiation::nGroups, flux_refined_scale);
      }
    }
#endif

}

void
HydroSim::avgDown ()
{
  BL_PROFILE("HydroSim::avgDown()");

  if (level == parent->finestLevel()) {
      return;
  }

  for (int k = 0; k < num_field_types; k++) {
      avgDown(k);
  }

}

void
HydroSim::avgDown (int state_indx)
{
    BL_PROFILE("HydroSim::avgDown(state_indx)");

    if (level == parent->finestLevel()) {
        return;
    }

    HydroSim& fine_lev = getLevel(level+1);

    const Geometry& fgeom = fine_lev.geom;
    const Geometry& cgeom =          geom;

    MultiFab&  S_crse   = get_new_data(state_indx);
    MultiFab&  S_fine   = fine_lev.get_new_data(state_indx);

    amrex::average_down(S_fine, S_crse,
                         fgeom, cgeom,
                         0, S_fine.nComp(), fine_ratio);
}

MultiFab&
HydroSim::make_fine_mask()
{
    BL_PROFILE("HydroSim::make_fine_mask()");

    BL_ASSERT(level > 0);

    if (fine_mask.empty()) {
        fine_mask = makeFineMask(parent->boxArray(level-1),
                                 parent->DistributionMap(level-1),
                                 parent->boxArray(level), crse_ratio,
                                 1.0,
                                 0.0);
    }

    return fine_mask;
}

iMultiFab&
HydroSim::make_boundary_mask (int ng)
{
    BL_PROFILE("HydroSim::make_boundary_mask()");

    const auto cached = std::find_if(interior_masks.begin(), interior_masks.end(),
                                     [ng] (const auto& ibm) { return ibm->nGrow() == ng; });
    if (cached != interior_masks.end()) {
        return **cached;
    }

    interior_masks.push_back(std::make_unique<iMultiFab>(grids, dmap, 1, ng));

    iMultiFab& imf = *interior_masks.back();

    int ghost_cover_flag = 0;
    int uncovered_flag = 1;

    imf.BuildMask(geom.Domain(), geom.periodicity(),
                  ghost_cover_flag, uncovered_flag, uncovered_flag, uncovered_flag);

    return imf;
}

void
HydroSim::fill_ghost_field(MultiFab& S, Real time, int ng)
{
  BL_PROFILE("HydroSim::fill_ghost_field()");

  BL_ASSERT(S.nGrow() >= ng);

  AmrLevel::FillPatch(*this, S, ng, time, State_Type, 0, NUM_STATE);
}
