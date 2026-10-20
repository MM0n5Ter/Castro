#include <AMReX_LO_BCTYPES.H>
#include <AMReX_ParmParse.H>
#include <Radiation.H>
#include <RadTransportSolver.H>
#include <radiation_helpers.H>

#include <opacity.H>

#include <format>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace amrex;

void Radiation::fillBndryRegister(BndryRegister& bdry, int level, Real time)
{
  BL_PROFILE("Radiation::fillBndryRegister");

  HydroSim      *hydrosim = (HydroSim*)&parent->getLevel(level);
  const BoxArray& grids = hydrosim->boxArray();
  const DistributionMapping& dmap = hydrosim->DistributionMap();
  const Geometry& geom  = parent->Geom(level);

  Real old_time = hydrosim->get_state_data(Radiation_Type).prevTime();
  Real new_time = hydrosim->get_state_data(Radiation_Type).curTime();
  Real eps = (new_time > old_time) ? 0.001*(new_time - old_time) : 1.0;

  BL_ASSERT( (time > old_time-eps) && (time < new_time + eps));

  MultiFab& S_new = hydrosim->get_new_data(Radiation_Type);
  MultiFab& S_old = hydrosim->get_old_data(Radiation_Type);

  if (!(!geom.isAnyPeriodic())) {

    int need_old_state = (time <= new_time - eps);
    int need_new_state = (time >= old_time + eps);

    int n_grow = 1;

    MultiFab old_state_buffer;
    MultiFab new_state_buffer;

    if (need_old_state) {
      old_state_buffer.define(grids, dmap, 1, n_grow);
      old_state_buffer.setVal(0.0);
      MultiFab::Copy(old_state_buffer, S_old, Rad, 0, 1, 0);
      old_state_buffer.FillBoundary(geom.periodicity());
    }

    if (need_new_state) {
      new_state_buffer.define(grids, dmap, 1, n_grow);
      new_state_buffer.setVal(0.0);
      MultiFab::Copy(new_state_buffer, S_new, Rad, 0, 1, 0);
      new_state_buffer.FillBoundary(geom.periodicity());
    }

    int n_ghost = 1;

    if (time > new_time - eps) {
      bdry.copyFrom(new_state_buffer, n_ghost, 0, 0, 1);
    }
    else if (time < old_time + eps) {
      bdry.copyFrom(old_state_buffer, n_ghost, 0, 0, 1);
    }
    else {
      Real a = (new_time - time) / (new_time - old_time);
      Real b = (time - old_time) / (new_time - old_time);
      bdry.linComb(a, old_state_buffer, 0, b, new_state_buffer, 0, 0, 1, n_ghost);
    }
  } else {
    int n_ghost = 0;

    if (time > new_time - eps) {
      bdry.copyFrom(S_new, n_ghost, 0, 0, Radiation::nGroups);
    }
    else if (time < old_time + eps) {
      bdry.copyFrom(S_old, n_ghost, 0, 0, Radiation::nGroups);
    }
    else {
      Real a = (new_time - time) / (new_time - old_time);
      Real b = (time - old_time) / (new_time - old_time);
      bdry.linComb(a, S_old, 0, b, S_new, 0, 0, Radiation::nGroups, n_ghost);
    }
  }
}

void Radiation::fetchBndryData(RadiativeBoundary& bd, MultiFab& Er,
                             Real time, int level)
{
  BL_PROFILE("Radiation::fetchBndryData");
  HydroSim      *hydrosim = (HydroSim*)&parent->getLevel(level);
  const BoxArray& grids = hydrosim->boxArray();
  const DistributionMapping& dmap = hydrosim->DistributionMap();

  if (!(level == 0)) {
    BoxArray coarse_boxes(grids);
    IntVect crse_ratio = parent->refRatio(level-1);
    coarse_boxes.coarsen(crse_ratio);
    BndryRegister crse_br(coarse_boxes, dmap, 0, 1, 1, 1);
    crse_br.setVal(1.0e30);
    fillBndryRegister(crse_br, level-1, time);

    bd.setBndryValues(crse_br, 0, Er, Rad, 0, 1, crse_ratio, rad_boundary);
  } else {
    bd.setBndryValues(Er, Rad, 0, 1, rad_boundary);
  }

  bd.setTime(time);

  bd.setBoundaryFluxConditions(rad_boundary);
}

void Radiation::fetchMultigridBndry(MultiGroupBoundary& multigroup_boundary, MultiFab& Er,
                               Real time, int level)
{
  BL_PROFILE("Radiation::fetchMultigridBndry");
  HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
  const BoxArray& grids = hydrosim->boxArray();
  const DistributionMapping& dmap = hydrosim->DistributionMap();

  if (!(level == 0)) {
    BoxArray coarse_boxes(grids);
    IntVect crse_ratio = parent->refRatio(level-1);
    coarse_boxes.coarsen(crse_ratio);
    BndryRegister crse_br(coarse_boxes, dmap, 0, 1, 1, Radiation::nGroups);
    crse_br.setVal(1.0e30);
    fillBndryRegister(crse_br, level-1, time);

    multigroup_boundary.setBndryValues(crse_br, 0, Er, 0,
                        0, Radiation::nGroups, crse_ratio, rad_boundary);
  } else {
    multigroup_boundary.setBndryValues(Er, 0, 0, Radiation::nGroups, rad_boundary);
  }

  multigroup_boundary.setTime(time);
  multigroup_boundary.setBoundaryFluxConditions(rad_boundary);
}

void Radiation::fetchMultigridBndryZeroRad(MultiGroupBoundary& multigroup_boundary, MultiFab& Er, int level)
{
  BL_PROFILE("Radiation::fetchMultigridBndryZeroRad");
  HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
  const BoxArray& grids = hydrosim->boxArray();
  const DistributionMapping& dmap = hydrosim->DistributionMap();

  if (!(level == 0)) {
    BoxArray coarse_boxes(grids);
    IntVect crse_ratio = parent->refRatio(level-1);
    coarse_boxes.coarsen(crse_ratio);
    BndryRegister crse_br(coarse_boxes, dmap, 0, 1, 1, 1);
    crse_br.setVal(0.0);

    multigroup_boundary.setBndryValues(crse_br, 0, Er, 0,
                        0, 1, crse_ratio, rad_boundary);
  } else {
    multigroup_boundary.setBndryValues(Er, 0, 0, 1, rad_boundary);
  }

  multigroup_boundary.setBoundaryFluxConditions(rad_boundary);
}

void Radiation::extendBorderValues(MultiFab& f, int indx)
{
    BL_PROFILE("Radiation::extendBorderValues");

    BL_ASSERT(f.nGrow() >= 1);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(f); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const Box& grown_valid_box = amrex::grow(bx, 1);

        Array4<Real> const face_flux_arr = f[mfi].array(indx);

        amrex::LoopOnCpu(grown_valid_box, [=] (int i, int j, int k) noexcept
        {

            if (i == bx.smallEnd(0)) {
                face_flux_arr(i-1,j,k) = 2.0_rt * face_flux_arr(i,j,k) - face_flux_arr(i+1,j,k);
            }

            if (i == bx.bigEnd(0)) {
                face_flux_arr(i+1,j,k) = 2.0_rt * face_flux_arr(i,j,k) - face_flux_arr(i-1,j,k);
            }

#if AMREX_SPACEDIM >= 2
            if (j == bx.smallEnd(1)) {
                face_flux_arr(i,j-1,k) = 2.0_rt * face_flux_arr(i,j,k) - face_flux_arr(i,j+1,k);
            }

            if (j == bx.bigEnd(1)) {
                face_flux_arr(i,j+1,k) = 2.0_rt * face_flux_arr(i,j,k) - face_flux_arr(i,j-1,k);
            }
#endif

#if AMREX_SPACEDIM == 3
            if (k == bx.smallEnd(2)) {
                face_flux_arr(i,j,k-1) = 2.0_rt * face_flux_arr(i,j,k) - face_flux_arr(i,j,k+1);
            }

            if (k == bx.bigEnd(2)) {
                face_flux_arr(i,j,k+1) = 2.0_rt * face_flux_arr(i,j,k) - face_flux_arr(i,j,k-1);
            }
#endif
        });
    }
}

void Radiation::reflux(int level)
{
  return;
}

void Radiation::prepare_delayed_reflux(int crse_level)
{
  BL_PROFILE("Radiation::prepare_delayed_reflux");
  if (verbose && ParallelDescriptor::IOProcessor()) {
    std::cout << "Radiation deferred sync setup for coarse level " << crse_level << "..." << std::endl;
  }

  int level = crse_level + 1;

  const Orientation ori(0, Orientation::low);

  int do_swap = (conservative_register_old[level] &&
                 (conservative_register_old[level]->coarsenedBoxes() ==
                  flux_accum_reg[level]->coarsenedBoxes()) &&
                 ((*conservative_register_old[level])[ori].DistributionMap() ==
                  (*flux_accum_reg[level])[ori].DistributionMap()));

  if (do_swap) {
      std::swap(conservative_register_old[level], flux_accum_reg[level]);
  }
  else {
      conservative_register_old[level] = std::move(flux_accum_reg[level]);
  }

  dt_previous[crse_level] = parent->dtLevel(crse_level);

  if (verbose && ParallelDescriptor::IOProcessor()) {
    if (apply_sync) {
      std::cout << "                                                   done"
           << std::endl;
    }
    else {
      std::cout << "                                                   zeroed out"
           << std::endl;
    }
  }
}

void Radiation::apply_delayed_reflux(int level, MultiFab& rhs, int indx)
{
  int fine_level = parent->finestLevel();
  HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
  const BoxArray& grids = hydrosim->boxArray();
  const DistributionMapping& dmap = hydrosim->DistributionMap();
  Real delta_t = parent->dtLevel(level);

  if (level < parent->maxLevel() &&
      conservative_register_old[level+1]) {

    FluxRegister& fine_correction_register = *conservative_register_old[level+1];

    if (indx == 0) {
        fine_correction_register.ClearInternalBorders(hydrosim->Geom());
    }

    Real scale = dt_previous[level] / delta_t;
    if (!apply_sync) scale = 0.0;

    fine_correction_register.Reflux(rhs, scale, indx, 0, 1, hydrosim->Geom());

    if (level+1 < parent->maxLevel() &&
        conservative_register_old[level+2]) {

      BoxArray fgrids;

      if (level+1 <= fine_level) {

        fgrids = parent->getLevel(level+1).boxArray();
      }

      const IntVect& rat = fine_correction_register.refRatio();
      BoxArray prior_cover_boxes = fine_correction_register.coarsenedBoxes();
      prior_cover_boxes.refine(rat);

      if (fgrids != prior_cover_boxes) {

        IntVect level_refine_ratio = IntVect::TheUnitVector();

        for (int flev = level+1; flev < parent->maxLevel(); flev++) {

          level_refine_ratio *= parent->refRatio(flev-1);

          if (conservative_register_old[flev+1]) {

            if (flev > level+1) {
              fgrids.refine(parent->refRatio(flev-1));
            }

            FluxRegister& fine_fine_register = *conservative_register_old[flev+1];
            BoxArray fine_fine_boxes = fine_fine_register.coarsenedBoxes();
            fine_fine_boxes.grow(1);

            const Box& domain = parent->Geom(flev).Domain();

            for (int i = 0; i < fine_fine_boxes.size(); i++) {
              fine_fine_boxes.set(i, (fine_fine_boxes[i] & domain));
            }

            if (fgrids.contains(fine_fine_boxes)) {

              break;
            }
{

              if (indx == 0) {
                  fine_fine_register.ClearInternalBorders(parent->Geom(flev));
              }

              BoxArray coarse_cover_boxes(fine_fine_boxes);

              coarse_cover_boxes.coarsen(level_refine_ratio);

              BoxArray restored_fine_boxes(coarse_cover_boxes);
              restored_fine_boxes.refine(level_refine_ratio);

              const DistributionMapping& fine_fine_dmap = fine_fine_register.DistributionMap();

              MultiFab fine_level_data(restored_fine_boxes, fine_fine_dmap, 1, 0);
              fine_level_data.setVal(0.0);

              Real scale = dt_previous[flev] / delta_t;
              if (!apply_sync) scale = 0.0;

              fine_fine_register.Reflux(fine_level_data, scale, indx, 0, 1,
                             parent->Geom(flev));

              MultiFab rhs_tmp(grids, dmap, 1, 0);
              rhs_tmp.setVal(0.0);

              amrex::average_down(fine_level_data, rhs_tmp, 0, 1, level_refine_ratio);

              rhs.plus(rhs_tmp, 0, 1, 0);
            }
          }
        }
      }

    }

  }

  IntVect level_refine_ratio = IntVect::TheUnitVector();

  for (int flev = level; flev > 0; flev--) {

    if (conservative_register_old[flev] &&
        dt_previous[flev-1] > 0.0) {

      FluxRegister& coarse_correction_register = *conservative_register_old[flev];

      const IntVect& rat = coarse_correction_register.refRatio();
      BoxArray prior_cover_boxes = coarse_correction_register.coarsenedBoxes();
      prior_cover_boxes.refine(rat);

      const BoxArray& fgrids = parent->getLevel(flev).boxArray();
      if (fgrids == prior_cover_boxes) {

        break;
      }
      else {

        prior_cover_boxes.refine(level_refine_ratio);

        if (prior_cover_boxes.contains(grids)) {
          break;
        }

        const DistributionMapping& coarse_correction_dmap = coarse_correction_register.DistributionMap();

        FluxRegister refined_correction_register(prior_cover_boxes, coarse_correction_dmap,
                                   IntVect::TheUnitVector(),
                                   level+1,
                                   1);
        level_refine_ratio *= rat;

        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
          const Orientation lo_face(dir,Orientation::low);
          const Orientation hi_face(dir,Orientation::high);

          for (FabSetIter fsi(refined_correction_register[lo_face]);
               fsi.isValid(); ++fsi) {

            coarse_to_fine_face(refined_correction_register[lo_face][fsi].array(),
                   coarse_correction_register[lo_face][fsi].array(indx),
                   dir, level_refine_ratio);
          }

          for (FabSetIter fsi(refined_correction_register[hi_face]);
               fsi.isValid(); ++fsi) {

            coarse_to_fine_face(refined_correction_register[hi_face][fsi].array(),
                   coarse_correction_register[hi_face][fsi].array(indx),
                   dir, level_refine_ratio);
          }
        }

        Real scale = dt_previous[flev-1] / delta_t;
        if (!apply_sync) scale = 0.0;

        refined_correction_register.Reflux(rhs, scale, 0, 0, 1, hydrosim->Geom());

      }

    }

  }

}
