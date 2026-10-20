#include <HydroSim.H>

#include <conduction_coeffs.H>

using std::string;

#include <diffusion_operator.H>
#include <AMReX_Geometry.H>

#include <AMReX_Gpu.H>

#include <state_indices.H>

#include <hydrosim_params.H>

#include <eos.H>

#include <conductivity.H>

static AMREX_GPU_HOST_DEVICE AMREX_INLINE
void
evaluate_zone_conductivity(int i, int j, int k,
                           Array4<Real const> const& field_arr,
                           eos_t& eos_state, Real& rhoinv) {

    eos_state.rho  = field_arr(i,j,k,URHO);
    rhoinv = 1.0_rt/eos_state.rho;

    eos_state.T = field_arr(i,j,k,UTEMP);
    eos_state.e = field_arr(i,j,k,UEINT) * rhoinv;
    for (int n = 0; n < NumSpec; n++) {
      eos_state.xn[n] = field_arr(i,j,k,UFS+n) * rhoinv;
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
      eos_state.aux[n] = field_arr(i,j,k,UFX+n) * rhoinv;
    }
#endif

    if (eos_state.e < 0.0_rt) {
      eos_state.T = hydrosim::small_temp;
      eos(eos_input_rt, eos_state);
    } else {
      eos(eos_input_re, eos_state);
    }


    if (eos_state.rho > hydrosim::conduction_cutoff_density) {
      conductivity(eos_state);

      if (eos_state.rho < hydrosim::conduction_cutoff_density_hi) {
        Real multiplier = (eos_state.rho - hydrosim::conduction_cutoff_density) /
          (hydrosim::conduction_cutoff_density_hi - hydrosim::conduction_cutoff_density);
        eos_state.conductivity = eos_state.conductivity * multiplier;
      }
    } else {
      eos_state.conductivity = 0.0_rt;
    }
}

void
fill_cell_conductivity(const Box& bx,
               Array4<Real const> const& field_arr,
               Array4<Real> const& cell_conductivity_arr) {

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    eos_t eos_state;
    Real rhoinv;
    evaluate_zone_conductivity(i, j, k, field_arr, eos_state, rhoinv);
    cell_conductivity_arr(i,j,k) = hydrosim::conduction_scale_factor * eos_state.conductivity;

  });
}

void
fill_conduction_coeff(const Box& bx,
                     Array4<Real const> const& field_arr,
                     Array4<Real> const& cell_conductivity_arr) {

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    eos_t eos_state;
    Real rhoinv;
    evaluate_zone_conductivity(i, j, k, field_arr, eos_state, rhoinv);
    cell_conductivity_arr(i,j,k) = hydrosim::conduction_scale_factor * eos_state.conductivity *
      rhoinv / eos_state.cv;

  });
}

void
HydroSim::computeConductionTerm (Real time, MultiFab& state_in, MultiFab& ConductionTermLocal)
{
    BL_PROFILE("HydroSim::computeConductionTerm()");

   Vector<std::unique_ptr<MultiFab> > coeffs(AMREX_SPACEDIM);
   for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
       coeffs[dir] = std::make_unique<MultiFab>(getEdgeBoxArray(dir), dmap, 1, 0);
   }

   MultiFab Temperature(grids, dmap, 1, 1);

   {
       FillPatchIterator fpi(*this, state_in, 1, time, State_Type, 0, NUM_STATE);
       MultiFab& grown_field = fpi.get_mf();

       MultiFab::Copy(Temperature, grown_field, UTEMP, 0, 1, 1);

#ifdef _OPENMP
#pragma omp parallel
#endif
       {
           FArrayBox cell_coeff(The_Async_Arena());;

           for (MFIter mfi(grown_field, TilingIfNotGPU()); mfi.isValid(); ++mfi)
           {

               const Box& bx = mfi.tilebox();


               const Box& obx = amrex::grow(bx, 1);
               cell_coeff.resize(obx, 1);

               Array4<Real> const cell_conductivity_arr = cell_coeff.array();
               Array4<Real const> const field_arr = grown_field.array(mfi);

               fill_cell_conductivity(obx, field_arr, cell_conductivity_arr);

               for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {

                   const Box& nbx = amrex::surroundingNodes(bx, idir);

                   Array4<Real> const edge_conductivity_arr = (*coeffs[idir]).array(mfi);

                   amrex::ParallelFor(nbx,
                   [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                   {
                       if (idir == 0) {
                           edge_conductivity_arr(i,j,k) = 0.5_rt * (cell_conductivity_arr(i,j,k) + cell_conductivity_arr(i-1,j,k));
                       } else if (idir == 1) {
                           edge_conductivity_arr(i,j,k) = 0.5_rt * (cell_conductivity_arr(i,j,k) + cell_conductivity_arr(i,j-1,k));
                       } else {
                           edge_conductivity_arr(i,j,k) = 0.5_rt * (cell_conductivity_arr(i,j,k) + cell_conductivity_arr(i,j,k-1));
                       }
                   });
               }
           }
       }

   }

   MultiFab CoarseTemperature;

   if (conduction_use_mlmg) {

       if (level > 0) {
           const BoxArray& coarse_box_array = getLevel(level-1).boxArray();
           const DistributionMapping& coarse_distribution_map = getLevel(level-1).DistributionMap();
           CoarseTemperature.define(coarse_box_array,coarse_distribution_map,1,1);
           FillPatch(getLevel(level-1),CoarseTemperature,1,time,State_Type,UTEMP,1);
       }

       diffusion->apply_stencil(level, Temperature, CoarseTemperature, ConductionTermLocal, coeffs);

   } else {
       const auto dx = geom.CellSizeArray();
       const auto dxinv = geom.InvCellSizeArray();
       const auto problo = geom.ProbLoArray();
       const int coord = geom.Coord();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
       for (MFIter mfi(Temperature, TilingIfNotGPU()); mfi.isValid(); ++mfi)
       {
           const Box& bx = mfi.tilebox();
           Array4<Real const> const& Temperature_array = Temperature.array(mfi);
           Array4<Real> const & ConductionTerm_array = ConductionTermLocal.array(mfi);

           Array4<const Real> const edge_conductivity_x = (*coeffs[0]).array(mfi);
#if AMREX_SPACEDIM >= 2
           Array4<const Real> const edge_conductivity_y = (*coeffs[1]).array(mfi);
#endif
#if AMREX_SPACEDIM == 3
           Array4<const Real> const edge_conductivity_z = (*coeffs[2]).array(mfi);
#endif

           ParallelFor(bx,
           [=] AMREX_GPU_DEVICE (int i, int j, int k)
           {
               for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
                   int il = i;
                   int jl = j;
                   int kl = k;

                   int ir = i;
                   int jr = j;
                   int kr = k;

                   Real inv_dx_sq = dxinv[idir]*dxinv[idir];
                   Real conductivity_right;
                   Real conductivity_left;

                   if (idir == 0) {
                       il = i - 1;
                       ir = i + 1;
                       conductivity_right = edge_conductivity_x(ir,jr,kr);
                       conductivity_left = edge_conductivity_x(i ,j ,k);
                   }
#if AMREX_SPACEDIM >= 2
                   else if (idir == 1) {
                       jl = j - 1;
                       jr = j + 1;
                       conductivity_right = edge_conductivity_y(ir,jr,kr);
                       conductivity_left = edge_conductivity_y(i ,j ,k);
                   }
#endif
#if AMREX_SPACEDIM == 3
                   else {
                       kl = k - 1;
                       kr = k + 1;
                       conductivity_right = edge_conductivity_z(ir,jr,kr);
                       conductivity_left = edge_conductivity_z(i ,j ,k);
                   }
#endif

#if AMREX_SPACEDIM < 3

                   if ((coord != 0) && (idir == 0)) {

                       Real rr = problo[idir] + static_cast<Real>(ir) * dx[idir];
                       Real rl = problo[idir] + static_cast<Real>(i) * dx[idir];
                       Real rc = problo[idir] + (static_cast<Real>(i) + 0.5_rt) * dx[idir];

                       if (coord == 1) {

                           conductivity_right *= rr;
                           conductivity_left *= rl;
                           inv_dx_sq *= 1.0_rt / rc;
                       } else {

                           conductivity_right *= rr * rr;
                           conductivity_left *= rl * rl;
                           inv_dx_sq *= 1.0_rt / (rc * rc) ;
                       }
                   } else if ((coord == 2) && (idir == 1)) {

                       Real rc = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
                       Real polar_hi = problo[idir] + static_cast<Real>(jr) * dx[idir];
                       Real polar_lo = problo[idir] + static_cast<Real>(j) * dx[idir];
                       Real polar_center = problo[idir] + (static_cast<Real>(j) + 0.5_rt) * dx[idir];

                       conductivity_right *= std::sin(polar_hi);
                       conductivity_left *= std::sin(polar_lo);
                       inv_dx_sq *= 1.0_rt / (rc * rc * std::sin(polar_center));
                   }
#endif
                   ConductionTerm_array(i,j,k) += inv_dx_sq *
                       (conductivity_right * (Temperature_array(ir,jr,kr) - Temperature_array(i ,j ,k )) -
                        conductivity_left * (Temperature_array(i ,j ,k ) - Temperature_array(il,jl,kl)));
               }
           });
       }
   }
}

void
HydroSim::add_conduction_to_source (MultiFab& external_source, MultiFab& state_in, MultiFab& ConductionTerm, Real t, Real term_weight)
{
    BL_PROFILE("HydroSim::add_temp_diffusion_to_sources()");

    ConductionTerm.setVal(0.);
    if (conduction_enabled == 1) {
        computeConductionTerm(t, state_in, ConductionTerm);
    }

    if (conduction_enabled == 1) {
       MultiFab::Saxpy(external_source,term_weight,ConductionTerm,0,UEDEN,1,0);
       MultiFab::Saxpy(external_source,term_weight,ConductionTerm,0,UEINT,1,0);
    }
}

void
HydroSim::assemble_old_conduction_source(MultiFab& source, MultiFab& state_in, Real time, Real dt)
{
    BL_PROFILE("HydroSim::assemble_old_conduction_source()");

    amrex::ignore_unused(dt);

    const Real strt_time = ParallelDescriptor::second();

    MultiFab ConductionTermLocal(grids, dmap, 1, 0);

    add_conduction_to_source(source, state_in, ConductionTermLocal, time);

    if (verbose > 1)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;
        amrex::Real level_real = level;
#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::assemble_old_conduction_source() time = " << run_time
                       << " on level " << level_real << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }
}

void
HydroSim::assemble_new_conduction_source(MultiFab& source, MultiFab& field_old, MultiFab& field_new, Real time, Real dt)
{
    BL_PROFILE("HydroSim::assemble_new_conduction_source()");

    const Real strt_time = ParallelDescriptor::second();

    MultiFab ConductionTermLocal(grids, dmap, 1, 0);

    Real term_weight = 0.5;

    add_conduction_to_source(source, field_new, ConductionTermLocal, time, term_weight);


    term_weight = -0.5;
    Real old_time = time - dt;

    add_conduction_to_source(source, field_old, ConductionTermLocal, old_time, term_weight);

    if (verbose > 1)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;
        amrex::Real level_real = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::assemble_new_conduction_source() time = " << run_time
                       << " on level " << level_real << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }
}
