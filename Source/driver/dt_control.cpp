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
#include <runtime_parameters.H>
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
#include <Diffusion.H>
#endif

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <extern_parameters.H>
#include <prob_parameters.H>

#include <ambient.H>
#include <hydrosim_limits.H>

#include <solver_constants.H>
#include <timestep.H>

#ifdef DIFFUSION
#include <conductivity.H>
#endif

#ifdef MHD
#include <mhd_helpers.H>
#endif

#ifdef ROTATION
#include <Rotation.H>
#endif

#ifdef REACTIONS
#include <actual_network.H>

#ifdef NEW_NETWORK_IMPLEMENTATION
#include <rhs.H>
#endif

#ifndef NEW_NETWORK_IMPLEMENTATION
#include <actual_rhs.H>
#endif
#endif

using namespace amrex;

Real
HydroSim::estTimeStep (int use_new_state)
{
    BL_PROFILE("HydroSim::estTimeStep()");

    if (constant_dt > 0.0) {
      return constant_dt;
    }

    Real estdt = max_dt;

    const MultiFab& activeState = use_new_state ? get_new_data(State_Type) : get_old_data(State_Type);
#ifdef MHD
    const MultiFab& Bx = use_new_state ? get_new_data(Bfield_Type_x) : get_old_data(Bfield_Type_x);
    const MultiFab& By = use_new_state ? get_new_data(Bfield_Type_y) : get_old_data(Bfield_Type_y);
    const MultiFab& Bz = use_new_state ? get_new_data(Bfield_Type_z) : get_old_data(Bfield_Type_z);
#endif
#ifdef RADIATION
    const MultiFab& radiationField = use_new_state ? get_new_data(Radiation_Type) : get_old_data(Radiation_Type);
#endif
    const auto geomdata = geom.data();

    bool exclude_covered_zones = false;

    if (level < parent->finestLevel() && parent->subcyclingMode() == "None") {
        exclude_covered_zones = true;
    }

    MultiFab tmp_coverMask;
    const MultiFab& coverMask = exclude_covered_zones ? getLevel(level+1).make_fine_mask() : tmp_coverMask;

    std::string limiter = "hydrosim.max_dt";

    Real dtlimit_hydro = max_dt / cfl;

    if (do_hydro)
    {

#ifdef RADIATION
        if (Radiation::rad_hydro_coupled) {

            Real ldtlimit_hydro = timestep::dtlimit_rad(activeState, radiationField, geomdata);
            ParallelDescriptor::ReduceRealMin(ldtlimit_hydro);
            dtlimit_hydro = amrex::min(dtlimit_hydro, ldtlimit_hydro) * cfl;
            if (verbose) {
                amrex::Print() << "...estimated hydro-limited timestep at level " << level << ": " << dtlimit_hydro << std::endl;
            }
        }
        else
        {
#endif

#ifdef MHD
          auto cfl_timestep = timestep::estdt<timestep::mhd>(geomdata, coverMask, activeState, Bx, By, Bz);
#else
          auto cfl_timestep = timestep::estdt<timestep::hydro>(geomdata, coverMask, activeState);
#endif

          amrex::ParallelAllReduce::Min(cfl_timestep, MPI_COMM_WORLD);
          dtlimit_hydro = amrex::min(dtlimit_hydro, cfl_timestep.value) * cfl;
          if (verbose) {
              amrex::Print() << "...estimated hydro-limited timestep at level " << level << ": " << dtlimit_hydro << std::endl;
              std::string index_label = "(i";
#if AMREX_SPACEDIM >= 2
              index_label += ",j";
#endif
#if AMREX_SPACEDIM == 3
              index_label += ",k";
#endif
              index_label += ")";

              amrex::Print() << "...hydro-limited CFL timestep constrained at " << index_label << " = " << cfl_timestep.index << std::endl;
          }

#ifdef RADIATION
        }
#endif

        if (dtlimit_hydro < estdt) {
            limiter = "hydro";
            estdt = dtlimit_hydro;
        }

    }

#ifdef DIFFUSION

    Real dtlimit_diffusion = max_dt / cfl;

    if (conduction_enabled)
    {
        auto diffusion_dt = timestep::estdt<timestep::diffusion>(geomdata, coverMask, activeState);
        ParallelAllReduce::Min(diffusion_dt, MPI_COMM_WORLD);
        dtlimit_diffusion = amrex::min(dtlimit_diffusion, diffusion_dt.value) * cfl;

        if (verbose) {
            amrex::Print() << "...estimated diffusion-limited timestep at level " << level << ": " << dtlimit_diffusion << std::endl;
            std::string index_label = "(i";
#if AMREX_SPACEDIM >= 2
            index_label += ",j";
#endif
#if AMREX_SPACEDIM == 3
            index_label += ",k";
#endif
            index_label += ")";

            amrex::Print() << "...diffusion-limited timestep constrained at " << index_label << " = " << diffusion_dt.index << std::endl;

        }
    }

    if (dtlimit_diffusion < estdt) {
        limiter = "diffusion";
        estdt = dtlimit_diffusion;
    }
#endif

#ifdef REACTIONS
    Real dtlimit_burn = max_dt;

    if (do_react && (hydrosim::burn_dt_energy < 1.e199_rt || hydrosim::burn_dt_massfrac < 1.e199_rt)) {

        auto burn_timestep = timestep::estdt<timestep::burning>(geomdata, coverMask, activeState);

        ParallelAllReduce::Min(burn_timestep, MPI_COMM_WORLD);
        dtlimit_burn = amrex::min(dtlimit_burn, burn_timestep.value);

        if (verbose) {
            amrex::Print() << "...estimated burning-limited timestep at level " << level << ": " << dtlimit_burn << std::endl;
            std::string index_label = "(i";
#if AMREX_SPACEDIM >= 2
            index_label += ",j";
#endif
#if AMREX_SPACEDIM == 3
            index_label += ",k";
#endif
            index_label += ")";

            amrex::Print() << "...burning-limited timestep constrained at " << index_label << " = " << burn_timestep.index << std::endl;
        }

        if (dtlimit_burn < estdt) {
            limiter = "burning";
            estdt = dtlimit_burn;
        }
    }
#endif

    if (verbose) {
        amrex::Print() << "HydroSim::estTimeStep (" << limiter << "-limited) at level " << level << ":  estdt = " << estdt << '\n' << std::endl;
    }

    return estdt;
}

void
HydroSim::computeInitialDt (int                   finest_level,
                          int                    ,
                          Vector<int>&           n_cycle,
                          const Vector<IntVect>&  ,
                          Vector<Real>&          dt_level,
                          Real                  stop_time)
{
    BL_PROFILE("HydroSim::computeInitialDt()");

    if (level > 0) {
      return;
    }

    int i;

    Real dt_0 = 1.0e+100;
    int n_factor = 1;
    for (i = 0; i <= finest_level; i++)
    {
        dt_level[i] = getLevel(i).initialTimeStep();
        n_factor   *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_level[i]);
    }

    const Real eps = 0.001*dt_0;
    Real cur_time  = state[State_Type].curTime();
    if (stop_time >= 0.0) {
        if ((cur_time + dt_0) > (stop_time - eps)) {
          dt_0 = stop_time - cur_time;
        }
    }

    n_factor = 1;
    for (i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void
HydroSim::computeNewDt (int                    finest_level,
                      int                     ,
                      Vector<int>&           n_cycle,
                      const Vector<IntVect>&  ,
                      Vector<Real>&          dt_min,
                      Vector<Real>&          dt_level,
                      Real                   stop_time,
                      int                    post_regrid_flag)
{
    BL_PROFILE("HydroSim::computeNewDt()");

    if (level > 0) {
        return;
    }

    Real dt_0 = 1.0e+100;
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        HydroSim& adv_level = getLevel(i);
        dt_min[i] = adv_level.estTimeStep();
    }

    if (constant_dt <= 0.0)
    {
       if (post_regrid_flag == 1)
       {
          for (int i = 0; i <= finest_level; i++)
          {
              dt_min[i] = std::min(dt_min[i],dt_level[i]);
          }
       }
       else
       {
          if (lastDtWriteLimited) {

              dt_min[0] = std::min(dt_min[0], lastDtBeforeWriteLimit);

              lastDtWriteLimited = 0;
              lastDtBeforeWriteLimit = 0.0;

          }
          else {

              for (int i = 0; i <= finest_level; i++)
              {
                  if (verbose && ParallelDescriptor::IOProcessor()) {
                    if (dt_min[i] > change_max*dt_level[i])
                      {
                          std::cout << "HydroSim::compute_new_dt : limiting dt at level "
                                    << i << '\n';
                          std::cout << " ... new dt computed: " << dt_min[i]
                                    << '\n';
                          std::cout << " ... but limiting to: "
                                    << change_max * dt_level[i] << " = " << change_max
                                    << " * " << dt_level[i] << '\n';
                      }
                  }
                  dt_min[i] = std::min(dt_min[i],change_max*dt_level[i]);
              }

          }
       }
    }

    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_min[i]);
    }

    if (plot_interval_exact) {

        const Real plot_per = parent->plotPer();

        if (plot_per > 0.0) {

            const Real cur_time = state[State_Type].curTime();

            const Real phaseOffset = std::fmod(cur_time, plot_per);

            Real nextPlotDt;

            if (std::abs(phaseOffset - plot_per) <= std::numeric_limits<Real>::epsilon() * cur_time) {
                nextPlotDt = plot_per + (plot_per - phaseOffset);
            }
            else {
                nextPlotDt = plot_per - phaseOffset;
            }

            if (nextPlotDt < dt_0) {
                lastDtWriteLimited = 1;
                lastDtBeforeWriteLimit = dt_0;
                dt_0 = nextPlotDt;

                const Real dtFloor = 1.e-4 * lastDtBeforeWriteLimit;
                dt_0 = std::max(dt_0, dtFloor);

                if (verbose) {
                  amrex::Print() << " ... limiting dt to " << dt_0 << " to hit the next plot interval.\n";
                }
            }

        }

    }

    if (small_interval_exact) {

        const Real small_plot_per = parent->smallplotPer();

        if (small_plot_per > 0.0) {

            const Real cur_time = state[State_Type].curTime();

            const Real phaseOffset = std::fmod(cur_time, small_plot_per);

            Real nextSmallPlotDt;

            if (std::abs(phaseOffset - small_plot_per) <= std::numeric_limits<Real>::epsilon() * cur_time) {
                nextSmallPlotDt = small_plot_per + (small_plot_per - phaseOffset);
            }
            else {
                nextSmallPlotDt = small_plot_per - phaseOffset;
            }

            if (nextSmallPlotDt < dt_0) {
                lastDtWriteLimited = 1;
                lastDtBeforeWriteLimit = dt_0;
                dt_0 = nextSmallPlotDt;

                const Real dtFloor = 1.e-4 * lastDtBeforeWriteLimit;
                dt_0 = std::max(dt_0, dtFloor);

                if (verbose) {
                    amrex::Print() << " ... limiting dt to " << dt_0 << " to hit the next smallplot interval.\n";
                }
            }

        }

    }

    const Real eps = std::numeric_limits<Real>::epsilon();
    Real cur_time = state[State_Type].curTime();
    if (stop_time >= 0.0) {
        if ((cur_time + dt_0) >= (stop_time - eps)) {
            dt_0 = stop_time - cur_time;
            if (verbose) {
                amrex::Print() << " ... limiting dt to " << dt_0 << " to hit the stop_time.\n";
            }
        }
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

Real
HydroSim::initialTimeStep ()
{
    Real start_dt_local  = 0.0;

    if (first_step_dt > 0.0)
    {
       start_dt_local = first_step_dt;
    }
    else
    {
       start_dt_local = init_dt_shrink*estTimeStep();
    }

    return start_dt_local;
}

#ifdef RADIATION
Real
timestep::dtlimit_rad (const MultiFab& activeState, const MultiFab& radiationField, const GeometryData& geomdata)
{

    ReduceOps<ReduceOpMin> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(activeState, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& tbox = mfi.tilebox();
        const Box& vbox = mfi.validbox();

        FArrayBox gPr;
        gPr.resize(tbox);
        global::radiation_instance->estimate_rad_gamma_pressure(activeState[mfi], radiationField[mfi], gPr, geomdata.CellSize(), vbox);

        auto u = activeState[mfi].array();
        auto rad_gamma_pressure_arr = gPr.array();


        reduce_op.eval(tbox, reduce_data,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            Real rhoInv = 1.0_rt / u(i,j,k,URHO);

            const auto* dx = geomdata.CellSize();
#if AMREX_SPACEDIM >= 2
            const auto* problo = geomdata.ProbLo();
            const auto coord = geomdata.Coord();
#endif

            eos_t eos_state;
            eos_state.rho = u(i,j,k,URHO);
            eos_state.T   = u(i,j,k,UTEMP);
            eos_state.e   = u(i,j,k,UEINT) * rhoInv;
            for (int n = 0; n < NumSpec; ++n) {
                eos_state.xn[n] = u(i,j,k,UFS+n) * rhoInv;
            }
#if NAUX_NET > 0
            for (int n = 0; n < NumAux; ++n) {
                eos_state.aux[n] = u(i,j,k,UFX+n) * rhoInv;
            }
#endif

            eos(eos_input_re, eos_state);

            Real c = eos_state.cs;
            c = std::sqrt(c * c + rad_gamma_pressure_arr(i,j,k) * rhoInv);

            Real ux = u(i,j,k,UMX) * rhoInv;
            Real uy = u(i,j,k,UMY) * rhoInv;
            Real uz = u(i,j,k,UMZ) * rhoInv;

            Real dt1 = dx[0] / (c + std::abs(ux));
#if AMREX_SPACEDIM >= 2
            Real dt2 = dx[1] / (c + std::abs(uy));
            if (coord == 2) {
                dt2 *= problo[0] + 0.5_rt * dx[0];
            }
#else
            Real dt2 = std::numeric_limits<Real>::max();
#endif
#if AMREX_SPACEDIM == 3
            Real dt3 = dx[2] / (c + std::abs(uz));
#else
            Real dt3 = std::numeric_limits<Real>::max();
#endif

            Real dt_min = amrex::min(dt1, dt2, dt3);

            return {dt_min};
        });

        Gpu::synchronize();
    }

    ReduceTuple hv = reduce_data.value();
    Real estdt = amrex::get<0>(hv);
    estdt = std::min(estdt, max_dt / hydrosim::cfl);

    return estdt;
}
#endif
