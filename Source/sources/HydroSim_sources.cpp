#include <algorithm>
#include <ranges>

#include <HydroSim.H>
#include <advection_util.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

using namespace amrex;

void
HydroSim::apply_source_to_state(MultiFab& target_state, const MultiFab& source, Real dt, int ng)
{
    BL_PROFILE("HydroSim::apply_source_to_state()");

    AMREX_ASSERT(source.nGrow() >= ng);
    AMREX_ASSERT(target_state.nGrow() >= ng);

    MultiFab::Saxpy(target_state, dt, source, 0, 0, source.nComp(), ng);
}

void
HydroSim::apply_source_and_clean (
#ifdef MHD
                                MultiFab& Bx, MultiFab& By, MultiFab& Bz,
#endif
                                MultiFab& state, const MultiFab& source, Real time, Real dt)
{
    apply_source_to_state(state, source, dt, 0);
    clean_state(
#ifdef MHD
                Bx, By, Bz,
#endif
                state, time, 0);
}

bool
HydroSim::source_flag(int src)
{
    switch(src) {

#ifdef SPONGE
    case sponge_src:
        if (do_sponge) {
            return true;
        } else {
            return false;
        }
#endif

    case ext_src:
        if (!(add_ext_src)) {
            return false;
        } else {
            return true;
        }
#ifndef MHD
    case thermo_src:
        if (time_integration_method == SpectralDeferredCorrections) {
            return true;
        } else {
          return false;
        }
#else
    case thermo_src:
        return true;
#endif

    case geom_src:
        if (!(geom.Coord() == 0 || use_geom_source == 0)) {
          return true;
        } else {
          return false;
        }

#ifdef DIFFUSION
    case diff_src:
        if (diffuse_temp &&
            !(time_integration_method == SpectralDeferredCorrections)) {
          return true;
        }
        else {
          return false;
        }
#endif

#ifdef HYBRID_MOMENTUM
    case hybrid_src:
        return true;
#endif

#ifdef GRAVITY
    case grav_src:
        if (do_grav) {
            return true;
        } else {
            return false;
        }
#endif

#ifdef ROTATION
    case rot_src:
        if (do_rotation) {
            return true;
        } else {
            return false;
        }
#endif

    default:
        return false;

    } // end switch
}

void
HydroSim::handle_old_sources(
#ifdef MHD
                MultiFab& Bx, MultiFab& By, MultiFab& Bz,
#endif
                MultiFab& source, MultiFab& state_old, MultiFab& state_new, Real time, Real dt, SourceUpdate source_update)
{

    BL_PROFILE("HydroSim::handle_old_sources()");

    const Real strt_time = ParallelDescriptor::second();

    // Construct the old-time sources.

    source.setVal(0.0, source.nGrow());

    if (!apply_sources()) {
        return;
    }

    for (int n = 0; n < num_src; ++n) {
        assemble_old_source(n, state_old, time, dt, source);
    }

    if (source_update == SourceUpdate::ApplyToState) {
        apply_source_and_clean(
#ifdef MHD
                               Bx, By, Bz,
#endif
                               state_new, source, time, dt);
    }

    // Fill the ghost cells (but only for CTU; the time would not be correct for true SDC, so
    // the FillPatch is handled separately in the SDC advance).

    if (hydrosim::time_integration_method == CornerTransportUpwind ||
        hydrosim::time_integration_method == SimplifiedSpectralDeferredCorrections) {
        AmrLevel::FillPatch(*this, source, source.nGrow(), time, Source_Type, 0, NSRC);
    }

    // Optionally print out diagnostic information about how much
    // these source terms changed the state.

    if (source_update == SourceUpdate::ApplyToState && print_update_diagnostics) {
      evaluate_and_print_source_change(get_old_data(Source_Type), dt, "old-time sources");
    }

    if (verbose > 0)
    {
        const int IOProc   = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;
        amrex::Real llevel = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::handle_old_sources() time = " << run_time
                       << " on level " << llevel << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }
}

advance_status
HydroSim::handle_old_sources (Real time, Real dt, SourceUpdate source_update)
{
    advance_status status {};

    MultiFab& S_new = get_new_data(State_Type);

    MultiFab& old_source = get_old_data(Source_Type);

#ifdef MHD
    MultiFab& Bx_old = get_old_data(Mag_Type_x);
    MultiFab& By_old = get_old_data(Mag_Type_y);
    MultiFab& Bz_old = get_old_data(Mag_Type_z);
#endif

    handle_old_sources(
#ifdef MHD
                   Bx_old, By_old, Bz_old,
#endif
                   old_source, Sborder, S_new, time, dt,
                   source_update);

    return status;
}

void
HydroSim::handle_new_sources(
#ifdef MHD
                MultiFab& Bx, MultiFab& By, MultiFab& Bz,
#endif
                MultiFab& source, MultiFab& state_old, MultiFab& state_new, Real time, Real dt, SourceUpdate source_update)
{

    BL_PROFILE("HydroSim::handle_new_sources()");

    const Real strt_time = ParallelDescriptor::second();

    source.setVal(0.0, NUM_GROW_SRC);

    if (!apply_sources()) {
        return;
    }

    // Construct the new-time source terms.

    for (int n = 0; n < num_src; ++n) {
        assemble_new_source(n, state_old, state_new, time, dt, source);
    }

    if (source_update == SourceUpdate::ApplyToState) {
        apply_source_and_clean(
#ifdef MHD
                               Bx, By, Bz,
#endif
                               state_new, source, time, dt);
    }


    // Optionally print out diagnostic information about how much
    // these source terms changed the state.

    if (print_update_diagnostics) {
      evaluate_and_print_source_change(get_new_data(Source_Type), dt, "new-time sources");
    }

    if (verbose > 0)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;
        amrex::Real llevel = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::handle_new_sources() time = " << run_time << " on level " << llevel << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }

}

advance_status
HydroSim::handle_new_sources (Real time, Real dt)
{
    advance_status status {};

    MultiFab& S_new = get_new_data(State_Type);

    MultiFab& new_source = get_new_data(Source_Type);

#ifdef MHD
    MultiFab& Bx_new = get_new_data(Mag_Type_x);
    MultiFab& By_new = get_new_data(Mag_Type_y);
    MultiFab& Bz_new = get_new_data(Mag_Type_z);
#endif

    handle_new_sources(
#ifdef MHD
                   Bx_new, By_new, Bz_new,
#endif
                   new_source, Sborder, S_new, time, dt);

    return status;
}

void
HydroSim::assemble_old_source(int src, MultiFab& state_in, Real time, Real dt, MultiFab& source)
{
    BL_PROFILE("HydroSim::assemble_old_source()");

    BL_ASSERT(src >= 0 && src < num_src);

    switch(src) {

#ifdef SPONGE
    case sponge_src:
        assemble_old_sponge_source(state_in, time, dt, source);
        break;
#endif

    case ext_src:
        assemble_old_ext_source(state_in, time, dt, source);
        break;

    case thermo_src:
        assemble_old_thermo_source(state_in, time, dt, source);
        break;

    case geom_src:
        assemble_old_geom_source(state_in, time, dt, source);
        break;

#ifdef DIFFUSION
    case diff_src:
        if (!(time_integration_method == SpectralDeferredCorrections)) {
          // for MOL or SDC, we'll compute a diffusive flux in the MOL routine
          assemble_old_diff_source(state_in, time, dt, source);
        }
        break;
#endif

#ifdef HYBRID_MOMENTUM
    case hybrid_src:
        assemble_old_hybrid_source(state_in, time, dt, source);
        break;
#endif

#ifdef GRAVITY
    case grav_src:
        assemble_old_gravity_source(state_in, time, dt, source);
        break;
#endif

#ifdef ROTATION
    case rot_src:
        assemble_old_rotation_source(state_in, time, dt, source);
        break;
#endif

    default:
        break;

    } // end switch
}

void
HydroSim::assemble_new_source(int src, MultiFab& state_old, MultiFab& state_new, Real time, Real dt, MultiFab& source)
{
    BL_PROFILE("HydroSim::assemble_new_source()");

    BL_ASSERT(src >= 0 && src < num_src);

    switch(src) {

#ifdef SPONGE
    case sponge_src:
        assemble_new_sponge_source(state_old, state_new, time, dt, source);
        break;
#endif

    case ext_src:
        assemble_new_ext_source(state_old, state_new, time, dt, source);
        break;

#ifdef MHD
    case thermo_src:
        assemble_new_thermo_source(state_old, state_new, time, dt, source);
        break;
#endif

    case geom_src:
        assemble_new_geom_source(state_old, state_new, time, dt, source);
        break;

#ifdef DIFFUSION
    case diff_src:
        assemble_new_diff_source(state_old, state_new, time, dt, source);
        break;
#endif

#ifdef HYBRID_MOMENTUM
    case hybrid_src:
        assemble_new_hybrid_source(state_old, state_new, time, dt, source);
        break;
#endif

#ifdef GRAVITY
    case grav_src:
        assemble_new_gravity_source(state_old, state_new, time, dt, source);
        break;
#endif

#ifdef ROTATION
    case rot_src:
        assemble_new_rotation_source(state_old, state_new, time, dt, source);
        break;
#endif

    default:
        break;

    } // end switch
}

// Returns whether any sources are actually applied.

bool
HydroSim::apply_sources()
{
    return std::ranges::any_of(std::views::iota(0, static_cast<int>(num_src)),
                               [this] (int src) { return source_flag(src); });
}

// Evaluate diagnostics quantities describing the effect of an
// update on the state. The optional parameter scope determines
// whether we want to do this calculation globally over all processes
// or locally just on this processor. The latter is useful if you
// are evaluating the contribution from multiple source changes at once
// and want to amortize the cost of the parallel reductions.
// Note that the resultant output is volume-weighted.

Vector<Real>
HydroSim::evaluate_source_change(const MultiFab& source, Real dt, SumScope scope)
{

  BL_PROFILE("HydroSim::evaluate_source_change()");

  Vector<Real> update(source.nComp(), 0.0);

  // Create a temporary array which will hold a single component
  // at a time of the volume-weighted source.

  MultiFab weighted_source(source.boxArray(), source.DistributionMap(), 1, 0);

  for (int n = 0; n < source.nComp(); ++n) {

    weighted_source.setVal(0.0);

    // Fill weighted_source with source x volume.

    MultiFab::AddProduct(weighted_source, source, n, volume, 0, 0, 1, 0);

    update[n] = weighted_source.sum(0, scope == SumScope::ThisRank) * dt;

  }

  return update;

}

// Calculate the changes to the state due to a source term,
// and also print the results.  We assume here that the source
// is lined up with the NUM_STATE components of State_Type
// because we are interested in printing changes to energy, mass, etc.

void
HydroSim::evaluate_and_print_source_change (const MultiFab& source, Real dt, const std::string& source_name)
{
    Vector<Real> update = evaluate_source_change(source, dt, SumScope::ThisRank);

#ifdef BL_LAZY
    Lazy::QueueReduction( [=] () mutable {
#endif
        if (!update.empty()) {
            ParallelDescriptor::ReduceRealSum(update.dataPtr(), static_cast<int>(update.size()), ParallelDescriptor::IOProcessorNumber());
        }

        if (ParallelDescriptor::IOProcessor()) {
            if (std::abs(update[URHO]) != 0.0 || std::abs(update[UEDEN]) != 0.0) {
                std::cout << std::endl << "  Contributions to the state from " << source_name << ":" << std::endl;

                std::cout << "       mass added: " << update[URHO] << std::endl;
                std::cout << "       xmom added: " << update[UMX] << std::endl;
                std::cout << "       ymom added: " << update[UMY] << std::endl;
                std::cout << "       zmom added: " << update[UMZ] << std::endl;
                std::cout << "       eint added: " << update[UEINT] << std::endl;
                std::cout << "       ener added: " << update[UEDEN] << std::endl;

                std::cout << std::endl;
            }
        }

#ifdef BL_LAZY
    });
#endif
}

#ifdef SHOCK_VAR
// Compute the shock flag from the time-level n data and store it in
// the USHOCK component of the old state and of Sborder.

void
HydroSim::store_old_shock_flag (Real time, Real dt)
{
    // We need to compute the old sources -- note that we will
    // recompute the old sources after the burn, so this is done here
    // only for evaluating the shock flag.

    handle_old_sources(time, dt, SourceUpdate::FillOnly);

    MultiFab& S_old = get_old_data(State_Type);
    MultiFab& old_source = get_old_data(Source_Type);

    FArrayBox shk(The_Async_Arena());
    FArrayBox q_tmp(The_Async_Arena()), qaux_tmp(The_Async_Arena());

    for (MFIter mfi(Sborder, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const Box& obx = amrex::grow(bx, 1);

        shk.resize(bx, 1);
#ifdef RADIATION
        q_tmp.resize(obx, NQ);
#else
        q_tmp.resize(obx, NQTHERM);
#endif
        qaux_tmp.resize(obx, NQAUX);

        Array4<Real> const shk_arr = shk.array();
        Array4<Real> const q_arr = q_tmp.array();
        Array4<Real> const qaux_arr = qaux_tmp.array();

        Array4<Real const> const Sborder_old_arr = Sborder.array(mfi);
        Array4<Real> const S_old_arr = S_old.array(mfi);
        Array4<Real> const old_src_arr = old_source.array(mfi);

        hydro::ctoprim(obx, time, Sborder_old_arr, {}, q_arr, qaux_arr);

        shock(bx, q_arr, old_src_arr, shk_arr);

        // now store it in S_old -- we'll fillpatch into Sborder in a bit

        // Note: we still compute the shock flag in the hydro for the
        // hybrid-Riemann (for now) since that version will have seen the
        // effect of the burning in the first dt), but that version is
        // never stored in State_Type

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            S_old_arr(i,j,k,USHOCK) = shk_arr(i,j,k);
        });

    }

    // we only computed the shock flag on the interior, but the first
    // burn needs ghost cells, so FillPatch just the shock flag

    if (Sborder.nGrow() > 0) {
      AmrLevel::FillPatch(*this, Sborder, Sborder.nGrow(), time, State_Type, USHOCK, 1, USHOCK);
    }
}
#endif

// Perform all operations that occur prior to computing the predictor sources
// and the hydro advance.

advance_status
HydroSim::pre_advance_operators (Real time, Real dt)  // NOLINT(readability-convert-member-functions-to-static)
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    advance_status status {};

    // If we are using gravity, solve for the potential and
    // gravitational field.  note: since reactions don't change
    // density, we can do this before or after the burn.

#ifdef GRAVITY
    assemble_old_gravity(time);
#endif

#ifdef SHOCK_VAR
    // we want to compute the shock flag that will be used
    // (optionally) in disabling reactions in shocks.  We compute this
    // only once per timestep using the time-level n data.

    store_old_shock_flag(time, dt);

#endif

    // If we are Strang splitting the reactions, do the old-time contribution now.

#ifndef TRUE_SDC
#ifdef REACTIONS
    status = handle_old_reactions(time, dt);

    if (status.success == false) {
        return status;
    }
#endif
#endif


    // Initialize the new-time data. This copy needs to come after all Strang-split operators.

    MultiFab& S_new = get_new_data(State_Type);

    MultiFab::Copy(S_new, Sborder, 0, 0, NUM_STATE, S_new.nGrow());

    return status;
}

// Perform all operations that occur after computing the predictor sources
// but before the hydro advance.

advance_status
HydroSim::pre_hydro_operators (Real time, Real dt)  // NOLINT(readability-convert-member-functions-to-static)
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    advance_status status {};

#ifdef REACTIONS
    // The SDC reactive source ghost cells on coarse levels might not
    // be in sync due to any average down done, so fill them here.

    if (hydrosim::time_integration_method == SimplifiedSpectralDeferredCorrections) {
        MultiFab& react_src = get_new_data(Simplified_SDC_React_Type);

        AmrLevel::FillPatch(*this, react_src, react_src.nGrow(), time + dt, Simplified_SDC_React_Type, 0, react_src.nComp());
    }
#endif

    return status;
}

// Perform all operations that occur after the hydro source
// but before the corrector sources.

advance_status
HydroSim::post_hydro_operators (Real time, Real dt)  // NOLINT(readability-convert-member-functions-to-static)
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    advance_status status {};

#ifdef GRAVITY
    assemble_new_gravity(time);
#endif

    return status;
}

// Perform all operations that occur after the corrector sources.

advance_status
HydroSim::post_advance_operators (Real time, Real dt)  // NOLINT(readability-convert-member-functions-to-static)
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    advance_status status {};

#ifndef TRUE_SDC
#ifdef REACTIONS
    status = handle_new_reactions(time, dt);
#endif
#endif

    return status;
}
