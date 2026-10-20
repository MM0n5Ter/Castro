#include <HydroSim.H>

#include <utility>

#ifdef RADIATION
#include <Radiation.H>
#endif

using namespace amrex;

void
HydroSim::assemble_old_source(int src, MultiFab& source, MultiFab& state_in, Real time, Real dt)
{
    BL_PROFILE("HydroSim::assemble_old_source()");

    BL_ASSERT(src >= 0 && src < num_forcing);

    switch(src) {

#ifdef SPONGE
    case damping_src:
        assemble_old_damping_source(source, state_in, time, dt);
        break;
#endif

    case external_source:
        assemble_old_external_source(source, state_in, time, dt);
        break;

    case thermodynamic_src:
        assemble_old_thermodynamic_source(source, state_in, time, dt);
        break;

    case geometry_source:
        assemble_old_geometry_source(source, state_in, time, dt);
        break;

#ifdef DIFFUSION
    case conduction_src:
        if (!(time_scheme == IterativeCorrection)) {
          assemble_old_conduction_source(source, state_in, time, dt);
        }
        break;
#endif

#ifdef HYBRID_MOMENTUM
    case hybrid_mom_source:
        assemble_old_hybrid_source(source, state_in, time, dt);
        break;
#endif

#ifdef GRAVITY
    case gravity_src:
        assemble_old_gravity_source(source, state_in, time, dt);
        break;
#endif

#ifdef ROTATION
    case rotation_src:
        assemble_old_rotation_source(source, state_in, time, dt);
        break;
#endif

    default:
        break;

    }
}

void
HydroSim::assemble_new_source(int src, MultiFab& source, MultiFab& field_old, MultiFab& field_new, Real time, Real dt)
{
    BL_PROFILE("HydroSim::assemble_new_source()");

    BL_ASSERT(src >= 0 && src < num_forcing);

    switch(src) {

#ifdef SPONGE
    case damping_src:
        assemble_new_damping_source(source, field_old, field_new, time, dt);
        break;
#endif

    case external_source:
        assemble_new_external_source(source, field_old, field_new, time, dt);
        break;

#ifdef MHD
    case thermodynamic_src:
        assemble_new_thermodynamic_source(source, field_old, field_new, time, dt);
        break;
#endif

    case geometry_source:
        assemble_new_geometry_source(source, field_old, field_new, time, dt);
        break;

#ifdef DIFFUSION
    case conduction_src:
        assemble_new_conduction_source(source, field_old, field_new, time, dt);
        break;
#endif

#ifdef HYBRID_MOMENTUM
    case hybrid_mom_source:
        assemble_new_hybrid_source(source, field_old, field_new, time, dt);
        break;
#endif

#ifdef GRAVITY
    case gravity_src:
        assemble_new_gravity_source(source, field_old, field_new, time, dt);
        break;
#endif

#ifdef ROTATION
    case rotation_src:
        assemble_new_rotation_source(source, field_old, field_new, time, dt);
        break;
#endif

    default:
        break;

    }
}

static void report_source_timing (const char* routine, Real start, int lev)
{
    const int IOProc   = ParallelDescriptor::IOProcessorNumber();
    amrex::Real run_time = ParallelDescriptor::second() - start;
    amrex::Real level_real = lev;

#ifdef BL_LAZY
    Lazy::QueueReduction( [=] () mutable {
#endif
    ParallelDescriptor::ReduceRealMax(run_time,IOProc);

    amrex::Print() << "HydroSim::" << routine << "() time = " << run_time
                   << " on level " << level_real << "\n" << "\n";
#ifdef BL_LAZY
    });
#endif
}

void
HydroSim::run_old_sources(
#ifdef MHD
                MultiFab& Bx, MultiFab& By, MultiFab& Bz,
#endif
                MultiFab& source, MultiFab& field_old, MultiFab& field_new, Real time, Real dt, bool modify_state_flag)
{

    BL_PROFILE("HydroSim::run_old_sources()");

    const Real strt_time = ParallelDescriptor::second();

    source.setVal(0.0, source.nGrow());

    if (!apply_source_terms()) {
        return;
    }

    for (int n = 0; n < num_forcing; ++n) {
        assemble_old_source(n, source, field_old, time, dt);
    }

    if (modify_state_flag) {
        apply_source_term_to_state(field_new, source, dt, 0);
        clean_state(
#ifdef MHD
                     Bx, By, Bz,
#endif
                     field_new, time, 0);
    }

    if (hydrosim::time_scheme == CornerTrace ||
        hydrosim::time_scheme == SimplifiedIterCorrection) {
        AmrLevel::FillPatch(*this, source, source.nGrow(), time, Forcing_Type, 0, NUM_SOURCE);
    }

    if (modify_state_flag && log_source_changes) {
      log_old_source_changes(dt);
    }

    if (verbose > 0)
    {
        report_source_timing("run_old_sources", strt_time, level);
    }
}

step_result
HydroSim::run_old_sources (Real time, Real dt, bool modify_state_flag)
{
    step_result status {};

    MultiFab& S_new = get_new_data(State_Type);

    MultiFab& old_forcing = get_old_data(Forcing_Type);

#ifdef MHD
    MultiFab& bfield_x_old = get_old_data(Bfield_Type_x);
    MultiFab& bfield_y_old = get_old_data(Bfield_Type_y);
    MultiFab& bfield_z_old = get_old_data(Bfield_Type_z);
#endif

    run_old_sources(
#ifdef MHD
                   bfield_x_old, bfield_y_old, bfield_z_old,
#endif
                   old_forcing, Sborder, S_new, time, dt,
                   modify_state_flag);

    return status;
}

void
HydroSim::run_new_sources(
#ifdef MHD
                MultiFab& Bx, MultiFab& By, MultiFab& Bz,
#endif
                MultiFab& source, MultiFab& field_old, MultiFab& field_new, Real time, Real dt, bool modify_state_flag)
{

    BL_PROFILE("HydroSim::run_new_sources()");

    const Real strt_time = ParallelDescriptor::second();

    source.setVal(0.0, SRC_GHOST_WIDTH);

    if (!apply_source_terms()) {
        return;
    }

    for (int n = 0; n < num_forcing; ++n) {
        assemble_new_source(n, source, field_old, field_new, time, dt);
    }

    if (modify_state_flag) {
        apply_source_term_to_state(field_new, source, dt, 0);
        clean_state(
#ifdef MHD
                     Bx, By, Bz,
#endif
                     field_new, time, 0);
    }

    if (log_source_changes) {
      log_new_source_changes(dt);
    }

    if (verbose > 0)
    {
        report_source_timing("run_new_sources", strt_time, level);
    }

}

step_result
HydroSim::run_new_sources (Real time, Real dt)
{
    step_result status {};

    MultiFab& S_new = get_new_data(State_Type);

    MultiFab& new_source = get_new_data(Forcing_Type);

#ifdef MHD
    MultiFab& magfield_x_new = get_new_data(Bfield_Type_x);
    MultiFab& magfield_y_new = get_new_data(Bfield_Type_y);
    MultiFab& magfield_z_new = get_new_data(Bfield_Type_z);
#endif

    run_new_sources(
#ifdef MHD
                   magfield_x_new, magfield_y_new, magfield_z_new,
#endif
                   new_source, Sborder, S_new, time, dt);

    return status;
}

bool
HydroSim::apply_source_terms()
{

    for (int n = 0; n < num_forcing; ++n) {
        if (source_enabled(n)) {
            return true;
        }
    }

    return false;

}

void
HydroSim::apply_source_term_to_state(MultiFab& state_to_update, const MultiFab& source, Real dt, int ng)
{
    BL_PROFILE("HydroSim::apply_source_term_to_state()");

    AMREX_ASSERT(source.nGrow() >= ng);
    AMREX_ASSERT(state_to_update.nGrow() >= ng);

    MultiFab::Saxpy(state_to_update, dt, source, 0, 0, source.nComp(), ng);
}

bool
HydroSim::source_enabled(int src)
{
    switch(src) {

#ifdef SPONGE
    case damping_src:
        return do_damping_zone != 0;
#endif

    case external_source:
        return enable_extra_source != 0;
#ifndef MHD
    case thermodynamic_src:
        return time_scheme == IterativeCorrection;
#else
    case thermodynamic_src:
        return true;
#endif

    case geometry_source:
        return !(geom.Coord() == 0 || use_curvilinear_src == 0);

#ifdef DIFFUSION
    case conduction_src:
        return conduction_enabled &&
            !(time_scheme == IterativeCorrection);
#endif

#ifdef HYBRID_MOMENTUM
    case hybrid_mom_source:
        return true;
#endif

#ifdef GRAVITY
    case gravity_src:
        return do_gravity_flag != 0;
#endif

#ifdef ROTATION
    case rotation_src:
        return do_frame_spin != 0;
#endif

    default:
        return false;

    }
}

void
HydroSim::log_old_source_changes(Real dt)
{
    const MultiFab& source = get_old_data(Forcing_Type);

    std::string source_name = "old-time sources";

    measure_and_log_source_change(source, dt, source_name);
}

void
HydroSim::log_new_source_changes(Real dt)
{
    const MultiFab& source = get_new_data(Forcing_Type);

    std::string source_name = "new-time sources";

    measure_and_log_source_change(source, dt, source_name);
}

static void
log_source_change(const Vector<Real>& update)
{

  if (ParallelDescriptor::IOProcessor()) {

    static const std::pair<const char*, int> source_labels[] = {
      {"       mass added: ", URHO},
      {"       xmom added: ", UMX},
      {"       ymom added: ", UMY},
      {"       zmom added: ", UMZ},
      {"       eint added: ", UEINT},
      {"       ener added: ", UEDEN}
    };

    for (const auto& [label, idx] : source_labels) {
      std::cout << label << update[idx] << std::endl;
    }

    std::cout << std::endl;

  }

}

void
HydroSim::measure_and_log_source_change (const MultiFab& source, Real dt, const std::string& source_name)
{
    bool local = true;
    Vector<Real> update = measure_source_change(source, dt, local);

#ifdef BL_LAZY
    Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealSum(update.dataPtr(), static_cast<int>(update.size()), ParallelDescriptor::IOProcessorNumber());

        if (ParallelDescriptor::IOProcessor()) {
            if (std::abs(update[URHO]) != 0.0 || std::abs(update[UEDEN]) != 0.0) {
                std::cout << std::endl << "  Contributions to the state from " << source_name << ":" << std::endl;

                log_source_change(update);
            }
        }

#ifdef BL_LAZY
    });
#endif
}

Vector<Real>
HydroSim::measure_source_change(const MultiFab& source, Real dt, bool local)
{

  BL_PROFILE("HydroSim::measure_source_change()");

  Vector<Real> update(source.nComp(), 0.0);

  MultiFab volume_weighted_component(source.boxArray(), source.DistributionMap(), 1, 0);

  for (int n = 0; n < source.nComp(); ++n) {

    volume_weighted_component.setVal(0.0);

    MultiFab::AddProduct(volume_weighted_component, source, n, volume, 0, 0, 1, 0);

    update[n] = volume_weighted_component.sum(0, local) * dt;

  }

  return update;

}
