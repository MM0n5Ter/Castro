#include <HydroSim.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

using namespace amrex;

void
HydroSim::apply_source_term_to_state(MultiFab& state_to_update, const MultiFab& source, Real dt, int ng)
{
    BL_PROFILE("HydroSim::apply_source_term_to_state()");

    AMREX_ASSERT(source.nGrow() >= ng);
    AMREX_ASSERT(state_to_update.nGrow() >= ng);

    MultiFab::Saxpy(state_to_update, dt, source, 0, 0, source.nComp(), ng);
}

void
HydroSim::blend_source_terms(MultiFab& S_new, MultiFab& source_prev, MultiFab &source_next, Real dt)
{
  BL_PROFILE("HydroSim::blend_source_terms()");


  MultiFab::Saxpy(S_new,-0.5*dt,source_prev,0,0,S_new.nComp(),0);
  MultiFab::Saxpy(S_new, 0.5*dt,source_next,0,0,S_new.nComp(),0);
}

bool
HydroSim::source_enabled(int src)
{
    switch(src) {

#ifdef SPONGE
    case damping_src:
        if (do_damping_zone) {
            return true;
        } else {
            return false;
        }
#endif

    case external_source:
        if (enable_extra_source) {
            return true;
        } else {
            return false;
        }
#ifndef MHD
    case thermodynamic_src:
        if (time_scheme == IterativeCorrection) {
            return true;
        } else {
          return false;
        }
#else
    case thermodynamic_src:
        return true;
#endif

    case geometry_source:
        if (geom.Coord() == 0 || use_curvilinear_src == 0) {
          return false;
        } else {
          return true;
        }

#ifdef DIFFUSION
    case conduction_src:
        if (conduction_enabled &&
            !(time_scheme == IterativeCorrection)) {
          return true;
        }
        else {
          return false;
        }
#endif

#ifdef HYBRID_MOMENTUM
    case hybrid_mom_source:
        return true;
#endif

#ifdef GRAVITY
    case gravity_src:
        if (do_gravity_flag) {
            return true;
        } else {
            return false;
        }
#endif

#ifdef ROTATION
    case rotation_src:
        if (do_frame_spin) {
            return true;
        } else {
            return false;
        }
#endif

    default:
        return false;

    }
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
      bool use_new_state = false;
      log_all_source_changes(dt, use_new_state);
    }

    if (verbose > 0)
    {
        const int IOProc   = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;
        amrex::Real level_real = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::run_old_sources() time = " << run_time
                       << " on level " << level_real << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
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
      bool use_new_state = true;
      log_all_source_changes(dt, use_new_state);
    }

    if (verbose > 0)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;
        amrex::Real level_real = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::run_new_sources() time = " << run_time << " on level " << level_real << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
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


void
HydroSim::log_source_change(const Vector<Real>& update)
{

  if (ParallelDescriptor::IOProcessor()) {

    std::cout << "       mass added: " << update[URHO] << std::endl;
    std::cout << "       xmom added: " << update[UMX] << std::endl;
    std::cout << "       ymom added: " << update[UMY] << std::endl;
    std::cout << "       zmom added: " << update[UMZ] << std::endl;
    std::cout << "       eint added: " << update[UEINT] << std::endl;
    std::cout << "       ener added: " << update[UEDEN] << std::endl;

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


void
HydroSim::log_all_source_changes(Real dt, bool use_new_state)
{
    const MultiFab& source = use_new_state ? get_new_data(Forcing_Type) : get_old_data(Forcing_Type);

    std::string source_name = use_new_state? "new-time sources" : "old-time sources";

    measure_and_log_source_change(source, dt, source_name);
}


step_result
HydroSim::before_advance_hooks (Real time, Real dt)
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    step_result status {};


#ifdef GRAVITY
    assemble_old_gravity(time);
#endif

#ifdef SHOCK_VAR


    const bool modify_state_flag{false};
    run_old_sources(time, dt, modify_state_flag);

    MultiFab& S_old = get_old_data(State_Type);
    MultiFab& old_forcing = get_old_data(Forcing_Type);

    FArrayBox shk(The_Async_Arena());
    FArrayBox prim_tmp(The_Async_Arena()), qthermo_tmp(The_Async_Arena());

    for (MFIter mfi(Sborder, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const Box& obx = amrex::grow(bx, 1);

        shk.resize(bx, 1);
#ifdef RADIATION
        prim_tmp.resize(obx, NQ);
#else
        prim_tmp.resize(obx, NQPRIMBASE);
#endif
        qthermo_tmp.resize(obx, NQEXTRAQ);

        Array4<Real> const shock_flag_arr = shk.array();
        Array4<Real> const prim_arr = prim_tmp.array();
        Array4<Real> const qthermo_arr = qthermo_tmp.array();

        Array4<Real const> const Ghost_state_old_arr = Sborder.array(mfi);
        Array4<Real> const state_old_arr = S_old.array(mfi);
        Array4<Real> const prev_source_arr = old_forcing.array(mfi);

        ctoprim(obx, time, Ghost_state_old_arr, prim_arr, qthermo_arr);

        shock(bx, prim_arr, prev_source_arr, shock_flag_arr);



        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            state_old_arr(i,j,k,UFLAG_SHOCK) = shock_flag_arr(i,j,k);
        });

    }


    if (Sborder.nGrow() > 0) {
      AmrLevel::FillPatch(*this, Sborder, Sborder.nGrow(), time, State_Type, UFLAG_SHOCK, 1, UFLAG_SHOCK);
    }

#endif


#ifndef TRUE_SDC
#ifdef REACTIONS
    status = apply_old_reactions(time, dt);

    if (status.success == false) {
        return status;
    }
#endif
#endif



    MultiFab& S_new = get_new_data(State_Type);

    MultiFab::Copy(S_new, Sborder, 0, 0, NUM_STATE, S_new.nGrow());

    return status;
}


step_result
HydroSim::before_hydro_hooks (Real time, Real dt)
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    step_result status {};

#ifdef REACTIONS

    if (hydrosim::time_scheme == SimplifiedIterCorrection) {
        MultiFab& reaction_source_arr = get_new_data(Burn_Corr_Type);

        AmrLevel::FillPatch(*this, reaction_source_arr, reaction_source_arr.nGrow(), time + dt, Burn_Corr_Type, 0, reaction_source_arr.nComp());
    }
#endif

    return status;
}


step_result
HydroSim::after_hydro_hooks (Real time, Real dt)
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    step_result status {};

#ifdef GRAVITY
    assemble_new_gravity(time);
#endif

    return status;
}


step_result
HydroSim::after_advance_hooks (Real time, Real dt)
{
    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    step_result status {};

#ifndef TRUE_SDC
#ifdef REACTIONS
    status = apply_new_reactions(time, dt);

    if (status.success == false) {
        return status;
    }
#endif
#endif

    return status;
}
