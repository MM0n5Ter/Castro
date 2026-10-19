#include <numbers>
#include <HydroSim.H>
#include <user_source.H>


#ifdef HYBRID_MOMENTUM
#include <hybrid.H>
#endif

using namespace amrex;

#ifdef SPONGE
void
HydroSim::add_damping_term(const Box& bx,
                     Array4<Real const> const& state_in,
                     Array4<Real> const& source,
                     Real dt) {

  Real alpha;
  if (damping_timescale > 0.0_rt) {
    alpha = dt / damping_timescale;
  } else {
    alpha = 0.0_rt;
  }

  auto dx = geom.CellSizeArray();
  auto problo = geom.ProbLoArray();
  auto geomdata = geom.data();

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real src[NUM_SOURCE] = {0.0};

    GpuArray<Real, 3> r;

    r[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];

#if AMREX_SPACEDIM >= 2
    r[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
#else
    r[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
    r[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
#else
    r[2] = 0.0_rt;
#endif

    Real rho = state_in(i,j,k,URHO);
    Real rhoInv = 1.0_rt / rho;


    Real delta_r = damping_upper_radius - damping_lower_radius;

    Real density_span = damping_lower_density - damping_upper_density;

    Real delta_p = damping_lower_pressure - damping_upper_pressure;


    Real damping_strength = 0.0_rt;

    if (damping_lower_radius >= 0.0_rt && damping_upper_radius > damping_lower_radius) {
      Real rad = distance(geomdata, r);

      if (rad < damping_lower_radius) {
        damping_strength = damping_lower_factor;

      } else if (rad >= damping_lower_radius && rad <= damping_upper_radius) {
        damping_strength = damping_lower_factor +
          0.5_rt * (damping_upper_factor - damping_lower_factor) *
          (1.0_rt - std::cos(std::numbers::pi * (rad - damping_lower_radius) / delta_r));

      } else {
        damping_strength = damping_upper_factor;
      }
    }



    if (damping_upper_density > 0.0_rt && damping_lower_density > 0.0_rt) {
      if (rho > damping_upper_density) {
        damping_strength = damping_lower_factor;

      } else if (rho <= damping_upper_density && rho >= damping_lower_density) {
        damping_strength = damping_lower_factor +
          0.5_rt * (damping_upper_factor - damping_lower_factor) *
          (1.0_rt - std::cos(std::numbers::pi * (rho - damping_upper_density) / density_span));

      } else {
        damping_strength = damping_upper_factor;
      }
    }



    if (damping_upper_pressure > 0.0_rt && damping_lower_pressure >= 0.0_rt) {

      eos_rep_t eos_state;

      eos_state.rho = state_in(i,j,k,URHO);
      eos_state.T = state_in(i,j,k,UTEMP);
      for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = state_in(i,j,k,UFS+n) * rhoInv;
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
        eos_state.aux[n] = state_in(i,j,k,UFX+n) * rhoInv;
      }
#endif

      eos(eos_input_rt, eos_state);

      Real p = eos_state.p;

      if (p > damping_upper_pressure) {
        damping_strength = damping_lower_factor;

      } else if (p <= damping_upper_pressure && p >= damping_lower_pressure) {
        damping_strength = damping_lower_factor +
          0.5_rt * (damping_upper_factor - damping_lower_factor) *
          (1.0_rt - std::cos(std::numbers::pi * (p - damping_upper_pressure) / delta_p));

      } else {
        damping_strength = damping_upper_factor;

      }
    }



    Real fac;
    if (damping_implicit == 1) {
       fac = -(1.0_rt - 1.0_rt / (1.0_rt + alpha * damping_strength));

    } else {
       fac = -alpha * damping_strength;

    }


    GpuArray<Real, 3> Sr;
    GpuArray<Real, 3> damping_target_velocity = {damping_target_x_velocity,
                                                damping_target_y_velocity,
                                                damping_target_z_velocity};
    for (int n = 0; n < 3; n++) {
      Sr[n] = (state_in(i,j,k,UMX+n) - rho * damping_target_velocity[n]) * fac / dt;
      src[UMX+n] = Sr[n];
    }


    Real SrE = 0.0;
    for (int n = 0; n < 3; n++) {
      SrE += state_in(i,j,k,UMX+n) * rhoInv * Sr[n];
    }

    src[UEDEN] = SrE;

#ifdef HYBRID_MOMENTUM
    GpuArray<Real, 3> Hybrid_momentum_source;
    compute_hybrid_momentum_source(r, Sr, Hybrid_momentum_source);
    for (int n = 0; n < 3; n++) {
      src[UMR+n] = Hybrid_momentum_source[n];
    }
#endif

    for (int n = 0; n < NUM_SOURCE; n++) {
      source(i,j,k,n) += src[n];
    }

  });
}

void
HydroSim::assemble_old_damping_source(MultiFab& source, MultiFab& state_in,
                                    Real time, Real dt)
{

    amrex::ignore_unused(source);
    amrex::ignore_unused(state_in);
    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

}

void
HydroSim::assemble_new_damping_source(MultiFab& source, const MultiFab& field_old, MultiFab& field_new,
                                    Real time, Real dt)
{

    amrex::ignore_unused(field_old);
    amrex::ignore_unused(time);

    const Real strt_time = ParallelDescriptor::second();

    if (!do_damping_zone) {
        return;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(field_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        add_damping_term(bx, field_new.array(mfi), source.array(mfi), dt);
    }

    if (verbose > 1)
    {
        const int IOProc   = ParallelDescriptor::IOProcessorNumber();
        Real      run_time = ParallelDescriptor::second() - strt_time;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        if (ParallelDescriptor::IOProcessor()) {
            std::cout << "HydroSim::assemble_new_damping_source() time = " << run_time << "\n" << "\n";
        }
#ifdef BL_LAZY
        });
#endif
    }
}
#endif

void
HydroSim::fill_thermodynamic_source (const MultiFab& state_in, MultiFab& thermodynamic_src)
{


  const auto dx = geom.CellSizeArray();
  const auto prob_lo = geom.ProbLoArray();

  auto coord = geom.Coord();


#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(thermodynamic_src, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

    const Box& bx = mfi.tilebox();

    Array4<Real const> const U = state_in.array(mfi);
    Array4<Real> const src = thermodynamic_src.array(mfi);


    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

      Real rp = prob_lo[0] + (static_cast<Real>(i) + 1.5_rt)*dx[0];
      Real rm = prob_lo[0] + (static_cast<Real>(i) - 0.5_rt)*dx[0];
      Real r = 0.5_rt*(rm + rp);

      if (coord == 0) {
        src(i,j,k,UEINT) = -0.5_rt*(U(i+1,j,k,UMX)/U(i+1,j,k,URHO)  -
                                    U(i-1,j,k,UMX)/U(i-1,j,k,URHO))/dx[0];

      } else if (coord == 1) {
        src(i,j,k,UEINT) = -0.5_rt*(rp*U(i+1,j,k,UMX)/U(i+1,j,k,URHO) -
                                    rm*U(i-1,j,k,UMX)/U(i-1,j,k,URHO))/(r*dx[0]);

      } else if (coord == 2) {
        src(i,j,k,UEINT) = -0.5_rt*(rp*rp*U(i+1,j,k,UMX)/U(i+1,j,k,URHO) -
                                    rm*rm*U(i-1,j,k,UMX)/U(i-1,j,k,URHO))/(r*r*dx[0]);
      }

#if AMREX_SPACEDIM >= 2
      if (coord == 2) {
          Real theta_plus = prob_lo[1] + (static_cast<Real>(j) + 1.5_rt)*dx[1];
          Real theta_minus = prob_lo[1] + (static_cast<Real>(j) - 0.5_rt)*dx[1];
          Real theta = 0.5_rt*(theta_minus + theta_plus);

          src(i,j,k,UEINT) += -0.5_rt*(std::sin(theta_plus)*U(i,j+1,k,UMY)/U(i,j+1,k,URHO) -
                                       std::sin(theta_minus)*U(i,j-1,k,UMY)/U(i,j-1,k,URHO)) /
                                      (r*std::sin(theta)*dx[1]);
      } else {
          src(i,j,k,UEINT) += -0.5_rt*(U(i,j+1,k,UMY)/U(i,j+1,k,URHO) -
                                       U(i,j-1,k,UMY)/U(i,j-1,k,URHO))/dx[1];
      }
#endif
#if AMREX_SPACEDIM == 3
      src(i,j,k,UEINT) += -0.5_rt*(U(i,j,k+1,UMZ)/U(i,j,k+1,URHO) -
                                   U(i,j,k-1,UMZ)/U(i,j,k-1,URHO))/dx[2];
#endif

      eos_rep_t eos_state;
      eos_state.rho = U(i,j,k,URHO);
      eos_state.T = U(i,j,k,UTEMP);
      for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = U(i,j,k,UFS+n)/U(i,j,k,URHO);
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
        eos_state.aux[n] = U(i,j,k,UFX+n)/U(i,j,k,URHO);
      }
#endif

      eos(eos_input_rt, eos_state);

      src(i,j,k,UEINT) = eos_state.p * src(i,j,k,UEINT);

    });
  }
}

void
HydroSim::assemble_old_thermodynamic_source(MultiFab& source, MultiFab& state_in,
                                    Real time, Real dt)
{

    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

#ifndef MHD
  if (!(time_scheme == IterativeCorrection)) {
      return;
  }
#endif

  const Real strt_time = ParallelDescriptor::second();

  MultiFab thermodynamic_src(grids, dmap, source.nComp(), 0);

  thermodynamic_src.setVal(0.0);

  fill_thermodynamic_source(state_in, thermodynamic_src);

  Real term_weight = 1.0;

  MultiFab::Saxpy(source, term_weight, thermodynamic_src, 0, 0, source.nComp(), 0);

  if (verbose > 1)
  {
      const int IOProc   = ParallelDescriptor::IOProcessorNumber();
      Real      run_time = ParallelDescriptor::second() - strt_time;

#ifdef BL_LAZY
      Lazy::QueueReduction( [=] () mutable {
#endif
      ParallelDescriptor::ReduceRealMax(run_time,IOProc);

      amrex::Print() << "HydroSim::assemble_old_thermodynamic_source() time = " << run_time << "\n" << "\n";
#ifdef BL_LAZY
      });
#endif
    }

}

void
HydroSim::assemble_new_thermodynamic_source(MultiFab& source, const MultiFab& field_old, MultiFab& field_new,
                                    Real time, Real dt)
{

    amrex::ignore_unused(source);
    amrex::ignore_unused(field_old);
    amrex::ignore_unused(field_new);
    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

#ifndef MHD
  if (!(time_scheme == IterativeCorrection)) {
      return;
  }

  amrex::Abort("you should not get here!");
#else

  const Real strt_time = ParallelDescriptor::second();

  MultiFab thermodynamic_src(grids, dmap, source.nComp(), 0);

  thermodynamic_src.setVal(0.0);

  fill_thermodynamic_source(field_old, thermodynamic_src);

  Real term_weight = -0.5;

  MultiFab::Saxpy(source, term_weight, thermodynamic_src, 0, 0, source.nComp(), 0);


  thermodynamic_src.setVal(0.0);

  term_weight = 0.5;

  FillPatchIterator fpi(*this, field_new, 1, time, State_Type, 0, NUM_STATE);
  MultiFab& grown_field = fpi.get_mf();

  fill_thermodynamic_source(grown_field, thermodynamic_src);

  MultiFab::Saxpy(source, term_weight, thermodynamic_src, 0, 0, source.nComp(), 0);

  if (verbose > 1)
  {
      const int IOProc   = ParallelDescriptor::IOProcessorNumber();
      Real      run_time = ParallelDescriptor::second() - strt_time;

#ifdef BL_LAZY
      Lazy::QueueReduction( [=] () mutable {
#endif
      ParallelDescriptor::ReduceRealMax(run_time,IOProc);

      if (ParallelDescriptor::IOProcessor())
          std::cout << "HydroSim::assemble_new_thermodynamic_source() time = " << run_time << "\n" << "\n";
#ifdef BL_LAZY
      });
#endif
    }


#endif
}

void
HydroSim::fill_external_source (const Real time, const Real dt,
                         const MultiFab& field_old, const MultiFab& field_new,
                         MultiFab& external_source)
{

    amrex::ignore_unused(field_old);

    GeometryData geomdata = geom.data();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(external_source, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        Array4<Real const> const snew = field_new.array(mfi);
        Array4<Real> const src = external_source.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            user_source(i, j, k, geomdata, snew, src, dt, time);
        });
    }
}

void
HydroSim::assemble_old_external_source(MultiFab& source, const MultiFab& state_in, Real time, Real dt)
{
    const Real strt_time = ParallelDescriptor::second();

    if (!enable_extra_source) {
        return;
    }

    MultiFab external_source(grids, dmap, source.nComp(), 0);

    external_source.setVal(0.0);

    fill_external_source(time, dt, state_in, state_in, external_source);

    Real term_weight = 1.0;

    MultiFab::Saxpy(source, term_weight, external_source, 0, 0, source.nComp(), 0);

    if (verbose > 1)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::assemble_old_external_source() time = " << run_time << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }
}

void
HydroSim::assemble_new_external_source(MultiFab& source, const MultiFab& field_old, const MultiFab& field_new, Real time, Real dt)
{
    const Real strt_time = ParallelDescriptor::second();

    if (!enable_extra_source) {
        return;
    }


    Real term_weight;

    MultiFab external_source(grids, dmap, source.nComp(), 0);

    external_source.setVal(0.0);


    Real old_time = time - dt;

    if (extra_source_implicit) {
        term_weight = -1.0;
    } else {
        term_weight = -0.5;
    }

    fill_external_source(old_time, dt, field_old, field_old, external_source);

    MultiFab::Saxpy(source, term_weight, external_source, 0, 0, source.nComp(), 0);


    external_source.setVal(0.0);

    if (extra_source_implicit) {
        term_weight = 1.0;
    } else {
        term_weight = 0.5;
    }

    fill_external_source(time, dt, field_old, field_new, external_source);

    MultiFab::Saxpy(source, term_weight, external_source, 0, 0, source.nComp(), 0);

    if (verbose > 1)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::assemble_new_external_source() time = " << run_time << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }
}
