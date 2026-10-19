#include <numbers>

#ifdef SPONGE
#include <HydroSim.H>

#ifdef HYBRID_MOMENTUM
#include <hybrid.H>
#endif

using namespace amrex;

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


#endif
