#include <HydroSim.H>
#include <HydroSim_math.H>

using namespace amrex;

void
HydroSim::fill_axisymmetric_geometry_source (Real time, Real dt, const MultiFab& conserved_state, MultiFab& geometry_source)
{


  amrex::ignore_unused(time);
  amrex::ignore_unused(dt);

  auto dx = geom.CellSizeArray();
  auto prob_lo = geom.ProbLoArray();

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(geometry_source, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

    const Box& bx = mfi.tilebox();

    Array4<Real const> const field_arr = conserved_state.array(mfi);

    Array4<Real> const src = geometry_source.array(mfi);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

      Real r = prob_lo[0] + (static_cast<Real>(i) + 0.5_rt)*dx[0];

      src(i,j,k,UMX) = field_arr(i,j,k,UMZ) * field_arr(i,j,k,UMZ) / (field_arr(i,j,k,URHO) * r);

      src(i,j,k,UMZ) = - field_arr(i,j,k,UMX) * field_arr(i,j,k,UMZ) / (field_arr(i,j,k,URHO) * r);

    });
  }
}

void
HydroSim::fill_spherical_geometry_source (Real time, Real dt, const MultiFab& conserved_state, MultiFab& geometry_source)
{


  amrex::ignore_unused(time);
  amrex::ignore_unused(dt);

  auto dx = geom.CellSizeArray();
  auto prob_lo = geom.ProbLoArray();

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(geometry_source, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

    const Box& bx = mfi.tilebox();

    Array4<Real const> const field_arr = conserved_state.array(mfi);

    Array4<Real> const src = geometry_source.array(mfi);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

      Real r = prob_lo[0] + (static_cast<Real>(i) + 0.5_rt)*dx[0];
      Real theta_cotangent{};
#if AMREX_SPACEDIM >= 2
      Real theta = prob_lo[1] + (static_cast<Real>(j) + 0.5_rt)*dx[1];
      theta_cotangent = cot(theta);
#endif

      src(i,j,k,UMX) = (field_arr(i,j,k,UMY) * field_arr(i,j,k,UMY) +
                        field_arr(i,j,k,UMZ) * field_arr(i,j,k,UMZ)) / (field_arr(i,j,k,URHO) * r);

      src(i,j,k,UMY) = (field_arr(i,j,k,UMZ) * field_arr(i,j,k,UMZ) * theta_cotangent -
                        field_arr(i,j,k,UMX) * field_arr(i,j,k,UMY)) / (field_arr(i,j,k,URHO) * r);

      src(i,j,k,UMZ) = (- field_arr(i,j,k,UMY) * field_arr(i,j,k,UMZ) * theta_cotangent -
                        field_arr(i,j,k,UMX) * field_arr(i,j,k,UMZ)) / (field_arr(i,j,k,URHO) * r);

    });
  }
}

static void fill_curvilinear_source (HydroSim& lev, int coord, Real time, Real dt,
                                     const MultiFab& state, MultiFab& dst)
{
  if (coord == 1) {
      lev.fill_axisymmetric_geometry_source(time, dt, state, dst);
  } else {
      lev.fill_spherical_geometry_source(time, dt, state, dst);
  }
}

void
HydroSim::assemble_old_geometry_source(MultiFab& source, const MultiFab& state_in, Real time, Real dt)
{

  amrex::ignore_unused(source);
  amrex::ignore_unused(state_in);
  amrex::ignore_unused(time);
  amrex::ignore_unused(dt);

  if (geom.Coord() == 0) {
      return;
  }

  if (use_curvilinear_src == 0) {
      return;
  }

#if AMREX_SPACEDIM <= 2
  const Real strt_time = ParallelDescriptor::second();

  MultiFab geometry_source(grids, dmap, source.nComp(), 0);

  geometry_source.setVal(0.0);

  fill_curvilinear_source(*this, geom.Coord(), time, dt, state_in, geometry_source);

  Real term_weight = 1.0;

  MultiFab::Saxpy(source, term_weight, geometry_source, 0, 0, source.nComp(), 0);

  if (verbose > 1)
  {
      const int IOProc = ParallelDescriptor::IOProcessorNumber();
      amrex::Real run_time = ParallelDescriptor::second() - strt_time;

#ifdef BL_LAZY
      Lazy::QueueReduction( [=] () mutable {
#endif
      ParallelDescriptor::ReduceRealMax(run_time,IOProc);

      amrex::Print() << "HydroSim::assemble_old_geometry_source() time = " << run_time << "\n" << "\n";
#ifdef BL_LAZY
      });
#endif
    }

#endif
}

void
HydroSim::assemble_new_geometry_source(MultiFab& source, const MultiFab& field_old,
                                  const MultiFab& field_new, Real time, Real dt)
{

  amrex::ignore_unused(source);
  amrex::ignore_unused(field_old);
  amrex::ignore_unused(field_new);
  amrex::ignore_unused(time);
  amrex::ignore_unused(dt);

  if (geom.Coord() == 0) {
      return;
  }

  if (use_curvilinear_src == 0) {
      return;
  }

#if AMREX_SPACEDIM <= 2
  const Real strt_time = ParallelDescriptor::second();

  MultiFab geometry_source(grids, dmap, source.nComp(), 0);

  geometry_source.setVal(0.0);

  Real old_time = time - dt;

  fill_curvilinear_source(*this, geom.Coord(), old_time, dt, field_old, geometry_source);

  Real term_weight = -0.5;

  MultiFab::Saxpy(source, term_weight, geometry_source, 0, 0, source.nComp(), 0);


  geometry_source.setVal(0.0);

  term_weight = 0.5;

  fill_curvilinear_source(*this, geom.Coord(), time, dt, field_new, geometry_source);

  MultiFab::Saxpy(source, term_weight, geometry_source, 0, 0, source.nComp(), 0);

  if (verbose > 1)
  {
      const int IOProc   = ParallelDescriptor::IOProcessorNumber();
      Real      run_time = ParallelDescriptor::second() - strt_time;

#ifdef BL_LAZY
      Lazy::QueueReduction( [=] () mutable {
#endif
      ParallelDescriptor::ReduceRealMax(run_time,IOProc);

      amrex::Print() << "HydroSim::assemble_new_geometry_source() time = " << run_time << "\n" << "\n";
#ifdef BL_LAZY
      });
#endif
    }

#endif
}
