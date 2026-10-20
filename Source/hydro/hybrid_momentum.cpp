#include <HydroSim.H>
#include <HydroSim_helpers.H>

#include <hybrid_momentum.H>

using namespace amrex;

void
HydroSim::assemble_old_hybrid_source(MultiFab& source, MultiFab& field_old, Real time, Real dt)
{

    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    BL_PROFILE("HydroSim::assemble_old_hybrid_source()");

    const Real strt_time = ParallelDescriptor::second();

    Real term_weight = 1.0;

    fill_hybrid_source_mf(source, field_old, term_weight);

    if (verbose > 1)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        if (ParallelDescriptor::IOProcessor())
            std::cout << "HydroSim::assemble_old_hybrid_source() time = " << run_time << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }
}



void
HydroSim::assemble_new_hybrid_source(MultiFab& source, MultiFab& field_old, MultiFab& field_new, Real time, Real dt)
{

    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    BL_PROFILE("HydroSim::assemble_new_hybrid_source()");

    const Real strt_time = ParallelDescriptor::second();


    Real term_weight = -0.5;

    fill_hybrid_source_mf(source, field_old, term_weight);


    term_weight = 0.5;

    fill_hybrid_source_mf(source, field_new, term_weight);

    if (verbose > 1)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        if (ParallelDescriptor::IOProcessor())
            std::cout << "HydroSim::assemble_new_hybrid_source() time = " << run_time << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }
}



void
HydroSim::fill_hybrid_source_mf(MultiFab& sources, const MultiFab& state_in, Real term_weight)
{
    BL_PROFILE("HydroSim::fill_hybrid_source_mf()");

    GeometryData geomdata = geom.data();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        const auto u = state_in.array(mfi);
        auto src = sources.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            GpuArray<Real, 3> loc;

            position(i, j, k, geomdata, loc);

            loc[0] -= problem::center[0];
            loc[1] -= problem::center[1];

            Real R = amrex::max(std::sqrt(loc[0] * loc[0] + loc[1] * loc[1]),
                                std::numeric_limits<Real>::min());

            Real rhoInv = 1.0_rt / u(i,j,k,URHO);
            Real InvRadius = 1.0_rt / R;

            src(i,j,k,UMR) = src(i,j,k,UMR) + term_weight * (rhoInv * InvRadius * InvRadius * InvRadius) *
                                              u(i,j,k,UML) * u(i,j,k,UML);

        });
    }
}



void
HydroSim::linear_to_split_momentum(MultiFab& state_in, int ng)
{
    BL_PROFILE("HydroSim::linear_to_split_momentum()");

    GeometryData geomdata = geom.data();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(ng);

        auto u = state_in.array(mfi);


        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            GpuArray<Real, 3> loc;

            position(i, j, k, geomdata, loc);

            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                loc[dir] -= problem::center[dir];

            GpuArray<Real, 3> cartesian_vector;

            for (int dir = 0; dir < 3; ++dir)
                cartesian_vector[dir] = u(i,j,k,UMX+dir);

            GpuArray<Real, 3> hybrid_vector;

            cartesian_to_hybrid(loc, cartesian_vector, hybrid_vector);

            for (int dir = 0; dir < 3; ++dir)
                u(i,j,k,UMR+dir) = hybrid_vector[dir];

        });
    }
}



void
HydroSim::split_to_linear_momentum(MultiFab& state_in, int ng)
{
    BL_PROFILE("HydroSim::split_to_linear_momentum()");

    GeometryData geomdata = geom.data();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(ng);

        auto u = state_in.array(mfi);


        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            GpuArray<Real, 3> loc;

            position(i, j, k, geomdata, loc);

            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
                loc[dir] -= problem::center[dir];

            GpuArray<Real, 3> hybrid_vector;

            for (int dir = 0; dir < 3; ++dir)
                hybrid_vector[dir] = u(i,j,k,UMR+dir);

            GpuArray<Real, 3> cartesian_vector;

            hybrid_to_cartesian(loc, hybrid_vector, cartesian_vector);

            for (int dir = 0; dir < 3; ++dir)
                u(i,j,k,UMX+dir) = cartesian_vector[dir];

        });
    }
}
