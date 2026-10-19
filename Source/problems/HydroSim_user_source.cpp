#include <HydroSim.H>
#include <user_source.H>

using namespace amrex;

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
