#include <HydroSim.H>
#include <HydroSim_helpers.H>

using namespace amrex;

void
HydroSim::assemble_old_rotation_source(MultiFab& source, MultiFab& state_in, Real time, Real dt)
{

    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::assemble_old_rotation_source()");

    const Real strt_time = ParallelDescriptor::second();


    if (!do_frame_spin) {
        return;

    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        rsrc(bx, state_in.array(mfi), source.array(mfi), dt);

    }

    if (verbose > 1)
    {
        const int IOProc   = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;
        amrex::Real level_real = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::assemble_old_rotation_source() time = " << run_time
                       << " on level " << level_real << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }
}



void
HydroSim::assemble_new_rotation_source(MultiFab& source, MultiFab& field_old, MultiFab& field_new, Real time, Real dt)
{

    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::assemble_new_rotation_source()");

    const Real strt_time = ParallelDescriptor::second();


    if (!do_frame_spin) {
        return;

    }


#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        for (MFIter mfi(field_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            apply_rotation_correction(bx,
                     field_old.array(mfi), field_new.array(mfi),
                     source.array(mfi),
                     (*mass_face_flux[0]).array(mfi), (*mass_face_flux[1]).array(mfi), (*mass_face_flux[2]).array(mfi),
                     dt, volume.array(mfi));
        }
    }

    if (verbose > 1)
    {
        const int IOProc   = ParallelDescriptor::IOProcessorNumber();
        Real      run_time = ParallelDescriptor::second() - strt_time;
        Real level_real = level;
#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::assemble_new_rotation_source() time = " << run_time << " on level " << level_real << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }
}
