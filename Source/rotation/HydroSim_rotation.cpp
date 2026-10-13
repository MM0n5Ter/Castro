#include <HydroSim.H>
#include <HydroSim_util.H>

using namespace amrex;

void
HydroSim::assemble_old_rotation_source(MultiFab& source, MultiFab& state_in, Real time, Real dt)
{

    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::assemble_old_rotation_source()");

    const Real strt_time = ParallelDescriptor::second();

    // Fill the rotation data.

    if (!do_rotation) {
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

    if (verbose > 1) {
        print_stage_time("HydroSim::assemble_old_rotation_source()", strt_time, true);
    }
}



void
HydroSim::assemble_new_rotation_source(MultiFab& source, MultiFab& state_old, MultiFab& state_new, Real time, Real dt)
{

    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::assemble_new_rotation_source()");

    const Real strt_time = ParallelDescriptor::second();

    // Fill the rotation data.

    if (!do_rotation) {
        return;

    }

    // Now do corrector part of rotation source term update

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        for (MFIter mfi(state_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            corrrsrc(bx,
                     state_old.array(mfi), state_new.array(mfi),
                     source.array(mfi),
                     (*mass_fluxes[0]).array(mfi), (*mass_fluxes[1]).array(mfi), (*mass_fluxes[2]).array(mfi),
                     dt, volume.array(mfi));
        }
    }

    if (verbose > 1) {
        print_stage_time("HydroSim::assemble_new_rotation_source()", strt_time, true);
    }
}
