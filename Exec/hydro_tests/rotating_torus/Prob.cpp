
#include <HydroSim.H>

using namespace amrex;

Real HydroSim::initial_moment_of_inertia = 0.0;

Real
HydroSim::compute_moment_of_inertia()
{


    int finest_level = parent->finestLevel();
    Real time = state[State_Type].curTime();
    Real dt = parent->dtLevel(0);

    Real moment = 0.0;

    bool local_only = true;

    for (int lev = 0; lev <= finest_level; ++lev) {

        HydroSim& hs_level = getLevel(lev);

        int idir = -1;
        moment += hs_level.locDistSquaredTotal("density", time, idir, local_only);

    }

    ParallelDescriptor::ReduceRealSum(moment);

    return moment;

}

void
HydroSim::user_post_init() {

    if (level != 0) return;

    initial_moment_of_inertia = compute_moment_of_inertia();

    amrex::Print() << std::endl << "  Moment of inertia = " << initial_moment_of_inertia << std::endl;

}

void
HydroSim::user_post_restart() {

    if (level != 0) return;

    initial_moment_of_inertia = compute_moment_of_inertia();

    amrex::Print() << std::endl << "  Moment of inertia = " << initial_moment_of_inertia << std::endl;

}

void
HydroSim::user_post_timestep() {

    if (level != 0) return;

    Real moment_of_inertia = compute_moment_of_inertia();

    amrex::Print() << std::endl << "  Moment of inertia = " << moment_of_inertia << std::endl;
    amrex::Print() << std::endl << "  Moment of inertia / initial moment of inertia = " << moment_of_inertia / initial_moment_of_inertia << std::endl;

}
