
#include <AMReX_LevelBld.H>
#include <HydroSim.H>

using namespace amrex;

class HydroSimBld
    :
    public LevelBld
{
    void variableSetUp () override;
    void variableCleanUp () override;
    AmrLevel *operator() () override;
    AmrLevel *operator() (Amr&            papa,
                          int             lev,
                          const Geometry& level_geom,
                          const BoxArray& ba,
                          const DistributionMapping& dm,
                          Real            time) override;
};

HydroSimBld HydroSim_bld;

LevelBld*
getLevelBld ()
{
    return &HydroSim_bld;
}

void
HydroSimBld::variableSetUp ()
{
    HydroSim::variableSetUp();
}

void
HydroSimBld::variableCleanUp ()
{
    HydroSim::variableCleanUp();
}

AmrLevel*
HydroSimBld::operator() ()
{
    return new HydroSim;
}

AmrLevel*
HydroSimBld::operator() (Amr&            papa,
                       int             lev,
                       const Geometry& level_geom,
                       const BoxArray& ba,
                       const DistributionMapping& dm,
                       Real            time)
{
    return new HydroSim(papa, lev, level_geom, ba, dm, time);
}
