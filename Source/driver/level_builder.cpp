
#include <AMReX_LevelBld.H>
#include <HydroSim.H>

using namespace amrex;

class HydroSimBuilder
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

HydroSimBuilder HydroSim_builder;

LevelBld*
getLevelBld ()
{
    return &HydroSim_builder;
}

void
HydroSimBuilder::variableSetUp ()
{
    HydroSim::variableSetUp();
}

void
HydroSimBuilder::variableCleanUp ()
{
    HydroSim::variableCleanUp();
}

AmrLevel*
HydroSimBuilder::operator() ()
{
    return new HydroSim;
}

AmrLevel*
HydroSimBuilder::operator() (Amr&            papa,
                       int             lev,
                       const Geometry& level_geom,
                       const BoxArray& ba,
                       const DistributionMapping& dm,
                       Real            time)
{
    return new HydroSim(papa, lev, level_geom, ba, dm, time);
}
