
#include <AMReX_REAL.H>

#include <HydroSim.H>

using namespace amrex;

void
HydroSim::det_peak_properties (amrex::Real time,
                             amrex::Real& rho_enuc_max, amrex::Real& peak_x)
{
    BL_PROFILE("HydroSim::det_peak_properties()");

    const auto dx = geom.CellSizeArray();
    const auto prob_lo = geom.ProbLoArray();

    auto mf = derive("rho_enuc", time, 0);
    BL_ASSERT(mf != nullptr);


    const amrex::Real level_rho_enuc_max = mf->max(0);
    const amrex::IntVect level_max_index = mf->maxIndex(0);


    const amrex::Real level_peak_x = prob_lo[0] +
        (static_cast<amrex::Real>(level_max_index[0]) + 0.5_rt) * dx[0];


    if (level_rho_enuc_max > rho_enuc_max) {
        rho_enuc_max = level_rho_enuc_max;
        peak_x = level_peak_x;
    }
}
