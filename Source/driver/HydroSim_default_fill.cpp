
#include <AMReX_BLFort.H>
#include <HydroSim.H>
#include <HydroSim_default_fill.H>
#include <AMReX_BCUtil.H>

using namespace amrex;

void default_bc_fill(Box const& bx, FArrayBox& data,
                     const int dcomp, const int numcomp,
                     Geometry const& geom, const Real time,
                     const Vector<BCRec>& bcr, const int bcomp,
                     const int scomp)
{
    GpuBndryFuncFab<HydroSimDefaultFill> gpu_bndry_func(HydroSimDefaultFill{});
    gpu_bndry_func(bx, data, dcomp, numcomp, geom, time, bcr, bcomp, scomp);
}
