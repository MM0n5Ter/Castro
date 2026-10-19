
#include <AMReX_BLFort.H>
#include <HydroSim.H>
#include <HydroSim_bc_fill_nd.H>
#include <HydroSim_generic_fill.H>
#include <user_bc_fill.H>

using namespace amrex;

void hs_statefill(Box const& bx, FArrayBox& data,
                  const int dcomp, const int numcomp,
                  Geometry const& geom, const Real time,
                  const Vector<BCRec>& bcr, const int bcomp,
                  const int scomp)
{


    Vector<BCRec> bcr_inflow_as_outflow{bcr};
    for (auto & bc : bcr_inflow_as_outflow) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (bc.lo(dir) == amrex::BCType::ext_dir) {
                bc.setLo(dir, amrex::BCType::foextrap);
            }
            if (bc.hi(dir) == amrex::BCType::ext_dir) {
                bc.setHi(dir, amrex::BCType::foextrap);
            }
        }
    }

    GpuBndryFuncFab<HydroSimGenericFill> gpu_bndry_func(HydroSimGenericFill{});
    gpu_bndry_func(bx, data, dcomp, numcomp, geom, time, bcr_inflow_as_outflow, bcomp, scomp);


    if (numcomp != NUM_STATE) {
        return;
    }


    ambient_region_fill(bx, data.array(dcomp), geom, bcr);


#if AMREX_SPACEDIM == 2
    if ((bcr[URHO].lo(0) == amrex::BCType::ext_dir && bcr[URHO].lo(1) == amrex::BCType::ext_dir) ||
        (bcr[URHO].lo(0) == amrex::BCType::ext_dir && bcr[URHO].hi(1) == amrex::BCType::ext_dir) ||
        (bcr[URHO].hi(0) == amrex::BCType::ext_dir && bcr[URHO].lo(1) == amrex::BCType::ext_dir) ||
        (bcr[URHO].hi(0) == amrex::BCType::ext_dir && bcr[URHO].hi(1) == amrex::BCType::ext_dir)) {
        amrex::Error("Error: external boundaries meeting at a corner not supported");
    }
#endif

#if AMREX_SPACEDIM == 3
    if ((bcr[URHO].lo(0) == amrex::BCType::ext_dir &&
         (bcr[URHO].lo(1) == amrex::BCType::ext_dir || bcr[URHO].lo(2) == amrex::BCType::ext_dir)) ||
        (bcr[URHO].lo(1) == amrex::BCType::ext_dir && bcr[URHO].lo(2) == amrex::BCType::ext_dir) ||
        (bcr[URHO].lo(0) == amrex::BCType::ext_dir &&
         (bcr[URHO].hi(1) == amrex::BCType::ext_dir || bcr[URHO].lo(2) == amrex::BCType::ext_dir)) ||
        (bcr[URHO].hi(1) == amrex::BCType::ext_dir && bcr[URHO].lo(2) == amrex::BCType::ext_dir) ||
        (bcr[URHO].lo(0) == amrex::BCType::ext_dir &&
         (bcr[URHO].lo(1) == amrex::BCType::ext_dir || bcr[URHO].hi(2) == amrex::BCType::ext_dir)) ||
        (bcr[URHO].lo(1) == amrex::BCType::ext_dir && bcr[URHO].hi(2) == amrex::BCType::ext_dir) ||
        (bcr[URHO].lo(0) == amrex::BCType::ext_dir &&
         (bcr[URHO].hi(1) == amrex::BCType::ext_dir || bcr[URHO].hi(2) == amrex::BCType::ext_dir)) ||
        (bcr[URHO].hi(1) == amrex::BCType::ext_dir && bcr[URHO].hi(2) == amrex::BCType::ext_dir) ||
        (bcr[URHO].hi(0) == amrex::BCType::ext_dir &&
         (bcr[URHO].lo(1) == amrex::BCType::ext_dir || bcr[URHO].lo(2) == amrex::BCType::ext_dir)) ||
        (bcr[URHO].lo(1) == amrex::BCType::ext_dir && bcr[URHO].lo(2) == amrex::BCType::ext_dir) ||
        (bcr[URHO].hi(0) == amrex::BCType::ext_dir &&
         (bcr[URHO].hi(1) == amrex::BCType::ext_dir || bcr[URHO].lo(2) == amrex::BCType::ext_dir)) ||
        (bcr[URHO].hi(1) == amrex::BCType::ext_dir && bcr[URHO].lo(2) == amrex::BCType::ext_dir) ||
        (bcr[URHO].hi(0) == amrex::BCType::ext_dir &&
         (bcr[URHO].lo(1) == amrex::BCType::ext_dir || bcr[URHO].hi(2) == amrex::BCType::ext_dir)) ||
        (bcr[URHO].lo(1) == amrex::BCType::ext_dir && bcr[URHO].hi(2) == amrex::BCType::ext_dir) ||
        (bcr[URHO].hi(0) == amrex::BCType::ext_dir &&
         (bcr[URHO].hi(1) == amrex::BCType::ext_dir || bcr[URHO].hi(2) == amrex::BCType::ext_dir)) ||
        (bcr[URHO].hi(1) == amrex::BCType::ext_dir && bcr[URHO].hi(2) == amrex::BCType::ext_dir)) {
        amrex::Error("Error: external boundaries meeting at a corner not supported");
    }
#endif

#ifdef GRAVITY
    hydrostatic_fill(bx, data.array(), geom, bcr, time);
#endif


    const auto state = data.array();


    Array1D<BCRec, 0, NUM_STATE - 1> bcs;
    for (int n = 0; n < numcomp; ++n) {
        bcs(n) = bcr[n];
    }

    const auto geomdata = geom.data();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        user_bc_fill(i, j, k, state, time, bcs, geomdata);
    });
}

