#include <AMReX_BLFort.H>
#include <HydroSim.H>
#include <boundary_fill.H>
#include <user_bc_fill.H>
#include <AMReX_BCUtil.H>
#include <ambient.H>

using namespace amrex;

void hs_generic_fill(Box const& bx, FArrayBox& data,
                     const int dcomp, const int numcomp,
                     Geometry const& geom, const Real time,
                     const Vector<BCRec>& bcr, const int bcomp,
                     const int scomp)
{
    GpuBndryFuncFab<HydroSimGenericFill> gpu_bndry_func(HydroSimGenericFill{});
    gpu_bndry_func(bx, data, dcomp, numcomp, geom, time, bcr, bcomp, scomp);
}

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


#if AMREX_SPACEDIM >= 2
    int ext_dirs = 0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (bcr[URHO].lo(d) == amrex::BCType::ext_dir || bcr[URHO].hi(d) == amrex::BCType::ext_dir) {
            ++ext_dirs;
        }
    }
    if (ext_dirs >= 2) {
        amrex::Error("Error: external boundaries meeting at a corner not supported");
    }
#endif

#ifdef GRAVITY
    hydrostatic_fill(bx, data.array(), geom, bcr, time);
#endif


    const auto state = data.array();


    AMREX_ALWAYS_ASSERT(static_cast<int>(bcr.size()) >= numcomp && numcomp <= NUM_STATE);

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

static GpuArray<bool, 2*AMREX_SPACEDIM> ambient_faces (const BCRec& bc)
{
    GpuArray<bool, 2*AMREX_SPACEDIM> faces;

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const bool on_axis = (hydrosim::ambient_bc_axis == dir || hydrosim::ambient_bc_axis == -1);
        faces[2*dir] = on_axis &&
                       (bc.lo(dir) == amrex::BCType::foextrap || bc.lo(dir) == amrex::BCType::hoextrap);
        faces[2*dir+1] = on_axis &&
                         (bc.hi(dir) == amrex::BCType::foextrap || bc.hi(dir) == amrex::BCType::hoextrap);
    }

    return faces;
}

static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool in_ambient_ghost (int i, int j, int k, GpuArray<bool, 2*AMREX_SPACEDIM> const& faces,
                       GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi)
{
    return (faces[0] && i < domlo[0]) ||
           (faces[1] && i > domhi[0])
#if AMREX_SPACEDIM >= 2
           ||
           (faces[2] && j < domlo[1]) ||
           (faces[3] && j > domhi[1])
#endif
#if AMREX_SPACEDIM == 3
           ||
           (faces[4] && k < domlo[2]) ||
           (faces[5] && k > domhi[2])
#endif
           ;
}

void
ambient_density_fill(const Box& bx, Array4<Real> const& state,
                Geometry const& geom, const Vector<BCRec>& bcr)
{


    if (! hydrosim::apply_ambient_bc) {
        return;
    }

    const BCRec bc = bcr[URHO];

    const auto domlo = geom.Domain().loVect3d();
    const auto domhi = geom.Domain().hiVect3d();

    const auto ambient_faces_arr = ambient_faces(bc);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

        if (in_ambient_ghost(i, j, k, ambient_faces_arr, domlo, domhi)) {
          state(i,j,k) = ambient::ambient_reference_state[URHO];
        }
    });
}

void
ambient_region_fill(const Box& bx, Array4<Real> const& state,
             Geometry const& geom, const Vector<BCRec>& bcr)
{


    if (! hydrosim::apply_ambient_bc) {
        return;
    }

    const BCRec bc = bcr[URHO];

    const auto domlo = geom.Domain().loVect3d();
    const auto domhi = geom.Domain().hiVect3d();

    const auto ambient_faces_arr = ambient_faces(bc);


    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if (in_ambient_ghost(i, j, k, ambient_faces_arr, domlo, domhi)) {
            for (int n = 0; n < NUM_STATE; ++n) {
                state(i,j,k,n) = ambient::ambient_reference_state[n];
            }

            if (hydrosim::ambient_outflow_mode == 1) {

                if (i < domlo[0]) {
                    state(i,j,k,UMX) = amrex::min(0.0_rt, state(domlo[0],j,k,UMX));
                    state(i,j,k,UMY) = 0.0_rt;
                    state(i,j,k,UMZ) = 0.0_rt;
                } else if (i > domhi[0]) {
                    state(i,j,k,UMX) = amrex::max(0.0_rt, state(domhi[0],j,k,UMX));
                    state(i,j,k,UMY) = 0.0_rt;
                    state(i,j,k,UMZ) = 0.0_rt;
                } else if (j < domlo[1]) {
                    state(i,j,k,UMX) = 0.0_rt;
                    state(i,j,k,UMY) = amrex::min(0.0_rt, state(i,domlo[1],k,UMY));
                    state(i,j,k,UMZ) = 0.0_rt;
                } else if (j > domhi[1]) {
                    state(i,j,k,UMX) = 0.0_rt;
                    state(i,j,k,UMY) = amrex::max(0.0_rt, state(i,domhi[1],k,UMY));
                    state(i,j,k,UMZ) = 0.0_rt;
                } else if (k < domlo[2]) {
                    state(i,j,k,UMX) = 0.0_rt;
                    state(i,j,k,UMY) = 0.0_rt;
                    state(i,j,k,UMZ) = amrex::min(0.0_rt, state(i,j,domlo[2],UMZ));
                } else if (k > domhi[2]) {
                    state(i,j,k,UMX) = 0.0_rt;
                    state(i,j,k,UMY) = 0.0_rt;
                    state(i,j,k,UMZ) = amrex::max(0.0_rt, state(i,j,domhi[2],UMZ));
                }

                state(i,j,k,UEDEN) = state(i,j,k,UEINT) +
                  0.5_rt * (state(i,j,k,UMX) * state(i,j,k,UMX) +
                            state(i,j,k,UMY) * state(i,j,k,UMY) +
                            state(i,j,k,UMZ) * state(i,j,k,UMZ)) /
                  state(i,j,k,URHO);

            }
        }
    });
}
