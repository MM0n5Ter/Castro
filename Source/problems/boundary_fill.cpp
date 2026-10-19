
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

    const bool ambient_face_x_lo = (hydrosim::ambient_bc_axis == 0 || hydrosim::ambient_bc_axis == -1) &&
                              (bc.lo(0) == amrex::BCType::foextrap || bc.lo(0) == amrex::BCType::hoextrap);
    const bool ambient_face_x_hi = (hydrosim::ambient_bc_axis == 0 || hydrosim::ambient_bc_axis == -1) &&
                              (bc.hi(0) == amrex::BCType::foextrap || bc.hi(0) == amrex::BCType::hoextrap);

#if AMREX_SPACEDIM >= 2
    const bool ambient_face_y_lo = (hydrosim::ambient_bc_axis == 1 || hydrosim::ambient_bc_axis == -1) &&
                              (bc.lo(1) == amrex::BCType::foextrap || bc.lo(1) == amrex::BCType::hoextrap);
    const bool ambient_face_y_hi = (hydrosim::ambient_bc_axis == 1 || hydrosim::ambient_bc_axis == -1) &&
                              (bc.hi(1) == amrex::BCType::foextrap || bc.hi(1) == amrex::BCType::hoextrap);
#endif

#if AMREX_SPACEDIM == 3
    const bool ambient_face_z_lo = (hydrosim::ambient_bc_axis == 2 || hydrosim::ambient_bc_axis == -1) &&
                              (bc.lo(2) == amrex::BCType::foextrap || bc.lo(2) == amrex::BCType::hoextrap);
    const bool ambient_face_z_hi = (hydrosim::ambient_bc_axis == 2 || hydrosim::ambient_bc_axis == -1) &&
                              (bc.hi(2) == amrex::BCType::foextrap || bc.hi(2) == amrex::BCType::hoextrap);
#endif


    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        if ((ambient_face_x_lo && i < domlo[0]) ||
            (ambient_face_x_hi && i > domhi[0])
#if AMREX_SPACEDIM >= 2
            ||
            (ambient_face_y_lo && j < domlo[1]) ||
            (ambient_face_y_hi && j > domhi[1])
#endif
#if AMREX_SPACEDIM == 3
            ||
            (ambient_face_z_lo && k < domlo[2]) ||
            (ambient_face_z_hi && k > domhi[2])
#endif
          ) {
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

    const bool ambient_face_x_lo = (hydrosim::ambient_bc_axis == 0 || hydrosim::ambient_bc_axis == -1) &&
                              (bc.lo(0) == amrex::BCType::foextrap || bc.lo(0) == amrex::BCType::hoextrap);
    const bool ambient_face_x_hi = (hydrosim::ambient_bc_axis == 0 || hydrosim::ambient_bc_axis == -1) &&
                              (bc.hi(0) == amrex::BCType::foextrap || bc.hi(0) == amrex::BCType::hoextrap);

#if AMREX_SPACEDIM >= 2
    const bool ambient_face_y_lo = (hydrosim::ambient_bc_axis == 1 || hydrosim::ambient_bc_axis == -1) &&
                              (bc.lo(1) == amrex::BCType::foextrap || bc.lo(1) == amrex::BCType::hoextrap);
    const bool ambient_face_y_hi = (hydrosim::ambient_bc_axis == 1 || hydrosim::ambient_bc_axis == -1) &&
                              (bc.hi(1) == amrex::BCType::foextrap || bc.hi(1) == amrex::BCType::hoextrap);
#endif

#if AMREX_SPACEDIM == 3
    const bool ambient_face_z_lo = (hydrosim::ambient_bc_axis == 2 || hydrosim::ambient_bc_axis == -1) &&
                              (bc.lo(2) == amrex::BCType::foextrap || bc.lo(2) == amrex::BCType::hoextrap);
    const bool ambient_face_z_hi = (hydrosim::ambient_bc_axis == 2 || hydrosim::ambient_bc_axis == -1) &&
                              (bc.hi(2) == amrex::BCType::foextrap || bc.hi(2) == amrex::BCType::hoextrap);
#endif

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

        if ((ambient_face_x_lo && i < domlo[0]) ||
            (ambient_face_x_hi && i > domhi[0])
#if AMREX_SPACEDIM >= 2
            ||
            (ambient_face_y_lo && j < domlo[1]) ||
            (ambient_face_y_hi && j > domhi[1])
#endif
#if AMREX_SPACEDIM == 3
            ||
            (ambient_face_z_lo && k < domlo[2]) ||
            (ambient_face_z_hi && k > domhi[2])
#endif
          ) {
          state(i,j,k) = ambient::ambient_reference_state[URHO];
        }
    });
}

