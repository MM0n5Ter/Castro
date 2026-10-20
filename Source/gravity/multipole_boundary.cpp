#include <cmath>
#include <limits>
#include <numbers>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <AMReX_ParmParse.H>
#include <Gravity.H>
#include <HydroSim.H>

#include <AMReX_FillPatchUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>

#include <hydrosim_limits.H>

#include <fundamental_constants.H>

#include <multipole_moments.H>
#include <SolverUtils.H>

using namespace amrex;

#if (AMREX_SPACEDIM == 3)
void
Gravity::fill_direct_sum_boundaries(int crse_level, int fine_level, const Vector<MultiFab*>& Rhs, MultiFab& phi)
{
    BL_PROFILE("Gravity::fill_direct_sum_boundaries()");

    BL_ASSERT(crse_level==0);

    const Real strt = ParallelDescriptor::second();

    const Geometry& crse_geom = parent->Geom(crse_level);

    const int* domlo = crse_geom.Domain().loVect();
    const int* domhi = crse_geom.Domain().hiVect();

    const int lowIndexXY[3] = {domlo[0]-1, domlo[1]-1, 0         };
    const int highIndexXY[3] = {domhi[0]+1, domhi[1]+1, 0         };

    const int lowIndexXZ[3] = {domlo[0]-1, 0         , domlo[2]-1};
    const int highIndexXZ[3] = {domhi[0]+1, 0         , domhi[2]+1};

    const int lowIndexYZ[3] = {0         , domlo[1]-1, domlo[2]-1};
    const int highIndexYZ[3] = {0         , domhi[1]+1, domhi[2]+1};

    const int bc_lo[3] = {domlo[0]-1, domlo[1]-1, domlo[2]-1};
    const int bc_hi[3] = {domhi[0]+1, domhi[1]+1, domhi[2]+1};

    GpuArray<Real, 3> boundary_spacing;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        boundary_spacing[n] = crse_geom.CellSizeArray()[n];
    }
    for (int n = AMREX_SPACEDIM; n < 3; ++n) {
        boundary_spacing[n] = 0.0_rt;
    }

    GpuArray<Real, 3> problo;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        problo[n] = crse_geom.ProbLoArray()[n];
    }
    for (int n = AMREX_SPACEDIM; n < 3; ++n) {
        problo[n] = 0.0_rt;
    }

    GpuArray<Real, 3> probhi;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        probhi[n] = crse_geom.ProbHiArray()[n];
    }
    for (int n = AMREX_SPACEDIM; n < 3; ++n) {
        probhi[n] = 0.0_rt;
    }

    IntVect lowCornerXY( lowIndexXY );
    IntVect highCornerXY  ( highIndexXY );
    IntVect lowCornerXZ( lowIndexXZ );
    IntVect highCornerXZ  ( highIndexXZ );
    IntVect lowCornerYZ( lowIndexYZ );
    IntVect highCornerYZ  ( highIndexYZ );

    Box faceBoxXY(lowCornerXY, highCornerXY);
    Box faceBoxXZ(lowCornerXZ, highCornerXZ);
    Box faceBoxYZ(lowCornerYZ, highCornerYZ);

    const long pointCountXY = faceBoxXY.numPts();
    const long pointCountXZ = faceBoxXZ.numPts();
    const long pointCountYZ = faceBoxYZ.numPts();

    FArrayBox boundaryXYLo(faceBoxXY);
    FArrayBox boundaryXYHi(faceBoxXY);
    FArrayBox boundaryXZLo(faceBoxXZ);
    FArrayBox boundaryXZHi(faceBoxXZ);
    FArrayBox boundaryYZLo(faceBoxYZ);
    FArrayBox boundaryYZHi(faceBoxYZ);

    boundaryXYLo.setVal<RunOn::Device>(0.0);
    boundaryXYHi.setVal<RunOn::Device>(0.0);
    boundaryXZLo.setVal<RunOn::Device>(0.0);
    boundaryXZHi.setVal<RunOn::Device>(0.0);
    boundaryYZLo.setVal<RunOn::Device>(0.0);
    boundaryYZHi.setVal<RunOn::Device>(0.0);

    int physbc_lo[3];
    int physbc_hi[3];

    for (int dir = 0; dir < 3; dir++) {
        physbc_lo[dir] = phys_bc->lo(dir);
        physbc_hi[dir] = phys_bc->hi(dir);
    }

    for (int lev = crse_level; lev <= fine_level; ++lev) {

        MultiFab source(Rhs[lev - crse_level]->boxArray(),
                        Rhs[lev - crse_level]->DistributionMap(),
                        1, 0);

        MultiFab::Copy(source, *Rhs[lev - crse_level], 0, 0, 1, 0);

        if (lev < fine_level) {
            const MultiFab& mask = dynamic_cast<HydroSim*>(&(parent->getLevel(lev+1)))->make_fine_mask();
            MultiFab::Multiply(source, mask, 0, 0, 1, 0);
        }

        const auto dx = parent->Geom(lev).CellSizeArray();

#ifdef _OPENMP
        int nthreads = omp_get_max_threads();
        Vector<std::unique_ptr<FArrayBox> > local_boundaryXYLo(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_boundaryXYHi(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_boundaryXZLo(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_boundaryXZHi(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_boundaryYZLo(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_boundaryYZHi(nthreads);
        for (int i=0; i<nthreads; i++) {
            local_boundaryXYLo[i].reset(new FArrayBox(faceBoxXY));
            local_boundaryXYHi[i].reset(new FArrayBox(faceBoxXY));
            local_boundaryXZLo[i].reset(new FArrayBox(faceBoxXZ));
            local_boundaryXZHi[i].reset(new FArrayBox(faceBoxXZ));
            local_boundaryYZLo[i].reset(new FArrayBox(faceBoxYZ));
            local_boundaryYZHi[i].reset(new FArrayBox(faceBoxYZ));
        }
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
            local_boundaryXYLo[tid]->setVal<RunOn::Gpu>(0.0);
            local_boundaryXYHi[tid]->setVal<RunOn::Gpu>(0.0);
            local_boundaryXZLo[tid]->setVal<RunOn::Gpu>(0.0);
            local_boundaryXZHi[tid]->setVal<RunOn::Gpu>(0.0);
            local_boundaryYZLo[tid]->setVal<RunOn::Gpu>(0.0);
            local_boundaryYZHi[tid]->setVal<RunOn::Gpu>(0.0);
#endif
            for (MFIter mfi(source, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box bx = mfi.tilebox();

                const auto rho = source[mfi].array();
                const auto vol = (*volume[lev])[mfi].array();

                GpuArray<bool, 3> doFoldedAddLo {false};
                GpuArray<bool, 3> doFoldedAddHi {false};
                bool doFoldedAdd {false};

                for (int b = 0; b < 3; ++b) {
                    if (physbc_lo[b] == amrex::PhysBCType::symmetry) {
                        doFoldedAddLo[b] = true;
                        doFoldedAdd      = true;
                    }

                    if (physbc_hi[b] == amrex::PhysBCType::symmetry) {
                        doFoldedAddHi[b] = true;
                        doFoldedAdd      = true;
                    }
                }

#ifdef _OPENMP
                auto boundaryXYLo_arr = local_boundaryXYLo[tid]->array();
                auto boundaryXYHi_arr = local_boundaryXYHi[tid]->array();
                auto boundaryXZLo_arr = local_boundaryXZLo[tid]->array();
                auto boundaryXZHi_arr = local_boundaryXZHi[tid]->array();
                auto boundaryYZLo_arr = local_boundaryYZLo[tid]->array();
                auto boundaryYZHi_arr = local_boundaryYZHi[tid]->array();
#else
                auto boundaryXYLo_arr = boundaryXYLo.array();
                auto boundaryXYHi_arr = boundaryXYHi.array();
                auto boundaryXZLo_arr = boundaryXZLo.array();
                auto boundaryXZHi_arr = boundaryXZHi.array();
                auto boundaryYZLo_arr = boundaryYZLo.array();
                auto boundaryYZHi_arr = boundaryYZHi.array();
#endif

                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    GpuArray<Real, 3> loc, boundaryLoc;
                    loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];

#if AMREX_SPACEDIM >= 2
                    loc[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1];
#else
                    loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
                    loc[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2];
#else
                    loc[2] = 0.0_rt;
#endif

                    for (int m = bc_lo[1]; m <= bc_hi[1]; ++m) {
                        if (m == bc_lo[1]) {
                            boundaryLoc[1] = problo[1];
                        }
                        else if (m == bc_hi[1]) {
                            boundaryLoc[1] = probhi[1];
                        }
                        else {
                            boundaryLoc[1] = problo[1] + (static_cast<Real>(m) + 0.5_rt) * boundary_spacing[1];
                        }
                        Real dy2 = (loc[1] - boundaryLoc[1]) * (loc[1] - boundaryLoc[1]);

                        for (int l = bc_lo[0]; l <= bc_hi[0]; ++l) {
                            if (l == bc_lo[0]) {
                                boundaryLoc[0] = problo[0];
                            }
                            else if (l == bc_hi[0]) {
                                boundaryLoc[0] = probhi[0];
                            }
                            else {
                                boundaryLoc[0] = problo[0] + (static_cast<Real>(l) + 0.5_rt) * boundary_spacing[0];
                            }
                            Real dx2 = (loc[0] - boundaryLoc[0]) * (loc[0] - boundaryLoc[0]);

                            boundaryLoc[2] = problo[2];
                            Real dz2 = (loc[2] - boundaryLoc[2]) * (loc[2] - boundaryLoc[2]);

                            Real r = std::sqrt(dx2 + dy2 + dz2);

                            Real dbc = -C::Gconst * rho(i,j,k) * vol(i,j,k) / r;

                            if (doFoldedAdd) {

                                dbc += accumulate_mirror_contribution(loc, boundaryLoc, problo, probhi,
                                                                rho(i,j,k), vol(i,j,k),
                                                                doFoldedAddLo, doFoldedAddHi);

                            }

                            Gpu::Atomic::Add(&boundaryXYLo_arr(l,m,0), dbc);

                            boundaryLoc[2] = probhi[2];
                            dz2 = (loc[2] - boundaryLoc[2]) * (loc[2] - boundaryLoc[2]);

                            r = std::sqrt(dx2 + dy2 + dz2);

                            dbc = -C::Gconst * rho(i,j,k) * vol(i,j,k) / r;

                            if (doFoldedAdd) {

                                dbc += accumulate_mirror_contribution(loc, boundaryLoc, problo, probhi,
                                                                rho(i,j,k), vol(i,j,k),
                                                                doFoldedAddLo, doFoldedAddHi);

                            }

                            Gpu::Atomic::Add(&boundaryXYHi_arr(l,m,0), dbc);

                        }

                    }

                    for (int n = bc_lo[2]; n <= bc_hi[2]; ++n) {
                        if (n == bc_lo[2]) {
                            boundaryLoc[2] = problo[2];
                        }
                        else if (n == bc_hi[2]) {
                            boundaryLoc[2] = probhi[2];
                        }
                        else {
                            boundaryLoc[2] = problo[2] + (static_cast<Real>(n) + 0.5_rt) * boundary_spacing[2];
                        }
                        Real dz2 = (loc[2] - boundaryLoc[2]) * (loc[2] - boundaryLoc[2]);

                        for (int l = bc_lo[0]; l <= bc_hi[0]; ++l) {
                            if (l == bc_lo[0]) {
                                boundaryLoc[0] = problo[0];
                            }
                            else if (l == bc_hi[0]) {
                                boundaryLoc[0] = probhi[0];
                            }
                            else {
                                boundaryLoc[0] = problo[0] + (static_cast<Real>(l) + 0.5_rt) * boundary_spacing[0];
                            }
                            Real dx2 = (loc[0] - boundaryLoc[0]) * (loc[0] - boundaryLoc[0]);

                            boundaryLoc[1] = problo[1];
                            Real dy2 = (loc[1] - boundaryLoc[1]) * (loc[1] - boundaryLoc[1]);

                            Real r = std::sqrt(dx2 + dy2 + dz2);

                            Real dbc = -C::Gconst * rho(i,j,k) * vol(i,j,k) / r;

                            if (doFoldedAdd) {

                                dbc += accumulate_mirror_contribution(loc, boundaryLoc, problo, probhi,
                                                                rho(i,j,k), vol(i,j,k),
                                                                doFoldedAddLo, doFoldedAddHi);

                            }

                            Gpu::Atomic::Add(&boundaryXZLo_arr(l,0,n), dbc);

                            boundaryLoc[1] = probhi[1];
                            dy2 = (loc[1] - boundaryLoc[1]) * (loc[1] - boundaryLoc[1]);

                            r = std::sqrt(dx2 + dy2 + dz2);

                            dbc = -C::Gconst * rho(i,j,k) * vol(i,j,k) / r;

                            if (doFoldedAdd) {

                                dbc += accumulate_mirror_contribution(loc, boundaryLoc, problo, probhi,
                                                                rho(i,j,k), vol(i,j,k),
                                                                doFoldedAddLo, doFoldedAddHi);

                            }

                            Gpu::Atomic::Add(&boundaryXZHi_arr(l,0,n), dbc);

                        }

                    }

                    for (int n = bc_lo[2]; n <= bc_hi[2]; ++n) {
                        if (n == bc_lo[2]) {
                            boundaryLoc[2] = problo[2];
                        }
                        else if (n == bc_hi[2]) {
                            boundaryLoc[2] = probhi[2];
                        }
                        else {
                            boundaryLoc[2] = problo[2] + (static_cast<Real>(n) + 0.5_rt) * boundary_spacing[2];
                        }
                        Real dz2 = (loc[2] - boundaryLoc[2]) * (loc[2] - boundaryLoc[2]);

                        for (int m = bc_lo[1]; m <= bc_hi[1]; ++m) {
                            if (m == bc_lo[1]) {
                                boundaryLoc[1] = problo[1];
                            }
                            else if (m == bc_hi[1]) {
                                boundaryLoc[1] = probhi[1];
                            }
                            else {
                                boundaryLoc[1] = problo[1] + (static_cast<Real>(m) + 0.5_rt) * boundary_spacing[1];
                            }
                            Real dy2 = (loc[1] - boundaryLoc[1]) * (loc[1] - boundaryLoc[1]);

                            boundaryLoc[0] = problo[0];
                            Real dx2 = (loc[0] - boundaryLoc[0]) * (loc[0] - boundaryLoc[0]);

                            Real r = std::sqrt(dx2 + dy2 + dz2);

                            Real dbc = -C::Gconst * rho(i,j,k) * vol(i,j,k) / r;

                            if (doFoldedAdd) {

                                dbc += accumulate_mirror_contribution(loc, boundaryLoc, problo, probhi,
                                                                rho(i,j,k), vol(i,j,k),
                                                                doFoldedAddLo, doFoldedAddHi);

                            }

                            Gpu::Atomic::Add(&boundaryYZLo_arr(0,m,n), dbc);

                            boundaryLoc[0] = probhi[0];
                            dx2 = (loc[0] - boundaryLoc[0]) * (loc[0] - boundaryLoc[0]);

                            r = std::sqrt(dx2 + dy2 + dz2);

                            dbc = -C::Gconst * rho(i,j,k) * vol(i,j,k) / r;

                            if (doFoldedAdd) {

                                dbc += accumulate_mirror_contribution(loc, boundaryLoc, problo, probhi,
                                                                rho(i,j,k), vol(i,j,k),
                                                                doFoldedAddLo, doFoldedAddHi);

                            }

                            Gpu::Atomic::Add(&boundaryYZHi_arr(0,m,n), dbc);

                        }

                    }

                });

            }

#ifdef _OPENMP
            Real* boundaryXYLo_data = boundaryXYLo.dataPtr();
            Real* boundaryXYHi_data = boundaryXYHi.dataPtr();
            Real* boundaryXZLo_data = boundaryXZLo.dataPtr();
            Real* boundaryXZHi_data = boundaryXZHi.dataPtr();
            Real* boundaryYZLo_data = boundaryYZLo.dataPtr();
            Real* boundaryYZHi_data = boundaryYZHi.dataPtr();
#pragma omp barrier
#pragma omp for nowait
            for (int i=0; i<pointCountXY; i++) {
                for (int it=0; it<nthreads; it++) {
                    const Real* pl = local_boundaryXYLo[it]->dataPtr();
                    const Real* ph = local_boundaryXYHi[it]->dataPtr();
                    boundaryXYLo_data[i] += pl[i];
                    boundaryXYHi_data[i] += ph[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<pointCountXZ; i++) {
                for (int it=0; it<nthreads; it++) {
                    const Real* pl = local_boundaryXZLo[it]->dataPtr();
                    const Real* ph = local_boundaryXZHi[it]->dataPtr();
                    boundaryXZLo_data[i] += pl[i];
                    boundaryXZHi_data[i] += ph[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<pointCountYZ; i++) {
                for (int it=0; it<nthreads; it++) {
                    const Real* pl = local_boundaryYZLo[it]->dataPtr();
                    const Real* ph = local_boundaryYZHi[it]->dataPtr();
                    boundaryYZLo_data[i] += pl[i];
                    boundaryYZHi_data[i] += ph[i];
                }
            }
#endif
        }

    }

    BL_ASSERT(pointCountXY <= std::numeric_limits<int>::max());
    BL_ASSERT(pointCountXZ <= std::numeric_limits<int>::max());
    BL_ASSERT(pointCountYZ <= std::numeric_limits<int>::max());

    ParallelDescriptor::ReduceRealSum(boundaryXYLo.dataPtr(), static_cast<int>(pointCountXY));
    ParallelDescriptor::ReduceRealSum(boundaryXYHi.dataPtr(), static_cast<int>(pointCountXY));
    ParallelDescriptor::ReduceRealSum(boundaryXZLo.dataPtr(), static_cast<int>(pointCountXZ));
    ParallelDescriptor::ReduceRealSum(boundaryXZHi.dataPtr(), static_cast<int>(pointCountXZ));
    ParallelDescriptor::ReduceRealSum(boundaryYZLo.dataPtr(), static_cast<int>(pointCountYZ));
    ParallelDescriptor::ReduceRealSum(boundaryYZHi.dataPtr(), static_cast<int>(pointCountYZ));

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx= mfi.growntilebox();

        auto p = phi[mfi].array();

        auto boundaryXYLo_arr = boundaryXYLo.array();
        auto boundaryXYHi_arr = boundaryXYHi.array();
        auto boundaryXZLo_arr = boundaryXZLo.array();
        auto boundaryXZHi_arr = boundaryXZHi.array();
        auto boundaryYZLo_arr = boundaryYZLo.array();
        auto boundaryYZHi_arr = boundaryYZHi.array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (i == bc_lo[0]) {
                p(i,j,k) = boundaryYZLo_arr(0,j,k);
            }

            if (i == bc_hi[0]) {
                p(i,j,k) = boundaryYZHi_arr(0,j,k);
            }

            if (j == bc_lo[1]) {
                p(i,j,k) = boundaryXZLo_arr(i,0,k);
            }

            if (j == bc_hi[1]) {
                p(i,j,k) = boundaryXZHi_arr(i,0,k);
            }

            if (k == bc_lo[2]) {
                p(i,j,k) = boundaryXYLo_arr(i,j,0);
            }

            if (k == bc_hi[2]) {
                p(i,j,k) = boundaryXYHi_arr(i,j,0);
            }
        });
    }

    if (gravity::verbose)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        Real      end    = ParallelDescriptor::second() - strt;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(end,IOProc);
        amrex::Print() << "Gravity::fill_direct_sum_boundaries() time = " << end << std::endl << std::endl;
#ifdef BL_LAZY
        });
#endif
    }

}
#endif

void
Gravity::init_multipole_field() const
{
    if (gravity::lnum < 0) {
        amrex::Abort("lnum negative");
    }

    if (gravity::lnum > multipole::max_moment_order) {
        amrex::Abort("lnum greater than max_moment_order");
    }

    int lo_bc[3];
    int hi_bc[3];

    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
      lo_bc[dir] = phys_bc->lo(dir);
      hi_bc[dir] = phys_bc->hi(dir);
    }
    for (int dir = AMREX_SPACEDIM; dir < 3; dir++)
    {
      lo_bc[dir] = -1;
      hi_bc[dir] = -1;
    }

    const auto problo = parent->Geom(0).ProbLoArray();
    const auto probhi = parent->Geom(0).ProbHiArray();

    multipole::volumeScale = 1.0_rt;
    multipole::parityScale = 1.0_rt;

    multipole::doFoldedAdd = false;

    for (int n = 0; n < 3; ++n) {
        multipole::doFoldedAddLo(n) = false;
        multipole::doFoldedAddHi(n) = false;

        multipole::doMirrorLo(n) = false;
        multipole::doMirrorHi(n) = false;
    }

    const Real wallDistTol = 1.0e-2_rt;

    for (int b = 0; b < AMREX_SPACEDIM; ++b) {

        if ((lo_bc[b] == amrex::PhysBCType::symmetry) && (parent->Geom(0).Coord() == 0)) {
            if (!(std::abs(problem::center[b] - problo[b]) < wallDistTol)) {
                multipole::doFoldedAddLo(b) = true;
                multipole::doFoldedAdd      = true;
            } else {
                multipole::volumeScale *= 2.0_rt;
                multipole::doMirrorLo(b) = true;
            }
        }

        if ((hi_bc[b] == amrex::PhysBCType::symmetry) && (parent->Geom(0).Coord() == 0)) {
            if (std::abs(problem::center[b] - probhi[b]) < wallDistTol) {
                multipole::volumeScale *= 2.0_rt;
                multipole::doMirrorHi(b) = true;
            }
            else {
                multipole::doFoldedAddHi(b) = true;
                multipole::doFoldedAdd      = true;
            }
        }

    }

    for (int l = 0; l <= multipole::max_moment_order; ++l) {
        multipole::axis_parity_table(l) = 1.0_rt;
        for (int m = 0; m <= multipole::max_moment_order; ++m) {
            multipole::factorialTable(l,m) = 0.0_rt;
            multipole::azimuth_parity_table(l,m) = 1.0_rt;
        }
    }

    for (int l = 0; l <= multipole::max_moment_order; ++l) {

        multipole::axis_parity_table(l) = 1.0_rt;

        if (l % 2 != 0) {
            if (AMREX_SPACEDIM == 3 && (multipole::doMirrorLo(2) || multipole::doMirrorHi(2))) {
                multipole::axis_parity_table(l) = 0.0_rt;
            }
            else if (AMREX_SPACEDIM == 2 && parent->Geom(0).Coord() == 1) {
                multipole::axis_parity_table(l) = 0.0_rt;
            }
            else if (AMREX_SPACEDIM == 1 && parent->Geom(0).Coord() == 2) {
                multipole::axis_parity_table(l) = 0.0_rt;
            }
        }

        if (l > 0) {
            if (AMREX_SPACEDIM == 1 && parent->Geom(0).Coord() == 2) {
                multipole::axis_parity_table(l) = 0.0_rt;
            }
        }

        for (int m = 1; m <= l; ++m) {

            if (AMREX_SPACEDIM == 3) {
                multipole::azimuth_parity_table(l,m) = 1.0_rt;
            }
            else if (AMREX_SPACEDIM == 2 && parent->Geom(0).Coord() == 1) {
                multipole::azimuth_parity_table(l,m) = 0.0_rt;
            }
            else if (AMREX_SPACEDIM == 1 && parent->Geom(0).Coord() == 2) {
                multipole::azimuth_parity_table(l,m) = 0.0_rt;
            }

            if ((l+m) % 2 != 0 && (multipole::doMirrorLo(2) || multipole::doMirrorHi(2))) {
                multipole::azimuth_parity_table(l,m) = 0.0_rt;
            }

            if (multipole::doMirrorLo(0) || multipole::doMirrorLo(1) || multipole::doMirrorHi(0) || multipole::doMirrorHi(1)) {
                multipole::azimuth_parity_table(l,m) = 0.0_rt;
            }

            multipole::factorialTable(l,m) = 2.0_rt * factorial(l-m) / factorial(l+m) * multipole::volumeScale;

        }

    }

    Real maxWidth = probhi[0] - problo[0];
    if (AMREX_SPACEDIM >= 2) {
        maxWidth = amrex::max(maxWidth, probhi[1] - problo[1]);
    }
    if (AMREX_SPACEDIM == 3) {
        maxWidth = amrex::max(maxWidth, probhi[2] - problo[2]);
    }

    multipole::rmax = 0.5_rt * maxWidth * std::sqrt(static_cast<Real>(AMREX_SPACEDIM));
}

void
Gravity::fill_multipole_boundaries(int crse_level, int fine_level, const Vector<MultiFab*>& Rhs, MultiFab& phi)
{
    BL_PROFILE("Gravity::fill_multipole_boundaries()");

    BL_ASSERT(crse_level == 0);

    BL_ASSERT(gravity::lnum >= 0);

    const Real strt = ParallelDescriptor::second();

#if (AMREX_SPACEDIM == 3)
    const int npts = level_point_count;
#else
    const int npts = 1;
#endif

    Box momentBox0( IntVect(AMREX_D_DECL(0, 0, 0)), IntVect(AMREX_D_DECL(gravity::lnum, 0,    npts-1)) );
    Box momentBoxC( IntVect(AMREX_D_DECL(0, 0, 0)), IntVect(AMREX_D_DECL(gravity::lnum, gravity::lnum, npts-1)) );
    Box momentBoxS( IntVect(AMREX_D_DECL(0, 0, 0)), IntVect(AMREX_D_DECL(gravity::lnum, gravity::lnum, npts-1)) );

    FArrayBox qL0(momentBox0);
    FArrayBox qLC(momentBoxC);
    FArrayBox qLS(momentBoxS);

    FArrayBox qU0(momentBox0);
    FArrayBox qUC(momentBoxC);
    FArrayBox qUS(momentBoxS);

    qL0.setVal<RunOn::Device>(0.0);
    qLC.setVal<RunOn::Device>(0.0);
    qLS.setVal<RunOn::Device>(0.0);
    qU0.setVal<RunOn::Device>(0.0);
    qUC.setVal<RunOn::Device>(0.0);
    qUS.setVal<RunOn::Device>(0.0);

#if (AMREX_SPACEDIM == 3)
    int surface_only_mode = 1;
#else
    const int surface_only_mode = 1;
#endif

    for (int lev = crse_level; lev <= fine_level; ++lev) {

        MultiFab source(Rhs[lev - crse_level]->boxArray(),
                        Rhs[lev - crse_level]->DistributionMap(), 1, 0);

        MultiFab::Copy(source, *Rhs[lev - crse_level], 0, 0, 1, 0);

        if (lev < fine_level) {
        auto *hydrosim_level = dynamic_cast<HydroSim*>(&(parent->getLevel(lev+1)));
        if (hydrosim_level != nullptr) {
        const MultiFab& mask = hydrosim_level->make_fine_mask();
        MultiFab::Multiply(source, mask, 0, 0, 1, 0);
        } else {
                amrex::Abort("unable to access mask");
            }
        }

        const auto dx = parent->Geom(lev).CellSizeArray();
        const auto problo = parent->Geom(lev).ProbLoArray();
        const auto probhi = parent->Geom(lev).ProbHiArray();
        int coord_type = parent->Geom(lev).Coord();

#ifdef _OPENMP
        int nthreads = omp_get_max_threads();
        Vector<std::unique_ptr<FArrayBox> > local_lowerMoment0(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_lowerMomentC(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_lowerMomentS(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_upperMoment0(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_upperMomentC(nthreads);
        Vector<std::unique_ptr<FArrayBox> > local_upperMomentS(nthreads);
        for (int i=0; i<nthreads; i++) {
            local_lowerMoment0[i].reset(new FArrayBox(momentBox0));
            local_lowerMomentC[i].reset(new FArrayBox(momentBoxC));
            local_lowerMomentS[i].reset(new FArrayBox(momentBoxS));
            local_upperMoment0[i].reset(new FArrayBox(momentBox0));
            local_upperMomentC[i].reset(new FArrayBox(momentBoxC));
            local_upperMomentS[i].reset(new FArrayBox(momentBoxS));
        }
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
            local_lowerMoment0[tid]->setVal<RunOn::Device>(0.0);
            local_lowerMomentC[tid]->setVal<RunOn::Device>(0.0);
            local_lowerMomentS[tid]->setVal<RunOn::Device>(0.0);
            local_upperMoment0[tid]->setVal<RunOn::Device>(0.0);
            local_upperMomentC[tid]->setVal<RunOn::Device>(0.0);
            local_upperMomentS[tid]->setVal<RunOn::Device>(0.0);
#endif
            for (MFIter mfi(source, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();

#ifdef _OPENMP
                auto lowerMoment0_arr = local_lowerMoment0[tid]->array();
                auto lowerMomentC_arr = local_lowerMomentC[tid]->array();
                auto lowerMomentS_arr = local_lowerMomentS[tid]->array();
                auto upperMoment0_arr = local_upperMoment0[tid]->array();
                auto upperMomentC_arr = local_upperMomentC[tid]->array();
                auto upperMomentS_arr = local_upperMomentS[tid]->array();
#else
                auto lowerMoment0_arr = qL0.array();
                auto lowerMomentC_arr = qLC.array();
                auto lowerMomentS_arr = qLS.array();
                auto upperMoment0_arr = qU0.array();
                auto upperMomentC_arr = qUC.array();
                auto upperMomentS_arr = qUS.array();
#endif

                auto rho = source[mfi].array();
                auto vol = (*volume[lev])[mfi].array();

                amrex::ParallelFor(amrex::Gpu::KernelInfo().setReduction(true), bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::Gpu::Handler const& handler) noexcept
                {

                    int nlo = 0;
                    if (surface_only_mode == 1) {
                        nlo = npts-1;
                    }

                    Real binScale = multipole::rmax / dx[0];

                    Real cutoff_cubed_inv = 1.0_rt / (multipole::rmax * multipole::rmax * multipole::rmax);

                    Real x = (problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0]) / multipole::rmax;

#if AMREX_SPACEDIM >= 2
                    Real y = (problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1]) / multipole::rmax;
#else
                    Real y = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
                    Real z = (problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2]) / multipole::rmax;
#else
                    Real z = 0.0_rt;
#endif

                    Real r = std::sqrt(x * x + y * y + z * z);

                    Real cosPolar{}, sampleAzimuth{};
                    int index{};

                    if (AMREX_SPACEDIM == 3) {
                        index = static_cast<int>(r * binScale);
                        cosPolar = z / r;
                        sampleAzimuth = std::atan2(y, x);
                    }
                    else if (AMREX_SPACEDIM == 2 && coord_type == 1) {
                        index = nlo;
                        cosPolar = y / r;
                        sampleAzimuth = z;
                    }
                    else if (AMREX_SPACEDIM == 1 && coord_type == 2) {
                        index = nlo;
                        cosPolar = 1.0_rt;
                        sampleAzimuth = 0.0_rt;
                    }

                    accumulate_moments(cosPolar, sampleAzimuth, r, rho(i,j,k), vol(i,j,k) * cutoff_cubed_inv,
                                  lowerMoment0_arr, lowerMomentC_arr, lowerMomentS_arr, upperMoment0_arr, upperMomentC_arr, upperMomentS_arr,
                                  npts, nlo, index, handler, true);

                    if (multipole::doFoldedAdd) {

                        accumulate_mirror_moments(x, y, z, problo, probhi,
                                                rho(i,j,k), vol(i,j,k) * cutoff_cubed_inv,
                                                lowerMoment0_arr, lowerMomentC_arr, lowerMomentS_arr, upperMoment0_arr, upperMomentC_arr, upperMomentS_arr,
                                                npts, nlo, index, handler);

                    }
                });
            }

#ifdef _OPENMP
            int np0 = momentBox0.numPts();
            int npC = momentBoxC.numPts();
            int npS = momentBoxS.numPts();
            Real* pL0 = qL0.dataPtr();
            Real* pLC = qLC.dataPtr();
            Real* pLS = qLS.dataPtr();
            Real* pU0 = qU0.dataPtr();
            Real* pUC = qUC.dataPtr();
            Real* pUS = qUS.dataPtr();
#pragma omp barrier
#pragma omp for nowait
            for (int i=0; i<np0; ++i)
            {
                for (int it=0; it<nthreads; it++) {
                    const Real* pp = local_lowerMoment0[it]->dataPtr();
                    pL0[i] += pp[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<npC; ++i)
            {
                for (int it=0; it<nthreads; it++) {
                    const Real* pp = local_lowerMomentC[it]->dataPtr();
                    pLC[i] += pp[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<npS; ++i)
            {
                for (int it=0; it<nthreads; it++) {
                    const Real* pp = local_lowerMomentS[it]->dataPtr();
                    pLS[i] += pp[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<np0; ++i)
            {
                for (int it=0; it<nthreads; it++) {
                  const Real* pp = local_upperMoment0[it]->dataPtr();
                  pU0[i] += pp[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<npC; ++i)
            {
                for (int it=0; it<nthreads; it++) {
                  const Real* pp = local_upperMomentC[it]->dataPtr();
                  pUC[i] += pp[i];
                }
            }
#pragma omp for nowait
            for (int i=0; i<npS; ++i)
            {
                for (int it=0; it<nthreads; it++) {
                    const Real* pp = local_upperMomentS[it]->dataPtr();
                    pUS[i] += pp[i];
                }
            }
#endif

        }

    }

    Real* lowerMoment0_ptr = qL0.dataPtr();
    Real* lowerMomentC_ptr = qLC.dataPtr();
    Real* lowerMomentS_ptr = qLS.dataPtr();

    FArrayBox lowerMoment0_host(The_Pinned_Arena());
    FArrayBox lowerMomentC_host(The_Pinned_Arena());
    FArrayBox lowerMomentS_host(The_Pinned_Arena());

    if (!ParallelDescriptor::UseGpuAwareMpi()) {
        if (The_Arena() == The_Managed_Arena()) {
            qL0.prefetchToHost();
            qLC.prefetchToHost();
            qLS.prefetchToHost();
        }
        else if (The_Arena() == The_Device_Arena()) {
            lowerMoment0_host.resize(momentBox0);
            lowerMomentC_host.resize(momentBoxC);
            lowerMomentS_host.resize(momentBoxS);

            lowerMoment0_host.copy<RunOn::Device>(qL0, momentBox0);
            lowerMomentC_host.copy<RunOn::Device>(qLC, momentBoxC);
            lowerMomentS_host.copy<RunOn::Device>(qLS, momentBoxS);

            lowerMoment0_ptr = lowerMoment0_host.dataPtr();
            lowerMomentC_ptr = lowerMomentC_host.dataPtr();
            lowerMomentS_ptr = lowerMomentS_host.dataPtr();
        }
    }

    Gpu::synchronize();

    ParallelDescriptor::ReduceRealSum(lowerMoment0_ptr, static_cast<int>(momentBox0.numPts()));
    ParallelDescriptor::ReduceRealSum(lowerMomentC_ptr, static_cast<int>(momentBoxC.numPts()));
    ParallelDescriptor::ReduceRealSum(lowerMomentS_ptr, static_cast<int>(momentBoxS.numPts()));

    if (!ParallelDescriptor::UseGpuAwareMpi()) {
        if (The_Arena() == The_Managed_Arena()) {
            qL0.prefetchToDevice();
            qLC.prefetchToDevice();
            qLS.prefetchToDevice();
        }
        else if (The_Arena() == The_Device_Arena()) {
            qL0.copy<RunOn::Device>(lowerMoment0_host, momentBox0);
            qLC.copy<RunOn::Device>(lowerMomentC_host, momentBoxC);
            qLS.copy<RunOn::Device>(lowerMomentS_host, momentBoxS);
        }
    }

    if (surface_only_mode != 1) {

        Real* upperMoment0_ptr = qU0.dataPtr();
        Real* upperMomentC_ptr = qUC.dataPtr();
        Real* upperMomentS_ptr = qUS.dataPtr();

        FArrayBox upperMoment0_host(The_Pinned_Arena());
        FArrayBox upperMomentC_host(The_Pinned_Arena());
        FArrayBox upperMomentS_host(The_Pinned_Arena());

        if (!ParallelDescriptor::UseGpuAwareMpi()) {
            if (The_Arena() == The_Managed_Arena()) {
                qU0.prefetchToHost();
                qUC.prefetchToHost();
                qUS.prefetchToHost();
            }
            else if (The_Arena() == The_Device_Arena()) {
                upperMoment0_host.resize(momentBox0);
                upperMomentC_host.resize(momentBoxC);
                upperMomentS_host.resize(momentBoxS);

                upperMoment0_host.copy<RunOn::Device>(qU0, momentBox0);
                upperMomentC_host.copy<RunOn::Device>(qUC, momentBoxC);
                upperMomentS_host.copy<RunOn::Device>(qUS, momentBoxS);

                upperMoment0_ptr = upperMoment0_host.dataPtr();
                upperMomentC_ptr = upperMomentC_host.dataPtr();
                upperMomentS_ptr = upperMomentS_host.dataPtr();
            }
        }

        Gpu::synchronize();

        ParallelDescriptor::ReduceRealSum(upperMoment0_ptr, static_cast<int>(momentBox0.numPts()));
        ParallelDescriptor::ReduceRealSum(upperMomentC_ptr, static_cast<int>(momentBoxC.numPts()));
        ParallelDescriptor::ReduceRealSum(upperMomentS_ptr, static_cast<int>(momentBoxS.numPts()));

        if (!ParallelDescriptor::UseGpuAwareMpi()) {
            if (The_Arena() == The_Managed_Arena()) {
                qU0.prefetchToDevice();
                qUC.prefetchToDevice();
                qUS.prefetchToDevice();
            }
            else if (The_Arena() == The_Device_Arena()) {
                qU0.copy<RunOn::Device>(upperMoment0_host, momentBox0);
                qUC.copy<RunOn::Device>(upperMomentC_host, momentBoxC);
                qUS.copy<RunOn::Device>(upperMomentS_host, momentBoxS);
            }
        }

    }

    const Box& domain = parent->Geom(crse_level).Domain();
    const auto dx = parent->Geom(crse_level).CellSizeArray();
    const auto problo = parent->Geom(crse_level).ProbLoArray();
    int coord_type = parent->Geom(crse_level).Coord();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox();

        auto lowerMoment0_arr = qL0.array();
        auto lowerMomentC_arr = qLC.array();
        auto lowerMomentS_arr = qLS.array();
        auto phi_arr = phi[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const int* domlo = domain.loVect();
            const int* domhi = domain.hiVect();

            int nlo = 0;
            if (surface_only_mode == 1) {
                nlo = npts-1;
            }

            Real cutoff_cubed = multipole::rmax * multipole::rmax * multipole::rmax;

            Real x;
            if (i > domhi[0]) {
                x = problo[0] + (static_cast<Real>(i  )         ) * dx[0] - problem::center[0];
            }
            else if (i < domlo[0]) {
                x = problo[0] + (static_cast<Real>(i+1)         ) * dx[0] - problem::center[0];
            }
            else {
                x = problo[0] + (static_cast<Real>(i  ) + 0.5_rt) * dx[0] - problem::center[0];
            }

            x = x / multipole::rmax;

#if AMREX_SPACEDIM >= 2
            Real y;
            if (j > domhi[1]) {
                y = problo[1] + (static_cast<Real>(j  )         ) * dx[1] - problem::center[1];
            }
            else if (j < domlo[1]) {
                y = problo[1] + (static_cast<Real>(j+1)         ) * dx[1] - problem::center[1];
            }
            else {
                y = problo[1] + (static_cast<Real>(j  ) + 0.5_rt) * dx[1] - problem::center[1];
            }
#else
            Real y = 0.0_rt;
#endif

            y = y / multipole::rmax;

#if AMREX_SPACEDIM == 3
            Real z;
            if (k > domhi[2]) {
                z = problo[2] + (static_cast<Real>(k  )         ) * dx[2] - problem::center[2];
            }
            else if (k < domlo[2]) {
                z = problo[2] + (static_cast<Real>(k+1)         ) * dx[2] - problem::center[2];
            }
            else {
                z = problo[2] + (static_cast<Real>(k  ) + 0.5_rt) * dx[2] - problem::center[2];
            }
#else
            Real z = 0.0;
#endif

            z = z / multipole::rmax;

            if (i < domlo[0] || i > domhi[0]
#if AMREX_SPACEDIM >= 2
                || j < domlo[1] || j > domhi[1]
#endif
#if AMREX_SPACEDIM >= 3
                || k < domlo[2] || k > domhi[2]
#endif
                ) {

                Real r = std::sqrt(x * x + y * y + z * z);

                if (r < 1.0e-12_rt) {
                    phi_arr(i,j,k) = 0.0_rt;
                    return;
                }

                Real cosPolar{}, sampleAzimuth{};
                if (AMREX_SPACEDIM == 3) {
                    cosPolar = z / r;
                    sampleAzimuth = std::atan2(y, x);
                }
                else if (AMREX_SPACEDIM == 2 && coord_type == 1) {
                    cosPolar = y / r;
                    sampleAzimuth = 0.0_rt;
                }

                phi_arr(i,j,k) = 0.0_rt;

                Real legendreValue, legendreValue1, legendreValue2;
                Real assocLegendreValue, assocLegendreValue1, assocLegendreValue2;

                for (int n = nlo; n <= npts - 1; ++n) {

                    for (int l = 0; l <= gravity::lnum; ++l) {

                        evaluateLegendreByDegree(l, legendreValue, legendreValue1, legendreValue2, cosPolar);

                        Real r_U = std::pow(r, -l-1);

                        phi_arr(i,j,k) += lowerMoment0_arr(l,0,n) * legendreValue * r_U * cutoff_cubed;

                    }

                    for (int m = 1; m <= gravity::lnum; ++m) {
                        for (int l = 1; l <= gravity::lnum; ++l) {

                            if (m > l) {
                                continue;
                            }

                            evaluateAssociatedLegendre(l, m, assocLegendreValue, assocLegendreValue1, assocLegendreValue2, cosPolar);

                            Real r_U = std::pow(r, -l-1);

                            phi_arr(i,j,k) += (lowerMomentC_arr(l,m,n) * std::cos(m * sampleAzimuth) + lowerMomentS_arr(l,m,n) * std::sin(m * sampleAzimuth)) *
                                              assocLegendreValue * r_U * cutoff_cubed;

                        }
                    }

                }

                phi_arr(i,j,k) = -C::Gconst * phi_arr(i,j,k) / multipole::rmax;
            }
        });
    }

    if (gravity::verbose)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        Real      end    = ParallelDescriptor::second() - strt;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(end,IOProc);
        amrex::Print() << "Gravity::fill_multipole_boundaries() time = " << end << std::endl << std::endl;
#ifdef BL_LAZY
        });
#endif
    }

}
