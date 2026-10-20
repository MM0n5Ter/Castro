#include <iomanip>

#include <numbers>

#include <HydroSim.H>
#include <HydroSim_helpers.H>

#ifdef GRAVITY
#include <Gravity.H>
#endif

#ifdef ROTATION
#include <Rotation.H>
#endif

using namespace amrex;

Real
HydroSim::volWeightedTotal (const std::string& name, Real time, bool local, bool exclude_fine)
{
    auto mf = derive(name, time, 0);

    BL_ASSERT(mf);

    return volWeightedTotal(*mf, 0, local, exclude_fine);
}

Real
HydroSim::volWeightedTotal (const MultiFab& mf, int comp, bool local, bool exclude_fine)
{
    BL_PROFILE("HydroSim::volWeightedTotal()");

    bool fine_mask_present = level < parent->finestLevel() && exclude_fine;

    MultiFab scratch_mf;
    const MultiFab& fine_cover_mf = fine_mask_present ? getLevel(level+1).make_fine_mask() : scratch_mf;

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& fab = mf[mfi].array(comp);
        auto const& vol = volume.array(mfi);
        auto const& mask = fine_mask_present ? fine_cover_mf.array(mfi) : Array4<Real>{};

        const Box& box = mfi.tilebox();

        reduce_op.eval(box, reduce_data,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            Real zoneWeight = fine_mask_present ? mask(i,j,k) : 1.0_rt;

            return {fab(i,j,k) * vol(i,j,k) * zoneWeight};
        });

    }

    ReduceTuple hv = reduce_data.value();
    Real sum = amrex::get<0>(hv);

    if (!local) {
        ParallelDescriptor::ReduceRealSum(sum);
    }

    return sum;
}

Real
HydroSim::locWeightedSum (const std::string& name, Real time, int idir, bool local)
{
    auto mf = derive(name, time, 0);

    BL_ASSERT(mf);

    return locWeightedSum(*mf, 0, idir, local);
}

Real
HydroSim::locWeightedSum (const MultiFab& mf, int comp, int idir, bool local)
{
    BL_PROFILE("HydroSim::locWeightedSum()");

    bool fine_mask_present = level < parent->finestLevel();

    MultiFab scratch_mf;
    const MultiFab& fine_cover_mf = fine_mask_present ? getLevel(level+1).make_fine_mask() : scratch_mf;

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    auto dx     = geom.CellSizeArray();
    auto problo = geom.ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& fab = mf[mfi].array(comp);
        auto const& vol = volume.array(mfi);
        auto const& mask = fine_mask_present ? fine_cover_mf.array(mfi) : Array4<Real>{};

        const Box& box = mfi.tilebox();

        reduce_op.eval(box, reduce_data,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            Real zoneWeight = fine_mask_present ? mask(i,j,k) : 1.0_rt;

            Real loc[3];

            loc[0] = problo[0] + (0.5_rt + i) * dx[0];

#if AMREX_SPACEDIM >= 2
            loc[1] = problo[1] + (0.5_rt + j) * dx[1];
#else
            loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
            loc[2] = problo[2] + (0.5_rt + k) * dx[2];
#else
            loc[2] = 0.0_rt;
#endif

            Real ds;

            if (idir == 0) {
                ds = fab(i,j,k) * vol(i,j,k) * zoneWeight * loc[0];
            }
            else if (idir == 1) {
                ds = fab(i,j,k) * vol(i,j,k) * zoneWeight * loc[1];
            }
            else {
                ds = fab(i,j,k) * vol(i,j,k) * zoneWeight * loc[2];
            }

            return {ds};
        });
    }

    ReduceTuple hv = reduce_data.value();
    Real sum = amrex::get<0>(hv);

    if (!local) {
        ParallelDescriptor::ReduceRealSum(sum);
    }

    return sum;
}

Real
HydroSim::volPairTotal (const std::string& name1,
                       const std::string& name2,
                       Real time, bool local)
{
    auto mf1 = derive(name1, time, 0);
    auto mf2 = derive(name2, time, 0);

    BL_ASSERT(mf1);
    BL_ASSERT(mf2);

    return volPairTotal(*mf1, *mf2, 0, 0, local);
}

Real
HydroSim::volPairTotal (const MultiFab& mf1,
                       const MultiFab& mf2,
                       int comp1, int comp2,
                       bool local)
{
    BL_PROFILE("HydroSim::volPairTotal()");

    bool fine_mask_present = level < parent->finestLevel();

    MultiFab scratch_mf;
    const MultiFab& fine_cover_mf = fine_mask_present ? getLevel(level+1).make_fine_mask() : scratch_mf;

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(mf1, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& fab1 = mf1[mfi].array(comp1);
        auto const& fab2 = mf2[mfi].array(comp2);
        auto const& vol  = volume.array(mfi);
        auto const& mask = fine_mask_present ? fine_cover_mf.array(mfi) : Array4<Real>{};

        const Box& box = mfi.tilebox();

        reduce_op.eval(box, reduce_data,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            Real zoneWeight = fine_mask_present ? mask(i,j,k) : 1.0_rt;

            return {fab1(i,j,k) * fab2(i,j,k) * vol(i,j,k) * zoneWeight};
        });
    }

    ReduceTuple hv = reduce_data.value();
    Real sum = amrex::get<0>(hv);

    if (!local) {
        ParallelDescriptor::ReduceRealSum(sum);
    }

    return sum;
}

Real
HydroSim::locDistSquaredTotal (const std::string& name,
                       Real               time,
                       int                idir,
                       bool               local)
{
    BL_PROFILE("HydroSim::locDistSquaredTotal()");

    auto mf = derive(name, time, 0);

    BL_ASSERT(mf);

    bool fine_mask_present = level < parent->finestLevel();

    MultiFab scratch_mf;
    const MultiFab& fine_cover_mf = fine_mask_present ? getLevel(level+1).make_fine_mask() : scratch_mf;

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    auto dx     = geom.CellSizeArray();
    auto problo = geom.ProbLoArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& fab = (*mf).array(mfi);
        auto const& mask = fine_mask_present ? fine_cover_mf.array(mfi) : Array4<Real>{};

        const Box& box = mfi.tilebox();

        reduce_op.eval(box, reduce_data,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            Real zoneWeight = fine_mask_present ? mask(i,j,k) : 1.0_rt;

            Real loc[3];

            loc[0] = problo[0] + (0.5_rt + i) * dx[0];

#if AMREX_SPACEDIM >= 2
            loc[1] = problo[1] + (0.5_rt + j) * dx[1];
#else
            loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
            loc[2] = problo[2] + (0.5_rt + k) * dx[2];
#else
            loc[2] = 0.0_rt;
#endif

            Real ds;

            if (idir == 0) {
                ds = fab(i,j,k) * zoneWeight * loc[0] * loc[0];
            }
            else if (idir == 1) {
                ds = fab(i,j,k) * zoneWeight * loc[1] * loc[1];
            }
            else if (idir == 2) {
                ds = fab(i,j,k) * zoneWeight * loc[2] * loc[2];
            }
            else {
                ds = fab(i,j,k) * zoneWeight * (loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2]);
            }

            return {ds};
        });
    }

    ReduceTuple hv = reduce_data.value();
    Real sum = amrex::get<0>(hv);

    if (!local) {
        ParallelDescriptor::ReduceRealSum(sum);
    }

    return sum;
}
