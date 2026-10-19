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



#ifdef GRAVITY
void
HydroSim::wave_strain (Real time,
                  Real& strain_plus_1, Real& strain_cross_1,
                  Real& strain_plus_2, Real& strain_cross_2,
                  Real& strain_plus_3, Real& strain_cross_3,
                  bool local) {

    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::wave_strain()");

    if (hydrosim::wave_obs_distance <= 0.0_rt) {
        return;
    }

    constexpr amrex::Real light_speed_quartic = amrex::Math::powi<4>(C::c_light);

    GeometryData geomdata = geom.data();

    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& gravity_new = get_new_data(Accel_Type);

    bool fine_mask_present = level < parent->finestLevel();

    MultiFab scratch_mf;
    const MultiFab& fine_cover_mf = fine_mask_present ? getLevel(level+1).make_fine_mask() : scratch_mf;


    ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum,
              ReduceOpSum, ReduceOpSum, ReduceOpSum,
              ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_op;
    ReduceData<Real, Real, Real,
               Real, Real, Real,
               Real, Real, Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& box = mfi.tilebox();

        auto rho = S_new[mfi].array(URHO);
        auto vol = volume.array(mfi);
        auto xmom = S_new[mfi].array(UMX);
        auto ymom = S_new[mfi].array(UMY);
        auto zmom = S_new[mfi].array(UMZ);
        auto accel_x = gravity_new[mfi].array(0);
        auto accel_y = gravity_new[mfi].array(1);
        auto accel_z = gravity_new[mfi].array(2);
        auto const& mask = fine_mask_present ? fine_cover_mf.array(mfi) : Array4<Real>{};


        reduce_op.eval(box, reduce_data,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            Real zoneWeight = fine_mask_present ? mask(i,j,k) : 1.0_rt;

            GpuArray<Real, 3> r;
            position(i, j, k, geomdata, r);

            for (int n = 0; n < 3; ++n) {
                r[n] -= problem::center[n];
            }

            Real rhoInv;
            if (rho(i,j,k) * zoneWeight > 0.0_rt) {
                rhoInv = 1.0_rt / rho(i,j,k);
            } else {
                rhoInv = 0.0_rt;
            }


            GpuArray<Real, 3> pos{r};
#ifdef ROTATION
            auto omega = get_angular_velocity(geomdata, j);
            pos = rotate_to_inertial_frame(r, omega, time);
#endif


            GpuArray<Real, 3> vel;
            vel[0] = xmom(i,j,k) * zoneWeight * rhoInv;
            vel[1] = ymom(i,j,k) * zoneWeight * rhoInv;
            vel[2] = zmom(i,j,k) * zoneWeight * rhoInv;

            GpuArray<Real, 3> inertial_velocity{vel};
#ifdef ROTATION
            frame_to_inertial_velocity(i, j, k, geomdata, time, inertial_velocity);
#endif

            GpuArray<Real, 3> g;
            g[0] = accel_x(i,j,k) * zoneWeight;
            g[1] = accel_y(i,j,k) * zoneWeight;
            g[2] = accel_z(i,j,k) * zoneWeight;


            GpuArray<Real, 3> inertial_accel{g};
#ifdef ROTATION
            inertial_accel = rotate_to_inertial_frame(g, omega, time);
#endif


            Real dM = 2.0_rt * rho(i,j,k) * vol(i,j,k) * zoneWeight;

            Array2D<Real, 0, 2, 0, 2> quadMomentSecondDeriv{0.0};

            if (AMREX_SPACEDIM == 3) {

                for (int m = 0; m < 3; ++m) {
                    for (int l = 0; l < 3; ++l) {
                        quadMomentSecondDeriv(l,m) += dM * (inertial_velocity[l] * inertial_velocity[m] + pos[l] * inertial_accel[m]);
                    }
                }

            } else {



                dM /= (2.0_rt * std::numbers::pi);

                quadMomentSecondDeriv(0,0) += dM * (2.0_rt * std::numbers::pi) * (inertial_velocity[1] * inertial_velocity[1] + pos[1] * inertial_accel[1]);
                quadMomentSecondDeriv(1,1) += dM * std::numbers::pi * (inertial_velocity[0] * inertial_velocity[0] + pos[0] * g[0]);
                quadMomentSecondDeriv(2,2) += dM * std::numbers::pi * (inertial_velocity[0] * inertial_velocity[0] + pos[0] * g[0]);

            }


            Array2D<Real, 0, 2, 0, 2> dQ;

            for (int l = 0; l < 3; ++l) {
                for (int m = 0; m < 3; ++m) {
                    dQ(l,m) = 0.5_rt * quadMomentSecondDeriv(l,m) + 0.5_rt * quadMomentSecondDeriv(m,l);
                    if (l == m) {
                        dQ(l,m) -= (1.0_rt / 3.0_rt) * quadMomentSecondDeriv(m,m);
                    }
                }
            }

            return {dQ(0,0), dQ(1,0), dQ(2,0),
                    dQ(0,1), dQ(1,1), dQ(2,1),
                    dQ(0,2), dQ(1,2), dQ(2,2)};
        });
    }

    Array2D<Real, 0, 2, 0, 2> Qtt;

    ReduceTuple hv = reduce_data.value();
    Qtt(0,0) = amrex::get<0>(hv);
    Qtt(1,0) = amrex::get<1>(hv);
    Qtt(2,0) = amrex::get<2>(hv);
    Qtt(0,1) = amrex::get<3>(hv);
    Qtt(1,1) = amrex::get<4>(hv);
    Qtt(2,1) = amrex::get<5>(hv);
    Qtt(0,2) = amrex::get<6>(hv);
    Qtt(1,2) = amrex::get<7>(hv);
    Qtt(2,2) = amrex::get<8>(hv);


    if (!local) {
        amrex::ParallelDescriptor::ReduceRealSum(Qtt.begin(), 9);
    }



    Real delta[3][3] = {{0.0}};

    for (int i = 0; i < 3; ++i) {
        delta[i][i] = 1.0;
    }


    for (int dir = 0; dir < 3; ++dir) {

        Real dist[3] = {0.0};
        dist[dir] = hydrosim::wave_obs_distance;

        Real r = std::sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]);

        const Real n[3] = {dist[0] / r, dist[1] / r, dist[2] / r};


        Real proj[3][3][3][3] = {{0.0}};

        for (int l = 0; l < 3; ++l) {
            for (int k = 0; k < 3; ++k) {
                for (int j = 0; j < 3; ++j) {
                    for (int i = 0; i < 3; ++i) {
                        proj[l][k][j][i] = (delta[k][i] - n[i] * n[k]) * (delta[l][j] - n[j] * n[l]) -
                                            0.5_rt * (delta[j][i] - n[i] * n[j]) * (delta[l][k] - n[k] * n[l]);
                    }
                }
            }
        }


        Real h[3][3] = {{0.0}};

        for (int l = 0; l < 3; ++l) {
            for (int k = 0; k < 3; ++k) {
                for (int j = 0; j < 3; ++j) {
                    for (int i = 0; i < 3; ++i) {
                        h[j][i] += proj[l][k][j][i] * Qtt(k, l);
                    }
                }
            }
        }

        r *= C::parsec * 1.e3_rt;

        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                h[j][i] *= 2.0_rt * C::Gconst / (light_speed_quartic * r);
            }
        }


        if (dir == 0) {

            strain_plus_1  += h[1][1];
            strain_cross_1 += h[2][1];

        }
        else if (dir == 1) {

            strain_plus_2  += h[2][2];
            strain_cross_2 += h[0][2];

        }
        else if (dir == 2) {

            strain_plus_3  += h[0][0];
            strain_cross_3 += h[1][0];

        }

    }
}
#endif
