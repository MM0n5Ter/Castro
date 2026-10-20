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

static void combine_level_bins (Real* total, const Vector<RealVector>& bins, int level, int n1d, Amr* parent)
{
    const Real* const target_bins = bins[level].dataPtr();

    amrex::ParallelFor(n1d,
    [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        total[i] = target_bins[i];
    });

    if (level > 0)
    {
        int ratio = parent->refRatio(level-1)[0];
        for (int lev = level-1; lev >= 0; lev--)
        {
            if (lev < level-1) {
                ratio *= parent->refRatio(lev)[0];
            }

            const Real* const cur_bins = bins[lev].dataPtr();

            amrex::ParallelFor(n1d/ratio,
            [=] AMREX_GPU_DEVICE (int i) noexcept
            {
                for (int n = 0; n < ratio; n++)
                {
                    total[ratio*i+n] += 1./double(ratio) * cur_bins[i];
                }
            });
        }
    }
}

void
Gravity::build_radial_field(int level, Real time, RealVector& radial_field)
{
    BL_PROFILE("Gravity::build_radial_field()");

    const Real strt = ParallelDescriptor::second();

    int diag_print_on = 0;

    Real total_over_levels = 0.;

    for (int lev = 0; lev <= level; lev++)
    {
        const Real t_old = LevelTable[lev]->get_state_data(State_Type).prevTime();
        const Real t_new = LevelTable[lev]->get_state_data(State_Type).curTime();
        const Real eps   = (t_new - t_old) * 1.e-6;

        MultiFab S(grids[lev],dmap[lev],NUM_STATE,0);

        if ( eps == 0.0 ) {
            MultiFab::Copy(S, LevelTable[lev]->get_new_data(State_Type), 0, 0, NUM_STATE, 0);
        }
        else if ( std::abs(time-t_old) < eps)
        {
            MultiFab::Copy(S, LevelTable[lev]->get_old_data(State_Type), 0, 0, NUM_STATE, 0);
        }
        else if ( std::abs(time-t_new) < eps)
        {
            MultiFab::Copy(S, LevelTable[lev]->get_new_data(State_Type), 0, 0, NUM_STATE, 0);
        }
        else if (time > t_old && time < t_new)
        {
            Real alpha   = (time - t_old)/(t_new - t_old);
            Real oneMinusAlpha = 1.0 - alpha;

            MultiFab::Copy(S, LevelTable[lev]->get_old_data(State_Type), 0, 0, NUM_STATE, 0);
            S.mult(oneMinusAlpha);

            MultiFab S_new(grids[lev],dmap[lev],NUM_STATE,0);
            MultiFab::Copy(S_new, LevelTable[lev]->get_new_data(State_Type), 0, 0, NUM_STATE, 0);
            S_new.mult(alpha);

            S.plus(S_new,0,NUM_STATE,0);
        }
        else
        {
            std::cout << " Level / Time in build_radial_field is: " << lev << " " << time  << std::endl;
            std::cout << " but old / new time      are: " << t_old << " " << t_new << std::endl;
            amrex::Abort("Problem in Gravity::build_radial_field");
        }

        if (lev < level)
        {
            auto* fine_level = dynamic_cast<HydroSim*>(&(parent->getLevel(lev+1)));
        if (fine_level != nullptr) {
        const MultiFab& mask = fine_level->make_fine_mask();
        for (int n = 0; n < NUM_STATE; ++n) {
            MultiFab::Multiply(S, mask, 0, n, 1, 0);
        }
        } else {
                amrex::Abort("unable to create mask");
            }
        }

        int n1d = static_cast<int>(radial_shell_mass[lev].size());

#ifdef GR_GRAV
        Real* const cur_pres_bins = radial_shell_pres[lev].dataPtr();
#endif
        Real* const cur_vol_bins = radial_shell_vol[lev].dataPtr();
        Real* const cur_mass_bins = radial_shell_mass[lev].dataPtr();

        amrex::ParallelFor(n1d,
        [=] AMREX_GPU_DEVICE (int i) noexcept
        {
#ifdef GR_GRAV
            cur_pres_bins[i] = 0.;
#endif
            cur_vol_bins[i] = 0.;
            cur_mass_bins[i] = 0.;
        });

#ifdef _OPENMP
        int nthreads = omp_get_max_threads();
#ifdef GR_GRAV
        Vector< RealVector > local_radial_pres(nthreads);
#endif
        Vector< RealVector > local_radial_mass(nthreads);
        Vector< RealVector > local_radial_vol (nthreads);
        for (int i=0; i<nthreads; i++) {
#ifdef GR_GRAV
            local_radial_pres[i].resize(n1d,0.0);
#endif
            local_radial_mass[i].resize(n1d,0.0);
            local_radial_vol [i].resize(n1d,0.0);
        }
#pragma omp parallel
#endif
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
#endif
            for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                FArrayBox& fab = S[mfi];

                tally_radial_profile(bx,
                                    fab.array(),
#ifdef _OPENMP
                                    local_radial_mass[tid],
                                    local_radial_vol[tid],
#ifdef GR_GRAV
                                    local_radial_pres[tid],
#endif
#else
                                    radial_shell_mass[lev],
                                    radial_shell_vol[lev],
#ifdef GR_GRAV
                                    radial_shell_pres[lev],
#endif
#endif
                                    n1d, lev);
            }

#ifdef _OPENMP
#pragma omp barrier
#pragma omp for
            for (int i=0; i<n1d; i++) {
                for (int it=0; it<nthreads; it++) {
#ifdef GR_GRAV
                    radial_shell_pres[lev][i] += local_radial_pres[it][i];
#endif
                    radial_shell_mass[lev][i] += local_radial_mass[it][i];
                    radial_shell_vol [lev][i] += local_radial_vol [it][i];
                }
            }
#endif
        }

        if (!ParallelDescriptor::UseGpuAwareMpi()) {
            Gpu::prefetchToHost(radial_shell_mass[lev].begin(), radial_shell_mass[lev].end());
            Gpu::prefetchToHost(radial_shell_vol[lev].begin(), radial_shell_vol[lev].end());
#ifdef GR_GRAV
            Gpu::prefetchToHost(radial_shell_pres[lev].begin(), radial_shell_pres[lev].end());
#endif
        }

        ParallelDescriptor::ReduceRealSum(radial_shell_mass[lev].dataPtr() ,n1d);
        ParallelDescriptor::ReduceRealSum(radial_shell_vol[lev].dataPtr()  ,n1d);
#ifdef GR_GRAV
        ParallelDescriptor::ReduceRealSum(radial_shell_pres[lev].dataPtr()  ,n1d);
#endif

        if (!ParallelDescriptor::UseGpuAwareMpi()) {
            Gpu::prefetchToDevice(radial_shell_mass[lev].begin(), radial_shell_mass[lev].end());
            Gpu::prefetchToDevice(radial_shell_vol[lev].begin(), radial_shell_vol[lev].end());
#ifdef GR_GRAV
            Gpu::prefetchToDevice(radial_shell_pres[lev].begin(), radial_shell_pres[lev].end());
#endif
        }

        if (diag_print_on > 0)
        {
            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            reduce_op.eval(n1d, reduce_data,
            [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
            {
                return {cur_mass_bins[i]};
            });

            ReduceTuple hv = reduce_data.value();
            Real sum = amrex::get<0>(hv);

            total_over_levels += sum;
        }
    }

    if (diag_print_on > 0) {
        amrex::Print() << "Gravity::build_radial_field: Sum of mass over all levels " << total_over_levels << std::endl;
    }

    int n1d = static_cast<int>(radial_shell_mass[level].size());
    RealVector radial_mass_total(n1d,0);

    Real* const mass_total = radial_mass_total.dataPtr();

    combine_level_bins(mass_total, radial_shell_mass, level, n1d, parent);

    if (diag_print_on > 0 && ParallelDescriptor::IOProcessor())
    {
        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

        reduce_op.eval(n1d, reduce_data,
        [=] AMREX_GPU_DEVICE (int i) -> ReduceTuple
        {
            return {mass_total[i]};
        });

        ReduceTuple hv = reduce_data.value();
        Real added_total = amrex::get<0>(hv);

        std::cout << "Gravity::build_radial_field: Sum of combined mass " << added_total << std::endl;
    }

    const Geometry& geom = parent->Geom(level);
    const Real* dx = geom.CellSize();
    Real dr        = dx[0] / static_cast<Real>(gravity::radial_bin_ratio);

    RealVector radial_vol_total(n1d,0);
    RealVector radial_density_total(n1d,0);

    Real* const vol_total = radial_vol_total.dataPtr();
    Real* const density_total = radial_density_total.dataPtr();

    combine_level_bins(vol_total, radial_shell_vol, level, n1d, parent);

    amrex::ParallelFor(n1d,
    [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        density_total[i] = mass_total[i];
        if (vol_total[i] > 0.) {
            density_total[i] /= vol_total[i];
        }
    });

#ifdef GR_GRAV
    RealVector radial_pres_total(n1d,0);

    Real* const pres_total = radial_pres_total.dataPtr();

    combine_level_bins(pres_total, radial_shell_pres, level, n1d, parent);

    amrex::ParallelFor(n1d,
    [=] AMREX_GPU_DEVICE (int i) noexcept
    {
        if (vol_total[i] > 0.) {
            pres_total[i] /= vol_total[i];
        }
    });
#endif

    const Real* const mass = radial_mass_total.dataPtr();
#ifdef GR_GRAV
    Real* const den = radial_density_total.dataPtr();
    Real* const pres = radial_pres_total.dataPtr();
#endif
    Real* const gravity_view = radial_field.dataPtr();

    Scan::PrefixSum<Real> (n1d,
        [=] AMREX_GPU_DEVICE (int i) -> Real
        {
            Real dM = 0.0;

            if (i > 0) {
                Real rlo = (static_cast<Real>(i-1)         ) * dr;
                Real rc  = (static_cast<Real>(i-1) + 0.5_rt) * dr;
                Real rhi = (static_cast<Real>(i-1) + 1.0_rt) * dr;

                Real shell_volume = (4.0_rt / 3.0_rt * std::numbers::pi) * dr * (rhi * rhi * rhi - rc  * rc  * rc);
                Real zone_volume  = (4.0_rt / 3.0_rt * std::numbers::pi) * dr * (rhi * rhi * rhi - rlo * rlo * rlo);
                dM = dM + (shell_volume / zone_volume) * mass[i-1];
            }

            Real rlo = (static_cast<Real>(i)         ) * dr;
            Real rc  = (static_cast<Real>(i) + 0.5_rt) * dr;
            Real rhi = (static_cast<Real>(i) + 1.0_rt) * dr;

            Real shell_volume = (4.0_rt / 3.0_rt * std::numbers::pi) * dr * (rc  * rc  * rc  - rlo * rlo * rlo);
            Real zone_volume  = (4.0_rt / 3.0_rt * std::numbers::pi) * dr * (rhi * rhi * rhi - rlo * rlo * rlo);
            dM = dM + (shell_volume / zone_volume) * mass[i];

            return dM;
        },
        [=] AMREX_GPU_DEVICE (int i, Real const& running_mass)
        {
            Real rc = (static_cast<Real>(i) + 0.5_rt) * dr;

            gravity_view[i] = -C::Gconst * running_mass / (rc * rc);

#ifdef GR_GRAV

            if (den[i] > 0.0_rt) {
                Real ga = (1.0_rt + pres[i] / (den[i] * C::c_light * C::c_light));
                Real gb = (1.0_rt + (4.0_rt * std::numbers::pi) * rc * rc * rc * pres[i] / (running_mass * C::c_light * C::c_light));
                Real gc = 1.0_rt / (1.0_rt - 2.0_rt * C::Gconst * running_mass / (rc * C::c_light * C::c_light));

                gravity_view[i] = gravity_view[i] * ga * gb * gc;
            }
#endif
        },
        Scan::Type::inclusive);

    if (gravity::verbose)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        Real      end    = ParallelDescriptor::second() - strt;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(end,IOProc);
        amrex::Print() << "Gravity::build_radial_field() time = " << end << std::endl << std::endl;
#ifdef BL_LAZY
        });
#endif
    }
}

void
Gravity::tally_radial_profile(const Box& bx,
                             Array4<Real const> const& u,
                             RealVector& radial_mass_tile,
                             RealVector& radial_vol_tile,
#ifdef GR_GRAV
                             RealVector& radial_pres_tile,
#endif
                             int n1d, int level) const
{
    const Geometry& geom = parent->Geom(level);

    GpuArray<Real, 3> dx, problo;
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        dx[i] = geom.CellSizeArray()[i];
        problo[i] = geom.ProbLoArray()[i];
    }
    for (int i = AMREX_SPACEDIM; i < 3; ++i) {
        dx[i] = 0.0_rt;
        problo[i] = 0.0_rt;
    }

    Real dr = dx[0] / static_cast<Real>(gravity::radial_bin_ratio);
    Real drinv = 1.0_rt / dr;

    const int coord_type = geom.Coord();
    const auto geomdata = geom.data();

    AMREX_ALWAYS_ASSERT(coord_type >= 0 && coord_type <= 2);

    Real symmetry_mult = 1.0_rt;

    if (coord_type == 0) {

        if ((std::abs(problem::center[0] - problo[0]) < 1.e-2_rt * dx[0]) &&
            (std::abs(problem::center[1] - problo[1]) < 1.e-2_rt * dx[1]) &&
            (std::abs(problem::center[2] - problo[2]) < 1.e-2_rt * dx[2])) {

            symmetry_mult = 8.0_rt;

        }

    } else if (coord_type == 1) {

        if (std::abs(problem::center[1] - problo[1]) < 1.e-2_rt * dx[1]) {

            symmetry_mult = 2.0_rt;

        }

    }

    Real fac = static_cast<Real>(gravity::radial_bin_ratio);

    Real dx_frac = dx[0] / fac;
    Real subcell_dy = dx[1] / fac;
    Real subcell_dz = dx[2] / fac;

    Real* const radial_mass_bins_ptr = radial_mass_tile.dataPtr();
    Real* const radial_vol_bins_ptr = radial_vol_tile.dataPtr();
#ifdef GR_GRAV
    Real* const radial_pres_bins_ptr = radial_pres_tile.dataPtr();
#endif

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        GpuArray<Real, 3> loc;
        loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];
        Real lo_i = problo[0] + static_cast<Real>(i) * dx[0] - problem::center[0];

        loc[1]= problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
        Real lo_j = problo[1] + static_cast<Real>(j) * dx[1] - problem::center[1];

        loc[2]= problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
        Real lo_k = problo[2] + static_cast<Real>(k) * dx[2] - problem::center[2];

        Real r = distance(geomdata, loc);
        int index = static_cast<int>(r * drinv);

        if (u(i,j,k,URHO) == 0.0_rt) {
            return;
        }

#ifdef GR_GRAV
        Real rhoInv = 1.0_rt / u(i,j,k,URHO);

        eos_t eos_state;

        eos_state.rho = u(i,j,k,URHO);
        eos_state.e   = u(i,j,k,UEINT) * rhoInv;
        eos_state.T   = u(i,j,k,UTEMP);
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = u(i,j,k,UFS+n) * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; ++n) {
            eos_state.aux[n] = u(i,j,k,UFX+n) * rhoInv;
        }
#endif

        eos(eos_input_re, eos_state);
#endif

        if (index > n1d - 1) {

#ifndef AMREX_USE_GPU
            if (level == 0) {
                std::cout << "   " << "\n";
                std::cout << ">>> Error: Gravity_nd::ca_compute_radial_mass " << i << " " << j << " " << k << "\n";
                std::cout << ">>> ... index too big: " << index << " > " << n1d-1 << "\n";
                std::cout << ">>> ... at (i,j,k)   : " << i << " " << j << " " << k << "\n";
                amrex::Abort("Error:: Gravity_nd.F90 :: ca_compute_radial_mass");
            }
#endif

        } else {

            for (int kk = 0; kk <= dg2 * (gravity::radial_bin_ratio - 1); ++kk) {
                Real zz   = lo_k + (static_cast<Real>(kk) + 0.5_rt) * subcell_dz;
                Real subcell_z_sq = zz * zz;

                for (int jj = 0; jj <= dg1 * (gravity::radial_bin_ratio - 1); ++jj) {
                    Real yy   = lo_j + (static_cast<Real>(jj) + 0.5_rt) * subcell_dy;
                    Real subcell_y_sq = yy * yy;

                    for (int ii = 0; ii <= gravity::radial_bin_ratio - 1; ++ii) {
                        Real xx    = lo_i + (static_cast<Real>(ii) + 0.5_rt) * dx_frac;
                        Real subcell_x_sq  = xx * xx;

                        r     = std::sqrt(subcell_x_sq + subcell_y_sq + subcell_z_sq);
                        index = static_cast<int>(r * drinv);

                        Real vol_frac{};

                        if (coord_type == 0) {

                            vol_frac = symmetry_mult * dx_frac * subcell_dy * subcell_dz;

                        } else if (coord_type == 1) {

                            vol_frac = 2.0_rt * std::numbers::pi * dx_frac * subcell_dy * symmetry_mult * xx;

                        } else if (coord_type == 2) {

                            Real rlo = std::abs(lo_i + static_cast<Real>(ii  ) * dx_frac);
                            Real rhi = std::abs(lo_i + static_cast<Real>(ii+1) * dx_frac);
                            vol_frac = (4.0_rt / 3.0_rt) * std::numbers::pi * (rhi * rhi * rhi - rlo * rlo * rlo);

                        }

                        if (index <= n1d - 1) {
                            Gpu::Atomic::Add(&radial_mass_bins_ptr[index], vol_frac * u(i,j,k,URHO));
                            Gpu::Atomic::Add(&radial_vol_bins_ptr[index], vol_frac);
#ifdef GR_GRAV
                            Gpu::Atomic::Add(&radial_pres_bins_ptr[index], vol_frac * eos_state.p);
#endif
                        }

                    }
                }
            }

        }

    });
}

void
Gravity::interpolate_monopole_field(int level, RealVector& radial_field, MultiFab& gravity_field_out) const
{
    BL_PROFILE("Gravity::interpolate_monopole_field()");

    int n1d = static_cast<int>(radial_field.size());

    const Geometry& geom = parent->Geom(level);
    const auto dx = geom.CellSizeArray();
    const Real dr = dx[0] / static_cast<Real>(gravity::radial_bin_ratio);

    const auto problo = geom.ProbLoArray();
    const auto geomdata = geom.data();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(gravity_field_out, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox();

        auto gravity_view = gravity_field_out.array(mfi);
        const Real* const radial_accel_ptr = radial_field.dataPtr();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            GpuArray<Real, 3> loc;

            loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];

#if AMREX_SPACEDIM >= 2
            loc[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
#else
            loc[1] = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
            loc[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
#else
            loc[2] = 0.0_rt;
#endif

            Real r = distance(geomdata, loc);

            int index = static_cast<int>(r / dr);

            Real cen = (static_cast<Real>(index) + 0.5_rt) * dr;
            Real xi = r - cen;

            Real accel_mag;

            if (index == 0) {

                Real slope = (radial_accel_ptr[index+1] - radial_accel_ptr[index]) / dr;
                accel_mag = radial_accel_ptr[index] + slope * xi;

            } else if (index == n1d-1) {

                Real slope = (radial_accel_ptr[index] - radial_accel_ptr[index-1]) / dr;
                accel_mag = radial_accel_ptr[index] + slope * xi;

            } else if (index > n1d-1) {

#ifndef AMREX_USE_GPU
                if (level == 0) {
                    std::cout << "PUT_RADIAL_GRAV: INDEX TOO BIG " << index << " > " << n1d-1 << "\n";
                    std::cout << "AT (i,j,k) " << i << " " << j << " " << k << "\n";
                    std::cout << "R / DR IS " << r << " " << dr << "\n";
                    amrex::Abort("Error:: Gravity.cpp :: interpolate_monopole_field");
                } else {
                }
#endif

            } else {

                Real ghi = radial_accel_ptr[index+1];
                Real gmd = radial_accel_ptr[index  ];
                Real glo = radial_accel_ptr[index-1];
                accel_mag = ( ghi -   2.0_rt * gmd + glo) * xi * xi / (2.0_rt * dr * dr) +
                           ( ghi                  - glo) * xi      / (2.0_rt * dr     ) +
                           (-ghi + 26.e0_rt * gmd - glo) / 24.e0_rt;

                Real stencil_min = amrex::min(gmd, amrex::min(glo, ghi));
                Real stencil_max = amrex::max(gmd, amrex::max(glo, ghi));
                accel_mag = amrex::max(accel_mag, stencil_min);
                accel_mag = amrex::min(accel_mag, stencil_max);

            }

            if (index <= n1d-1) {

                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    gravity_view(i,j,k,n) = accel_mag * (loc[n] / r);
                }

            }

        });
    }
}

void
Gravity::include_point_source_field (int level, MultiFab& phi, MultiFab& gravity_field_out) const
{
    BL_PROFILE("Gravity::include_point_source_field()");

    const auto dx     = parent->Geom(level).CellSizeArray();
    const auto problo = parent->Geom(level).ProbLoArray();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(gravity_field_out, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox();

        Array4<Real> const grav_vec_arr = gravity_field_out.array(mfi);
        Array4<Real> const phi_arr = phi.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {

            Real x = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];
#if AMREX_SPACEDIM >= 2
            Real y = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
#else
            Real y = 0.0;
#endif
#if AMREX_SPACEDIM == 3
            Real z = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
#else
            Real z = 0.0;
#endif

            if(hydrosim::point_mass_is_shifted == 1)
            {
                Real offset_radius = hydrosim::point_mass_shift;

                if(AMREX_SPACEDIM == 1)
                {
                    x += offset_radius;
                }
                else if(AMREX_SPACEDIM ==2)
                {
                    y += offset_radius;
                }
                else if(AMREX_SPACEDIM == 3)
                {
                    z += offset_radius;
                }

            }

            Real rsq = x * x + y * y + z * z;
            Real radial_point_accel = -C::Gconst * hydrosim::point_mass / rsq;

            Real inv_dist = 1.e0_rt / std::sqrt(rsq);

            if (phi_arr.contains(i,j,k)) {
                phi_arr(i,j,k) -= C::Gconst * hydrosim::point_mass * inv_dist;
            }

            grav_vec_arr(i,j,k,0) += radial_point_accel * (x * inv_dist);
            grav_vec_arr(i,j,k,1) += radial_point_accel * (y * inv_dist);
            grav_vec_arr(i,j,k,2) += radial_point_accel * (z * inv_dist);
        });
    }
}
