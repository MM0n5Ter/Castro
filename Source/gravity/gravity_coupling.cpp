#include <HydroSim.H>

#include <Gravity.H>

#ifdef HYBRID_MOMENTUM
#include <HydroSim_helpers.H>
#include <hybrid_momentum.H>
#endif

using namespace amrex;

void
HydroSim::assemble_old_gravity (Real time)
{
    BL_PROFILE("HydroSim::assemble_old_gravity()");

    const Real strt_time = ParallelDescriptor::second();

    MultiFab& gravity_old_mf = get_old_data(Accel_Type);
    MultiFab& phi_old = get_old_data(Potential_Type);

    if (gravity->get_solver_kind() != "PoissonGrav") {
        phi_old.setVal(0.0);
    }

    if (!do_gravity_flag) {

        gravity_old_mf.setVal(0.0);

        return;

    }

    auto composite_fix_applies = [&] () {
        return gravity->CompositeFixEnabled() && level < parent->finestLevel() && level <= gravity->get_solve_level_cap();
    };

    bool solve_at_old_time = true;

    if (parent->subcyclingMode() == "None" || parent->finestLevel() == 0) {
        solve_at_old_time = false;
    }

    if (gravity->get_solver_kind() == "PoissonGrav" && solve_at_old_time)
    {

        MultiFab comp_phi;
        Vector<std::unique_ptr<MultiFab> > composite_grad_phi(AMREX_SPACEDIM);

        if (composite_fix_applies()) {

            comp_phi.define(phi_old.boxArray(), phi_old.DistributionMap(), phi_old.nComp(), phi_old.nGrow());
            MultiFab::Copy(comp_phi, phi_old, 0, 0, phi_old.nComp(), phi_old.nGrow());

            for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                composite_grad_phi[n] = std::make_unique<MultiFab>(getEdgeBoxArray(n), dmap, 1, 0);
                MultiFab::Copy(*composite_grad_phi[n], *gravity->get_potential_gradient_prev(level)[n], 0, 0, 1, 0);
            }

        }

        if (hydrosim::verbose && ParallelDescriptor::IOProcessor()) {
            std::cout << "... old-time level Poisson gravity solve at level " << level << std::endl << std::endl;
        }

        const PotentialTime epoch = PotentialTime::Old;

        gravity->solve_potential_level(level,
                               phi_old,
                               amrex::GetVecOfPtrs(gravity->get_potential_gradient_prev(level)),
                               epoch);

        if (composite_fix_applies()) {

            gravity->build_composite_excess_grad(level,
                                                      comp_phi,
                                                      amrex::GetVecOfPtrs(composite_grad_phi),
                                                      composite_minus_level_phi,
                                                      composite_minus_level_grad_phi);

            MultiFab::Copy(phi_old, comp_phi, 0, 0, phi_old.nComp(), phi_old.nGrow());

            for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                MultiFab::Copy(*gravity->get_potential_gradient_prev(level)[n], *composite_grad_phi[n], 0, 0, 1, 0);
            }

        }

        if (gravity->get_self_check_flag() == 1) {

            if (hydrosim::verbose && ParallelDescriptor::IOProcessor()) {
                std::cout << " " << '\n';
                std::cout << "... testing phi_gradient_new after doing single level solve " << '\n';
            }

            gravity->verify_level_grad_phi_before(level);

        }

    }

    gravity->read_old_gravity_field(level, gravity_old_mf, time);

    if (verbose > 0)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;
        amrex::Real level_real = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::assemble_old_gravity() time = " << run_time << " on level "
                       << level_real << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }
}

void
HydroSim::assemble_new_gravity (Real time)
{
    BL_PROFILE("HydroSim::assemble_new_gravity()");

    const Real strt_time = ParallelDescriptor::second();

    MultiFab& gravity_new = get_new_data(Accel_Type);
    MultiFab& phi_new = get_new_data(Potential_Type);

    if (gravity->get_solver_kind() != "PoissonGrav") {
        phi_new.setVal(0.0);
    }

    if (!do_gravity_flag) {

        gravity_new.setVal(0.0);

        return;

    }

    if (gravity->get_solver_kind() == "PoissonGrav")
    {
        if (level == 0 && parent->subcyclingMode() == "None") {
            if (hydrosim::verbose > 0) {
                amrex::Print() << "\n... new-time composite Poisson gravity solve from level " << level << " to level " << parent->finestLevel() << std::endl << std::endl;
            }

            for (int lev = level; lev <= parent->finestLevel(); ++lev) {
                MultiFab& level_phi_old = getLevel(lev).get_old_data(Potential_Type);
                MultiFab& level_phi_new = getLevel(lev).get_new_data(Potential_Type);

                MultiFab::Copy(level_phi_new, level_phi_old, 0, 0, 1, level_phi_new.nGrow());
            }

            gravity->multilevel_potential_solve(level, parent->finestLevel());
        }
        else if (parent->subcyclingMode() != "None") {

            MultiFab& phi_old = get_old_data(Potential_Type);

            MultiFab::Copy(phi_new, phi_old, 0, 0, 1, phi_new.nGrow());

            if (gravity->CompositeFixEnabled() && level < parent->finestLevel() && level <= gravity->get_solve_level_cap()) {
                phi_new.minus(composite_minus_level_phi, 0, 1, 0);
            }

            if (hydrosim::verbose && ParallelDescriptor::IOProcessor()) {
                std::cout << "... new-time level Poisson gravity solve at level " << level << std::endl << std::endl;
            }

            const PotentialTime epoch = PotentialTime::New;

            gravity->solve_potential_level(level,
                                   phi_new,
                                   amrex::GetVecOfPtrs(gravity->get_potential_gradient_curr(level)),
                                   epoch);

            if (gravity->CompositeFixEnabled() == 1 && level < parent->finestLevel() && level <= gravity->get_solve_level_cap()) {

                if (gravity->get_self_check_flag() == 1) {

                    if (hydrosim::verbose && ParallelDescriptor::IOProcessor()) {
                        std::cout << " " << '\n';
                        std::cout << "... testing phi_gradient_new before adding composite_minus_level_grad_phi " << '\n';
                    }

                    gravity->verify_level_grad_phi_now(level);

                }

                phi_new.plus(composite_minus_level_phi, 0, 1, 0);
                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    gravity->get_potential_gradient_curr(level)[n]->plus(*composite_minus_level_grad_phi[n], 0, 1, 0);
                }

                if (gravity->get_self_check_flag() == 1) {

                    if (hydrosim::verbose && ParallelDescriptor::IOProcessor()) {
                        std::cout << " " << '\n';
                        std::cout << "... testing phi_gradient_new after adding composite_minus_level_grad_phi " << '\n';
                    }

                    gravity->verify_level_grad_phi_now(level);

                }

            }
        }
    }

    gravity->get_new_gravity_vector(level, gravity_new, time);

    if (gravity->get_solver_kind() == "PoissonGrav" && level <= gravity->get_solve_level_cap() && parent->subcyclingMode() != "None") {

        if (gravity->CompositeFixEnabled() == 1 && level < parent->finestLevel()) {

            if (gravity->NoSync() == 0) {

                phi_new.minus(composite_minus_level_phi, 0, 1, 0);

                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    gravity->get_potential_gradient_curr(level)[n]->minus(*composite_minus_level_grad_phi[n], 0, 1, 0);
                }

            }

            composite_minus_level_phi.clear();
            composite_minus_level_grad_phi.clear();

        }

    }

    if (verbose > 0)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;
        amrex::Real level_real = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::assemble_new_gravity() time = " << run_time
                       << " on level " << level_real << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }
}

static AMREX_GPU_HOST_DEVICE AMREX_INLINE
bool in_central_window (int i, int j, int k, const Box& bx,
                        GpuArray<Real, AMREX_SPACEDIM> const& problo,
                        GpuArray<Real, AMREX_SPACEDIM> const& dx)
{
    const Real eps = 1.e-8_rt;

    int center_i = static_cast<int>(std::floor((problem::center[0] - problo[0]) / dx[0] + eps));
#if AMREX_SPACEDIM >= 2
    int center_j = static_cast<int>(std::floor((problem::center[1] - problo[1]) / dx[1] + eps));
#endif
#if AMREX_SPACEDIM == 3
    int center_k = static_cast<int>(std::floor((problem::center[2] - problo[2]) / dx[2] + eps));
#endif


    const int box_size = 2;

    int istart = amrex::max(center_i - box_size, bx.smallEnd(0));
    int iend = amrex::min(center_i + box_size - 1, bx.bigEnd(0));
#if AMREX_SPACEDIM >= 2
    int jstart = amrex::max(center_j - box_size, bx.smallEnd(1));
    int jend = amrex::min(center_j + box_size - 1, bx.bigEnd(1));
#else
    int jstart = 0;
    int jend = 0;
#endif
#if AMREX_SPACEDIM == 3
    int k_window_lo = amrex::max(center_k - box_size, bx.smallEnd(2));
    int kend = amrex::min(center_k + box_size - 1, bx.bigEnd(2));
#else
    int k_window_lo = 0;
    int kend = 0;
#endif

    return i >= istart && i <= iend &&
           j >= jstart && j <= jend &&
           k >= k_window_lo && k <= kend;
}

void
HydroSim::centralmass_update(Real time, Real dt)
{

    amrex::ignore_unused(time);
    amrex::ignore_unused(dt);

    int finest_level = parent->finestLevel();

    if (level == finest_level && point_mass_correct_solution)
    {

        MultiFab& S_old = get_old_data(State_Type);
        MultiFab& S_new = get_new_data(State_Type);

        const auto dx = geom.CellSizeArray();
        const auto problo = geom.ProbLoArray();

        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

            const Box& bx = mfi.tilebox();

            Array4<Real const> const uin  = S_old.array(mfi);
            Array4<Real const> const uout = S_new.array(mfi);
            Array4<Real const> const vol  = volume.array(mfi);

            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                Real partial_mass_delta = 0.0_rt;

                if (in_central_window(i, j, k, bx, problo, dx)) {

                    partial_mass_delta = vol(i,j,k) * (uout(i,j,k,URHO) - uin(i,j,k,URHO));

                }

                return partial_mass_delta;

            });

        }

        ReduceTuple hv = reduce_data.value();
        Real center_mass_delta = amrex::get<0>(hv);

        ParallelDescriptor::ReduceRealSum(center_mass_delta);

        if (center_mass_delta > 0.0)
        {

            if (verbose > 1) {
                amrex::Print() << "  Updating point mass from " << point_mass << ", by " << center_mass_delta
                               << ", to " << point_mass + center_mass_delta << std::endl << std::endl;
            }

            point_mass += center_mass_delta;

#ifdef _OPENMP
#pragma omp parallel
#endif
            for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();

                Array4<Real const> const uin = S_old.array(mfi);
                Array4<Real> const uout = S_new.array(mfi);

                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {

                    if (in_central_window(i, j, k, bx, problo, dx)) {

                        for (int n = 0; n < NUM_STATE; ++n) {
                            uout(i,j,k,n) = uin(i,j,k,n);
                        }

                    }

                });
            }
        }
    }
}
