#include <numbers>

#include <HydroSim.H>
#include <fundamental_constants.H>
#include <Gravity.H>
#include <Rotation.H>

using namespace amrex;

#ifdef GRAVITY
#ifdef ROTATION

namespace scf {

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    Real scale_factor (int i, int j, int k,
                       const GeometryData& geomdata,
                       GpuArray<Real, 3> anchor_point)
    {
        const auto *dx = geomdata.CellSize();

        GpuArray<Real, 3> r = {0.0};
        position(i, j, k, geomdata, r);
        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            r[n] -= problem::center[n];
        }

        // Do a trilinear interpolation to find the contribution from
        // this grid point. Limit so that only the nearest zone centers
        // can participate. This implies that the maximum allowable
        // distance from the target location is 0.5 * dx.

        GpuArray<Real, 3> rr = {0.0};

        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            rr[n] = std::abs(r[n] - anchor_point[n]) / dx[n];
        }

        Real scale;

        if (rr[0] > 1.0_rt || rr[1] > 1.0_rt || rr[2] > 1.0_rt) {
            scale = 0.0;
        }
        else {
            scale = (1.0_rt - rr[0]) * (1.0_rt - rr[1]) * (1.0_rt - rr[2]);
        }

        return scale;
    }

}

void HydroSim::scf_solve() {

    AMREX_ASSERT(level == 0);

    // First do some sanity checks.

    // Maximum density must be set.

    if (scf_peak_density <= 0) {
        amrex::Error("hydrosim.scf_peak_density must be set for SCF relaxation.");
    }

    // Equatorial radius and polar radius must both
    // be positive, and the polar radius cannot be
    // larger than the equatorial radius.

    if (scf_equator_radius <= 0) {
        amrex::Error("Equatorial radius must be positive for SCF relaxation.");
    }

    if (scf_pole_radius <= 0) {
        amrex::Error("Polar radius must be positive for SCF relaxation.");
    }

    if (scf_pole_radius > scf_equator_radius) {
        amrex::Error("Polar radius must not be larger than equatorial radius for SCF relaxation.");
    }

    // Now do the SCF solve.

    run_equilibrium_solve();

    // Update the gravitational field. Since we don't need this
    // during the iterations, we only do it once at the end.

    for (int lev = 0; lev <= parent->finestLevel(); ++lev)
    {
        MultiFab& gravity_new = getLevel(lev).get_new_data(Accel_Type);
        gravity->get_new_gravity_vector(lev, gravity_new, getLevel(lev).state[Accel_Type].curTime());
    }

    // Perform a final pass to ensure we're consistent with AMR.

    for (int lev = parent->finestLevel()-1; lev >= 0; --lev) {
        getLevel(lev).avgDown();
    }

}

void
HydroSim::run_equilibrium_solve()
{

    const int finest_level = parent->finestLevel();
    const int num_amr_levels = finest_level + 1;

    // Do the initial relaxation setup. We need to fix two points
    // to uniquely determine an equilibrium configuration for a
    // rotating star. We are provided the distance from the center,
    // and the axis we're measuring on.

    GpuArray<Real, 3> anchor_point_A = {problem::center[0], problem::center[1], problem::center[2]};
    GpuArray<Real, 3> anchor_point_B = {problem::center[0], problem::center[1], problem::center[2]};

    // Note that the sign is somewhat arbitrary here.

    anchor_point_A[0] += hydrosim::scf_equator_radius;
    anchor_point_B[2] += hydrosim::scf_pole_radius;

    // To do the relaxation loop, we need to know the target maximum
    // enthalpy on the domain. So we'll loop through the grid and
    // calculate what it should be, based on the density, temperature,
    // and composition. We do not specify the latter two -- we assume
    // that these are filled as desired in initdata. So, to get the
    // maximum enthalpy, we'll loop through the grid, using the composition
    // and temperature in each zone, combined with the requested maximum
    // density, call the EOS to calculate the enthalpy for that combination,
    // and then take the maximum of that. (Note that the current density on
    // the grid is treated as an initial guess and does not need to reach
    // the requested maximum.) The SCF procedure is only intended for a setup
    // with uniform temperature and composition, so in that case this procedure
    // should yield the same result as just picking a zone at random and using
    // its temperature and composition. The reason for doing it this way is so
    // that the user doesn't have to request temperature and composition in their
    // inputs file (which is messy in general) -- they can tell us what composition
    // they want by filling it in the initialization.

    Real target_enthalpy_max = 0.0;

    for (int lev = 0; lev <= finest_level; ++lev) {

        MultiFab& field_new = getLevel(lev).get_new_data(State_Type);

        ReduceOps<ReduceOpMax> reduce_op;
        ReduceData<Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(field_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

            const Box& bx = mfi.tilebox();

            auto state_arr = field_new[mfi].array();

            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                eos_t eos_state;

                eos_state.rho = hydrosim::scf_peak_density;
                eos_state.T   = state_arr(i,j,k,UTEMP);
                for (int n = 0; n < NumSpec; ++n) {
                    eos_state.xn[n] = state_arr(i,j,k,UFS+n) / state_arr(i,j,k,URHO);
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; ++n) {
                    eos_state.aux[n] = state_arr(i,j,k,UFX+n) / state_arr(i,j,k,URHO);
                }
#endif

                eos(eos_input_rt, eos_state);

                return {eos_state.h};
            });

        }

        ReduceTuple hv = reduce_data.value();
        target_enthalpy_max = amrex::max(target_enthalpy_max, amrex::get<0>(hv));

    }

    ParallelDescriptor::ReduceRealMax(target_enthalpy_max);

    Vector< std::unique_ptr<MultiFab> > psi(num_amr_levels);
    Vector< std::unique_ptr<MultiFab> > enthalpy(num_amr_levels);
    Vector< std::unique_ptr<MultiFab> > level_states(num_amr_levels);
    Vector< std::unique_ptr<MultiFab> > phi(num_amr_levels);

    // Iterate until the system is relaxed by filling the level data
    // and then doing a multilevel gravity solve.

    int ctr = 1;

    while (ctr <= scf_iter_cap) {

        Real time = getLevel(0).state[State_Type].curTime();

        // Construct a local MultiFab for the rotational psi.
        // This does not change over the loop iterations, but
        // this (and the data to follow) must be constructed
        // inside the loop on each iteration because of regrids.

        for (int lev = 0; lev <= finest_level; ++lev) {

            psi[lev] = std::make_unique<MultiFab>(getLevel(lev).grids, getLevel(lev).dmap, 1, 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
            for (MFIter mfi((*psi[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {

                const Box& bx = mfi.tilebox();

                fill_scaled_centrifugal_potential(bx, (*psi[lev]).array(mfi), time);

            }

        }

        // Construct a local MultiFab for the enthalpy.

        for (int lev = 0; lev <= finest_level; ++lev) {
            enthalpy[lev] = std::make_unique<MultiFab>(getLevel(lev).grids, getLevel(lev).dmap, 1, 0);
        }

        // Construct a local MultiFab for the state data. We do
        // this because we have to mask out the state several times
        // in the below calculation, and it's easiest to have a scratch
        // copy of the data to work with.

        for (int lev = 0; lev <= finest_level; ++lev) {
            level_states[lev] = std::make_unique<MultiFab>(getLevel(lev).grids, getLevel(lev).dmap, NUM_STATE, 0);
            phi[lev] = std::make_unique<MultiFab>(getLevel(lev).grids, getLevel(lev).dmap, 1, 0);
        }

        // Copy in the state data. Mask it out on coarse levels.

        for (int lev = 0; lev <= finest_level; ++lev) {

            MultiFab::Copy((*level_states[lev]), getLevel(lev).get_new_data(State_Type), 0, 0, NUM_STATE, 0);
            MultiFab::Copy((*phi[lev]), getLevel(lev).get_new_data(Potential_Type), 0, 0, 1, 0);

            if (lev < finest_level) {
                const MultiFab& mask = getLevel(lev+1).make_fine_mask();

                for (int n = 0; n < NUM_STATE; ++n) {
                    MultiFab::Multiply((*level_states[lev]), mask, 0, n, 1, 0);
                }

                MultiFab::Multiply((*phi[lev]), mask, 0, 0, 1, 0);
            }

        }

        // First step is to find the rotational frequency.

        Real grav_sum_A = 0.0;
        Real centrifugal_sum_A = 0.0;
        Real grav_sum_B = 0.0;
        Real centrifugal_sum_B = 0.0;

        for (int lev = 0; lev <= finest_level; ++lev) {

            auto geomdata = parent->Geom(lev).data();

            ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_op;
            ReduceData<Real, Real, Real, Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
            for (MFIter mfi((*phi[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {

                const Box& bx = mfi.tilebox();

                auto phi_arr = (*phi[lev])[mfi].array();
                auto centrifugal_arr = (*psi[lev])[mfi].array();

                reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    Real weight_A = scf::scale_factor(i, j, k, geomdata, anchor_point_A);

                    Real weighted_grav_A = weight_A * phi_arr(i,j,k);
                    Real weighted_centrifugal_A = weight_A * centrifugal_arr(i,j,k);

                    Real weight_B = scf::scale_factor(i, j, k, geomdata, anchor_point_B);

                    Real weighted_grav_B = weight_B * phi_arr(i,j,k);
                    Real weighted_centrifugal_B = weight_B * centrifugal_arr(i,j,k);

                    return {weighted_grav_A, weighted_centrifugal_A, weighted_grav_B, weighted_centrifugal_B};
                });
            }

            ReduceTuple hv = reduce_data.value();
            grav_sum_A += amrex::get<0>(hv);
            centrifugal_sum_A += amrex::get<1>(hv);
            grav_sum_B += amrex::get<2>(hv);
            centrifugal_sum_B += amrex::get<3>(hv);

        }

        ParallelDescriptor::ReduceRealSum(grav_sum_A);
        ParallelDescriptor::ReduceRealSum(centrifugal_sum_A);
        ParallelDescriptor::ReduceRealSum(grav_sum_B);
        ParallelDescriptor::ReduceRealSum(centrifugal_sum_B);

        // Now update the square of the rotation frequency, following Hachisu.
        // Deal carefully with the special case where grav_sum_A and grav_sum_B are equal -- we assume
        // this is the non-rotating case, and we want to handle that gracefully. Since the
        // rotation module knows to treat spin_period = 0 as effectively disabling
        // rotation, we'll use that approach.

        if (std::abs(scf_pole_radius - scf_equator_radius) / std::abs(scf_equator_radius) < 1.e-6) {

            spin_period = 0.0;

        } else {

            Real angular_speed_sq = -(grav_sum_A - grav_sum_B) / (centrifugal_sum_A - centrifugal_sum_B);

            if (angular_speed_sq < 0.0 && ParallelDescriptor::IOProcessor()) {
                std::cerr << "Omega squared = " << angular_speed_sq << " is negative in the relaxation step; aborting." << std::endl;
                amrex::Error();
            }

            Real omega = std::sqrt(angular_speed_sq);

            // Rotational period is 2 pi / omega.
            // Let's also be sure not to let the period
            // change by too much in a single iteration.

            spin_period = amrex::min(1.1_rt * spin_period,
                                           amrex::max(0.9_rt * spin_period,
                                                      2.0_rt * std::numbers::pi / omega));

        }


        // Second step is to evaluate the Bernoulli constant.

        Real bernoulli = 0.0;

        for (int lev = 0; lev <= finest_level; ++lev) {

            auto geomdata = parent->Geom(lev).data();

            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
            for (MFIter mfi((*phi[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {

                const Box& bx = mfi.tilebox();

                auto phi_arr = (*phi[lev])[mfi].array();

                reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    const auto coord = geomdata.Coord();

                    GpuArray<Real, 3> r = {0.0};
                    position(i, j, k, geomdata, r);
                    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                        r[n] -= problem::center[n];
                    }

                    Real scale = scf::scale_factor(i, j, k, geomdata, anchor_point_A);

                    auto omega = get_angular_velocity(geomdata, j);
                    Real local_bernoulli_const = scale * (phi_arr(i,j,k) + centrifugal_potential(r, omega, coord));

                    return {local_bernoulli_const};
                });

            }

            ReduceTuple hv = reduce_data.value();
            bernoulli += amrex::get<0>(hv);

        }

        ParallelDescriptor::ReduceRealSum(bernoulli);



        // Third step is to construct the enthalpy field and
        // find the maximum enthalpy for the star.

        for (int lev = 0; lev <= finest_level; ++lev) {

            auto geomdata = parent->Geom(lev).data();

            enthalpy[lev]->setVal(0.0);

#ifdef _OPENMP
#pragma omp parallel
#endif
            for (MFIter mfi((*phi[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {

                const Box& bx = mfi.tilebox();

                auto enthalpy_field_arr = (*enthalpy[lev])[mfi].array();
                auto phi_arr = (*phi[lev])[mfi].array();

                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    const auto coord = geomdata.Coord();

                    // The Bernoulli equation says that energy is conserved:
                    // enthalpy + gravitational potential + rotational potential = const
                    // We already have the constant, so our goal is to construct the enthalpy field.

                    GpuArray<Real, 3> r = {0.0};
                    position(i, j, k, geomdata, r);
                    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                        r[n] -= problem::center[n];
                    }

                    auto omega = get_angular_velocity(geomdata, j);
                    enthalpy_field_arr(i,j,k) = bernoulli - phi_arr(i,j,k) - centrifugal_potential(r, omega, coord);
                });

            }

        }

        Real current_enthalpy_max = 0.0;
        Real current_density_max = 0.0;

        for (int lev = 0; lev <= finest_level; ++lev) {

            current_enthalpy_max = std::max(current_enthalpy_max, enthalpy[lev]->max(0));
            current_density_max = std::max(current_density_max, level_states[lev]->max(URHO));

        }

        Real Max_change_norm = 0.0;

        // Finally, update the density using the enthalpy field.

        for (int lev = 0; lev <= finest_level; ++lev) {

            ReduceOps<ReduceOpMax> reduce_op;
            ReduceData<Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
            for (MFIter mfi((*level_states[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {

                const Box& bx = mfi.tilebox();

                auto enthalpy_field_arr = (*enthalpy[lev])[mfi].array();
                auto state_arr = (*level_states[lev])[mfi].array();

                reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    Real norm = 0.0;

                    // We only want to call the EOS for zones with enthalpy > 0.
                    // For distances far enough from the center, the rotation
                    // term can overcome the other terms and make the enthalpy
                    // spuriously negative. If the enthalpy is negative, we just
                    // leave the zone alone -- this should be ambient material.

                    if (enthalpy_field_arr(i,j,k) > 0.0 && state_arr(i,j,k,URHO) > 0.0) {

                        Real density_prev = state_arr(i,j,k,URHO);

                        // Rescale the enthalpy by the maximum allowed value.

                        enthalpy_field_arr(i,j,k) = target_enthalpy_max * (enthalpy_field_arr(i,j,k) / current_enthalpy_max);

                        eos_t eos_state;

                        eos_state.rho = state_arr(i,j,k,URHO); // Initial guess for the EOS
                        eos_state.T   = state_arr(i,j,k,UTEMP);
                        for (int n = 0; n < NumSpec; ++n) {
                            eos_state.xn[n] = state_arr(i,j,k,UFS+n) / state_arr(i,j,k,URHO);
                        }
#if NAUX_NET > 0
                        for (int n = 0; n < NumAux; ++n) {
                            eos_state.aux[n] = state_arr(i,j,k,UFX+n) / state_arr(i,j,k,URHO);
                        }
#endif
                        eos_state.h   = enthalpy_field_arr(i,j,k);

                        eos(eos_input_th, eos_state);

                        state_arr(i,j,k,URHO)  = eos_state.rho;
                        state_arr(i,j,k,UTEMP) = eos_state.T;
                        state_arr(i,j,k,UEINT) = state_arr(i,j,k,URHO) * eos_state.e;
                        for (int n = 0; n < NumSpec; ++n) {
                            state_arr(i,j,k,UFS+n) = state_arr(i,j,k,URHO) * eos_state.xn[n];
                        }

                        state_arr(i,j,k,UMX) = 0.0;
                        state_arr(i,j,k,UMY) = 0.0;
                        state_arr(i,j,k,UMZ) = 0.0;

                        state_arr(i,j,k,UEDEN) = state_arr(i,j,k,UEINT);

                        // Convergence test

                        // Zones only participate in this test if they have a density
                        // that is above a certain fraction of the peak, to avoid
                        // oscillations in low density zones stalling convergence.

                        Real drho = std::abs(state_arr(i,j,k,URHO) - density_prev) / density_prev;

                        if (state_arr(i,j,k,URHO) / current_density_max > 1.0e-3_rt) {
                            norm = drho;
                        }
                    }

                    return {norm};
                });

            }

            ReduceTuple hv = reduce_data.value();
            Max_change_norm = amrex::max(Max_change_norm, amrex::get<0>(hv));

        }

        ParallelDescriptor::ReduceRealMax(Max_change_norm);

        // Copy state data back to its source, and synchronize it on coarser levels.

        for (int lev = 0; lev <= finest_level; ++lev) {
            MultiFab::Copy(getLevel(lev).get_new_data(State_Type), (*level_states[lev]), 0, 0, NUM_STATE, 0);
            MultiFab::Copy(getLevel(lev).get_new_data(Potential_Type), (*phi[lev]), 0, 0, 1, 0);
        }

        for (int lev = finest_level-1; lev >= 0; --lev) {
            getLevel(lev).avgDown();
        }

        // Since we've changed the density distribution on the grid, regrid.

        bool do_io = false;
        if (finest_level > 0) {
            parent->RegridOnly(time, do_io);
        }

        // Update the gravitational field -- only after we've completed cleaning up the state above.

        gravity->multilevel_potential_solve(0, finest_level);

        // Update diagnostic quantities.

        Real kinetic_energy_sum = 0.0;
        Real potential_energy_sum = 0.0;
        Real internal_energy_sum = 0.0;
        Real mass = 0.0;

        for (int lev = 0; lev <= finest_level; ++lev) {

            auto geomdata = parent->Geom(lev).data();

            ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_op;
            ReduceData<Real, Real, Real, Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
            for (MFIter mfi((*level_states[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {

                const Box& bx = mfi.tilebox();

                auto state_arr = (*level_states[lev])[mfi].array();
                auto phi_arr = (*phi[lev])[mfi].array();

                reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    Real dM = 0.0, dK = 0.0, dU = 0.0, dE = 0.0;

                    if (state_arr(i,j,k,URHO) > 0.0)
                    {
                        const auto coord = geomdata.Coord();

                        GpuArray<Real, 3> r = {0.0};
                        position(i, j, k, geomdata, r);
                        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                            r[n] -= problem::center[n];
                        }

                        Real dV = cell_geometry::volume(i, j, k, geomdata);
                        dM = state_arr(i,j,k,URHO) * dV;

                        auto omega = get_angular_velocity(geomdata, j);
                        dK = centrifugal_potential(r, omega, coord) * dM;

                        dU = 0.5_rt * phi_arr(i,j,k) * dM;

                        eos_t eos_state;

                        eos_state.rho = state_arr(i,j,k,URHO);
                        eos_state.T   = state_arr(i,j,k,UTEMP);
                        for (int n = 0; n < NumSpec; ++n) {
                            eos_state.xn[n] = state_arr(i,j,k,UFS+n) / state_arr(i,j,k,URHO);
                        }
#if NAUX_NET > 0
                        for (int n = 0; n < NumAux; ++n) {
                            eos_state.aux[n] = state_arr(i,j,k,UFX+n) / state_arr(i,j,k,URHO);
                        }
#endif

                        eos(eos_input_rt, eos_state);

                        dE = eos_state.p * dV;
                    }

                    return {dM, dK, dU, dE};
                });

            }

            ReduceTuple hv = reduce_data.value();
            mass += amrex::get<0>(hv);
            kinetic_energy_sum += amrex::get<1>(hv);
            potential_energy_sum += amrex::get<2>(hv);
            internal_energy_sum += amrex::get<3>(hv);

        }

        ParallelDescriptor::ReduceRealSum(kinetic_energy_sum);
        ParallelDescriptor::ReduceRealSum(potential_energy_sum);
        ParallelDescriptor::ReduceRealSum(internal_energy_sum);
        ParallelDescriptor::ReduceRealSum(mass);

        Real virial_mismatch = std::abs(2.0 * kinetic_energy_sum + potential_energy_sum + 3.0 * internal_energy_sum) / std::abs(potential_energy_sum);

        // Now check to see if we're converged.

        int is_relaxed = 0;

        if (Max_change_norm < scf_residual_tol) {
            is_relaxed = 1;
        }

        if (ParallelDescriptor::IOProcessor()) {

            // Grab the value for the solar mass.

            std::cout << std::endl << std::endl;
            std::cout << "   Relaxation iterations completed: " << ctr << std::endl;
            std::cout << "   L-infinity norm of residual (relative to old state): " << Max_change_norm << std::endl;
            std::cout << "   Rotational period (s): " << spin_period << std::endl;
            std::cout << "   Kinetic energy: " << kinetic_energy_sum << std::endl;
            std::cout << "   Potential energy: " << potential_energy_sum << std::endl;
            std::cout << "   Internal energy: " << internal_energy_sum << std::endl;
            std::cout << "   Virial error: " << virial_mismatch << std::endl;
            std::cout << "   Mass: " << mass / C::M_solar << " solar masses" << std::endl;

            if (is_relaxed == 1) {
                std::cout << "  Relaxation completed!" << std::endl;
            }

            std::cout << std::endl << std::endl;

        }

        if (is_relaxed == 1) {
            break;
        }

        ctr++;

    }

}
#endif
#endif
