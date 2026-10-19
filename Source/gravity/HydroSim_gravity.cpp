#include <HydroSim.H>

#include <Gravity.H>

#ifdef HYBRID_MOMENTUM
#include <HydroSim_helpers.H>
#include <hybrid.H>
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


    bool solve_at_old_time = true;

    if (parent->subcyclingMode() == "None" || parent->finestLevel() == 0) {
        solve_at_old_time = false;
    }

    if (gravity->get_solver_kind() == "PoissonGrav" && solve_at_old_time)
    {

        MultiFab comp_phi;
        Vector<std::unique_ptr<MultiFab> > composite_grad_phi(AMREX_SPACEDIM);

        if (gravity->CompositeFixEnabled() && level < parent->finestLevel() && level <= gravity->get_solve_level_cap()) {

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

        int use_new_state = 0;


        gravity->solve_potential_level(level,
                               phi_old,
                               amrex::GetVecOfPtrs(gravity->get_potential_gradient_prev(level)),
                               use_new_state);

        if (gravity->CompositeFixEnabled() && level < parent->finestLevel() && level <= gravity->get_solve_level_cap()) {


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

            int use_new_state = 1;

            gravity->solve_potential_level(level,
                                   phi_new,
                                   amrex::GetVecOfPtrs(gravity->get_potential_gradient_curr(level)),
                                   use_new_state);

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

void HydroSim::assemble_old_gravity_source(MultiFab& source, MultiFab& state_in, Real time, Real dt)
{

    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::assemble_old_gravity_source()");

    const Real strt_time = ParallelDescriptor::second();

    const MultiFab& gravity_old_mf = get_old_data(Accel_Type);

    if (!do_gravity_flag) {
        return;
    }


#ifdef HYBRID_MOMENTUM
    GeometryData geomdata = geom.data();
#endif

    AMREX_ALWAYS_ASSERT(hydrosim::gravity_source_mode >= 1 && hydrosim::gravity_source_mode <= 4);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state_in, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        Array4<Real const> const uold = state_in.array(mfi);
        Array4<Real const> const gravity_view = gravity_old_mf.array(mfi);
        Array4<Real> const source_view = source.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {

            GpuArray<Real, NUM_STATE> snew;
            for (int n = 0; n < NUM_STATE; ++n) {
                snew[n] = 0.0_rt;
            }


            GpuArray<Real, NUM_SOURCE> src;
            for (int n = 0; n < NUM_SOURCE; ++n) {
                src[n] = 0.0_rt;
            }


            Real rho    = uold(i,j,k,URHO);
            Real rhoInv = 1.0_rt / rho;

            for (int n = 0; n < NUM_STATE; ++n) {
                snew[n] = uold(i,j,k,n);
            }

            Real kinetic_energy_old = 0.5_rt * (snew[UMX] * snew[UMX] + snew[UMY] * snew[UMY] + snew[UMZ] * snew[UMZ]) * rhoInv;

            GpuArray<Real, 3> Sr;
            for (int n = 0; n < 3; ++n) {
                Sr[n] = rho * gravity_view(i,j,k,n);

                src[UMX+n] = Sr[n];

                snew[UMX+n] += dt * src[UMX+n];
            }

#ifdef HYBRID_MOMENTUM
            GpuArray<Real, 3> loc;
            position(i, j, k, geomdata, loc);
            for (int n = 0; n < 3; ++n) {
                loc[n] -= problem::center[n];
            }

            GpuArray<Real, 3> hybrid_mom_source;

            compute_hybrid_momentum_source(loc, Sr, hybrid_mom_source);

            for (int n = 0; n < 3; ++n) {
                 src[UMR+n] = hybrid_mom_source[n];
                 snew[UMR+n] += dt * src[UMR+n];
            }
#endif

            Real SrE{};

            if (hydrosim::gravity_source_mode == 1 || hydrosim::gravity_source_mode == 2) {


                SrE = (uold(i,j,k,UMX) * Sr[0] + uold(i,j,k,UMY) * Sr[1] + uold(i,j,k,UMZ) * Sr[2]) * rhoInv;

            } else if (hydrosim::gravity_source_mode == 3) {

                Real kinetic_energy_updated = 0.5_rt * (snew[UMX] * snew[UMX] + snew[UMY] * snew[UMY] + snew[UMZ] * snew[UMZ]) * rhoInv;
                SrE = kinetic_energy_updated - kinetic_energy_old;

            } else if (hydrosim::gravity_source_mode == 4) {


                SrE = (uold(i,j,k,UMX) * Sr[0] + uold(i,j,k,UMY) * Sr[1] + uold(i,j,k,UMZ) * Sr[2]) * rhoInv;

            }

            src[UEDEN] = SrE;

            snew[UEDEN] += dt * SrE;


            for (int n = 0; n < NUM_SOURCE; ++n) {
                source_view(i,j,k,n) += src[n];
            }

        });

    }

    if (hydrosim::verbose > 1)
    {
        const int IOProc   = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;
        amrex::Real level_real = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::assemble_old_gravity_source() time = " << run_time
                       << " on level " << level_real << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }
}

void HydroSim::assemble_new_gravity_source(MultiFab& source, MultiFab& field_old, MultiFab& field_new,
                                          Real time, Real dt)
{

    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::assemble_new_gravity_source()");

    const Real strt_time = ParallelDescriptor::second();

    MultiFab& gravity_old_mf = get_old_data(Accel_Type);
    MultiFab& gravity_new = get_new_data(Accel_Type);

    if (!do_gravity_flag) {
        return;
    }

    GpuArray<Real, 3> dx;
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        dx[i] = geom.CellSizeArray()[i];
    }
    for (int i = AMREX_SPACEDIM; i < 3; ++i) {
        dx[i] = 0.0_rt;
    }

#ifdef HYBRID_MOMENTUM
    GeometryData geomdata = geom.data();
#endif

    AMREX_ALWAYS_ASSERT(hydrosim::gravity_source_mode >= 1 && hydrosim::gravity_source_mode <= 4);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        for (MFIter mfi(field_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            Array4<Real const> const uold  = field_old.array(mfi);
            Array4<Real const> const unew  = field_new.array(mfi);
            Array4<Real const> const gold  = gravity_old_mf.array(mfi);
            Array4<Real const> const gravity_new_view  = gravity_new.array(mfi);
            Array4<Real const> const vol   = volume.array(mfi);
            Array4<Real const> const xface_flux = (*mass_face_flux[0]).array(mfi);
            Array4<Real const> const flux1 = (*mass_face_flux[1]).array(mfi);
            Array4<Real const> const flux2 = (*mass_face_flux[2]).array(mfi);
            Array4<Real> const source_view  = source.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                GpuArray<Real, NUM_SOURCE> src{};

                Real halfInvDt = 0.5_rt / dt;


                Real rho_at_old    = uold(i,j,k,URHO);
                Real rho_old_inv = 1.0_rt / uold(i,j,k,URHO);

                Real rho_at_new    = unew(i,j,k,URHO);
                Real rho_new_inv = 1.0_rt / unew(i,j,k,URHO);


                GpuArray<Real, NUM_STATE> snew{};
                for (int n = 0; n < NUM_STATE; ++n) {
                    snew[n] = unew(i,j,k,n);
                }

                Real kinetic_energy_old = 0.5_rt * (snew[UMX] * snew[UMX] + snew[UMY] * snew[UMY] + snew[UMZ] * snew[UMZ]) * rho_new_inv;


                GpuArray<Real, 3> vold;
                for (int n = 0; n < 3; ++n) {
                    vold[n] = uold(i,j,k,UMX+n) * rho_old_inv;
                }

                GpuArray<Real, 3> mom_source_old;
                for (int n = 0; n < 3; ++n) {
                    mom_source_old[n] = rho_at_old * gold(i,j,k,n);
                }

                Real energy_source_old = vold[0] * mom_source_old[0] + vold[1] * mom_source_old[1] + vold[2] * mom_source_old[2];


                GpuArray<Real, 3> velocity_new;
                for (int n = 0; n < 3; ++n) {
                    velocity_new[n] = snew[UMX+n] * rho_new_inv;
                }

                GpuArray<Real, 3> mom_source_new;
                for (int n = 0; n < 3; ++n) {
                    mom_source_new[n] = rho_at_new * gravity_new_view(i,j,k,n);
                }

                Real energy_source_new = velocity_new[0] * mom_source_new[0] + velocity_new[1] * mom_source_new[1] + velocity_new[2] * mom_source_new[2];


                GpuArray<Real, 3> mom_source_corr;
                for (int n = 0; n < 3; ++n) {
                    mom_source_corr[n] = 0.5_rt * (mom_source_new[n] - mom_source_old[n]);
                }


                for (int n = 0; n < 3; ++n) {
                    src[UMX+n] = mom_source_corr[n];
                    snew[UMX+n] += dt * src[UMX+n];
                }

#ifdef HYBRID_MOMENTUM
                GpuArray<Real, 3> loc;
                position(i, j, k, geomdata, loc);
                for (int n = 0; n < 3; ++n) {
                    loc[n] -= problem::center[n];
                }

                GpuArray<Real, 3> hybrid_mom_source;

                compute_hybrid_momentum_source(loc, mom_source_corr, hybrid_mom_source);

                for (int n = 0; n < 3; ++n) {
                    src[UMR+n] = hybrid_mom_source[n];
                    snew[UMR+n] += dt * src[UMR+n];
                }
#endif


                Real energy_source_corr{};

                if (hydrosim::gravity_source_mode == 1) {


                    energy_source_corr = 0.5_rt * (energy_source_new - energy_source_old);

                } else if (hydrosim::gravity_source_mode == 2) {


                    for (int n = 0; n < 3; ++n) {
                        velocity_new[n] = snew[UMX+n] * rho_new_inv;
                    }
                    energy_source_new = velocity_new[0] * mom_source_new[0] + velocity_new[1] * mom_source_new[1] + velocity_new[2] * mom_source_new[2];

                    energy_source_corr = 0.5_rt * (energy_source_new - energy_source_old);

                } else if (hydrosim::gravity_source_mode == 3) {


                    Real kinetic_energy_updated = 0.5_rt * (snew[UMX] * snew[UMX] + snew[UMY] * snew[UMY] + snew[UMZ] * snew[UMZ]) * rho_new_inv;
                    energy_source_corr = kinetic_energy_updated - kinetic_energy_old;

                } else if (hydrosim::gravity_source_mode == 4) {


                    energy_source_corr = - energy_source_old;



                    GpuArray<Real, 3> g;
                    for (int n = 0; n < 3; ++n) {
                        g[n] = 0.5_rt * (gravity_new_view(i,j,k,n) + gold(i,j,k,n));
                    }

                    Real gxl = 0.5_rt * (g[0] + 0.5_rt * (gravity_new_view(i-1*dg0,j,k,0) + gold(i-1*dg0,j,k,0)));
                    Real gxr = 0.5_rt * (g[0] + 0.5_rt * (gravity_new_view(i+1*dg0,j,k,0) + gold(i+1*dg0,j,k,0)));

                    Real gyl = 0.5_rt * (g[1] + 0.5_rt * (gravity_new_view(i,j-1*dg1,k,1) + gold(i,j-1*dg1,k,1)));
                    Real gyr = 0.5_rt * (g[1] + 0.5_rt * (gravity_new_view(i,j+1*dg1,k,1) + gold(i,j+1*dg1,k,1)));

                    Real gzl = 0.5_rt * (g[2] + 0.5_rt * (gravity_new_view(i,j,k-1*dg2,2) + gold(i,j,k-1*dg2,2)));
                    Real gzr = 0.5_rt * (g[2] + 0.5_rt * (gravity_new_view(i,j,k+1*dg2,2) + gold(i,j,k+1*dg2,2)));

                    energy_source_corr += halfInvDt * (xface_flux(i      ,j,k) * gxl * dx[0] +
                                         xface_flux(i+1*dg0,j,k) * gxr * dx[0] +
                                         flux1(i,j      ,k) * gyl * dx[1] +
                                         flux1(i,j+1*dg1,k) * gyr * dx[1] +
                                         flux2(i,j,k      ) * gzl * dx[2] +
                                         flux2(i,j,k+1*dg2) * gzr * dx[2]) / vol(i,j,k);

                }

                src[UEDEN] = energy_source_corr;

                snew[UEDEN] += dt * energy_source_corr;


                for (int n = 0; n < NUM_SOURCE; ++n) {
                    source_view(i,j,k,n) += src[n];
                }
            });
        }
    }

    if (hydrosim::verbose > 1)
    {
        const int IOProc = ParallelDescriptor::IOProcessorNumber();
        amrex::Real run_time = ParallelDescriptor::second() - strt_time;
        amrex::Real level_real = level;

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif
        ParallelDescriptor::ReduceRealMax(run_time,IOProc);

        amrex::Print() << "HydroSim::assemble_new_gravity_source() time = " << run_time
                       << " on level " << level_real << "\n" << "\n";
#ifdef BL_LAZY
        });
#endif
    }
}
