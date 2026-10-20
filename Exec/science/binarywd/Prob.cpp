#include <HydroSim.H>
#include <HydroSim_helpers.H>

#include <Gravity.H>

#include <AMReX_ParmParse.H>
#include <AMReX_buildInfo.H>

#include <binary_setup.H>

#include <array>
#include <fstream>

using namespace amrex;

void HydroSim::user_post_init() {

  using namespace binarywd;
  using namespace problem;

  user_post_timestep();

}

void HydroSim::user_post_restart() {

  using namespace binarywd;
  using namespace problem;

  Real time = state[State_Type].curTime();

  if (time == 0.0) {

      if (parent->NumDataLogs() > 0) {

          bool do_sums = false;

          const std::string datalogname = parent->DataLogName(0);

          {
              std::ifstream log(datalogname, std::ios::ate);
              do_sums = (log.tellg() == 0);
          }

          if (do_sums && (sum_step_period > 0 || sum_time_period > 0))
              tally_integrated_quantities();

      }

  }

  update_roche_radii();

}

void
HydroSim::user_post_timestep()
{
    BL_PROFILE("HydroSim::user_post_timestep()");

    using namespace binarywd;
    using namespace problem;

    if (level != 0) return;

    Real time = state[State_Type].curTime();
    Real dt = parent->dtLevel(0);

    if (time == 0.0) dt = 0.0;

    wd_update(time, dt);

    update_relaxation(time, dt);

    problem_sums();
}

void
HydroSim::update_relaxation(Real time, Real dt) {

    using namespace binarywd;
    using namespace problem;

    if (problem::problem != 1 || relaxation_is_done || mass_P <= 0.0 || mass_S <= 0.0 || dt <= 0.0) return;

    int coarse_level = 0;
    int finest_level = parent->finestLevel();
    int num_amr_levels = finest_level + 1;

    Vector< std::unique_ptr<MultiFab> > force(num_amr_levels);

    for (int lev = coarse_level; lev <= finest_level; ++lev) {

        const Real old_time = getLevel(lev).state[State_Type].prevTime();
        const Real new_time = getLevel(lev).state[State_Type].curTime();

        const Real ldt = new_time - old_time;

        force[lev] = std::make_unique<MultiFab>(getLevel(lev).grids, getLevel(lev).dmap, NUM_STATE, 0);
        force[lev]->setVal(0.0);

        MultiFab& S_new = getLevel(lev).get_new_data(State_Type);

        getLevel(lev).assemble_old_gravity_source(*force[lev], S_new, new_time, ldt);

        if (lev < parent->finestLevel()) {
            const MultiFab& mask = getLevel(lev+1).make_fine_mask();

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
            for (MFIter mfi(*force[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.tilebox();

                auto F = (*force[lev])[mfi].array();
                auto m = mask[mfi].array();

                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    for (int n = 0; n < NUM_STATE; ++n) {
                        F(i,j,k,n) *= m(i,j,k);
                    }
                });
            }
        }

    }

    std::array<Real, 3> force_P{}, force_S{};

    for (int lev = coarse_level; lev <= finest_level; ++lev) {

        MultiFab& S_new = getLevel(lev).get_new_data(State_Type);

        MultiFab& vol = getLevel(lev).Volume();

        auto geomdata = getLevel(lev).geom.data();

        const int coord_type = geom.Coord();

        const int* lo_bc = phys_bc.lo();

        const bool symm_lo_x = (lo_bc[0] == amrex::PhysBCType::symmetry);
        const bool symm_lo_y = (lo_bc[1] == amrex::PhysBCType::symmetry);
        const bool symm_lo_z = (lo_bc[2] == amrex::PhysBCType::symmetry);

        ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum> reduce_op;
        ReduceData<Real, Real, Real, Real, Real, Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();

            Array4<Real const> const mass_density_arr   = S_new[mfi].array(URHO);
            Array4<Real const> const force_arr = (*force[lev]).array(mfi);
            Array4<Real const> const cell_volume_arr   = vol.array(mfi);

            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                GpuArray<Real, 3> dF;
                for (int n = 0; n < 3; ++n) {
                    dF[n] = cell_volume_arr(i,j,k) * force_arr(i,j,k,UMX+n);
                }

                if (coord_type == 0) {

                    if (symm_lo_x) {
                        dF[0] = 0.0_rt;
                        dF[1] = 2.0_rt * dF[1];
                        dF[2] = 2.0_rt * dF[2];
                    }

                    if (symm_lo_y) {
                        dF[0] = 2.0_rt * dF[0];
                        dF[1] = 0.0_rt;
                        dF[2] = 2.0_rt * dF[2];
                    }

                    if (symm_lo_z) {
                        dF[0] = 2.0_rt * dF[0];
                        dF[1] = 2.0_rt * dF[1];
                        dF[2] = 0.0_rt;
                    }

                } else if (coord_type == 1) {

                    dF[0] = 0.0_rt;

                }

                Real primary_factor = 0.0_rt;
                Real secondary_factor = 0.0_rt;

                if (stellar_mask(i, j, k, geomdata, mass_density_arr(i,j,k), StarRole::Primary) > 0.0_rt) {

                    primary_factor = 1.0_rt;

                } else if (stellar_mask(i, j, k, geomdata, mass_density_arr(i,j,k), StarRole::Secondary) > 0.0_rt) {

                    secondary_factor = 1.0_rt;

                }

                return {dF[0] * primary_factor,
                        dF[1] * primary_factor,
                        dF[2] * primary_factor,
                        dF[0] * secondary_factor,
                        dF[1] * secondary_factor,
                        dF[2] * secondary_factor};
            });

        }

        ReduceTuple hv = reduce_data.value();

        force_P[0] += amrex::get<0>(hv);
        force_P[1] += amrex::get<1>(hv);
        force_P[2] += amrex::get<2>(hv);

        force_S[0] += amrex::get<3>(hv);
        force_S[1] += amrex::get<4>(hv);
        force_S[2] += amrex::get<5>(hv);
    }

    amrex::ParallelDescriptor::ReduceRealSum({force_P[0], force_P[1], force_P[2], force_S[0], force_S[1], force_S[2]});

    auto vector_magnitude = [] (const Real* v) {
        return std::sqrt(std::pow(v[0], 2) + std::pow(v[1], 2) + std::pow(v[2], 2));
    };

    Real fp = vector_magnitude(force_P.data());
    Real fs = vector_magnitude(force_S.data());

    Real ap = vector_magnitude(com_P);
    Real as = vector_magnitude(com_S);

    Real omega = 0.5 * (std::sqrt((fp / mass_P) / ap) + std::sqrt((fs / mass_S) / as));

    Real period = 2.0 * M_PI / omega;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::cout << "\n";
        std::cout << "  Updating the rotational period from " << spin_period << " s to " << period << " s." << "\n";
        std::cout << "\n";
    }

    spin_period = period;

    GpuArray<Real, 3> L1, L2, L3;
    find_equilibrium_points(mass_P, mass_S, com_P, com_S, L1, L2, L3);

    const auto dx = geom.CellSizeArray();
    const auto coord = geom.Coord();
    GeometryData geomdata = geom.data();

    MultiFab& phi_new = get_new_data(Potential_Type);

    Real potential;

    {

        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(phi_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

            const Box& bx = mfi.tilebox();

            Array4<Real const> const phi = phi_new[mfi].array();

            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                GpuArray<Real, 3> r;
                position(i, j, k, geomdata, r);

                auto angular_velocity_vec = get_angular_velocity(geomdata, j);
                Real phiEff = phi(i,j,k) + centrifugal_potential(r, angular_velocity_vec, coord);

                for (int n = 0; n < 3; ++n) {
                    r[n] -= L1[n];
                }

                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    r[n] /= dx[n];
                }
                for (int n = AMREX_SPACEDIM; n < 3; ++n) {
                    r[n] = 0.0_rt;
                }

                Real dP = 0.0_rt;

                if ((r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) < 1.0_rt) {
                    dP = (1.0_rt - std::abs(r[0])) * (1.0_rt - std::abs(r[1])) * (1.0_rt - std::abs(r[2])) * phiEff;
                }

                return dP;

            });

        }

        ReduceTuple hv = reduce_data.value();
        potential = amrex::get<0>(hv);

        amrex::ParallelDescriptor::ReduceRealSum(potential);
    }

    {

        MultiFab& S_new = get_new_data(State_Type);

        ReduceOps<ReduceOpSum> reduce_op;
        ReduceData<Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

            const Box& bx = mfi.tilebox();

            Array4<Real const> const u = S_new.array(mfi);
            Array4<Real const> const phi = phi_new.array(mfi);

            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                GpuArray<Real, 3> r;
                position(i, j, k, geomdata, r);

                auto angular_velocity_vec = get_angular_velocity(geomdata, j);
                Real phiEff = phi(i,j,k) + centrifugal_potential(r, angular_velocity_vec, coord);

                Real done = 0.0_rt;

                if (phiEff > potential && u(i,j,k,URHO) > relaxation_density_cutoff) {
                    done = 1.0_rt;
                }

                return done;
            });

        }

        ReduceTuple hv = reduce_data.value();
        Real is_done = amrex::get<0>(hv);

        amrex::ParallelDescriptor::ReduceRealSum(is_done);

        if (is_done > 0.0) {
            relaxation_is_done = 1;
            amrex::Print() << "Disabling relaxation at time " << time
                           << "s because the critical density threshold has been passed."
                           << std::endl;
        }

    }

    if (relaxation_cutoff_time > 0.0 && time > relaxation_cutoff_time * std::max(t_ff_P, t_ff_S)) {
        relaxation_is_done = 1;
        amrex::Print() << "Disabling relaxation at time " << time
                       << "s because the maximum number of dynamical timescales has passed."
                       << std::endl;
    }

    if (relaxation_is_done == 1 && radial_damping_velocity_factor > 0.0_rt && radial_damping_is_done != 1) {
        amrex::Print() << "Radial damping force will now take effect." << std::endl;
    }
}

void HydroSim::writeGitHashes(std::ostream& log) {

  const char* hydrosim_hash       = buildInfoGetGitHash(1);
  const char* framework_hash        = buildInfoGetGitHash(2);
  const char* physics_lib_hash = buildInfoGetGitHash(3);

  log << "# HydroSim       git hash: " << hydrosim_hash       << std::endl;
  log << "# AMReX        git hash: " << framework_hash        << std::endl;
  log << "# Microphysics git hash: " << physics_lib_hash << std::endl;

}
