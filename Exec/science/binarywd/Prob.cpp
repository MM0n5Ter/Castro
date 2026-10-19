#include <HydroSim.H>
#include <HydroSim_helpers.H>

#include <Gravity.H>

#include <AMReX_ParmParse.H>
#include <AMReX_buildInfo.H>

#include <binarywd_util.H>
#include <binarywd_data.H>
#include <binary.H>

#include <fstream>

using namespace amrex;

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
HydroSim::wd_update (Real time, Real dt)
{
    BL_PROFILE("HydroSim::wd_update()");

    using namespace binarywd;
    using namespace problem;


    BL_ASSERT(level == 0 || (!parent->subCycle() && level == parent->finestLevel()));

    for (int i = 0; i < 3; i++) {
      com_P[i] += vel_P[i] * dt;
      com_S[i] += vel_S[i] * dt;
    }



    Real old_mass_P = mass_P;
    Real old_mass_S = mass_S;

    ReduceOps<ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum,
              ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum, ReduceOpSum,
              ReduceOpSum, ReduceOpSum> reduce_op;
    ReduceData<Real, Real, Real, Real, Real, Real,
               Real, Real, Real, Real, Real, Real,
               Real, Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (int lev = 0; lev <= parent->finestLevel(); lev++) {

        HydroSim& c_lev = getLevel(lev);
        MultiFab& S_new = c_lev.get_new_data(State_Type);

        GeometryData geomdata = c_lev.geom.data();

        const auto problo = c_lev.geom.ProbLoArray();
        const auto probhi = c_lev.geom.ProbHiArray();
        int coord_type = c_lev.geom.Coord();

        GpuArray<bool, 3> symm_bound_lo{false};
        GpuArray<bool, 3> symm_bound_hi{false};

        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            if (phys_bc.lo()[n] == amrex::PhysBCType::symmetry) {
                symm_bound_lo[n] = true;
            }
            if (phys_bc.hi()[n] == amrex::PhysBCType::symmetry) {
                symm_bound_hi[n] = true;
            }
        }

        bool fine_mask_present = true;
        if (lev == parent->finestLevel()) {
            fine_mask_present = false;
        }

        MultiFab scratch_mf;
        const MultiFab& fine_cover_mf = fine_mask_present ? getLevel(lev+1).make_fine_mask() : scratch_mf;

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto rho   = S_new[mfi].array(URHO);
            auto xmom  = S_new[mfi].array(UMX);
            auto ymom  = S_new[mfi].array(UMY);
            auto zmom  = S_new[mfi].array(UMZ);
            auto vol   = c_lev.volume[mfi].array();
            auto level_mask = fine_mask_present ? fine_cover_mf[mfi].array() : Array4<Real>{};

            const Box& box  = mfi.tilebox();


            reduce_op.eval(box, reduce_data,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
            {


                GpuArray<Real, 3> r;
                position(i, j, k, geomdata, r);


                GpuArray<Real, 3> rSymmetric{r[0], r[1], r[2]};
                for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                    if (symm_bound_lo[n]) {
                        rSymmetric[n] = rSymmetric[n] + (problo[n] - rSymmetric[n]);
                    }
                    if (symm_bound_hi[n]) {
                        rSymmetric[n] = rSymmetric[n] + (rSymmetric[n] - probhi[n]);
                    }
                }

                Real zoneWeight = 1.0;

                if (fine_mask_present) {
                    zoneWeight = level_mask(i,j,k);
                }

                Real dm = rho(i,j,k) * vol(i,j,k) * zoneWeight;

                Real dmSymmetric = dm;
                GpuArray<Real, 3> momSymmetric{xmom(i,j,k) * zoneWeight,
                                               ymom(i,j,k) * zoneWeight,
                                               zmom(i,j,k) * zoneWeight};

                if (coord_type == 0) {

                    if (symm_bound_lo[0]) {
                        dmSymmetric *= 2.0_rt;
                        for (int n = 0; n < 3; ++n) {
                            momSymmetric[n] *= 2.0_rt;
                        }
                    }

                    if (symm_bound_lo[1]) {
                        dmSymmetric *= 2.0_rt;
                        for (int n = 0; n < 3; ++n) {
                            momSymmetric[n] *= 2.0_rt;
                        }
                    }

                    if (symm_bound_lo[2]) {
                        dmSymmetric *= 2.0_rt;
                        for (int n = 0; n < 3; ++n) {
                            momSymmetric[n] *= 2.0_rt;
                        }
                    }

                }

                Real primary_factor = 0.0_rt;
                Real secondary_factor = 0.0_rt;

                if (stellar_mask(i, j, k, geomdata, rho(i,j,k), true) * zoneWeight > 0.0_rt) {

                    primary_factor = 1.0_rt;

                }
                else if (stellar_mask(i, j, k, geomdata, rho(i,j,k), false) * zoneWeight > 0.0_rt) {

                    secondary_factor = 1.0_rt;

                }

                Real com_P_x = dmSymmetric * rSymmetric[0] * primary_factor;
                Real com_P_y = dmSymmetric * rSymmetric[1] * primary_factor;
                Real com_P_z = dmSymmetric * rSymmetric[2] * primary_factor;

                Real com_S_x = dmSymmetric * rSymmetric[0] * secondary_factor;
                Real com_S_y = dmSymmetric * rSymmetric[1] * secondary_factor;
                Real com_S_z = dmSymmetric * rSymmetric[2] * secondary_factor;

                Real vel_P_x = momSymmetric[0] * vol(i,j,k) * primary_factor;
                Real vel_P_y = momSymmetric[1] * vol(i,j,k) * primary_factor;
                Real vel_P_z = momSymmetric[2] * vol(i,j,k) * primary_factor;

                Real vel_S_x = momSymmetric[0] * vol(i,j,k) * secondary_factor;
                Real vel_S_y = momSymmetric[1] * vol(i,j,k) * secondary_factor;
                Real vel_S_z = momSymmetric[2] * vol(i,j,k) * secondary_factor;

                Real m_P = dmSymmetric * primary_factor;
                Real m_S = dmSymmetric * secondary_factor;

                return {com_P_x, com_P_y, com_P_z, com_S_x, com_S_y, com_S_z,
                        vel_P_x, vel_P_y, vel_P_z, vel_S_x, vel_S_y, vel_S_z,
                        m_P, m_S};
            });

        }

    }


    bool local_only = true;

    for (int i = 0; i <= 6; ++i) {
        HydroSim::volInBoundary(time, vol_P[i], vol_S[i], std::pow(10.0,i), local_only);
    }


    ReduceTuple hv = reduce_data.value();

    com_P[0] = amrex::get<0>(hv);
    com_P[1] = amrex::get<1>(hv);
    com_P[2] = amrex::get<2>(hv);
    com_S[0] = amrex::get<3>(hv);
    com_S[1] = amrex::get<4>(hv);
    com_S[2] = amrex::get<5>(hv);
    vel_P[0] = amrex::get<6>(hv);
    vel_P[1] = amrex::get<7>(hv);
    vel_P[2] = amrex::get<8>(hv);
    vel_S[0] = amrex::get<9>(hv);
    vel_S[1] = amrex::get<10>(hv);
    vel_S[2] = amrex::get<11>(hv);
    mass_P   = amrex::get<12>(hv);
    mass_S   = amrex::get<13>(hv);

    const int sum_reduce_count = 28;
    Real sum_reduce_buffer[sum_reduce_count] = { 0.0 };

    for (int i = 0; i <= 6; ++i) {
        sum_reduce_buffer[i  ] = vol_P[i];
        sum_reduce_buffer[i+7] = vol_S[i];
    }

    sum_reduce_buffer[14] = mass_P;
    sum_reduce_buffer[15] = mass_S;

    for (int i = 0; i <= 2; ++i) {
        sum_reduce_buffer[i+16] = com_P[i];
        sum_reduce_buffer[i+19] = com_S[i];
        sum_reduce_buffer[i+22] = vel_P[i];
        sum_reduce_buffer[i+25] = vel_S[i];
    }

    amrex::ParallelDescriptor::ReduceRealSum(sum_reduce_buffer, sum_reduce_count);

    for (int i = 0; i <= 6; ++i) {
        vol_P[i] = sum_reduce_buffer[i  ];
        vol_S[i] = sum_reduce_buffer[i+7];
    }

    mass_P = sum_reduce_buffer[14];
    mass_S = sum_reduce_buffer[15];

    for (int i = 0; i <= 2; ++i) {
        com_P[i] = sum_reduce_buffer[i+16];
        com_S[i] = sum_reduce_buffer[i+19];
        vel_P[i] = sum_reduce_buffer[i+22];
        vel_S[i] = sum_reduce_buffer[i+25];
    }


    for (int i = 0; i <= 6; ++i) {
        rad_P[i] = std::pow(vol_P[i] * 3.0 / 4.0 / M_PI, 1.0/3.0);
        rad_S[i] = std::pow(vol_S[i] * 3.0 / 4.0 / M_PI, 1.0/3.0);
    }


    for (int i = 0; i < 3; i++) {
        if (mass_P > 0.0) {
            com_P[i] = com_P[i] / mass_P;
            vel_P[i] = vel_P[i] / mass_P;
        }

        if (mass_S > 0.0) {
            com_S[i] = com_S[i] / mass_S;
            vel_S[i] = vel_S[i] / mass_S;
        }
    }


#if (AMREX_SPACEDIM == 1)
    mass_P = old_mass_P;
    mass_S = old_mass_S;
#endif

    if (mass_P > 0.0 && dt > 0.0) {
        mdot_P = (mass_P - old_mass_P) / dt;
    }
    else {
        mdot_P = 0.0;
    }

    if (mass_S > 0.0 && dt > 0.0) {
        mdot_S = (mass_S - old_mass_S) / dt;
    }
    else {
        mdot_S = 0.0;
    }


    if (mass_P > 0.0 && vol_P[2] > 0.0) {
        rho_avg_P = mass_P / vol_P[2];
        t_ff_P = std::sqrt(3.0 * M_PI / (32.0 * C::Gconst * rho_avg_P));
    }

    if (mass_S > 0.0 && vol_S[2] > 0.0) {
        rho_avg_S = mass_S / vol_S[2];
        t_ff_S = std::sqrt(3.0 * M_PI / (32.0 * C::Gconst * rho_avg_S));
    }


    update_roche_radii();
}




void HydroSim::volInBoundary (Real time, Real& vol_P, Real& vol_S, Real rho_cutoff, bool local)
{

    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::volInBoundary()");

    using namespace binarywd;
    using namespace problem;

    BL_ASSERT(level == 0);

    vol_P = 0.0;
    vol_S = 0.0;

    for (int lev = 0; lev <= parent->finestLevel(); lev++) {

      HydroSim& c_lev = getLevel(lev);

      auto geomdata = c_lev.geom.data();
      MultiFab& S_new = c_lev.get_new_data(State_Type);

      bool fine_mask_present = true;
      if (lev == parent->finestLevel()) {
          fine_mask_present = false;
      }

      MultiFab scratch_mf;
      const MultiFab& fine_cover_mf = fine_mask_present ? getLevel(lev+1).make_fine_mask() : scratch_mf;

      ReduceOps<ReduceOpSum, ReduceOpSum> reduce_op;
      ReduceData<Real, Real> reduce_data(reduce_op);
      using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
      for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
          auto rho   = S_new[mfi].array(URHO);
          auto vol   = c_lev.volume[mfi].array();
          auto level_mask = fine_mask_present ? fine_cover_mf[mfi].array() : Array4<Real>{};

          const Box& box  = mfi.tilebox();


          reduce_op.eval(box, reduce_data,
          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
          {
              Real primary_factor = 0.0_rt;
              Real secondary_factor = 0.0_rt;

              Real zoneWeight = 1.0;
              if (fine_mask_present) {
                  zoneWeight = level_mask(i,j,k);
              }

              if (rho(i,j,k) * zoneWeight > rho_cutoff) {

                  if (stellar_mask(i, j, k, geomdata, rho(i,j,k), true) * zoneWeight > 0.0_rt) {

                      primary_factor = 1.0_rt;

                  }
                  else if (stellar_mask(i, j, k, geomdata, rho(i,j,k), false) * zoneWeight > 0.0_rt) {

                      secondary_factor = 1.0_rt;

                  }

              }

              return {vol(i,j,k) * zoneWeight * primary_factor,
                      vol(i,j,k) * zoneWeight * secondary_factor};
          });

      }

      ReduceTuple hv = reduce_data.value();

      vol_P += amrex::get<0>(hv);
      vol_S += amrex::get<1>(hv);

    }

    if (!local)
      amrex::ParallelDescriptor::ReduceRealSum({vol_P, vol_S});

}




Real HydroSim::dot_product(const Real a[], const Real b[]) {

  Real c = 0.0;

  c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

  return c;

}





Real HydroSim::norm(const Real a[]) {

  Real n = 0.0;

  n = std::sqrt(dot_product(a, a));

  return n;

}



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

          std::ifstream log;
          log.open(datalogname, std::ios::ate);

          if (log.tellg() == 0) {
              do_sums = true;
          }
          log.close();

          if (do_sums && (sum_step_period > 0 || sum_time_period > 0))
              tally_integrated_quantities();

      }

  }


  update_roche_radii();

}



void HydroSim::writeGitHashes(std::ostream& log) {

  const char* hydrosim_hash       = buildInfoGetGitHash(1);
  const char* framework_hash        = buildInfoGetGitHash(2);
  const char* physics_lib_hash = buildInfoGetGitHash(3);

  log << "# HydroSim       git hash: " << hydrosim_hash       << std::endl;
  log << "# AMReX        git hash: " << framework_hash        << std::endl;
  log << "# Microphysics git hash: " << physics_lib_hash << std::endl;

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


    Real force_P[3] = { 0.0 };
    Real force_S[3] = { 0.0 };

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

                if (stellar_mask(i, j, k, geomdata, mass_density_arr(i,j,k), true) > 0.0_rt) {

                    primary_factor = 1.0_rt;

                } else if (stellar_mask(i, j, k, geomdata, mass_density_arr(i,j,k), false) > 0.0_rt) {

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


    Real fp = std::sqrt(std::pow(force_P[0], 2) + std::pow(force_P[1], 2) + std::pow(force_P[2], 2));
    Real fs = std::sqrt(std::pow(force_S[0], 2) + std::pow(force_S[1], 2) + std::pow(force_S[2], 2));

    Real ap = std::sqrt(std::pow(com_P[0], 2) + std::pow(com_P[1], 2) + std::pow(com_P[2], 2));
    Real as = std::sqrt(std::pow(com_S[0], 2) + std::pow(com_S[1], 2) + std::pow(com_S[2], 2));

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



void
HydroSim::problem_sums ()
{
    using namespace binarywd;
    using namespace problem;

    if (level > 0) return;

    Real time = state[State_Type].curTime();

    int timestep = parent->levelSteps(0);


    Real mdot = 0.5 * (std::abs(mdot_P) + std::abs(mdot_S));


    Real wd_dist[3] = { 0.0 };
    Real wd_dist_init[3] = { 0.0 };

    Real separation = 0.0;
    Real angle = 0.0;


    Real com_P_mag = 0.0;
    Real com_S_mag = 0.0;

    Real vel_P_mag = 0.0;
    Real vel_S_mag = 0.0;

    Real vel_P_rad = 0.0;
    Real vel_S_rad = 0.0;

    Real vel_P_phi = 0.0;
    Real vel_S_phi = 0.0;

    int dataprecision = 16;

    int sci_column_width      = 25;
    int fixed_column_width      = 25;
    int integer_column_width      = 12;

    wd_dist_init[problem::axis_1 - 1] = 1.0;

    com_P_mag += std::pow(std::pow(com_P[0],2) + std::pow(com_P[1],2) + std::pow(com_P[2],2), 0.5);
    com_S_mag += std::pow(std::pow(com_S[0],2) + std::pow(com_S[1],2) + std::pow(com_S[2],2), 0.5);
    vel_P_mag += std::pow(std::pow(vel_P[0],2) + std::pow(vel_P[1],2) + std::pow(vel_P[2],2), 0.5);
    vel_S_mag += std::pow(std::pow(vel_S[0],2) + std::pow(vel_S[1],2) + std::pow(vel_S[2],2), 0.5);

#if (AMREX_SPACEDIM == 3)
    if (mass_P > 0.0) {
      vel_P_rad = (com_P[problem::axis_1 - 1] / com_P_mag) * vel_P[problem::axis_1 - 1] +
                  (com_P[problem::axis_2 - 1] / com_P_mag) * vel_P[problem::axis_2 - 1];
      vel_P_phi = (com_P[problem::axis_1 - 1] / com_P_mag) * vel_P[problem::axis_2 - 1] -
                  (com_P[problem::axis_2 - 1] / com_P_mag) * vel_P[problem::axis_1 - 1];
    }

    if (mass_S > 0.0) {
      vel_S_rad = (com_S[problem::axis_1 - 1] / com_S_mag) * vel_S[problem::axis_1 - 1] +
                  (com_S[problem::axis_2 - 1] / com_S_mag) * vel_S[problem::axis_2 - 1];
      vel_S_phi = (com_S[problem::axis_1 - 1] / com_S_mag) * vel_S[problem::axis_2 - 1] -
                  (com_S[problem::axis_2 - 1] / com_S_mag) * vel_S[problem::axis_1 - 1];
    }
#else
    if (mass_P > 0.0) {
      vel_P_rad = vel_P[problem::axis_1 - 1];
      vel_P_phi = vel_P[problem::axis_3 - 1];
    }

    if (mass_S > 0.0) {
      vel_S_rad = vel_S[problem::axis_1 - 1];
      vel_S_phi = vel_S[problem::axis_3 - 1];
    }
#endif

    if (mass_P > 0.0 && mass_S > 0.0) {


      for (int i = 0; i < 3; i++) {
          wd_dist[i] = com_S[i] - com_P[i];
      }

      separation = norm(wd_dist);


      angle = std::atan2(wd_dist[problem::axis_2 - 1] - wd_dist_init[problem::axis_2 - 1],
                         wd_dist[problem::axis_1 - 1] - wd_dist_init[problem::axis_1 - 1]) * 180.0 / M_PI;


      if (angle < 0.0) angle += 360.0;

    }


    if (amrex::ParallelDescriptor::IOProcessor())
    {


      if (parent->NumDataLogs() > 0) {

          std::ostream& log = parent->DataLog(0);

          if (log.good()) {

              if (time == 0.0) {


                  writeGitHashes(log);

                  int n = 0;

                  std::ostringstream header;

                  header << std::setw(integer_column_width) << "#   TIMESTEP";              ++n;
                  header << std::setw(fixed_column_width) << "                     TIME"; ++n;

                  header << std::setw(sci_column_width) << "              WD DISTANCE"; ++n;
                  header << std::setw(fixed_column_width) << "                 WD ANGLE"; ++n;
                  header << std::setw(sci_column_width) << "                     ADOT"; ++n;
                  header << std::setw(sci_column_width) << "                     MDOT"; ++n;

                  header << std::endl;

                  log << std::setw(integer_column_width) << "#   COLUMN 1";
                  log << std::setw(fixed_column_width) << "                        2";

                  for (int i = 3; i <= n; ++i) {
                      log << std::setw(sci_column_width) << i;
                  }

                  log << std::endl;

                  log << header.str();

              }

              log << std::fixed;

              log << std::setw(integer_column_width)                                     << timestep;
              log << std::setw(fixed_column_width) << std::setprecision(dataprecision) << time;

              log << std::scientific;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << separation;

              log << std::fixed;
              log << std::setw(fixed_column_width) << std::setprecision(dataprecision) << angle;

              log << std::scientific;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << vel_P_rad + vel_S_rad;

              log << std::scientific;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << mdot;

              log << std::endl;

          }
      }


      if (parent->NumDataLogs() > 1) {

          std::ostream& log = parent->DataLog(1);

          if (log.good()) {

              if (time == 0.0) {


                  writeGitHashes(log);

                  int n = 0;

                  std::ostringstream header;

                  header << std::setw(integer_column_width) << "#   TIMESTEP";              ++n;
                  header << std::setw(fixed_column_width) << "                     TIME"; ++n;
                  header << std::setw(sci_column_width) << "             PRIMARY MASS"; ++n;
                  header << std::setw(sci_column_width) << "             PRIMARY MDOT"; ++n;
                  header << std::setw(sci_column_width) << "          PRIMARY MAG COM"; ++n;
#if (AMREX_SPACEDIM == 3)
                  header << std::setw(sci_column_width) << "            PRIMARY X COM"; ++n;
                  header << std::setw(sci_column_width) << "            PRIMARY Y COM"; ++n;
                  header << std::setw(sci_column_width) << "            PRIMARY Z COM"; ++n;
#else
                  header << std::setw(sci_column_width) << "            PRIMARY R COM"; ++n;
                  header << std::setw(sci_column_width) << "            PRIMARY Z COM"; ++n;
#endif
                  header << std::setw(sci_column_width) << "          PRIMARY MAG VEL"; ++n;
                  header << std::setw(sci_column_width) << "          PRIMARY RAD VEL"; ++n;
                  header << std::setw(sci_column_width) << "          PRIMARY ANG VEL"; ++n;
#if (AMREX_SPACEDIM == 3)
                  header << std::setw(sci_column_width) << "            PRIMARY X VEL"; ++n;
                  header << std::setw(sci_column_width) << "            PRIMARY Y VEL"; ++n;
                  header << std::setw(sci_column_width) << "            PRIMARY Z VEL"; ++n;
#else
                  header << std::setw(sci_column_width) << "            PRIMARY R VEL"; ++n;
                  header << std::setw(sci_column_width) << "            PRIMARY Z VEL"; ++n;
#endif
                  header << std::setw(sci_column_width) << "       PRIMARY T_FREEFALL"; ++n;
                  header << std::setw(sci_column_width) << "     PRIMARY ROCHE RADIUS"; ++n;
                  for (int i = 0; i <= 6; ++i) {
                      header << "       PRIMARY 1E" << i << " RADIUS";          ++n;
                  }

                  header << std::endl;

                  log << std::setw(integer_column_width) << "#   COLUMN 1";
                  log << std::setw(fixed_column_width) << "                        2";

                  for (int i = 3; i <= n; ++i) {
                      log << std::setw(sci_column_width) << i;
                  }

                  log << std::endl;

                  log << header.str();

              }

              log << std::fixed;

              log << std::setw(integer_column_width)                                     << timestep;
              log << std::setw(fixed_column_width) << std::setprecision(dataprecision) << time;

              log << std::scientific;

              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << mass_P;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << mdot_P;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << com_P_mag;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << com_P[0];
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << com_P[1];
#if (AMREX_SPACEDIM == 3)
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << com_P[2];
#endif
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << vel_P_mag;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << vel_P_rad;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << vel_P_phi;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << vel_P[0];
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << vel_P[1];
#if (AMREX_SPACEDIM == 3)
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << vel_P[2];
#endif
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << t_ff_P;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << roche_rad_P;
              for (int i = 0; i <= 6; ++i) {
                  log << std::setw(sci_column_width) << std::setprecision(dataprecision) << rad_P[i];
              }

              log << std::endl;

          }

      }


      if (parent->NumDataLogs() > 2) {

          std::ostream& log = parent->DataLog(2);

          if (log.good()) {

              if (time == 0.0) {


                  writeGitHashes(log);

                  int n = 0;

                  std::ostringstream header;

                  header << std::setw(integer_column_width) << "#   TIMESTEP";              ++n;
                  header << std::setw(fixed_column_width) << "                     TIME"; ++n;
                  header << std::setw(sci_column_width) << "           SECONDARY MASS"; ++n;
                  header << std::setw(sci_column_width) << "           SECONDARY MDOT"; ++n;
                  header << std::setw(sci_column_width) << "        SECONDARY MAG COM"; ++n;
#if (AMREX_SPACEDIM == 3)
                  header << std::setw(sci_column_width) << "          SECONDARY X COM"; ++n;
                  header << std::setw(sci_column_width) << "          SECONDARY Y COM"; ++n;
                  header << std::setw(sci_column_width) << "          SECONDARY Z COM"; ++n;
#else
                  header << std::setw(sci_column_width) << "          SECONDARY R COM"; ++n;
                  header << std::setw(sci_column_width) << "          SECONDARY Z COM"; ++n;
#endif
                  header << std::setw(sci_column_width) << "        SECONDARY MAG VEL"; ++n;
                  header << std::setw(sci_column_width) << "        SECONDARY RAD VEL"; ++n;
                  header << std::setw(sci_column_width) << "        SECONDARY ANG VEL"; ++n;
#if (AMREX_SPACEDIM == 3)
                  header << std::setw(sci_column_width) << "          SECONDARY X VEL"; ++n;
                  header << std::setw(sci_column_width) << "          SECONDARY Y VEL"; ++n;
                  header << std::setw(sci_column_width) << "          SECONDARY Z VEL"; ++n;
#else
                  header << std::setw(sci_column_width) << "          SECONDARY R VEL"; ++n;
                  header << std::setw(sci_column_width) << "          SECONDARY Z VEL"; ++n;
#endif
                  header << std::setw(sci_column_width) << "     SECONDARY T_FREEFALL"; ++n;
                  header << std::setw(sci_column_width) << "   SECONDARY ROCHE RADIUS"; ++n;
                  for (int i = 0; i <= 6; ++i) {
                      header << "     SECONDARY 1E" << i << " RADIUS";          ++n;
                  }

                  header << std::endl;

                  log << std::setw(integer_column_width) << "#   COLUMN 1";
                  log << std::setw(fixed_column_width) << "                        2";

                  for (int i = 3; i <= n; ++i) {
                      log << std::setw(sci_column_width) << i;
                  }

                  log << std::endl;

                  log << header.str();

              }

              log << std::fixed;

              log << std::setw(integer_column_width)                                     << timestep;
              log << std::setw(fixed_column_width) << std::setprecision(dataprecision) << time;

              log << std::scientific;

              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << mass_S;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << mdot_S;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << com_S_mag;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << com_S[0];
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << com_S[1];
#if (AMREX_SPACEDIM == 3)
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << com_S[2];
#endif
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << vel_S_mag;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << vel_S_rad;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << vel_S_phi;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << vel_S[0];
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << vel_S[1];
#if (AMREX_SPACEDIM == 3)
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << vel_S[2];
#endif
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << t_ff_S;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << roche_rad_S;
              for (int i = 0; i <= 6; ++i) {
                  log << std::setw(sci_column_width) << std::setprecision(dataprecision) << rad_S[i];
              }

              log << std::endl;

          }

      }


      if (parent->NumDataLogs() > 3) {

          std::ostream& log = parent->DataLog(3);

          if (log.good()) {

              if (time == 0.0) {


                  writeGitHashes(log);

                  int n = 0;

                  std::ostringstream header;

                  header << std::setw(integer_column_width) << "#   TIMESTEP";              ++n;
                  header << std::setw(fixed_column_width) << "                     TIME"; ++n;
                  header << std::setw(sci_column_width) << "          ROTATION PERIOD"; ++n;
                  header << std::setw(sci_column_width) << "       ROTATION FREQUENCY"; ++n;

                  header << std::endl;

                  log << std::setw(integer_column_width) << "#   COLUMN 1";
                  log << std::setw(fixed_column_width) << "                        2";

                  for (int i = 3; i <= n; ++i) {
                      log << std::setw(sci_column_width) << i;
                  }

                  log << std::endl;

                  log << header.str();

              }

              log << std::fixed;

              log << std::setw(integer_column_width)                                     << timestep;
              log << std::setw(fixed_column_width) << std::setprecision(dataprecision) << time;

              log << std::scientific;

              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << spin_period;
              log << std::setw(sci_column_width) << std::setprecision(dataprecision) << (2.0 * M_PI / spin_period);

              log << std::endl;

          }

      }

    }
}
