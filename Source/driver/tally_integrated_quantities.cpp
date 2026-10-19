#include <iomanip>

#include <HydroSim.H>

#ifdef GRAVITY
#include <Gravity.H>
#endif

#include <user_diagnostics.H>

using namespace amrex;

void
HydroSim::tally_integrated_quantities ()
{
    if (verbose <= 0) {
        return;
    }

    BL_PROFILE("HydroSim::tally_integrated_quantities()");

    bool local_only = true;

    int finest_level = parent->finestLevel();
    Real time        = state[State_Type].curTime();
    Real dt          = parent->dtLevel(0);
    if (time == 0.0) {
        dt = 0.0;
    }
    int timestep     = parent->levelSteps(0);
    Real mass        = 0.0;
    Real mom[3]      = { 0.0 };
    Real angular_momentum_sum[3]  = { 0.0 };
#ifdef HYBRID_MOMENTUM
    Real hybrid_momentum_sum[3]  = { 0.0 };
#endif
    Real com[3]      = { 0.0 };
    Real center_velocity[3]  = { 0.0 };
    Real rho_e       = 0.0;
    Real rho_kinetic       = 0.0;
    Real rho_E       = 0.0;
#ifdef GRAVITY
    Real rho_potential     = 0.0;
    Real energy_budget = 0.0;
#endif

    Real T_max       = 0.0;
    Real rho_max     = 0.0;
    Real timescale_ratio_max   = 0.0;

    int output_precision = 16;

    int sci_column_width     = 25;
    int fixed_column_width     = 25;
    int integer_column_width     = 12;

    for (int lev = 0; lev <= finest_level; lev++)
    {
        HydroSim& hs_level = getLevel(lev);
        const MultiFab& S_new = hs_level.get_new_data(State_Type);
#ifdef GRAVITY
        MultiFab& phi_new = hs_level.get_new_data(Potential_Type);
#endif
#ifdef REACTIONS
        MultiFab& react_new = hs_level.get_new_data(Burning_Type);
#endif

        mass   += hs_level.volWeightedTotal(S_new, URHO, local_only);
        mom[0] += hs_level.volWeightedTotal(S_new, UMX, local_only);
        mom[1] += hs_level.volWeightedTotal(S_new, UMY, local_only);
        mom[2] += hs_level.volWeightedTotal(S_new, UMZ, local_only);

        angular_momentum_sum[0] += hs_level.volWeightedTotal("angular_momentum_x", time, local_only);
        angular_momentum_sum[1] += hs_level.volWeightedTotal("angular_momentum_y", time, local_only);
        angular_momentum_sum[2] += hs_level.volWeightedTotal("angular_momentum_z", time, local_only);

#ifdef HYBRID_MOMENTUM
        hybrid_momentum_sum[0] += hs_level.volWeightedTotal(S_new, UMR, local_only);
        hybrid_momentum_sum[1] += hs_level.volWeightedTotal(S_new, UML, local_only);
        hybrid_momentum_sum[2] += hs_level.volWeightedTotal(S_new, UMP, local_only);
#endif

        com[0] += hs_level.locWeightedSum(S_new, URHO, 0, local_only);
        com[1] += hs_level.locWeightedSum(S_new, URHO, 1, local_only);
        com[2] += hs_level.locWeightedSum(S_new, URHO, 2, local_only);

        rho_e += hs_level.volWeightedTotal(S_new, UEINT, local_only);
        rho_kinetic += hs_level.volWeightedTotal("kineng", time, local_only);
        rho_E += hs_level.volWeightedTotal(S_new, UEDEN, local_only);
#ifdef GRAVITY
        if (gravity->get_solver_kind() == "PoissonGrav") {
            rho_potential += hs_level.volPairTotal(S_new, phi_new, URHO, 0, local_only);
        }
#endif


#ifdef REACTIONS
        auto dx = hs_level.geom.CellSizeArray();

        Real dd = 0.0_rt;
#if AMREX_SPACEDIM == 1
        dd = dx[0];
#elif AMREX_SPACEDIM == 2
        dd = amrex::min(dx[0], dx[1]);
#else
        dd = amrex::min(dx[0], dx[1], dx[2]);
#endif
#endif

        bool fine_mask_present = true;
        if (lev == parent->finestLevel()) {
            fine_mask_present = false;
        }

        MultiFab scratch_mf;
        const MultiFab& fine_cover_mf = fine_mask_present ? getLevel(lev+1).make_fine_mask() : scratch_mf;

        ReduceOps<ReduceOpMax, ReduceOpMax, ReduceOpMax> reduce_op;
        ReduceData<Real, Real, Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();

            auto U = S_new[mfi].array();
#ifdef REACTIONS
            auto R = react_new[mfi].array();
#endif

            const auto & level_mask = fine_mask_present ? fine_cover_mf[mfi].array() : Array4<Real>{};

            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                Real zoneWeight = 1.0;
                if (fine_mask_present) {
                    zoneWeight = level_mask(i,j,k);
                }

                Real T = U(i,j,k,UTEMP) * zoneWeight;
                Real rho = U(i,j,k,URHO) * zoneWeight;
                Real timescale_ratio = 0.0_rt;

#ifdef REACTIONS
                Real enuc = std::abs(R(i,j,k,0)) / U(i,j,k,URHO);

                if (enuc > 1.e-100_rt && zoneWeight == 1.0) {

                    Real rhoInv = 1.0_rt / rho;

                    eos_rep_t eos_state;
                    eos_state.rho = rho;
                    eos_state.T   = T;
                    eos_state.e   = U(i,j,k,UEINT) * rhoInv;
                    for (int n = 0; n < NumSpec; ++n) {
                        eos_state.xn[n] = U(i,j,k,UFS+n) * rhoInv;
                    }
#if NAUX_NET > 0
                    for (int n = 0; n < NumAux; ++n) {
                        eos_state.aux[n] = U(i,j,k,UFX+n) * rhoInv;
                    }
#endif

                    eos(eos_input_re, eos_state);

                    Real t_e = eos_state.e / enuc;
                    Real t_s = dd / eos_state.cs;

                    timescale_ratio = t_s / t_e;
                }
#endif

                return {T, rho, timescale_ratio};
            });

        }

        ReduceTuple hv = reduce_data.value();

        T_max = amrex::max(T_max, amrex::get<0>(hv));
        rho_max = amrex::max(rho_max, amrex::get<1>(hv));
#ifdef REACTIONS
        timescale_ratio_max = amrex::max(timescale_ratio_max, amrex::get<2>(hv));
#endif

    }

    if (verbose > 0)
    {

#ifdef HYBRID_MOMENTUM
#ifdef GRAVITY
       const int nfoo = 17;
#else
       const int nfoo = 16;
#endif
#else
#ifdef GRAVITY
       const int nfoo = 14;
#else
       const int nfoo = 13;
#endif
#endif

        Real foo[nfoo] = {mass, mom[0], mom[1], mom[2],
                          com[0], com[1], com[2],
                          angular_momentum_sum[0], angular_momentum_sum[1], angular_momentum_sum[2],
#ifdef HYBRID_MOMENTUM
                          hybrid_momentum_sum[0], hybrid_momentum_sum[1], hybrid_momentum_sum[2],
#endif
#ifdef GRAVITY
                          rho_e, rho_kinetic, rho_E, rho_potential};
#else
                          rho_e, rho_kinetic, rho_E};
#endif

        const int max_reduce_count = 3;

        Real max_reduce_buffer[max_reduce_count] = {T_max, rho_max, timescale_ratio_max};

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif

        ParallelDescriptor::ReduceRealSum(foo, nfoo, ParallelDescriptor::IOProcessorNumber());

        ParallelDescriptor::ReduceRealMax(max_reduce_buffer, max_reduce_count, ParallelDescriptor::IOProcessorNumber());

        if (ParallelDescriptor::IOProcessor()) {

            int i = 0;
            mass       = foo[i++];
            mom[0]     = foo[i++];
            mom[1]     = foo[i++];
            mom[2]     = foo[i++];
            com[0]     = foo[i++];
            com[1]     = foo[i++];
            com[2]     = foo[i++];
            angular_momentum_sum[0] = foo[i++];
            angular_momentum_sum[1] = foo[i++];
            angular_momentum_sum[2] = foo[i++];
#ifdef HYBRID_MOMENTUM
            hybrid_momentum_sum[0] = foo[i++];
            hybrid_momentum_sum[1] = foo[i++];
            hybrid_momentum_sum[2] = foo[i++];
#endif
            rho_e      = foo[i++];
            rho_kinetic      = foo[i++];
            rho_E      = foo[i++];
#ifdef GRAVITY
            rho_potential    = foo[i++];

            std::string gravity_model_name = gravity->get_solver_kind();
            if (gravity_model_name == "PoissonGrav" || gravity_model_name == "MonopoleGrav") {
                energy_budget = 0.5 * rho_potential + rho_E;
            }
            else {
                energy_budget = rho_potential + rho_E;
            }
#endif

            for (int idir = 0; idir < 3; idir++) {
                com[idir]     = com[idir] / mass;
                center_velocity[idir] = mom[idir] / mass;
            }

            i = 0;
            T_max     = max_reduce_buffer[i++];
            rho_max   = max_reduce_buffer[i++];
            timescale_ratio_max = max_reduce_buffer[i++];

            std::cout << '\n';
            std::cout << "TIME= " << time << " MASS        = "   << mass      << '\n';
            std::cout << "TIME= " << time << " XMOM        = "   << mom[0]    << '\n';
            std::cout << "TIME= " << time << " YMOM        = "   << mom[1]    << '\n';
            std::cout << "TIME= " << time << " ZMOM        = "   << mom[2]    << '\n';
            std::cout << "TIME= " << time << " ANG MOM X   = "   << angular_momentum_sum[0] << '\n';
            std::cout << "TIME= " << time << " ANG MOM Y   = "   << angular_momentum_sum[1] << '\n';
            std::cout << "TIME= " << time << " ANG MOM Z   = "   << angular_momentum_sum[2] << '\n';
#ifdef HYBRID_MOMENTUM
            std::cout << "TIME= " << time << " HYB MOM R   = "   << hybrid_momentum_sum[0] << '\n';
            std::cout << "TIME= " << time << " HYB MOM L   = "   << hybrid_momentum_sum[1] << '\n';
            std::cout << "TIME= " << time << " HYB MOM P   = "   << hybrid_momentum_sum[2] << '\n';
#endif
            std::cout << "TIME= " << time << " RHO*e       = "   << rho_e     << '\n';
            std::cout << "TIME= " << time << " RHO*K       = "   << rho_kinetic     << '\n';
            std::cout << "TIME= " << time << " RHO*E       = "   << rho_E     << '\n';
#ifdef GRAVITY
            std::cout << "TIME= " << time << " RHO*PHI     = "   << rho_potential   << '\n';
            std::cout << "TIME= " << time << " TOTAL ENERGY= "   << energy_budget << '\n';
#endif
            std::cout << "TIME= " << time << " CENTER OF MASS X-LOC = " << com[0]     << '\n';
            std::cout << "TIME= " << time << " CENTER OF MASS X-VEL = " << center_velocity[0] << '\n';

            std::cout << "TIME= " << time << " CENTER OF MASS Y-LOC = " << com[1]     << '\n';
            std::cout << "TIME= " << time << " CENTER OF MASS Y-VEL = " << center_velocity[1] << '\n';

            std::cout << "TIME= " << time << " CENTER OF MASS Z-LOC = " << com[2]     << '\n';
            std::cout << "TIME= " << time << " CENTER OF MASS Z-VEL = " << center_velocity[2] << '\n';

            std::cout << "TIME= " << time << " MAXIMUM TEMPERATURE  = " << T_max << '\n';
            std::cout << "TIME= " << time << " MAXIMUM DENSITY      = " << rho_max << '\n';
#ifdef REACTIONS
            std::cout << "TIME= " << time << " MAXIMUM T_S / T_E    = " << timescale_ratio_max << '\n';
#endif

            std::ostream& primary_data_log = *HydroSim::run_logs[0];

            if (primary_data_log.good()) {

               if (time == 0.0) {

                   std::ostringstream header;

                   int n = 0;

                   header << std::setw(integer_column_width) << "#   TIMESTEP";              ++n;
                   header << std::setw(fixed_column_width) << "                     TIME"; ++n;
                   header << std::setw(sci_column_width) << "                     MASS"; ++n;
                   header << std::setw(sci_column_width) << "                     XMOM"; ++n;
                   header << std::setw(sci_column_width) << "                     YMOM"; ++n;
                   header << std::setw(sci_column_width) << "                     ZMOM"; ++n;
                   header << std::setw(sci_column_width) << "              ANG. MOM. X"; ++n;
                   header << std::setw(sci_column_width) << "              ANG. MOM. Y"; ++n;
                   header << std::setw(sci_column_width) << "              ANG. MOM. Z"; ++n;
#if (AMREX_SPACEDIM == 3)
#ifdef HYBRID_MOMENTUM
                   header << std::setw(sci_column_width) << "              HYB. MOM. R"; ++n;
                   header << std::setw(sci_column_width) << "              HYB. MOM. L"; ++n;
                   header << std::setw(sci_column_width) << "              HYB. MOM. P"; ++n;
#endif
#endif
                   header << std::setw(sci_column_width) << "              KIN. ENERGY"; ++n;
                   header << std::setw(sci_column_width) << "              INT. ENERGY"; ++n;
                   header << std::setw(sci_column_width) << "               GAS ENERGY"; ++n;
#ifdef GRAVITY
                   header << std::setw(sci_column_width) << "             GRAV. ENERGY"; ++n;
                   header << std::setw(sci_column_width) << "             TOTAL ENERGY"; ++n;
#endif
                   header << std::setw(sci_column_width) << "     CENTER OF MASS X-LOC"; ++n;
                   header << std::setw(sci_column_width) << "     CENTER OF MASS Y-LOC"; ++n;
                   header << std::setw(sci_column_width) << "     CENTER OF MASS Z-LOC"; ++n;
                   header << std::setw(sci_column_width) << "     CENTER OF MASS X-VEL"; ++n;
                   header << std::setw(sci_column_width) << "     CENTER OF MASS Y-VEL"; ++n;
                   header << std::setw(sci_column_width) << "     CENTER OF MASS Z-VEL"; ++n;
                   header << std::setw(sci_column_width) << "      MAXIMUM TEMPERATURE"; ++n;
                   header << std::setw(sci_column_width) << "          MAXIMUM DENSITY"; ++n;
#ifdef REACTIONS
                   header << std::setw(sci_column_width) << "        MAXIMUM T_S / T_E"; ++n;
#endif

                   header << std::endl;

                   primary_data_log << std::setw(integer_column_width) << "#   COLUMN 1";
                   primary_data_log << std::setw(fixed_column_width) << 2;

                   for (int icol = 3; icol <= n; ++icol) {
                       primary_data_log << std::setw(sci_column_width) << icol;
                   }

                   primary_data_log << std::endl;

                   primary_data_log << header.str();
               }

               primary_data_log << std::fixed;

               primary_data_log << std::setw(integer_column_width) <<  timestep;

               if (time < 1.e-4_rt || time > 1.e4_rt) {
                   primary_data_log << std::scientific;
               } else {
                   primary_data_log << std::fixed;
               }

               primary_data_log << std::setw(fixed_column_width) <<  std::setprecision(output_precision) << time;

               primary_data_log << std::scientific;

               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << mass;
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << mom[0];
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << mom[1];
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << mom[2];
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << angular_momentum_sum[0];
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << angular_momentum_sum[1];
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << angular_momentum_sum[2];
#ifdef HYBRID_MOMENTUM
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << hybrid_momentum_sum[0];
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << hybrid_momentum_sum[1];
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << hybrid_momentum_sum[2];
#endif
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << rho_kinetic;
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << rho_e;
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << rho_E;
#ifdef GRAVITY
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << rho_potential;
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << energy_budget;
#endif
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << com[0];
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << com[1];
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << com[2];
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << center_velocity[0];
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << center_velocity[1];
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << center_velocity[2];
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << T_max;
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << rho_max;
#ifdef REACTIONS
               primary_data_log << std::setw(sci_column_width) <<  std::setprecision(output_precision) << timescale_ratio_max;
#endif

               primary_data_log << std::endl;

            }
        }
#ifdef BL_LAZY
        });
#endif
    }

#ifdef GRAVITY
    {

        Real strain_plus_1  = 0.0;
        Real strain_cross_1 = 0.0;

        Real strain_plus_2  = 0.0;
        Real strain_cross_2 = 0.0;

        Real strain_plus_3  = 0.0;
        Real strain_cross_3 = 0.0;

        for (int lev = 0; lev <= finest_level; lev++)
        {
            HydroSim& hs_level = getLevel(lev);

#if (AMREX_SPACEDIM > 1)
            hs_level.wave_strain(time, strain_plus_1, strain_cross_1, strain_plus_2, strain_cross_2, strain_plus_3, strain_cross_3, local_only);
#endif

        }

        const int sum_reduce_count = 6;

        amrex::Vector<Real> sum_reduce_buffer(sum_reduce_count);

        sum_reduce_buffer[0] = strain_plus_1;
        sum_reduce_buffer[1] = strain_cross_1;
        sum_reduce_buffer[2] = strain_plus_2;
        sum_reduce_buffer[3] = strain_cross_2;
        sum_reduce_buffer[4] = strain_plus_3;
        sum_reduce_buffer[5] = strain_cross_3;

        amrex::ParallelDescriptor::ReduceRealSum(sum_reduce_buffer.dataPtr(), sum_reduce_count);

        strain_plus_1   = sum_reduce_buffer[0];
        strain_cross_1  = sum_reduce_buffer[1];
        strain_plus_2   = sum_reduce_buffer[2];
        strain_cross_2  = sum_reduce_buffer[3];
        strain_plus_3   = sum_reduce_buffer[4];
        strain_cross_3  = sum_reduce_buffer[5];

        if (ParallelDescriptor::IOProcessor()) {

            std::ostream& log = *HydroSim::run_logs[1];


            if (time == 0.0) {

                int n = 0;

                std::ostringstream header;

                header << std::setw(integer_column_width) << "#   TIMESTEP";              ++n;
                header << std::setw(fixed_column_width) << "                     TIME"; ++n;

                header << std::setw(sci_column_width) << "                  h_+ (x)"; ++n;
                header << std::setw(sci_column_width) << "                  h_x (x)"; ++n;
                header << std::setw(sci_column_width) << "                  h_+ (y)"; ++n;
                header << std::setw(sci_column_width) << "                  h_x (y)"; ++n;
                header << std::setw(sci_column_width) << "                  h_+ (z)"; ++n;
                header << std::setw(sci_column_width) << "                  h_x (z)"; ++n;

                header << std::endl;

                log << std::setw(integer_column_width) << "#   COLUMN 1";
                log << std::setw(fixed_column_width) << 2;

                for (int i = 3; i <= n; ++i) {
                    log << std::setw(sci_column_width) << i;
                }

                log << std::endl;

                log << header.str();

            }

            log << std::fixed;

            log << std::setw(integer_column_width)                                    << timestep;

            if (time < 1.e-4_rt || time > 1.e4_rt) {
                log << std::scientific;
            }
            else {
                log << std::fixed;
            }

            log << std::setw(fixed_column_width) << std::setprecision(output_precision) << time;

            log << std::scientific;

            log << std::setw(sci_column_width) << std::setprecision(output_precision) << strain_plus_1;
            log << std::setw(sci_column_width) << std::setprecision(output_precision) << strain_cross_1;
            log << std::setw(sci_column_width) << std::setprecision(output_precision) << strain_plus_2;
            log << std::setw(sci_column_width) << std::setprecision(output_precision) << strain_cross_2;
            log << std::setw(sci_column_width) << std::setprecision(output_precision) << strain_plus_3;
            log << std::setw(sci_column_width) << std::setprecision(output_precision) << strain_cross_3;

            log << std::endl;

        }

    }
#endif


    {
        std::vector<Real> per_species_total(NumSpec);
        std::vector<std::string> species_labels(NumSpec);


        for (int i = 0; i < NumSpec; i++) {
            species_labels[i] = desc_lst[State_Type].name(UFS+i);
            species_labels[i] = species_labels[i].substr(4,std::string::npos);
            per_species_total[i] = 0.0;
        }


        for (int lev = 0; lev <= finest_level; ++lev) {
            const MultiFab& S_new = getLevel(lev).get_new_data(State_Type);
            for (int i = 0; i < NumSpec; ++i) {
                per_species_total[i] += getLevel(lev).volWeightedTotal(S_new, UFS + i, local_only) / C::M_solar;
            }
        }

        int sum_reduce_count = NumSpec;

        amrex::Vector<Real> sum_reduce_buffer(sum_reduce_count);

        for (int i = 0; i < NumSpec; ++i) {
            sum_reduce_buffer[i] = per_species_total[i];
        }

        amrex::ParallelDescriptor::ReduceRealSum(sum_reduce_buffer.dataPtr(), sum_reduce_count);

        for (int i = 0; i < NumSpec; ++i) {
            per_species_total[i] = sum_reduce_buffer[i];
        }

        if (ParallelDescriptor::IOProcessor()) {

            std::ostream& log = *HydroSim::run_logs[2];

            if (time == 0.0) {

                int n = 0;

                std::ostringstream header;

                header << std::setw(integer_column_width) << "#   TIMESTEP";              ++n;
                header << std::setw(fixed_column_width) << "                     TIME"; ++n;

                for (int i = 0; i < NumSpec; i++) {
                    header << std::setw(sci_column_width) << ("Mass " + species_labels[i]); ++n;
                }

                header << std::endl;

                log << std::setw(integer_column_width) << "#   COLUMN 1";
                log << std::setw(fixed_column_width) << 2;

                for (int i = 3; i <= n; ++i) {
                    log << std::setw(sci_column_width) << i;
                }

                log << std::endl;

                log << header.str();

            }

            log << std::fixed;

            log << std::setw(integer_column_width)                                    << timestep;

            if (time < 1.e-4_rt || time > 1.e4_rt) {
                log << std::scientific;
            } else {
                log << std::fixed;
            }

            log << std::setw(fixed_column_width) << std::setprecision(output_precision) << time;

            log << std::scientific;

            for (int i = 0; i < NumSpec; i++) {
                log << std::setw(sci_column_width) << std::setprecision(output_precision) << per_species_total[i];
            }

            log << std::endl;

        }

    }


    {

        Real wall_time = 0.0;

        if (time > 0.0) {
            wall_time = amrex::ParallelDescriptor::second() - wall_clock_start;
        }


#ifdef AMREX_USE_GPU
        Long gpu_mem_free_MB = Gpu::Device::freeMemAvailable() / (1024 * 1024);
        ParallelDescriptor::ReduceLongMin(gpu_mem_free_MB, ParallelDescriptor::IOProcessorNumber());

        Long gpu_mem_used_MB = (Gpu::Device::totalGlobalMem() - Gpu::Device::freeMemAvailable()) / (1024 * 1024);
        ParallelDescriptor::ReduceLongMax(gpu_mem_used_MB, ParallelDescriptor::IOProcessorNumber());
#endif


        int peak_subcycle_count = 0;
        if (time > 0.0_rt) {
            for (int lev = 0; lev <= parent->finestLevel(); ++lev) {
                peak_subcycle_count = std::max(peak_subcycle_count, getLevel(lev).num_sub_taken);
            }
        }

        if (ParallelDescriptor::IOProcessor()) {

            std::ostream& log = *HydroSim::run_logs[3];

            if (time == 0.0) {

                int n = 0;

                std::ostringstream header;

                header << std::setw(integer_column_width) << "#   TIMESTEP";              ++n;
                header << std::setw(fixed_column_width) << "                     TIME"; ++n;
                header << std::setw(fixed_column_width) << "                       DT"; ++n;
                header << std::setw(integer_column_width) << "  FINEST LEV";              ++n;
                header << std::setw(fixed_column_width) << "  MAX NUMBER OF SUBCYCLES"; ++n;
                header << std::setw(sci_column_width) << " COARSE TIMESTEP WALLTIME"; ++n;
#ifdef AMREX_USE_GPU
                header << std::setw(sci_column_width) << "  MAXIMUM GPU MEMORY USED"; ++n;
                header << std::setw(sci_column_width) << "  MINIMUM GPU MEMORY FREE"; ++n;
#endif

                header << std::endl;

                log << std::setw(integer_column_width) << "#   COLUMN 1";
                log << std::setw(fixed_column_width) << 2;

                for (int i = 3; i < 4; ++i) {
                    log << std::setw(fixed_column_width) << i;
                }

                log << std::setw(integer_column_width) << 4;
                log << std::setw(fixed_column_width) << 5;

                for (int i = 6; i <= n; ++i) {
                    log << std::setw(sci_column_width) << i;
                }

                log << std::endl;

                log << header.str();

            }

            log << std::fixed;

            log << std::setw(integer_column_width)                                    << timestep;

            if (time < 1.e-4_rt || time > 1.e4_rt) {
                log << std::scientific;
            } else {
                log << std::fixed;
            }

            log << std::setw(fixed_column_width) << std::setprecision(output_precision) << time;

            log << std::fixed;

            log << std::setw(fixed_column_width) << std::setprecision(output_precision) << dt;
            log << std::setw(integer_column_width)                                    << parent->finestLevel();
            log << std::setw(fixed_column_width)                                    << peak_subcycle_count;
            log << std::setw(sci_column_width) << std::setprecision(output_precision) << wall_time;
#ifdef AMREX_USE_GPU
            log << std::setw(sci_column_width)                                    << gpu_mem_used_MB;
            log << std::setw(sci_column_width)                                    << gpu_mem_free_MB;
#endif

            log << std::endl;

        }

    }

    user_diagnostics();
}
