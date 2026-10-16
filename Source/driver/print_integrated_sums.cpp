#include <algorithm>
#include <array>
#include <iomanip>

#include <HydroSim.H>

#ifdef GRAVITY
#include <Gravity.H>
#endif

#include <user_diagnostics.H>

using namespace amrex;

void
HydroSim::print_integrated_sums ()
{
    if (verbose <= 0) {
        return;
    }

    BL_PROFILE("HydroSim::print_integrated_sums()");

    const SumScope sum_scope = SumScope::ThisRank;

    int finest_level = parent->finestLevel();
    Real time        = state[State_Type].curTime();
    Real dt          = parent->dtLevel(0);
    if (time == 0.0) {
        dt = 0.0; // dtLevel returns the next timestep for t = 0, so overwrite
    }
    int timestep     = parent->levelSteps(0);
    Real mass        = 0.0;
    std::array<Real, 3> mom{};
    std::array<Real, 3> ang_mom{};
#ifdef HYBRID_MOMENTUM
    std::array<Real, 3> hyb_mom{};
#endif
    std::array<Real, 3> com{};
    std::array<Real, 3> com_vel{};
    Real rho_e       = 0.0;
    Real rho_K       = 0.0;
    Real rho_E       = 0.0;
#ifdef GRAVITY
    Real rho_phi     = 0.0;
    Real total_energy = 0.0;
#endif

    Real T_max       = 0.0;
    Real rho_max     = 0.0;
    Real ts_te_max   = 0.0;

    // reduction computing the three maxima above on each level
    using MaxReduceOps = ReduceOps<ReduceOpMax, ReduceOpMax, ReduceOpMax>;
    using MaxReduceData = ReduceData<Real, Real, Real>;

    int col_precision = 16;

    int col_width     = 25; // Floating point data in scientific notation
    int time_col_width     = 25; // Floating point data not in scientific notation
    int step_col_width     = 12; // Integer data

#ifdef GRAVITY
    const bool poisson_gravity = gravity->get_gravity_type() == "FullPoisson";
#endif

    for (int lev = 0; lev <= finest_level; lev++)
    {
        HydroSim& sim_lev = getLevel(lev);
        const MultiFab& S_new = sim_lev.get_new_data(State_Type);
#ifdef GRAVITY
        MultiFab& phi_new = sim_lev.get_new_data(Potential_Type);
#endif
#ifdef REACTIONS
        MultiFab& R_new = sim_lev.get_new_data(Reactions_Type);
#endif

        mass   += sim_lev.volumeIntegralSum(S_new, URHO, sum_scope);
        mom[0] += sim_lev.volumeIntegralSum(S_new, UMX, sum_scope);
        mom[1] += sim_lev.volumeIntegralSum(S_new, UMY, sum_scope);
        mom[2] += sim_lev.volumeIntegralSum(S_new, UMZ, sum_scope);

        ang_mom[0] += sim_lev.volumeIntegralSum("angular_momentum_x", time, sum_scope);
        ang_mom[1] += sim_lev.volumeIntegralSum("angular_momentum_y", time, sum_scope);
        ang_mom[2] += sim_lev.volumeIntegralSum("angular_momentum_z", time, sum_scope);

#ifdef HYBRID_MOMENTUM
        hyb_mom[0] += sim_lev.volumeIntegralSum(S_new, UMR, sum_scope);
        hyb_mom[1] += sim_lev.volumeIntegralSum(S_new, UML, sum_scope);
        hyb_mom[2] += sim_lev.volumeIntegralSum(S_new, UMP, sum_scope);
#endif

        com[0] += sim_lev.positionWeightedSum(S_new, URHO, 0, sum_scope);
        com[1] += sim_lev.positionWeightedSum(S_new, URHO, 1, sum_scope);
        com[2] += sim_lev.positionWeightedSum(S_new, URHO, 2, sum_scope);

        rho_e += sim_lev.volumeIntegralSum(S_new, UEINT, sum_scope);
        rho_K += sim_lev.volumeIntegralSum("kineng", time, sum_scope);
        rho_E += sim_lev.volumeIntegralSum(S_new, UEDEN, sum_scope);
#ifdef GRAVITY
        if (poisson_gravity) {
            rho_phi += sim_lev.volumeWeightedProductSum(S_new, phi_new, URHO, 0, sum_scope);
        }
#endif

        // Compute extrema

#ifdef REACTIONS
        auto dx = sim_lev.geom.CellSizeArray();

        Real dd = 0.0_rt;
#if AMREX_SPACEDIM == 1
        dd = dx[0];
#elif AMREX_SPACEDIM == 2
        dd = amrex::min(dx[0], dx[1]);
#else
        dd = amrex::min(dx[0], dx[1], dx[2]);
#endif
#endif

        bool mask_available = true;
        if (lev == parent->finestLevel()) {
            mask_available = false;
        }

        MultiFab tmp_mf;
        const MultiFab& mask_mf = mask_available ? getLevel(lev+1).build_fine_mask() : tmp_mf;

        MaxReduceOps reduce_op;
        MaxReduceData reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();

            auto U = S_new[mfi].array();
#ifdef REACTIONS
            auto R = R_new[mfi].array();
#endif

            const auto & level_mask = mask_available ? mask_mf[mfi].array() : Array4<Real>{};

            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                Real maskFactor = 1.0;
                if (mask_available) {
                    maskFactor = level_mask(i,j,k);
                }

                Real T = U(i,j,k,UTEMP) * maskFactor;
                Real rho = U(i,j,k,URHO) * maskFactor;
                Real ts_te = 0.0_rt;

#ifdef REACTIONS
                Real enuc = std::abs(R(i,j,k,0)) / U(i,j,k,URHO);

                if (enuc > 1.e-100_rt && maskFactor == 1.0) {

                    Real rhoInv = 1.0_rt / rho;

                    // Calculate sound speed
                    eos_rep_t eos_state;
                    eos_state.rho = rho;
                    eos_state.T   = T;
                    eos_state.e   = U(i,j,k,UEINT) * rhoInv;
                    for (int n = 0; n < NumSpec; ++n) {
                        eos_state.xn[n] = U(i,j,k,USPEC+n) * rhoInv;
                    }
#if NAUX_NET > 0
                    for (int n = 0; n < NumAux; ++n) {
                        eos_state.aux[n] = U(i,j,k,UAUXV+n) * rhoInv;
                    }
#endif

                    eos(eos_input_re, eos_state);

                    Real t_e = eos_state.e / enuc;
                    Real t_s = dd / eos_state.cs;

                    ts_te = t_s / t_e;
                }
#endif

                return {T, rho, ts_te};
            });

        }

        ReduceTuple hv = reduce_data.value();

        T_max = amrex::max(T_max, amrex::get<0>(hv));
        rho_max = amrex::max(rho_max, amrex::get<1>(hv));
#ifdef REACTIONS
        ts_te_max = amrex::max(ts_te_max, amrex::get<2>(hv));
#endif

    }

    if (verbose > 0)
    {

#ifdef HYBRID_MOMENTUM
#ifdef GRAVITY
       const int n_reduce_vals = 17;
#else
       const int n_reduce_vals = 16;
#endif
#else
#ifdef GRAVITY
       const int n_reduce_vals = 14;
#else
       const int n_reduce_vals = 13;
#endif
#endif

        std::array<Real, n_reduce_vals> foo = {mass, mom[0], mom[1], mom[2],
                                      com[0], com[1], com[2],
                                      ang_mom[0], ang_mom[1], ang_mom[2],
#ifdef HYBRID_MOMENTUM
                                      hyb_mom[0], hyb_mom[1], hyb_mom[2],
#endif
#ifdef GRAVITY
                                      rho_e, rho_K, rho_E, rho_phi};
#else
                                      rho_e, rho_K, rho_E};
#endif

        const int n_max_vals = 3;

        std::array<Real, n_max_vals> max_vals = {T_max, rho_max, ts_te_max};

#ifdef BL_LAZY
        Lazy::QueueReduction( [=] () mutable {
#endif

        ParallelDescriptor::ReduceRealSum(foo.data(), n_reduce_vals, ParallelDescriptor::IOProcessorNumber());

        ParallelDescriptor::ReduceRealMax(max_vals.data(), n_max_vals, ParallelDescriptor::IOProcessorNumber());

        if (ParallelDescriptor::IOProcessor()) {

            int i = 0;
            mass       = foo[i++];
            mom[0]     = foo[i++];
            mom[1]     = foo[i++];
            mom[2]     = foo[i++];
            com[0]     = foo[i++];
            com[1]     = foo[i++];
            com[2]     = foo[i++];
            ang_mom[0] = foo[i++];
            ang_mom[1] = foo[i++];
            ang_mom[2] = foo[i++];
#ifdef HYBRID_MOMENTUM
            hyb_mom[0] = foo[i++];
            hyb_mom[1] = foo[i++];
            hyb_mom[2] = foo[i++];
#endif
            rho_e      = foo[i++];
            rho_K      = foo[i++];
            rho_E      = foo[i++];
#ifdef GRAVITY
            rho_phi    = foo[i++];

            // Total energy is 1/2 * rho * phi + rho * E for self-gravity,
            // and rho * phi + rho * E for externally-supplied gravity.
            std::string gravity_type = gravity->get_gravity_type();
            if (gravity_type == "FullPoisson" || gravity_type == "SphericalMonopole") {
                total_energy = 0.5 * rho_phi + rho_E;
            }
            else {
                total_energy = rho_phi + rho_E;
            }
#endif

            const auto per_unit_mass = [mass] (Real x) { return x / mass; };
            std::transform(std::begin(com), std::end(com), std::begin(com), per_unit_mass);
            std::transform(std::begin(mom), std::end(mom), std::begin(com_vel), per_unit_mass);

            i = 0;
            T_max     = max_vals[i++];
            rho_max   = max_vals[i++];
            ts_te_max = max_vals[i++];    // NOLINT(clang-analyzer-deadcode.DeadStores)

            auto print_sum = [time] (const char* label, Real value) {
                std::cout << "TIME= " << time << label << value << '\n';
            };

            std::cout << '\n';
            print_sum(" MASS        = ", mass);
            print_sum(" XMOM        = ", mom[0]);
            print_sum(" YMOM        = ", mom[1]);
            print_sum(" ZMOM        = ", mom[2]);
            print_sum(" ANG MOM X   = ", ang_mom[0]);
            print_sum(" ANG MOM Y   = ", ang_mom[1]);
            print_sum(" ANG MOM Z   = ", ang_mom[2]);
#ifdef HYBRID_MOMENTUM
            print_sum(" HYB MOM R   = ", hyb_mom[0]);
            print_sum(" HYB MOM L   = ", hyb_mom[1]);
            print_sum(" HYB MOM P   = ", hyb_mom[2]);
#endif
            print_sum(" RHO*e       = ", rho_e);
            print_sum(" RHO*K       = ", rho_K);
            print_sum(" RHO*E       = ", rho_E);
#ifdef GRAVITY
            print_sum(" RHO*PHI     = ", rho_phi);
            print_sum(" TOTAL ENERGY= ", total_energy);
#endif
            print_sum(" CENTER OF MASS X-LOC = ", com[0]);
            print_sum(" CENTER OF MASS X-VEL = ", com_vel[0]);

            print_sum(" CENTER OF MASS Y-LOC = ", com[1]);
            print_sum(" CENTER OF MASS Y-VEL = ", com_vel[1]);

            print_sum(" CENTER OF MASS Z-LOC = ", com[2]);
            print_sum(" CENTER OF MASS Z-VEL = ", com_vel[2]);

            print_sum(" MAXIMUM TEMPERATURE  = ", T_max);
            print_sum(" MAXIMUM DENSITY      = ", rho_max);
#ifdef REACTIONS
            print_sum(" MAXIMUM T_S / T_E    = ", ts_te_max);
#endif

            std::ostream& diag_log = *HydroSim::data_logs[0];

            if (diag_log.good()) {

               if (time == 0.0) {

                   std::ostringstream header;

                   int n = 0;

                   header << std::setw(step_col_width) << "#   TIMESTEP";              ++n;
                   header << std::setw(time_col_width) << "                     TIME"; ++n;
                   header << std::setw(col_width) << "                     MASS"; ++n;
                   header << std::setw(col_width) << "                     XMOM"; ++n;
                   header << std::setw(col_width) << "                     YMOM"; ++n;
                   header << std::setw(col_width) << "                     ZMOM"; ++n;
                   header << std::setw(col_width) << "              ANG. MOM. X"; ++n;
                   header << std::setw(col_width) << "              ANG. MOM. Y"; ++n;
                   header << std::setw(col_width) << "              ANG. MOM. Z"; ++n;
#if (AMREX_SPACEDIM == 3)
#ifdef HYBRID_MOMENTUM
                   header << std::setw(col_width) << "              HYB. MOM. R"; ++n;
                   header << std::setw(col_width) << "              HYB. MOM. L"; ++n;
                   header << std::setw(col_width) << "              HYB. MOM. P"; ++n;
#endif
#endif
                   header << std::setw(col_width) << "              KIN. ENERGY"; ++n;
                   header << std::setw(col_width) << "              INT. ENERGY"; ++n;
                   header << std::setw(col_width) << "               GAS ENERGY"; ++n;
#ifdef GRAVITY
                   header << std::setw(col_width) << "             GRAV. ENERGY"; ++n;
                   header << std::setw(col_width) << "             TOTAL ENERGY"; ++n;
#endif
                   header << std::setw(col_width) << "     CENTER OF MASS X-LOC"; ++n;
                   header << std::setw(col_width) << "     CENTER OF MASS Y-LOC"; ++n;
                   header << std::setw(col_width) << "     CENTER OF MASS Z-LOC"; ++n;
                   header << std::setw(col_width) << "     CENTER OF MASS X-VEL"; ++n;
                   header << std::setw(col_width) << "     CENTER OF MASS Y-VEL"; ++n;
                   header << std::setw(col_width) << "     CENTER OF MASS Z-VEL"; ++n;
                   header << std::setw(col_width) << "      MAXIMUM TEMPERATURE"; ++n;
                   header << std::setw(col_width) << "          MAXIMUM DENSITY"; ++n;
#ifdef REACTIONS
                   header << std::setw(col_width) << "        MAXIMUM T_S / T_E"; ++n;
#endif

                   header << std::endl;

                   diag_log << std::setw(step_col_width) << "#   COLUMN 1";
                   diag_log << std::setw(time_col_width) << 2;

                   for (int icol = 3; icol <= n; ++icol) {
                       diag_log << std::setw(col_width) << icol;
                   }

                   diag_log << std::endl;

                   diag_log << header.str();
               }

               // Write the quantities at this time
               diag_log << std::fixed;

               diag_log << std::setw(step_col_width) <<  timestep;

               if (time < 1.e-4_rt || time > 1.e4_rt) {
                   diag_log << std::scientific;
               } else {
                   diag_log << std::fixed;
               }

               diag_log << std::setw(time_col_width) <<  std::setprecision(col_precision) << time;

               diag_log << std::scientific;

               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << mass;
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << mom[0];
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << mom[1];
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << mom[2];
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << ang_mom[0];
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << ang_mom[1];
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << ang_mom[2];
#ifdef HYBRID_MOMENTUM
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << hyb_mom[0];
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << hyb_mom[1];
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << hyb_mom[2];
#endif
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << rho_K;
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << rho_e;
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << rho_E;
#ifdef GRAVITY
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << rho_phi;
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << total_energy;
#endif
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << com[0];
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << com[1];
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << com[2];
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << com_vel[0];
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << com_vel[1];
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << com_vel[2];
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << T_max;
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << rho_max;
#ifdef REACTIONS
               diag_log << std::setw(col_width) <<  std::setprecision(col_precision) << ts_te_max;
#endif

               diag_log << std::endl;

            }
        }
#ifdef BL_LAZY
        });
#endif
    }

#ifdef GRAVITY
    // Gravity diagnostics
    {
        // Gravitational wave amplitudes

        Real h_plus_1  = 0.0;
        Real h_cross_1 = 0.0;

        Real h_plus_2  = 0.0;
        Real h_cross_2 = 0.0;

        Real h_plus_3  = 0.0;
        Real h_cross_3 = 0.0;

        for (int lev = 0; lev <= finest_level; lev++)
        {
            HydroSim& sim_lev = getLevel(lev);

#if (AMREX_SPACEDIM > 1)
            // Gravitational wave signal. Each level returns its own contribution,
            // so we add them up here.
            const auto strain = sim_lev.wave_strain(time, sum_scope);

            h_plus_1  += strain.plus[0];
            h_cross_1 += strain.cross[0];
            h_plus_2  += strain.plus[1];
            h_cross_2 += strain.cross[1];
            h_plus_3  += strain.plus[2];
            h_cross_3 += strain.cross[2];
#endif

        }

        const int n_sum_vals = 6;

        amrex::Vector<Real> sum_vals(n_sum_vals);

        sum_vals[0] = h_plus_1;
        sum_vals[1] = h_cross_1;
        sum_vals[2] = h_plus_2;
        sum_vals[3] = h_cross_2;
        sum_vals[4] = h_plus_3;
        sum_vals[5] = h_cross_3;

        amrex::ParallelDescriptor::ReduceRealSum(sum_vals.dataPtr(), n_sum_vals);

        h_plus_1   = sum_vals[0];
        h_cross_1  = sum_vals[1];
        h_plus_2   = sum_vals[2];
        h_cross_2  = sum_vals[3];
        h_plus_3   = sum_vals[4];
        h_cross_3  = sum_vals[5];

        if (ParallelDescriptor::IOProcessor()) {

            std::ostream& log = *HydroSim::data_logs[1];

            // Write header row

            if (time == 0.0) {

                int n = 0;

                std::ostringstream header;

                header << std::setw(step_col_width) << "#   TIMESTEP";              ++n;
                header << std::setw(time_col_width) << "                     TIME"; ++n;

                header << std::setw(col_width) << "                  h_+ (x)"; ++n;
                header << std::setw(col_width) << "                  h_x (x)"; ++n;
                header << std::setw(col_width) << "                  h_+ (y)"; ++n;
                header << std::setw(col_width) << "                  h_x (y)"; ++n;
                header << std::setw(col_width) << "                  h_+ (z)"; ++n;
                header << std::setw(col_width) << "                  h_x (z)"; ++n;

                header << std::endl;

                log << std::setw(step_col_width) << "#   COLUMN 1";
                log << std::setw(time_col_width) << 2;

                for (int i = 3; i <= n; ++i) {
                    log << std::setw(col_width) << i;
                }

                log << std::endl;

                log << header.str();

            }

            log << std::fixed;

            log << std::setw(step_col_width)                                    << timestep;

            if (time < 1.e-4_rt || time > 1.e4_rt) {
                log << std::scientific;
            }
            else {
                log << std::fixed;
            }

            log << std::setw(time_col_width) << std::setprecision(col_precision) << time;

            log << std::scientific;

            log << std::setw(col_width) << std::setprecision(col_precision) << h_plus_1;
            log << std::setw(col_width) << std::setprecision(col_precision) << h_cross_1;
            log << std::setw(col_width) << std::setprecision(col_precision) << h_plus_2;
            log << std::setw(col_width) << std::setprecision(col_precision) << h_cross_2;
            log << std::setw(col_width) << std::setprecision(col_precision) << h_plus_3;
            log << std::setw(col_width) << std::setprecision(col_precision) << h_cross_3;

            log << std::endl;

        }

    }
#endif

    // Species

    {
        std::vector<Real> species_mass(NumSpec);
        std::vector<std::string> species_names(NumSpec);

        // Species names

        for (int i = 0; i < NumSpec; i++) {
            species_names[i] = desc_lst[State_Type].name(USPEC+i);
            species_names[i] = species_names[i].substr(4,std::string::npos);
            species_mass[i] = 0.0;
        }

        // Integrated mass of all species on the domain

        for (int lev = 0; lev <= finest_level; ++lev) {
            const MultiFab& S_new = getLevel(lev).get_new_data(State_Type);
            for (int i = 0; i < NumSpec; ++i) {
                species_mass[i] += getLevel(lev).volumeIntegralSum(S_new, USPEC + i, sum_scope) / C::M_solar;
            }
        }

        int n_sum_vals = NumSpec;

        amrex::Vector<Real> sum_vals(n_sum_vals);

        for (int i = 0; i < NumSpec; ++i) {
            sum_vals[i] = species_mass[i];
        }

        amrex::ParallelDescriptor::ReduceRealSum(sum_vals.dataPtr(), n_sum_vals);

        for (int i = 0; i < NumSpec; ++i) {
            species_mass[i] = sum_vals[i];
        }

        if (ParallelDescriptor::IOProcessor()) {

            std::ostream& log = *HydroSim::data_logs[2];

            if (time == 0.0) {

                int n = 0;

                std::ostringstream header;

                header << std::setw(step_col_width) << "#   TIMESTEP";              ++n;
                header << std::setw(time_col_width) << "                     TIME"; ++n;

                for (int i = 0; i < NumSpec; i++) {
                    header << std::setw(col_width) << ("Mass " + species_names[i]); ++n;
                }

                header << std::endl;

                log << std::setw(step_col_width) << "#   COLUMN 1";
                log << std::setw(time_col_width) << 2;

                for (int i = 3; i <= n; ++i) {
                    log << std::setw(col_width) << i;
                }

                log << std::endl;

                log << header.str();

            }

            log << std::fixed;

            log << std::setw(step_col_width)                                    << timestep;

            if (time < 1.e-4_rt || time > 1.e4_rt) {
                log << std::scientific;
            } else {
                log << std::fixed;
            }

            log << std::setw(time_col_width) << std::setprecision(col_precision) << time;

            log << std::scientific;

            for (int i = 0; i < NumSpec; i++) {
                log << std::setw(col_width) << std::setprecision(col_precision) << species_mass[i];
            }

            log << std::endl;

        }

    }

    // Information about the AMR driver.

    {
        // Calculate wall time for the step.

        Real wall_time = 0.0;

        if (time > 0.0) {
            wall_time = amrex::ParallelDescriptor::second() - wall_time_start;
        }

        // Calculate GPU memory consumption.

#ifdef AMREX_USE_GPU
        Long gpu_size_free_MB = Gpu::Device::freeMemAvailable() / (1024 * 1024);
        ParallelDescriptor::ReduceLongMin(gpu_size_free_MB, ParallelDescriptor::IOProcessorNumber());

        Long gpu_size_used_MB = (Gpu::Device::totalGlobalMem() - Gpu::Device::freeMemAvailable()) / (1024 * 1024);
        ParallelDescriptor::ReduceLongMax(gpu_size_used_MB, ParallelDescriptor::IOProcessorNumber());
#endif

        // Calculate maximum number of advance subcycles across all levels.

        int max_num_subcycles = 0;
        if (time > 0.0_rt) {
            for (int lev = 0; lev <= parent->finestLevel(); ++lev) {
                max_num_subcycles = std::max(max_num_subcycles, getLevel(lev).subcycles_used);
            }
        }

        if (ParallelDescriptor::IOProcessor()) {

            std::ostream& log = *HydroSim::data_logs[3];

            if (time == 0.0) {

                int n = 0;

                std::ostringstream header;

                header << std::setw(step_col_width) << "#   TIMESTEP";              ++n;
                header << std::setw(time_col_width) << "                     TIME"; ++n;
                header << std::setw(time_col_width) << "                       DT"; ++n;
                header << std::setw(step_col_width) << "  FINEST LEV";              ++n;
                header << std::setw(time_col_width) << "  MAX NUMBER OF SUBCYCLES"; ++n;
                header << std::setw(col_width) << " COARSE TIMESTEP WALLTIME"; ++n;
#ifdef AMREX_USE_GPU
                header << std::setw(col_width) << "  MAXIMUM GPU MEMORY USED"; ++n;
                header << std::setw(col_width) << "  MINIMUM GPU MEMORY FREE"; ++n;
#endif

                header << std::endl;

                log << std::setw(step_col_width) << "#   COLUMN 1";
                log << std::setw(time_col_width) << 2;

                for (int i = 3; i < 4; ++i) {
                    log << std::setw(time_col_width) << i;
                }

                log << std::setw(step_col_width) << 4; // Handle the finest lev column
                log << std::setw(time_col_width) << 5; // Handle the subcycle count column

                for (int i = 6; i <= n; ++i) {
                    log << std::setw(col_width) << i;
                }

                log << std::endl;

                log << header.str();

            }

            log << std::fixed;

            log << std::setw(step_col_width)                                    << timestep;

            if (time < 1.e-4_rt || time > 1.e4_rt) {
                log << std::scientific;
            } else {
                log << std::fixed;
            }

            log << std::setw(time_col_width) << std::setprecision(col_precision) << time;

            log << std::fixed;

            log << std::setw(time_col_width) << std::setprecision(col_precision) << dt;
            log << std::setw(step_col_width)                                    << parent->finestLevel();
            log << std::setw(time_col_width)                                    << max_num_subcycles;
            log << std::setw(col_width) << std::setprecision(col_precision) << wall_time;
#ifdef AMREX_USE_GPU
            log << std::setw(col_width)                                    << gpu_size_used_MB;
            log << std::setw(col_width)                                    << gpu_size_free_MB;
#endif

            log << std::endl;

        }

    }

    user_diagnostics();
}
