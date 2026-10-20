#include <HydroSim.H>
#include <HydroSim_helpers.H>

#ifdef RADIATION
#include <Radiation.H>
#include <flux_limiter.H>
#endif

using namespace amrex;

void
HydroSim::apply_tangential_pass(const Box& bx,
                            int idir_tan, int idir_normal, int d,
                            Array4<Real const> const& prim_arr,
                            Array4<Real> const& q_out_arr,
                            Array4<Real const> const& qthermo_arr,
                            Array4<Real const> const& flux_tan,
#ifdef RADIATION
                            Array4<Real const> const& radiative_flux_tan,
#endif
                            Array4<Real const> const& q_t,
#if AMREX_SPACEDIM == 2
                            Array4<Real const> const& area_t,
                            Array4<Real const> const& vol,
#endif
                            Real hdt, Real third_dt_dx)
{




    amrex::ignore_unused(hdt);

#if AMREX_SPACEDIM == 2
    int coord = geom.Coord();
    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
#endif

    bool density_reset_enabled = transverse_revert_density;
    bool reset_eint_flag = transverse_rebuild_rhoe;
    bool apply_eos = transverse_eos_pressure;
    Real pressure_floor = pressure_cutoff;
    Real temperature_guess = eos_temp_guess;
    Real species_fraction_tol = massfrac_check_tolerance;

    GpuArray<int, 3> lo_shift{0, 0, 0};
    GpuArray<int, 3> hi_shift{0, 0, 0};

    hi_shift[idir_tan] = 1;

    lo_shift[idir_normal] += d;
    hi_shift[idir_normal] += d;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {


        const int il = i + lo_shift[0];
        const int jl = j + lo_shift[1];
        const int kl = k + lo_shift[2];

        const int ir = i + hi_shift[0];
        const int jr = j + hi_shift[1];
        const int kr = k + hi_shift[2];


#if AMREX_SPACEDIM == 2
        const Real volinv = 1.0_rt / vol(il,jl,kl);
        const Real density_partial = prim_arr(i,j,k,QRHO) - hdt * (area_t(ir,jr,kr) * flux_tan(ir,jr,kr,URHO) -
                                                area_t(il,jl,kl) * flux_tan(il,jl,kl,URHO)) * volinv;
#else
        const Real density_partial = prim_arr(i,j,k,QRHO) - third_dt_dx * (flux_tan(ir,jr,kr,URHO) - flux_tan(il,jl,kl,URHO));
#endif

        for (int passive_index = 0; passive_index < npassive; passive_index++) {
            const int n = passive_cons_index(passive_index);
            const int nqp = passive_prim_index(passive_index);
#if AMREX_SPACEDIM == 2
            const Real passive_cons_partial = prim_arr(i,j,k,QRHO) * prim_arr(i,j,k,nqp) -
                hdt * (area_t(ir,jr,kr) * flux_tan(ir,jr,kr,n) -
                       area_t(il,jl,kl) * flux_tan(il,jl,kl,n)) * volinv;
#else
            const Real passive_cons_partial = prim_arr(i,j,k,QRHO) * prim_arr(i,j,k,nqp) -
                third_dt_dx * (flux_tan(ir,jr,kr,n) - flux_tan(il,jl,kl,n));
#endif
            q_out_arr(i,j,k,nqp) = passive_cons_partial / density_partial;
        }

        const Real pgp  = q_t(ir,jr,kr,GDPRESSURE);
        const Real pgm  = q_t(il,jl,kl,GDPRESSURE);
        const Real ugp  = q_t(ir,jr,kr,GDU+idir_tan);
        const Real ugm  = q_t(il,jl,kl,GDU+idir_tan);

#ifdef RADIATION
        Real lambda[NGROUPS];
        Real rad_energy_plus[NGROUPS];
        Real rad_energy_minus[NGROUPS];

        for (int g = 0; g < NGROUPS; ++g) {
            lambda[g] = qthermo_arr(il,jl,kl,QFLUXLIM+g);
            rad_energy_plus[g] = q_t(ir,jr,kr,GDRADENG+g);
            rad_energy_minus[g] = q_t(il,jl,kl,GDRADENG+g);
        }
#endif


#if AMREX_SPACEDIM == 2
        Real dup = area_t(ir,jr,kr) * pgp * ugp - area_t(il,jl,kl) * pgm * ugm;
        Real du = area_t(ir,jr,kr) * ugp - area_t(il,jl,kl) * ugm;
#else
        Real dup = pgp * ugp - pgm * ugm;
        Real du = ugp - ugm;
#endif
        Real pav = 0.5_rt * (pgp + pgm);
#ifdef RADIATION
        Real uav = 0.5_rt * (ugp + ugm);
#endif

#ifdef RADIATION
        Real gamma1_cell = qthermo_arr(il,jl,kl,QGAMMA1G);
#else
        Real gamma1_cell = qthermo_arr(il,jl,kl,QGAMMA1);
#endif

#ifdef RADIATION
        Real rad_lambda_jump[NGROUPS];
        Real rad_work_term[NGROUPS];
        Real der[NGROUPS];

        Real radiation_mom_delta = 0.0_rt;
        Real dre = 0.0_rt;

        for (int g = 0; g < NGROUPS; ++g) {
            rad_lambda_jump[g] = lambda[g] * (rad_energy_plus[g] - rad_energy_minus[g]);
            radiation_mom_delta += -third_dt_dx * rad_lambda_jump[g];
            rad_work_term[g] = uav * rad_lambda_jump[g];
            dre += -third_dt_dx * rad_work_term[g];
        }

        if (radiation::frequency_advection_mode == 1 && radiation::moving_frame) {
            for (int g = 0; g < NGROUPS; ++g) {
                Real eddington_factor = Eddington_factor(lambda[g]);
                Real f1 = 0.5_rt * (1.0_rt - eddington_factor);
                der[g] = third_dt_dx * uav * f1 * (rad_energy_plus[g] - rad_energy_minus[g]);
            }
        }
        else if (radiation::frequency_advection_mode == 2) {
#if AMREX_SPACEDIM == 2
            Real divu = (area_t(ir,jr,kr) * ugp - area_t(il,jl,kl) * ugm) * volinv;
            for (int g = 0; g < NGROUPS; g++) {
                Real eddington_factor = Eddington_factor(lambda[g]);
                Real f1 = 0.5_rt * (1.0_rt - eddington_factor);
                der[g] = -hdt * f1 * 0.5_rt * (rad_energy_plus[g] + rad_energy_minus[g]) * divu;
            }
#else
            for (int g = 0; g < NGROUPS; g++) {
                Real eddington_factor = Eddington_factor(lambda[g]);
                Real f1 = 0.5_rt * (1.0_rt - eddington_factor);
                der[g] = third_dt_dx * f1 * 0.5_rt * (rad_energy_plus[g] + rad_energy_minus[g]) * (ugm - ugp);
            }
#endif
        }
        else {
            for (int g = 0; g < NGROUPS; g++) {
                der[g] = third_dt_dx * rad_work_term[g];
            }
        }
#endif

        Real rrn = prim_arr(i,j,k,QRHO);
        Real run = rrn * prim_arr(i,j,k,QU);
        Real rvn = rrn * prim_arr(i,j,k,QV);
        Real rwn = rrn * prim_arr(i,j,k,QW);
        Real kinetic_energy_base = 0.5_rt * rrn * (prim_arr(i,j,k,QU) * prim_arr(i,j,k,QU) + prim_arr(i,j,k,QV) * prim_arr(i,j,k,QV) + prim_arr(i,j,k,QW) * prim_arr(i,j,k,QW));
        Real ren = prim_arr(i,j,k,QEINTDENS) + kinetic_energy_base;
#ifdef RADIATION
        Real ern[NGROUPS];
        for (int g = 0; g < NGROUPS; ++g) {
            ern[g] = prim_arr(i,j,k,QRADENG+g);
        }
#endif

#if AMREX_SPACEDIM == 2

        Real density_new = density_partial;


        Real momentum_x_new = run - hdt * (area_t(ir,jr,kr) * flux_tan(ir,jr,kr,UMX) -
                                   area_t(il,jl,kl) * flux_tan(il,jl,kl,UMX)) * volinv;
        if (idir_tan == 0 && !momentum_flux_has_pressure(0, idir_tan, coord)) {
            momentum_x_new = momentum_x_new - third_dt_dx * (pgp - pgm);
        }
        Real momentum_y_new = rvn - hdt * (area_t(ir,jr,kr) * flux_tan(ir,jr,kr,UMY) -
                                   area_t(il,jl,kl) * flux_tan(il,jl,kl,UMY)) * volinv;
        if (idir_tan == 1 && !momentum_flux_has_pressure(1, idir_tan, coord)) {
            Real r = problo[0] + static_cast<Real>(il + 0.5_rt) * dx[0];
            momentum_y_new = momentum_y_new - third_dt_dx / r * (pgp - pgm);
        }
        Real momentum_z_new = rwn - hdt * (area_t(ir,jr,kr) * flux_tan(ir,jr,kr,UMZ) -
                                   area_t(il,jl,kl) * flux_tan(il,jl,kl,UMZ)) * volinv;
        Real energy_new = ren - hdt * (area_t(ir,jr,kr) * flux_tan(ir,jr,kr,UEDEN) -
                                   area_t(il,jl,kl) * flux_tan(il,jl,kl,UEDEN)) * volinv;

#ifdef RADIATION
        if (idir_tan == 0) {
            Real rad_lambda_jump_sum = 0.0_rt;
            for (int g = 0; g < NGROUPS; ++g)
                rad_lambda_jump_sum = rad_lambda_jump_sum + rad_lambda_jump[g];

            momentum_x_new = momentum_x_new - 0.5_rt * hdt * (area_t(ir,jr,kr) + area_t(il,jl,kl)) * rad_lambda_jump_sum * volinv;
        }
        else {
            momentum_y_new = momentum_y_new + radiation_mom_delta;
        }

        energy_new = energy_new + dre;

        Real rad_energy_new_group[NGROUPS];
        for (int g = 0; g < NGROUPS; ++g) {
            rad_energy_new_group[g] = ern[g] - hdt * (area_t(ir,jr,kr) * radiative_flux_tan(ir,jr,kr,g) -
                                        area_t(il,jl,kl) * radiative_flux_tan(il,jl,kl,g)) * volinv + der[g];
        }
#endif

#else
        Real density_new = density_partial;
        Real momentum_x_new = run - third_dt_dx * (flux_tan(ir,jr,kr,UMX)   - flux_tan(il,jl,kl,UMX));
        Real momentum_y_new = rvn - third_dt_dx * (flux_tan(ir,jr,kr,UMY)   - flux_tan(il,jl,kl,UMY));
        Real momentum_z_new = rwn - third_dt_dx * (flux_tan(ir,jr,kr,UMZ)   - flux_tan(il,jl,kl,UMZ));
        Real energy_new = ren - third_dt_dx * (flux_tan(ir,jr,kr,UEDEN) - flux_tan(il,jl,kl,UEDEN));
#ifdef RADIATION
        momentum_x_new = momentum_x_new + radiation_mom_delta;
        energy_new = energy_new + dre;
        Real rad_energy_new_group[NGROUPS];
        for (int g = 0; g < NGROUPS; ++g) {
            rad_energy_new_group[g] = ern[g] - third_dt_dx * (radiative_flux_tan(ir,jr,kr,g) - radiative_flux_tan(il,jl,kl,g)) + der[g];
        }
#endif
#endif

        bool state_was_reset = false;

        auto revert_to_input_zone = [&] () {
            density_new = rrn;
            momentum_x_new = run;
            momentum_y_new = rvn;
            momentum_z_new = rwn;
            energy_new = ren;
            for (int passive_index = 0; passive_index < npassive; ++passive_index) {
                int nqp = passive_prim_index(passive_index);
                q_out_arr(i,j,k,nqp) = prim_arr(i,j,k,nqp);
            }
#ifdef RADIATION
            for (int g = 0; g < NGROUPS; ++g) {
                rad_energy_new_group[g] = ern[g];
            }
#endif
            state_was_reset = true;
        };

        if (density_reset_enabled && density_new < 0.0_rt) {
            revert_to_input_zone();
        }


        for (int n = 0; n < NumSpec; ++n) {
            if (q_out_arr(i,j,k,n+QFS) > 1.0_rt + species_fraction_tol ||
                q_out_arr(i,j,k,n+QFS) < -species_fraction_tol) {
                revert_to_input_zone();
                break;
            }
        }

        q_out_arr(i,j,k,QRHO) = density_new;
        const Real rhoinv = 1.0_rt / density_new;

        q_out_arr(i,j,k,QU) = momentum_x_new * rhoinv;
        q_out_arr(i,j,k,QV) = momentum_y_new * rhoinv;
        q_out_arr(i,j,k,QW) = momentum_z_new * rhoinv;

        Real kinetic_energy_new = 0.5_rt * (momentum_x_new * momentum_x_new + momentum_y_new * momentum_y_new + momentum_z_new * momentum_z_new) * rhoinv;
        q_out_arr(i,j,k,QEINTDENS) = energy_new - kinetic_energy_new;

        if (!state_was_reset) {

            if (reset_eint_flag == 1 && q_out_arr(i,j,k,QEINTDENS) <= 0.0_rt) {
#if AMREX_SPACEDIM == 2
                q_out_arr(i,j,k,QEINTDENS) = prim_arr(i,j,k,QEINTDENS) - hdt * (area_t(ir,jr,kr) * flux_tan(ir,jr,kr,UEINT) -
                                                            area_t(il,jl,kl) * flux_tan(il,jl,kl,UEINT) +
                                                            pav * du) * volinv;
#else
                q_out_arr(i,j,k,QEINTDENS) = prim_arr(i,j,k,QEINTDENS) - third_dt_dx * (flux_tan(ir,jr,kr,UEINT) - flux_tan(il,jl,kl,UEINT) + pav * du);
#endif
            }


            if (q_out_arr(i,j,k,QEINTDENS) <= 0.0_rt) {
                q_out_arr(i,j,k,QEINTDENS) = prim_arr(i,j,k,QEINTDENS);
            }


            if (apply_eos) {

                eos_rep_t eos_state;
                eos_state.rho = density_new;
                eos_state.e = q_out_arr(i,j,k,QEINTDENS) * rhoinv;
                eos_state.T = temperature_guess;

                for (int n = 0; n < NumSpec; n++) {
                    eos_state.xn[n] = q_out_arr(i,j,k,QFS+n);
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                    eos_state.aux[n] = q_out_arr(i,j,k,QFX+n);
                }
#endif
                eos(eos_input_re, eos_state);

                Real pres_new = eos_state.p;
                q_out_arr(i,j,k,QPRES) = amrex::max(pres_new, pressure_floor);

            } else {

#if AMREX_SPACEDIM == 2
                Real pres_new = prim_arr(i,j,k,QPRES) - hdt * (dup + pav * du * (gamma1_cell - 1.0_rt)) * volinv;
#else
                Real pres_new = prim_arr(i,j,k,QPRES) - third_dt_dx * (dup + pav * du * (gamma1_cell - 1.0_rt));
#endif
                q_out_arr(i,j,k,QPRES) = amrex::max(pres_new, pressure_floor);
            }
        }
        else {
            q_out_arr(i,j,k,QPRES) = prim_arr(i,j,k,QPRES);
            q_out_arr(i,j,k,QEINTDENS) = prim_arr(i,j,k,QEINTDENS);
        }

#ifdef RADIATION
        for (int g = 0; g < NGROUPS; ++g) {
            q_out_arr(i,j,k,QRADENG + g) = rad_energy_new_group[g];
        }

        q_out_arr(i,j,k,QPTOTAL) = q_out_arr(i,j,k,QPRES);
        for (int g = 0; g < NGROUPS; ++g) {
            q_out_arr(i,j,k,QPTOTAL) += lambda[g] * rad_energy_new_group[g];
        }

        q_out_arr(i,j,k,QETOTDENS) = q_out_arr(i,j,k,QEINTDENS);
        for (int g = 0; g < NGROUPS; ++g) {
            q_out_arr(i,j,k,QETOTDENS) += q_out_arr(i,j,k,QRADENG + g);
        }
#endif

    });

}

void
HydroSim::apply_tangential_correction(const Box& bx,
                           int idir_normal, int idir_tan1, int idir_tan2, int d,
                           Array4<Real const> const& prim_arr,
                           Array4<Real> const& q_out_arr,
                           Array4<Real const> const& qthermo_arr,
                           Array4<Real const> const& flux_tan1,
#ifdef RADIATION
                           Array4<Real const> const& radiative_flux_tan1,
#endif
                           Array4<Real const> const& flux_tan2,
#ifdef RADIATION
                           Array4<Real const> const& radiative_flux_tan2,
#endif
                           Array4<Real const> const& q_tan1,
                           Array4<Real const> const& q_tan2,
                           Real third_dt_dx_tan1, Real third_dt_dx_tan2)
{

    bool density_reset_enabled = transverse_revert_density;
    bool reset_eint_flag = transverse_rebuild_rhoe;
    bool apply_eos = transverse_eos_pressure;
    Real pressure_floor = pressure_cutoff;
    Real temperature_guess = eos_temp_guess;
    Real species_fraction_tol = massfrac_check_tolerance;

    GpuArray<int, 3> normal_shift{0, 0, 0};
    GpuArray<int, 3> t1_left{0, 0, 0};
    GpuArray<int, 3> t1_right{0, 0, 0};
    GpuArray<int, 3> t2_left{0, 0, 0};
    GpuArray<int, 3> t2_right{0, 0, 0};

    normal_shift[idir_normal] = d;
    t1_left[idir_normal] = d;
    t2_left[idir_normal] = d;
    t1_right[idir_normal] = d;
    t2_right[idir_normal] = d;

    t1_right[idir_normal == 0 ? 1 : 0] += 1;
    t2_right[idir_normal == 2 ? 1 : 2] += 1;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

        const int iln = i + normal_shift[0];
        const int jln = j + normal_shift[1];
        const int kln = k + normal_shift[2];

        const int trans1_left_i = i + t1_left[0];
        const int trans1_left_j = j + t1_left[1];
        const int trans1_left_k = k + t1_left[2];

        const int trans1_right_i = i + t1_right[0];
        const int trans1_right_j = j + t1_right[1];
        const int trans1_right_k = k + t1_right[2];

        const int trans2_left_i = i + t2_left[0];
        const int trans2_left_j = j + t2_left[1];
        const int trans2_left_k = k + t2_left[2];

        const int trans2_right_i = i + t2_right[0];
        const int trans2_right_j = j + t2_right[1];
        const int trans2_right_k = k + t2_right[2];


        auto corrected = [&] (Real base, int comp) {
            return base - third_dt_dx_tan1 * (flux_tan1(trans1_right_i,trans1_right_j,trans1_right_k,comp) -
                                              flux_tan1(trans1_left_i,trans1_left_j,trans1_left_k,comp))
                        - third_dt_dx_tan2 * (flux_tan2(trans2_right_i,trans2_right_j,trans2_right_k,comp) -
                                              flux_tan2(trans2_left_i,trans2_left_j,trans2_left_k,comp));
        };

        const Real rrn = prim_arr(i,j,k,QRHO);
        const Real density_partial = corrected(rrn, URHO);

        for (int passive_index = 0; passive_index < npassive; ++passive_index) {
            const int n = passive_cons_index(passive_index);
            const int nqp = passive_prim_index(passive_index);

            Real passive_cons_base = rrn * prim_arr(i,j,k,nqp);
            Real passive_cons_next = corrected(passive_cons_base, n);

            q_out_arr(i,j,k,nqp) = passive_cons_next / density_partial;
        }


        const Real pres_trans1_right  = q_tan1(trans1_right_i,trans1_right_j,trans1_right_k,GDPRESSURE);
        const Real pres_trans1_left  = q_tan1(trans1_left_i,trans1_left_j,trans1_left_k,GDPRESSURE);
        const Real vel_trans1_right  = q_tan1(trans1_right_i,trans1_right_j,trans1_right_k,GDU+idir_tan1);
        const Real vel_trans1_left  = q_tan1(trans1_left_i,trans1_left_j,trans1_left_k,GDU+idir_tan1);
#ifdef RADIATION
        Real rad_energy_trans1_right[NGROUPS];
        Real rad_energy_trans1_left[NGROUPS];
        for (int g = 0; g < NGROUPS; ++g) {
            rad_energy_trans1_right[g] = q_tan1(trans1_right_i,trans1_right_j,trans1_right_k,GDRADENG+g);
            rad_energy_trans1_left[g] = q_tan1(trans1_left_i,trans1_left_j,trans1_left_k,GDRADENG+g);
        }
#endif

        const Real pres_trans2_right  = q_tan2(trans2_right_i,trans2_right_j,trans2_right_k,GDPRESSURE);
        const Real pres_trans2_left  = q_tan2(trans2_left_i,trans2_left_j,trans2_left_k,GDPRESSURE);
        const Real vel_trans2_right  = q_tan2(trans2_right_i,trans2_right_j,trans2_right_k,GDU+idir_tan2);
        const Real vel_trans2_left  = q_tan2(trans2_left_i,trans2_left_j,trans2_left_k,GDU+idir_tan2);
#ifdef RADIATION
        Real rad_energy_trans2_right[NGROUPS];
        Real rad_energy_trans2_left[NGROUPS];
        for (int g = 0; g < NGROUPS; ++g) {
            rad_energy_trans2_right[g] = q_tan2(trans2_right_i,trans2_right_j,trans2_right_k,GDRADENG+g);
            rad_energy_trans2_left[g] = q_tan2(trans2_left_i,trans2_left_j,trans2_left_k,GDRADENG+g);
        }
#endif

        const Real pres_vel_diff_trans1 = pres_trans1_right * vel_trans1_right - pres_trans1_left * vel_trans1_left;
        const Real pres_avg_trans1 = 0.5_rt * (pres_trans1_right + pres_trans1_left);
        const Real vel_diff_trans1 = vel_trans1_right - vel_trans1_left;
#ifdef RADIATION
        const Real pres_correction_trans1 = third_dt_dx_tan1 * (pres_vel_diff_trans1 + pres_avg_trans1 * vel_diff_trans1 * (qthermo_arr(iln,jln,kln,QGAMMA1G) - 1.0_rt));
#else
        const Real pres_correction_trans1 = third_dt_dx_tan1 * (pres_vel_diff_trans1 + pres_avg_trans1 * vel_diff_trans1 * (qthermo_arr(iln,jln,kln,QGAMMA1) - 1.0_rt));
#endif

        const Real pres_vel_diff_trans2 = pres_trans2_right * vel_trans2_right - pres_trans2_left * vel_trans2_left;
        const Real pres_avg_trans2 = 0.5_rt * (pres_trans2_right + pres_trans2_left);
        const Real vel_diff_trans2 = vel_trans2_right - vel_trans2_left;
#ifdef RADIATION
        const Real pres_correction_trans2 = third_dt_dx_tan2 * (pres_vel_diff_trans2 + pres_avg_trans2 * vel_diff_trans2 * (qthermo_arr(iln,jln,kln,QGAMMA1G) - 1.0_rt));
#else
        const Real pres_correction_trans2 = third_dt_dx_tan2 * (pres_vel_diff_trans2 + pres_avg_trans2 * vel_diff_trans2 * (qthermo_arr(iln,jln,kln,QGAMMA1) - 1.0_rt));
#endif

#ifdef RADIATION
        Real lambda[NGROUPS];
        Real rad_lambda_jump_trans1[NGROUPS];
        Real rad_lambda_jump_trans2[NGROUPS];
        Real mom_delta_trans1 = 0.0_rt;
        Real mom_delta_trans2 = 0.0_rt;
        Real rad_work_trans1[NGROUPS];
        Real rad_work_trans2[NGROUPS];
        Real dre = 0.0_rt;

        for (int g = 0; g < NGROUPS; ++g) {
            lambda[g] = qthermo_arr(iln,jln,kln,QFLUXLIM+g);
            rad_lambda_jump_trans1[g] = lambda[g] * (rad_energy_trans1_right[g] - rad_energy_trans1_left[g]);
            rad_lambda_jump_trans2[g] = lambda[g] * (rad_energy_trans2_right[g] - rad_energy_trans2_left[g]);
            mom_delta_trans1 -= third_dt_dx_tan1 * rad_lambda_jump_trans1[g];
            mom_delta_trans2 -= third_dt_dx_tan2 * rad_lambda_jump_trans2[g];
            rad_work_trans1[g] = 0.5_rt * (vel_trans1_right + vel_trans1_left) * rad_lambda_jump_trans1[g];
            rad_work_trans2[g] = 0.5_rt * (vel_trans2_right + vel_trans2_left) * rad_lambda_jump_trans2[g];
            dre -= third_dt_dx_tan1 * rad_work_trans1[g] + third_dt_dx_tan2 * rad_work_trans2[g];
        }

        Real der[NGROUPS];

        if (radiation::frequency_advection_mode == 1 && radiation::moving_frame) {
            for (int g = 0; g < NGROUPS; ++g) {
                Real eddington_factor = Eddington_factor(lambda[g]);
                Real f1 = 0.5_rt * (1.0_rt - eddington_factor);
                der[g] = f1 * (third_dt_dx_tan1 * 0.5_rt * (vel_trans1_right + vel_trans1_left) * (rad_energy_trans1_right[g] - rad_energy_trans1_left[g]) +
                               third_dt_dx_tan2 * 0.5_rt * (vel_trans2_right + vel_trans2_left) * (rad_energy_trans2_right[g] - rad_energy_trans2_left[g]));
            }
        }
        else if (radiation::frequency_advection_mode == 2) {
            for (int g = 0; g < NGROUPS; ++g) {
                Real eddington_factor = Eddington_factor(lambda[g]);
                Real f1 = 0.5_rt * (1.0_rt - eddington_factor);
                der[g] = f1 * (third_dt_dx_tan1 * 0.5_rt * (rad_energy_trans1_right[g] + rad_energy_trans1_left[g]) * (vel_trans1_left - vel_trans1_right) +
                               third_dt_dx_tan2 * 0.5_rt * (rad_energy_trans2_right[g] + rad_energy_trans2_left[g]) * (vel_trans2_left - vel_trans2_right));
            }
        }
        else {
            for (int g = 0; g < NGROUPS; ++g) {
                der[g] = third_dt_dx_tan1 * rad_work_trans1[g] + third_dt_dx_tan2 * rad_work_trans2[g];
            }
        }
#endif

        Real run = rrn * prim_arr(i,j,k,QU);
        Real rvn = rrn * prim_arr(i,j,k,QV);
        Real rwn = rrn * prim_arr(i,j,k,QW);
        Real kinetic_energy_base = 0.5_rt * rrn * (prim_arr(i,j,k,QU) * prim_arr(i,j,k,QU) + prim_arr(i,j,k,QV) * prim_arr(i,j,k,QV) + prim_arr(i,j,k,QW) * prim_arr(i,j,k,QW));
        Real ren = prim_arr(i,j,k,QEINTDENS) + kinetic_energy_base;
#ifdef RADIATION
        Real ern[NGROUPS];
        for (int g = 0; g < NGROUPS; ++g) {
            ern[g] = prim_arr(i,j,k,QRADENG+g);
        }
#endif

        Real density_new = density_partial;
        Real momentum_x_new = corrected(run, UMX);
        Real momentum_y_new = corrected(rvn, UMY);
        Real momentum_z_new = corrected(rwn, UMZ);
        Real energy_new = corrected(ren, UEDEN);
#ifdef RADIATION
        if (idir_normal == 0) {
            momentum_y_new = momentum_y_new + mom_delta_trans1;
            momentum_z_new = momentum_z_new + mom_delta_trans2;
        }
        else if (idir_normal == 1) {
            momentum_x_new = momentum_x_new + mom_delta_trans1;
            momentum_z_new = momentum_z_new + mom_delta_trans2;
        }
        else {
            momentum_x_new = momentum_x_new + mom_delta_trans1;
            momentum_y_new = momentum_y_new + mom_delta_trans2;
        }
        energy_new = energy_new + dre;

        Real rad_energy_new_group[NGROUPS];
        for (int g = 0; g < NGROUPS; ++g) {
            rad_energy_new_group[g] = ern[g] - third_dt_dx_tan1 * (radiative_flux_tan1(trans1_right_i,trans1_right_j,trans1_right_k,g) -
                                             radiative_flux_tan1(trans1_left_i,trans1_left_j,trans1_left_k,g))
                               - third_dt_dx_tan2 * (radiative_flux_tan2(trans2_right_i,trans2_right_j,trans2_right_k,g) -
                                             radiative_flux_tan2(trans2_left_i,trans2_left_j,trans2_left_k,g))
                               + der[g];
        }
#endif

        bool state_was_reset = false;
        if (density_reset_enabled && density_new < 0.0_rt) {
            density_new = rrn;
            momentum_x_new = run;
            momentum_y_new = rvn;
            momentum_z_new = rwn;
            energy_new = ren;
            for (int passive_index = 0; passive_index < npassive; ++passive_index) {
                int nqp = passive_prim_index(passive_index);
                q_out_arr(i,j,k,nqp) = prim_arr(i,j,k,nqp);
            }
#ifdef RADIATION
            for (int g = 0; g < NGROUPS; ++g) {
                rad_energy_new_group[g] = ern[g];
            }
#endif
            state_was_reset = true;
        }


        for (int n = 0; n < NumSpec; ++n) {
            if (q_out_arr(i,j,k,n+QFS) > 1.0_rt + species_fraction_tol ||
                q_out_arr(i,j,k,n+QFS) < -species_fraction_tol) {
                density_new = rrn;
                momentum_x_new = run;
                momentum_y_new = rvn;
                momentum_z_new = rwn;
                energy_new = ren;
                for (int passive_index = 0; passive_index < npassive; ++passive_index) {
                    int nqp = passive_prim_index(passive_index);
                    q_out_arr(i,j,k,nqp) = prim_arr(i,j,k,nqp);
                }
#ifdef RADIATION
                for (int g = 0; g < NGROUPS; ++g) {
                    rad_energy_new_group[g] = ern[g];
                }
#endif
                state_was_reset = true;
                break;
            }
        }

        q_out_arr(i,j,k,QRHO) = density_new;

        q_out_arr(i,j,k,QU) = momentum_x_new / density_new;
        q_out_arr(i,j,k,QV) = momentum_y_new / density_new;
        q_out_arr(i,j,k,QW) = momentum_z_new / density_new;

        const Real kinetic_energy_new = 0.5_rt * (momentum_x_new * momentum_x_new + momentum_y_new * momentum_y_new + momentum_z_new * momentum_z_new) / density_new;
        q_out_arr(i,j,k,QEINTDENS) = energy_new - kinetic_energy_new;

        if (!state_was_reset) {
            if (reset_eint_flag == 1 && q_out_arr(i,j,k,QEINTDENS) <= 0.0_rt) {
                q_out_arr(i,j,k,QEINTDENS) = prim_arr(i,j,k,QEINTDENS)
                                 - third_dt_dx_tan1 * (flux_tan1(trans1_right_i,trans1_right_j,trans1_right_k,UEINT) -
                                               flux_tan1(trans1_left_i,trans1_left_j,trans1_left_k,UEINT) + pres_avg_trans1 * vel_diff_trans1)
                                 - third_dt_dx_tan2 * (flux_tan2(trans2_right_i,trans2_right_j,trans2_right_k,UEINT) -
                                               flux_tan2(trans2_left_i,trans2_left_j,trans2_left_k,UEINT) + pres_avg_trans2 * vel_diff_trans2);
            }


            if (q_out_arr(i,j,k,QEINTDENS) <= 0.0_rt) {
                q_out_arr(i,j,k,QEINTDENS) = prim_arr(i,j,k,QEINTDENS);
            }


            if (apply_eos) {

                eos_rep_t eos_state;
                eos_state.rho = density_new;
                eos_state.e = q_out_arr(i,j,k,QEINTDENS) / density_new;
                eos_state.T = temperature_guess;

                for (int n = 0; n < NumSpec; n++) {
                    eos_state.xn[n] = q_out_arr(i,j,k,QFS+n);
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                    eos_state.aux[n] = q_out_arr(i,j,k,QFX+n);
                }
#endif
                eos(eos_input_re, eos_state);
                q_out_arr(i,j,k,QPRES) = eos_state.p;

            } else {

                Real pres_new = prim_arr(i,j,k,QPRES) - pres_correction_trans1 - pres_correction_trans2;
                q_out_arr(i,j,k,QPRES) = pres_new;
            }

        }
        else {
            q_out_arr(i,j,k,QPRES) = prim_arr(i,j,k,QPRES);
            q_out_arr(i,j,k,QEINTDENS) = prim_arr(i,j,k,QEINTDENS);
        }

        q_out_arr(i,j,k,QPRES) = amrex::max(q_out_arr(i,j,k,QPRES), pressure_floor);

#ifdef RADIATION
        for (int g = 0; g < NGROUPS; ++g) {
            q_out_arr(i,j,k,QRADENG+g) = rad_energy_new_group[g];
        }

        q_out_arr(i,j,k,QPTOTAL) = q_out_arr(i,j,k,QPRES);
        for (int g = 0; g < NGROUPS; ++g) {
            q_out_arr(i,j,k,QPTOTAL) += lambda[g] * rad_energy_new_group[g];
        }

        q_out_arr(i,j,k,QETOTDENS) = q_out_arr(i,j,k,QEINTDENS);
        for (int g = 0; g < NGROUPS; ++g) {
            q_out_arr(i,j,k,QETOTDENS) += q_out_arr(i,j,k,QRADENG+g);
        }
#endif

    });

}

void
HydroSim::tangential_pass(const Box& bx,
                     int idir_tan, int idir_normal,
                     Array4<Real const> const& qm,
                     Array4<Real> const& qmo,
                     Array4<Real const> const& qp,
                     Array4<Real> const& qpo,
                     Array4<Real const> const& qthermo_arr,
                     Array4<Real const> const& flux_tan,
#ifdef RADIATION
                     Array4<Real const> const& radiative_flux_tan,
#endif
                     Array4<Real const> const& q_t,
#if AMREX_SPACEDIM == 2
                     Array4<Real const> const& area_t,
                     Array4<Real const> const& vol,
#endif
                     Real hdt, Real third_dt_dx)
{

    apply_tangential_pass(bx, idir_tan, idir_normal, -1,
                        qm, qmo,
                        qthermo_arr,
                        flux_tan,
#ifdef RADIATION
                        radiative_flux_tan,
#endif
                        q_t,
#if AMREX_SPACEDIM == 2
                        area_t,
                        vol,
#endif
                        hdt, third_dt_dx);

    apply_tangential_pass(bx, idir_tan, idir_normal, 0,
                        qp, qpo,
                        qthermo_arr,
                        flux_tan,
#ifdef RADIATION
                        radiative_flux_tan,
#endif
                        q_t,
#if AMREX_SPACEDIM == 2
                        area_t,
                        vol,
#endif
                        hdt, third_dt_dx);

}

void
HydroSim::tangential_correction(const Box& bx,
                    int idir_normal, int idir_tan1, int idir_tan2,
                    Array4<Real const> const& qm,
                    Array4<Real> const& qmo,
                    Array4<Real const> const& qp,
                    Array4<Real> const& qpo,
                    Array4<Real const> const& qthermo_arr,
                    Array4<Real const> const& flux_tan1,
#ifdef RADIATION
                    Array4<Real const> const& radiative_flux_tan1,
#endif
                    Array4<Real const> const& flux_tan2,
#ifdef RADIATION
                    Array4<Real const> const& radiative_flux_tan2,
#endif
                    Array4<Real const> const& q_tan1,
                    Array4<Real const> const& q_tan2,
                    Real third_dt_dx_tan1, Real third_dt_dx_tan2)
{

    apply_tangential_correction(bx, idir_normal, idir_tan1, idir_tan2, -1,
                       qm, qmo,
                       qthermo_arr,
                       flux_tan1,
#ifdef RADIATION
                       radiative_flux_tan1,
#endif
                       flux_tan2,
#ifdef RADIATION
                       radiative_flux_tan2,
#endif
                       q_tan1,
                       q_tan2,
                       third_dt_dx_tan1, third_dt_dx_tan2);

    apply_tangential_correction(bx, idir_normal, idir_tan1, idir_tan2, 0,
                       qp, qpo,
                       qthermo_arr,
                       flux_tan1,
#ifdef RADIATION
                       radiative_flux_tan1,
#endif
                       flux_tan2,
#ifdef RADIATION
                       radiative_flux_tan2,
#endif
                       q_tan1,
                       q_tan2,
                       third_dt_dx_tan1, third_dt_dx_tan2);

}
