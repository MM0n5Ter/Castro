#ifndef RIEMANN_RAREFACTION_H
#define RIEMANN_RAREFACTION_H

#include <utility>

#include <AMReX_REAL.H>

#include <eos_type.H>
#include <eos.H>


AMREX_INLINE
void
riemann_invariant_rhs(const amrex::Real p, const amrex::Real tau, const amrex::Real u,
                      const amrex::Real* xn, const int iwave,
                      amrex::Real& T,
                      amrex::Real& dtaudp, amrex::Real& dudp) {

    amrex::ignore_unused(u);




    eos_rep_t eos_state;
    eos_state.rho = 1.0_rt / tau;
    eos_state.p = p;
    for (int n = 0; n < NumSpec; ++n) {
        eos_state.xn[n] = xn[n];
    }
    eos_state.T = T;

    eos(eos_input_rp, eos_state);

    T = eos_state.T;

    amrex::Real C = std::sqrt(eos_state.gam1 * p / tau);

    dtaudp = -1.0_rt / (C * C);

    if (iwave == 1) {
        dudp = -1.0_rt / C;
    } else if (iwave == 3) {
        dudp = 1.0_rt / C;
    }

}


AMREX_INLINE
void
riemann_invariant_rhs2(const amrex::Real u, const amrex::Real tau, const amrex::Real p,
                       const amrex::Real* xn, const int iwave,
                       amrex::Real& T,
                       amrex::Real& dtaudu, amrex::Real& dpdu) {




    amrex::ignore_unused(u);

    eos_rep_t eos_state;
    eos_state.rho = 1.0_rt / tau;
    eos_state.p = p;
    for (int n = 0; n < NumSpec; ++n) {
        eos_state.xn[n] = xn[n];
    }
    eos_state.T = T;

    eos(eos_input_rp, eos_state);

    T = eos_state.T;

    amrex::Real C = std::sqrt(eos_state.gam1 * p / tau);

    if (iwave == 3) {
       dpdu = C;
       dtaudu = -1.0_rt / C;

    } else if (iwave == 1) {
        dpdu = -C;
        dtaudu = 1.0_rt / C;
    }
}


AMREX_INLINE
std::pair<amrex::Real, amrex::Real>
single_step_p(const amrex::Real pstar0, const amrex::Real dp,
              const amrex::Real u0, const amrex::Real tau0,
              const amrex::Real* xn, const int iwave, amrex::Real& T) {


    amrex::Real dtaudp1, dudp1;
    riemann_invariant_rhs(pstar0, tau0, u0, xn, iwave, T, dtaudp1, dudp1);

    amrex::Real dtaudp2, dudp2;
    riemann_invariant_rhs(pstar0+0.5*dp, tau0+0.5*dp*dtaudp1, u0+0.5*dp*dudp1,
                          xn, iwave, T, dtaudp2, dudp2);

    amrex::Real dtaudp3, dudp3;
    riemann_invariant_rhs(pstar0+0.5*dp, tau0+0.5*dp*dtaudp2, u0+0.5*dp*dudp2,
                          xn, iwave, T, dtaudp3, dudp3);

    amrex::Real dtaudp4, dudp4;
    riemann_invariant_rhs(pstar0+dp, tau0+dp*dtaudp3, u0+dp*dudp3,
                          xn, iwave, T, dtaudp4, dudp4);

    amrex::Real u = u0 + (1.0_rt/6.0_rt) * dp * (dudp1 + 2.0_rt * dudp2 + 2.0_rt * dudp3 + dudp4);
    amrex::Real tau = tau0 + (1.0_rt/6.0_rt) * dp * (dtaudp1 + 2.0_rt * dtaudp2 + 2.0_rt * dtaudp3 + dtaudp4);

    return {u, tau};
}


AMREX_INLINE
std::pair<amrex::Real, amrex::Real>
single_step_u(const amrex::Real u0, const amrex::Real du,
              const amrex::Real tau0, const amrex::Real p0,
              const amrex::Real* xn, const int iwave, amrex::Real& T) {


    amrex::Real dtaudu1, dpdu1;
    riemann_invariant_rhs2(u0, tau0, p0, xn, iwave, T, dtaudu1, dpdu1);

    amrex::Real dtaudu2, dpdu2;
    riemann_invariant_rhs2(u0+0.5*du, tau0+0.5*du*dtaudu1, p0+0.5*du*dpdu1,
                           xn, iwave, T, dtaudu2, dpdu2);

    amrex::Real dtaudu3, dpdu3;
    riemann_invariant_rhs2(u0+0.5*du, tau0+0.5*du*dtaudu2, p0+0.5*du*dpdu2,
                           xn, iwave, T, dtaudu3, dpdu3);

    amrex::Real dtaudu4, dpdu4;
    riemann_invariant_rhs2(u0+du, tau0+du*dtaudu3, p0+du*dpdu3,
                           xn, iwave, T, dtaudu4, dpdu4);

    amrex::Real p = p0 + (1.0_rt/6.0_rt) * du * (dpdu1 + 2.0_rt * dpdu2 + 2.0_rt * dpdu3 + dpdu4);
    amrex::Real tau = tau0 + (1.0_rt/6.0_rt) * du * (dtaudu1 + 2.0_rt * dtaudu2 + 2.0_rt * dtaudu3 + dtaudu4);

    return {tau, p};

}

AMREX_INLINE
void
rarefaction(const amrex::Real pstar,
            const amrex::Real rho_s, const amrex::Real u_s, const amrex::Real p_s,
            const amrex::Real* xn, const int iwave,
            amrex::Real& Z_s, amrex::Real& W_s, amrex::Real& rhostar) {


    const double S1{0.9};
    const double S2{4.0};


    amrex::Real tau = 1.0_rt / rho_s;
    amrex::Real u = u_s;
    amrex::Real p = p_s;

    amrex::Real dp = (pstar - p_s) / static_cast<amrex::Real>(100);

    amrex::Real T = hydrosim::eos_temp_guess;



    amrex::Real dp_new{dp};
    int nstep{0};

    while (p > pstar) {

        amrex::Real u0{u};
        amrex::Real tau0{tau};

        amrex::Real u_new;
        amrex::Real tau_new;

        amrex::Real rel_error = std::numeric_limits<double>::max();


        while (rel_error > solver_constants::integral_tolerance) {
            dp = dp_new;
            if (p + dp < pstar) {
                dp = pstar - p;
            }

            amrex::Real u_tmp;
            amrex::Real tau_tmp;
            std::tie(u_tmp, tau_tmp) = single_step_p(p, 0.5*dp, u0, tau0, xn, iwave, T);
            std::tie(u_new, tau_new) = single_step_p(p + 0.5*dp, 0.5*dp, u_tmp, tau_tmp, xn, iwave, T);

            auto [u_single, tau_single] = single_step_p(p, dp, u0, tau0, xn, iwave, T);

            amrex::Real u_err = std::abs(u_new - u_single);
            if (u0 != 0.0) {
                u_err /= u0;
            }
            amrex::Real tau_err = std::abs((tau_new - tau_single) / tau0);

            rel_error = std::max(u_err, tau_err);


            double dp_est = S1 * dp * std::pow(std::abs(solver_constants::integral_tolerance/rel_error), 0.2);
            dp_new = dp_est;

        }

        p += dp;
        u = u_new;
        tau = tau_new;

        nstep++;
    }


    eos_rep_t eos_state;
    eos_state.rho = 1.0_rt / tau;
    eos_state.p = p;
    for (int n = 0; n < NumSpec; ++n) {
        eos_state.xn[n] = xn[n];
    }
    eos_state.T = hydrosim::eos_temp_guess;

    eos(eos_input_rp, eos_state);

    Z_s = std::sqrt(eos_state.gam1 * p / tau);


    if (u == u_s) {
        W_s = Z_s;
    } else {
        W_s = std::abs(pstar - p_s) / std::abs(u - u_s);
    }

    rhostar = 1.0_rt / tau;
}


AMREX_INLINE
void
rarefaction_to_u(const amrex::Real rho_s, const amrex::Real u_s, const amrex::Real p_s,
                 const amrex::Real* xn, const int iwave, const amrex::Real xi,
                 amrex::Real& rho, amrex::Real& p, amrex::Real& u) {

    const int npts = 1000;


    const double S1{0.9};


    amrex::Real tau = 1.0_rt / rho_s;
    u = u_s;
    p = p_s;


    eos_rep_t eos_state;
    eos_state.rho = 1.0_rt / tau;
    eos_state.p = p;
    for (int n = 0; n < NumSpec; ++n) {
        eos_state.xn[n] = xn[n];
    }
    eos_state.T = hydrosim::eos_temp_guess;

    eos(eos_input_rp, eos_state);

    amrex::Real c = std::sqrt(eos_state.gam1 * p * tau);

    amrex::Real ustop = (iwave == 1) ? xi + c : xi - c;


    amrex::Real du = (ustop - u_s) / static_cast<amrex::Real>(10);

    bool finished = false;


    amrex::Real T = eos_state.T;

    amrex::Real du_new{du};

    while (! finished) {

        amrex::Real p0{p};
        amrex::Real tau0{tau};

        amrex::Real p_new;
        amrex::Real tau_new;

        amrex::Real rel_error = std::numeric_limits<double>::max();


        while (rel_error > solver_constants::integral_tolerance) {
            du = du_new;

            amrex::Real p_tmp;
            amrex::Real tau_tmp;
            std::tie(tau_tmp, p_tmp) = single_step_u(u, 0.5*du, tau0, p0, xn, iwave, T);
            std::tie(tau_new, p_new) = single_step_u(u + 0.5*du, 0.5*du, tau_tmp, p_tmp, xn, iwave, T);

            auto [tau_single, p_single] = single_step_u(u, du, tau0, p0, xn, iwave, T);

            amrex::Real p_err = std::abs((p_new - p_single) / p0);
            amrex::Real tau_err = std::abs((tau_new - tau_single) / tau0);

            rel_error = std::max(p_err, tau_err);



            amrex::Real du_est{};
            if (rel_error > 0) {
                du_est = S1 * du * std::pow(std::abs(solver_constants::integral_tolerance / rel_error), 0.2);
            } else {
                du_est = 10.0 * du;
            }
            du_new = du_est;

        }

        u += du;
        p = p_new;
        tau = tau_new;


        eos_rep_t eos_state;
        eos_state.rho = 1.0_rt/tau;
        eos_state.p = p;
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = xn[n];
        }
        eos_state.T = hydrosim::eos_temp_guess;

        eos(eos_input_rp, eos_state);

        c = std::sqrt(eos_state.gam1 * p * tau);

        ustop = (iwave == 1) ? xi + c : xi - c;


        if (du_new * u > 0.0_rt) {
            while (std::abs(u + du_new) > std::abs(ustop) && du_new != 0.0_rt) {
                du_new *= 0.5_rt;
            }
        } else {
            if (u > 0.0_rt) {
                while (u + du_new < ustop && du_new != 0.0_rt) {
                    du_new *= 0.5_rt;
                }

            } else {
                while (u + du_new > ustop && du_new != 0.0_rt) {
                    du_new *= 0.5_rt;
                }
            }
        }

        if (std::abs(du_new) < solver_constants::velocity_tolerance * std::abs(u) + solver_constants::velocity_tolerance) {
            finished = true;
        }

    }

    rho = 1.0_rt / tau;
}

#endif
