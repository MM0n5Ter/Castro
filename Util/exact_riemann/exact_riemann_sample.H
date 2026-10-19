#ifndef RIEMANN_SAMPLE_MODULE_H
#define RIEMANN_SAMPLE_MODULE_H

#include <AMReX_REAL.H>

#include <solver_constants.H>
#include <exact_riemann_rarefaction.H>

using namespace amrex::literals;

AMREX_INLINE
void
riemann_sample(const amrex::Real rho_l, const amrex::Real u_l, const amrex::Real p_l, const amrex::Real* xn_l,
               const amrex::Real rho_r, const amrex::Real u_r, const amrex::Real p_r, const amrex::Real* xn_r,
               const amrex::Real ustar, const amrex::Real pstar,
               const amrex::Real W_l, const amrex::Real W_r,
               const amrex::Real x, const amrex::Real xjump, const amrex::Real time,
               amrex::Real& rho, amrex::Real& u, amrex::Real& p, amrex::Real* xn) {


    eos_t eos_state;
    eos_state.rho = rho_l;
    eos_state.p = p_l;
    for (int n = 0; n < NumSpec; ++n) {
        eos_state.xn[n] = xn_l[n];
    }
    eos_state.T = hydrosim::eos_temp_guess;

    eos(eos_input_rp, eos_state);

    amrex::Real cs_l = std::sqrt(eos_state.gam1 * p_l / rho_l);

    eos_state.rho = rho_r;
    eos_state.p = p_r;
    for (int n = 0; n < NumSpec; ++n) {
        eos_state.xn[n] = xn_r[n];
    }
    eos_state.T = hydrosim::eos_temp_guess;

    eos(eos_input_rp, eos_state);

    amrex::Real cs_r = std::sqrt(eos_state.gam1 * p_r / rho_r);




    amrex::Real xi = (x - xjump) / time;


    amrex::Real chi = std::copysign(1.0_rt, xi - ustar);

    amrex::Real rho_s, u_s, p_s, W_s, cs_s;
    amrex::Real uhat_s, xihat, uhat_star;

    if (chi == -1.0_rt) {
        rho_s = rho_l;
        u_s = u_l;
        p_s = p_l;

        W_s = W_l;

        cs_s = cs_l;

        for (int n = 0; n < NumSpec; ++n) {
            xn[n] = xn_l[n];
        }

        uhat_s = chi * u_s;
        xihat = chi * xi;
        uhat_star = chi * ustar;

    } else if (chi == 1.0_rt) {
        rho_s = rho_r;
        u_s = u_r;
        p_s = p_r;

        W_s = W_r;

        cs_s = cs_r;

        for (int n = 0; n < NumSpec; ++n) {
            xn[n] = xn_r[n];
        }

        uhat_s = chi * u_s;
        xihat = chi * xi;
        uhat_star = chi * ustar;

    } else {
        amrex::Error("Not implemented");

    }



    amrex::Real rhostar, Z_temp;

    if (pstar > p_s) {

        rhostar = 1.0_rt / (1.0_rt / rho_s - (pstar - p_s)/ (W_s * W_s));

    } else {
        amrex::Real Z_temp, W_temp;
        if (chi == -1.0_rt) {
            rarefaction(pstar, rho_s, u_s, p_s, xn, 1, Z_temp, W_temp, rhostar);
        } else {
            rarefaction(pstar, rho_s, u_s, p_s, xn, 3, Z_temp, W_temp, rhostar);
        }

    }


    eos_state.rho = rhostar;
    eos_state.p = pstar;
    for (int n = 0; n < NumSpec; ++n) {
        eos_state.xn[n] = xn[n];
    }
    eos_state.T = hydrosim::eos_temp_guess;

    eos(eos_input_rp, eos_state);

    amrex::Real cs_star = std::sqrt(eos_state.gam1 * pstar / rhostar);


    amrex::Real lambdahat_s, lambdahat_star;

    if (pstar <= p_s) {
        lambdahat_s = uhat_s + cs_s;
        lambdahat_star = uhat_star + cs_star;
    } else {
        lambdahat_s = uhat_s + W_s / rho_s;
        lambdahat_star = lambdahat_s;
    }

    if (xihat <= lambdahat_star) {
        p = pstar;
        rho = rhostar;
        u = ustar;

    } else if (xihat > lambdahat_s) {
        p = p_s;
        rho = rho_s;
        u = u_s;

    } else {





        if (chi == -1.0_rt) {
            rarefaction_to_u(rho_s, u_s, p_s, xn, 1, xi, rho, p, u);

        } else if (chi == 1.0_rt) {
            rarefaction_to_u(rho_s, u_s, p_s, xn, 3, xi, rho, p, u);

        }
    }

}
#endif
