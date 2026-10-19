#ifndef RIEMANN_STAR_STATE_H
#define RIEMANN_STAR_STATE_H

#include <AMReX_REAL.H>

#include <solver_constants.H>
#include <exact_riemann_shock.H>
#include <exact_riemann_rarefaction.H>

using namespace amrex::literals;

AMREX_INLINE
int
riemann_star_state(const amrex::Real rho_l, const amrex::Real u_l, const amrex::Real p_l, const amrex::Real* xn_l,
                   const amrex::Real rho_r, const amrex::Real u_r, const amrex::Real p_r, const amrex::Real* xn_r,
                   amrex::Real& ustar, amrex::Real& pstar, amrex::Real& W_l, amrex::Real& W_r) {

    const amrex::Real tol = 1.e-10_rt;
    const amrex::Real smallp = 1.e-8_rt;
    const amrex::Real SMALL = 1.e-8_rt;



    eos_t eos_state;
    eos_state.rho = rho_l;
    eos_state.p = p_l;
    for (int n = 0; n < NumSpec; ++n) {
        eos_state.xn[n] = xn_l[n];
    }
    eos_state.T = hydrosim::eos_temp_guess;

    eos(eos_input_rp, eos_state);

    amrex::Real cs_l = std::sqrt(eos_state.gam1 * p_l / rho_l);

    amrex::Real gammaE_l = p_l / (rho_l * eos_state.e) + 1.0_rt;
    amrex::Real gammaC_l = eos_state.gam1;

    eos_state.rho = rho_r;
    eos_state.p = p_r;
    for (int n = 0; n < NumSpec; ++n) {
        eos_state.xn[n] = xn_r[n];
    }
    eos_state.T = hydrosim::eos_temp_guess;

    eos(eos_input_rp, eos_state);

    amrex::Real cs_r = std::sqrt(eos_state.gam1 * p_r / rho_r);

    amrex::Real gammaE_r = p_r / (rho_r * eos_state.e) + 1.0_rt;
    amrex::Real gammaC_r = eos_state.gam1;

    amrex::Real gammaE_bar = 0.5_rt * (gammaE_l + gammaE_r);
    amrex::Real gammaC_bar = 0.5_rt * (gammaC_l + gammaC_r);





    W_l = rho_l * cs_l;
    W_r = rho_r * cs_r;


    if (W_l == W_r) {
        pstar = 0.5_rt * (p_l + p_r + W_l * (u_l - u_r));
    } else {
        pstar = ((W_r * p_l + W_l * p_r) + W_l * W_r * (u_l - u_r)) / (W_l + W_r);
    }

    pstar = std::max(pstar, smallp);




    amrex::Real star_velocity_left, star_velocity_right;
    bool converged = false;

    int iter = 1;
    while (! converged && iter < hydrosim::shock_star_max_iters) {


        amrex::Real Z_l, Z_r;


        if (pstar - p_l > SMALL * p_l) {
            shock(pstar, rho_l, u_l, p_l, xn_l, gammaE_bar, gammaC_bar, Z_l, W_l);
        } else {
            amrex::Real rhostar_dummy;
            rarefaction(pstar, rho_l, u_l, p_l, xn_l, 1, Z_l, W_l, rhostar_dummy);
        }


        if (pstar - p_r > SMALL * p_r) {
            shock(pstar, rho_r, u_r, p_r, xn_r, gammaE_bar, gammaC_bar, Z_r, W_r);
        } else {
            amrex::Real rhostar_dummy;
            rarefaction(pstar, rho_r, u_r, p_r, xn_r, 3, Z_r, W_r, rhostar_dummy);
        }

        star_velocity_left = u_l - (pstar - p_l) / W_l;
        star_velocity_right = u_r + (pstar - p_r) / W_r;

        amrex::Real pstar_new = pstar - Z_l * Z_r * (star_velocity_right - star_velocity_left) / (Z_l + Z_r);


        amrex::Real error = std::abs(pstar_new - pstar);

        if (error < tol * pstar) {
            converged = true;
        }

        pstar = amrex::Clamp(pstar_new, 0.5 * pstar, 2.0 * pstar);

        iter++;
    }

    if (! converged) {
        amrex::Error("star state did not converge");
    }

    ustar = 0.5_rt * (star_velocity_left + star_velocity_right);

    return iter;

}
#endif
