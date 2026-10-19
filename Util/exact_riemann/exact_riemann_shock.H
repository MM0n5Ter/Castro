#ifndef RIEMANN_SHOCK_H
#define RIEMANN_SHOCK_H

#include <AMReX_REAL.H>

#include <eos_type.H>
#include <eos.H>

#include <solver_constants.H>

AMREX_INLINE
void
W_s_shock(const amrex::Real W_s, const amrex::Real pstar,
          const amrex::Real rho_s, const amrex::Real p_s, const amrex::Real e_s,
          const amrex::Real* xn,
          amrex::Real& rhostar_s, eos_rep_t& eos_state, amrex::Real& f, amrex::Real& fprime) {


    amrex::Real taustar_s = (1.0_rt / rho_s) - (pstar - p_s) / (W_s * W_s);
    rhostar_s = 1.0_rt / taustar_s;


    eos_state.rho = rhostar_s;
    eos_state.p = pstar;
    for (int n = 0; n < NumSpec; ++n) {
        eos_state.xn[n] = xn[n];
    }
    eos_state.T = hydrosim::eos_temp_guess;

    eos(eos_input_rp, eos_state);


    f = W_s * W_s * (eos_state.e - e_s) - 0.5_rt * (pstar * pstar - p_s * p_s);


    amrex::Real dedrho_p = eos_state.dedr - eos_state.dedT * eos_state.dpdr / eos_state.dpdT;

    fprime = 2.0_rt * W_s * (eos_state.e - e_s) -
        2.0_rt * dedrho_p * (pstar - p_s) * amrex::Math::powi<2>(rhostar_s) / W_s;

}


AMREX_INLINE
bool
newton_shock(amrex::Real& W_s, const amrex::Real pstar,
             const amrex::Real rho_s, const amrex::Real p_s, const amrex::Real e_s,
             const amrex::Real* xn,
             amrex::Real* rhostar_hist, amrex::Real* Ws_hist,
             eos_rep_t& eos_state) {


    bool converged = false;

    int iter = 1;
    while (! converged && iter < hydrosim::shock_star_max_iters) {

        amrex::Real rhostar_s, f, fprime;

        W_s_shock(W_s, pstar, rho_s, p_s, e_s, xn,
                  rhostar_s, eos_state, f, fprime);

        amrex::Real dW = -f / fprime;

        if (std::abs(dW) < hydrosim::riemann_star_tol * W_s) {
            converged = true;
        }

        W_s = amrex::Clamp(W_s + dW, 0.5_rt * W_s, 2.0_rt * W_s);


        rhostar_hist[iter] = rhostar_s;
        Ws_hist[iter] = W_s;

        iter++;
    }

    return converged;
}

AMREX_INLINE
void
shock(const amrex::Real pstar,
      const amrex::Real rho_s, const amrex::Real u_s, const amrex::Real p_s,
      const amrex::Real* xn,
      const amrex::Real gammaE_bar, const amrex::Real gammaC_bar,
      amrex::Real& Z_s, amrex::Real& W_s) {

    amrex::ignore_unused(u_s);

    amrex::Real rhostar_hist[solver_constants::STAR_HISTORY_DEPTH], Ws_hist[solver_constants::STAR_HISTORY_DEPTH];




    eos_rep_t eos_state;
    eos_state.rho = rho_s;
    eos_state.p = p_s;
    for (int n = 0; n < NumSpec; ++n) {
        eos_state.xn[n] = xn[n];
    }
    eos_state.T = hydrosim::eos_temp_guess;

    eos(eos_input_rp, eos_state);

    amrex::Real e_s = eos_state.e;


    amrex::Real gammaE_s = p_s / (rho_s * e_s) + 1.0_rt;

    amrex::Real gammaE_star = gammaE_s +
        2.0_rt * (1.0_rt - gammaE_bar / gammaC_bar) * (gammaE_bar - 1.0_rt) *
        (pstar - p_s) / (pstar + p_s);


    if (pstar - p_s < solver_constants::pressure_tolerance * p_s) {
        W_s = std::sqrt(eos_state.gam1 * p_s * rho_s);
    } else {
       W_s = std::sqrt((pstar - p_s) *
                       (pstar + 0.5_rt * (gammaE_star - 1.0_rt) * (pstar + p_s)) /
                       (pstar / rho_s - (gammaE_star - 1.0_rt) / (gammaE_s - 1.0_rt) * p_s / rho_s));
    }


    amrex::Real taustar_s = (1.0_rt / rho_s) - (pstar - p_s) / (W_s * W_s);
    amrex::Real rhostar_s;

    if (taustar_s < 0.0_rt) {
        rhostar_s = solver_constants::small;
        W_s = std::sqrt((pstar - p_s) / (1.0_rt / rho_s - 1.0_rt / rhostar_s));
    }


    auto converged = newton_shock(W_s, pstar, rho_s, p_s, e_s, xn,
                                  rhostar_hist, Ws_hist,
                                  eos_state);



    if (! converged) {
        for (int i = 0; i < hydrosim::shock_star_max_iters; ++i) {
            std::cout << i << " " << rhostar_hist[i] << " " << Ws_hist[i] << std::endl;
        }

        amrex::Error("shock did not converge");
    }



    taustar_s = (1.0_rt / rho_s) - (pstar - p_s) / (W_s * W_s);
    rhostar_s = 1.0_rt / taustar_s;


    amrex::Real C = std::sqrt(eos_state.gam1 * pstar * rhostar_s);

    amrex::Real p_e = eos_state.dpdT / eos_state.dedT;
    amrex::Real p_rho = eos_state.dpdr - eos_state.dpdT * eos_state.dedr / eos_state.dedT;

    amrex::Real p_tau = -amrex::Math::powi<2>(rhostar_s) * p_rho;

    amrex::Real dW2dpstar = (C*C - W_s*W_s) * W_s * W_s /
        ((0.5_rt * (pstar + p_s) * p_e - p_tau) * (pstar - p_s));

    amrex::Real dWdpstar = 0.5_rt * dW2dpstar / W_s;


    Z_s = W_s * W_s / (W_s - dWdpstar * (pstar - p_s));

}

#endif
