#ifndef HYDROSIM_FLUXLIMITER_H
#define HYDROSIM_FLUXLIMITER_H

#include <AMReX_REAL.H>
#include <cmath>

#include <HydroSim_helpers.H>

#include <radiation_params.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE inline
amrex::Real Eddington_factor(Real lambda)
{

    Real f;

    if (radiation::closure == 0) {
        f = lambda;
    }
    else if (radiation::closure == 1) {
        f = 1.0_rt / 3.0_rt;
    }
    else if (radiation::closure == 2) {
        f = 1.e0_rt - 2.e0_rt * lambda;
    }
    else if (radiation::closure == 3) {
        if (radiation::limiter == 0) {
            f = 1.0_rt / 3.0_rt;
        }
        else if (radiation::limiter < 10) {
            f = 0.5e0_rt * amrex::max(0.0_rt, (1.e0_rt - 3.e0_rt*  lambda)) +
                std::sqrt(amrex::max(0.0_rt, (1.e0_rt - 3.e0_rt * lambda)) *
                          (1.e0_rt + 5.e0_rt * lambda));
            f = lambda + f * f;
        }
        else if (radiation::limiter < 20) {
            f = 1.0e0_rt - 5.e0_rt * lambda + 9.e0_rt * lambda * lambda;
        }
        else if (radiation::limiter < 30) {
            f = 1.0e0_rt + lambda - 9.e0_rt * lambda * lambda;
        }
        else if (radiation::limiter < 40) {
            if (lambda > 2.0_rt / 9.0_rt) {
                f = 1.0_rt / 3.0_rt;
            }
            else {
                f = 1.e0_rt + 3.e0_rt * lambda - 2.e0_rt * std::sqrt(2.e0_rt * lambda);
            }
        }
#ifndef AMREX_USE_GPU
        else {
            amrex::Error("Unknown limiter");
        }
#endif
    }
    else if (radiation::closure == 4) {
        if (radiation::limiter == 0) {
            f = 1.0_rt / 3.0_rt;
        }
        else if (radiation::limiter < 10) {
            f = amrex::max(0.0_rt, (1.e0_rt - 3.e0_rt * lambda)) +
                 std::sqrt(amrex::max(0.0_rt, (1.e0_rt - 3.e0_rt * lambda)) *
                           (1.e0_rt + 5.e0_rt * lambda));
            f = (1.0_rt / 3.0_rt) + (1.0_rt / 6.0_rt) * f * f;
        }
        else if (radiation::limiter < 20) {
            f = (1.0_rt / 3.0_rt) + (2.0_rt / 3.0_rt) * (1.0e0_rt - 6.e0_rt * lambda + 9.e0_rt * lambda * lambda);
        }
        else if (radiation::limiter < 30) {
            f = (1.0_rt / 3.0_rt) + (2.0_rt / 3.0_rt) * (1.0e0_rt - 9.e0_rt * lambda * lambda);
        }
        else if (radiation::limiter < 40) {
            if (lambda > 2.0_rt / 9.0_rt) {
                f = (5.0_rt / 9.0_rt) - (2.0_rt / 3.0_rt) * lambda;
            }
            else {
                f = (1.0_rt / 3.0_rt) + (2.0_rt / 3.0_rt) * (1.e0_rt + 2.e0_rt * lambda - 2.e0_rt * std::sqrt(2.e0_rt * lambda));
            }
        }
#ifndef AMREX_USE_GPU
        else {
            amrex::Error("Unknown limiter");
        }
#endif
    }

    return f;

}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real LimitedDiffusionAlpha(Real lam)
{
    Real one_minus_term = amrex::max(0.0_rt, 1.0_rt - 3.0_rt * lam);

    Real R;

    if (radiation::limiter == 0) {
        R = 0.0_rt;
    }
    else if (radiation::limiter < 10) {
        R = (one_minus_term + std::sqrt(one_minus_term * (1.0_rt + 5.0_rt * lam))) / (2.0_rt * lam + 1.e-50_rt);
    }
    else if (radiation::limiter < 20) {
        R = one_minus_term / (lam + 1.e-50_rt);
    }
    else if (radiation::limiter < 30) {
        R = std::sqrt(one_minus_term * (1.0_rt + 3.0_rt * lam)) / (lam + 1.e-50_rt);
    }
    else if (radiation::limiter < 40) {
        if (lam > 2.0_rt / 9.0_rt) {
            R = std::sqrt(one_minus_term / 3.0_rt) / (lam + 1.e-50_rt);
        }
        else {
            R = 1.0_rt / (lam + 1.e-50_rt) - std::sqrt(2.0_rt / (lam + 1.e-50_rt));
        }
    }
    else {
#ifndef AMREX_USE_GPU
        amrex::Error("Unknown limiter");
#endif
    }

    Real alpha;

    if (R < 1.e-6_rt) {
        alpha = 0.25_rt;
    }
    else if (R > 300.0_rt) {
        alpha = 0.5_rt;
    }
    else {
        Real cr = std::cosh(R);
        alpha = cr * std::log(cr) / (2.0_rt * R * std::sinh(R));
    }

    return alpha;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real LimitedDiffusionLambda (Real r)
{
    Real lambda;

    if (radiation::limiter == 0) {
        lambda = 1.e0_rt / 3.e0_rt;
    }
    else if (radiation::limiter < 10) {
        lambda = (2.e0_rt + r) / (6.e0_rt + r * (3.e0_rt + r));
    }
    else if (radiation::limiter < 20) {
        lambda = 1.e0_rt / (3.e0_rt + r);
    }
    else if (radiation::limiter < 30) {
        lambda = 1.e0_rt / std::sqrt(9.e0_rt + r * r);
    }
    else if (radiation::limiter < 40) {
        if (r < 1.5e0_rt) {
            lambda = 2.e0_rt / (3.e0_rt + std::sqrt(9.e0_rt + 12.e0_rt * r * r));
        } else {
            lambda = 1.e0_rt / (1.e0_rt + r + std::sqrt(1.e0_rt + 2.e0_rt * r));
        }
    }
    else {
#ifndef AMREX_USE_GPU
        amrex::Error("Unknown limiter type");
#endif
    }

    return lambda;
}

#endif
