#ifndef planck_integrals_H
#define planck_integrals_H

#include <fundamental_constants.H>

namespace planck_integrals {
    const Real pi = 3.1415926535897932384626e0_rt;
    const Real planck_prefactor = C::a_rad * 15.e0_rt / (pi * pi * pi * pi);
    const Real planck_total_integral = (pi * pi * pi * pi) / 15.e0_rt;
    const Real tol = 1.0e-10_rt;

    const Real mid_switch_x = 2.061981e0_rt;
    const Real small_arg_cutoff = 1.e-5_rt;
    const Real large_arg_cutoff = 100.e0_rt;
}



AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real Li(int n, Real z)
{
    const int kmax = 18;

    // z here is really exp(-h nu / kT), and nu and T are positive,
    // so it satisfies the range of convergence.

    Real L = z;

    for (int k = 2; k <= kmax; ++k) {

        Real t = std::pow(z, k) / std::pow(k, n);

        L = L + t;

        // Terminate the series when the additional terms
        // become small enough that they approach roundoff error.

        if (t / L < planck_integrals::tol) {
            break;
        }

    }

    return L;
}



AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real planck_tail_integral(Real x)
{
    // Note that since we define z == exp(-x), then
    // x = -ln(z), so the signs below are consistent
    // with that (i.e. the terms
    // with odd powers of x have reversed sign).
    //
    // We are only evaluating this at a specific frequency,
    // and relying on the recursion relation to give the
    // correct absolute result.

    Real z = std::exp(-x);

    Real I = planck_integrals::planck_total_integral -
             (x * x * x * Li(1,z) + 3.e0_rt * x * x * Li(2,z) +
              6.e0_rt * x * Li(3,z) + 6.e0_rt * Li(4,z));

    return I;
}



AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real planck_series_sum(Real x)
{
    Real x2 = x * x;
    Real x3 = x * x * x;

    Real I = x3 / 3.0_rt - x2 * x2 / 8.0_rt;

    Real running_power = x3;

    const int SeriesTermCount = 8;
    const Real coefs[SeriesTermCount+1] = {1.e0_rt / 60.e0_rt,
                                  -1.e0_rt / 5040.e0_rt,
                                  1.e0_rt / 272160.e0_rt,
                                  -1.e0_rt / 13305600.e0_rt,
                                  1.e0_rt / 622702080.e0_rt,
                                  -691.e0_rt / 19615115520000.e0_rt,
                                  1.e0_rt / 1270312243200.e0_rt,
                                  -3617.e0_rt / 202741834014720000.e0_rt,
                                  43867.e0_rt / 107290978560589824000.e0_rt};

    for (int n = 0; n <= SeriesTermCount; ++n) {

        running_power = running_power * x2;
        Real t = coefs[n] * running_power;
        I = I + t;

        // Terminate the series expansion early if the terms
        // approach numerical roundoff error.

        if (std::abs(t / I) < planck_integrals::tol) {
            break;
        }

    }

    return I;
}



AMREX_GPU_HOST_DEVICE AMREX_INLINE
void PlanckAntiderivSlope (Real T, Real nu, Real& B, Real& dplanck_group_diff)
{
    // This routine evaluates the incomplete integral of the Planck function:
    // integral_value(x) = int_{0}^{x} b(x') dx'
    //
    // This incomplete integral can be used to recursively obtain the planck_integrals
    // function for all groups using:
    // b_g = integral_value(x_{g+1}) - integral_value(x_{g})
    //
    // The incomplete integral is defined in terms of polylogarithm
    // functions evaluated at exp(-x) where x = h * nu / k_B T.
    // We evaluate those polylogarithm functions directly when x is sufficiently
    // large, approximating them with a truncated infinite series; but, when x is
    // sufficiently small, we instead evaluate a series expansion of the original
    // integral. The threshold between these two approaches is set so that there
    // is no discontinuity of the Planck function at the threshold.

    // We set the threshold at the value 2.061981, above which we evaluate the
    // polylogarithm functions and below which we use the series expansion.
    //
    // We can also use this method to evaluate the derivative of the Planck function
    // with respect to temperature.

    Real x = C::hplanck * nu / (C::k_B * T);

    if (x > planck_integrals::large_arg_cutoff) {

        // If x is sufficiently large, this is effectively an integral over
        // all frequencies, and we know analytically that this is Stefan-Boltzmann
        // radiation, so we can save time (and avoid difficulties with evaluating
        // the exponential numerically for large argument) by evaluating it directly.

        B = C::a_rad * std::pow(T, 4);
        dplanck_group_diff = 4.0_rt * C::a_rad * std::pow(T, 3);

    }
    else if (x < planck_integrals::small_arg_cutoff) {

        // When x is very small, the integral is effectively over zero frequency
        // space and we can approximate it to zero.

        B = 0.0_rt;
        dplanck_group_diff = 0.0_rt;

    }
    else {

        Real integral_value;
        if (x > planck_integrals::mid_switch_x) {
            integral_value = planck_tail_integral(x);
        }
        else {
            integral_value = planck_series_sum(x);
        }

        B = planck_integrals::planck_prefactor * std::pow(T, 4) * integral_value;

        // Note that the second term in square brackets is the one
        // we care about for this group, and the third term is from
        // the recursive relationship between groups. Since we are
        // integrating upwards from zero frequency, the sign is reversed.

        Real part = std::pow(x, 4) / (std::exp(x) - 1.0_rt);
        dplanck_group_diff = planck_integrals::planck_prefactor * std::pow(T, 3) * (4.0_rt * integral_value - part);

    }
}


AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real PlanckAntiderivative(Real T, Real nu)
{
    Real B;

    // See comments above; this function evaluates B but does not
    // evaluate dB/dT.

    Real x = C::hplanck * nu / (C::k_B * T);

    if (x > planck_integrals::large_arg_cutoff) {

        B = C::a_rad * std::pow(T, 4);

    }
    else if (x < planck_integrals::small_arg_cutoff) {

        B = 0.e0_rt;

    }
    else {

        Real integral_value;
        if (x > planck_integrals::mid_switch_x) {
            integral_value = planck_tail_integral(x);
        }
        else {
            integral_value = planck_series_sum(x);
        }

        B = planck_integrals::planck_prefactor * std::pow(T, 4) * integral_value;

    }

    return B;
}



AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real PlanckBandIntegral(Real T, Real nu0, Real nu1)
{
    Real B = PlanckAntiderivative(T, nu1) - PlanckAntiderivative(T, nu0);

    return B;
}

#endif
