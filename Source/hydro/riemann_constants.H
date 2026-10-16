#ifndef RIEMANN_CONSTANTS_H
#define RIEMANN_CONSTANTS_H

#include <AMReX_REAL.H>

using namespace amrex::literals;

struct riemann_constants {
    static constexpr amrex::Real small_p_frac = 1.e-10_rt;
    static constexpr amrex::Real small = 1.e-8_rt;
    static constexpr amrex::Real smallu = 1.e-12_rt;
    static constexpr amrex::Real riemann_integral_tol = 1.e-8_rt;
    static constexpr amrex::Real riemann_u_tol = 1.e-6_rt;
    static constexpr amrex::Real riemann_p_tol = 1.e-8_rt;
    static constexpr int HISTORY_SIZE=40;
    static constexpr int PSTAR_BISECT_FACTOR = 5;
};

#endif
