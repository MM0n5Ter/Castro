#ifndef SOLVER_CONSTANTS_H
#define SOLVER_CONSTANTS_H

#include <AMReX_REAL.H>

using namespace amrex::literals;

namespace solver_constants {
    constexpr amrex::Real small_rel_pressure = 1.e-10_rt;
    constexpr amrex::Real small = 1.e-8_rt;
    constexpr amrex::Real small_velocity_ratio = 1.e-12_rt;
    constexpr amrex::Real integral_tolerance = 1.e-8_rt;
    constexpr amrex::Real velocity_tolerance = 1.e-6_rt;
    constexpr amrex::Real pressure_tolerance = 1.e-8_rt;
    constexpr int STAR_HISTORY_DEPTH=40;
    constexpr int STAR_BISECT_SCALE = 5;
}

#endif
