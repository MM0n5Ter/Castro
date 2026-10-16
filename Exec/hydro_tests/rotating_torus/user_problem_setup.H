#ifndef user_problem_setup_H
#define user_problem_setup_H

#include <prob_parameters.H>
#include <eos.H>
#include <HydroSim_math.H>
#include <Rotation.H>
#include <HydroSim_util.H>

AMREX_INLINE
void user_problem_setup ()
{
    const Geometry& dgeom = DefaultGeometry();

    const Real* problo = dgeom.ProbLo();
    const Real* probhi = dgeom.ProbHi();

#ifdef ROTATION
    auto omega = rotation::get_omega();
#else
    // Provide a dummy value so that we can compile without rotation.
    Real omega = 2.0_rt * M_PI;
#endif

    // Figure out R_0, the maximum pressure radius.

    problem::density_maximum_radius = std::pow(C::Gconst * point_mass / (omega * omega), 1.0_rt/3.0_rt);

    // Maximum and minimum vertical extent of the torus is the same as the radial extent

    problem::torus_width = 0.5_rt * (problem::outer_radius - problem::inner_radius);
    problem::torus_center = problem::inner_radius + problem::torus_width;

}

#endif
