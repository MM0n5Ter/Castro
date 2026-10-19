#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>
#include <HydroSim_math.H>
#include <Rotation.H>
#include <HydroSim_helpers.H>

AMREX_INLINE
void user_initialize ()
{
    const Geometry& default_geom = DefaultGeometry();

    const Real* problo = default_geom.ProbLo();
    const Real* probhi = default_geom.ProbHi();

#ifdef ROTATION
    auto omega = get_angular_speed();
#else
    Real omega = 2.0_rt * M_PI;
#endif


    problem::density_maximum_radius = std::pow(C::Gconst * point_mass / (omega * omega), 1.0_rt/3.0_rt);


    problem::torus_width = 0.5_rt * (problem::outer_radius - problem::inner_radius);
    problem::torus_center = problem::inner_radius + problem::torus_width;

}

#endif
