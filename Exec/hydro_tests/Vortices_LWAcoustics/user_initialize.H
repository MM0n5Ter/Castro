#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_INLINE
void user_initialize ()
{
    const Geometry& default_geom = DefaultGeometry();
    const Real* problo = default_geom.ProbLo();
    const Real* probhi = default_geom.ProbHi();

    // Define rho_0

    problem::rho_0 = std::pow(problem::ref_pressure, 1.0_rt/eos_rp::eos_gamma);

    // Define c_0

    problem::c_0 = std::sqrt(eos_rp::eos_gamma * problem::ref_pressure / problem::rho_0);

    // Define r_c, radius of each vortex

    problem::r_c = problem::ratio_c * problem::r_0;

    // Define circ

    problem::circ  = problem::r_circ * problem::r_0 * problem::c_0; //4e0_rt*M_PI*r_0*c_0*mach

    // Center of first vortex

    problem::x_c1 = 0.5_rt * probhi[0];
    problem::y_c1 = 0.5_rt * probhi[1] + problem::r_0;

    // Center of second vortex

    problem::x_c2 = 0.5_rt * probhi[0];
    problem::y_c2 = 0.5_rt * probhi[1] - problem::r_0;
}
#endif
