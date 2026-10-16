#ifndef user_problem_setup_H
#define user_problem_setup_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_INLINE
void user_problem_setup ()
{

    problem::t0 = problem::tau0 / (problem::epsilon * C::c_light * problem::kapbar);
    problem::x0 = problem::x0 / problem::kapbar;
    problem::qn[0] = C::a_rad * std::pow(problem::Temp0, 4) *
        C::c_light * problem::kapbar * problem::Q * problem::p0;
    problem::qn[1] = C::a_rad * std::pow(problem::Temp0, 4) *
        C::c_light * problem::kapbar * problem::Q * problem::p1;
}
#endif

