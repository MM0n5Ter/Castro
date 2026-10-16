#ifndef user_problem_setup_H
#define user_problem_setup_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_INLINE
void user_problem_setup ()
{

    // Force a different pressure choice for problem 5

    if (problem::problem == 5) {
        problem::pressure = 10.0;
    }
}
#endif
