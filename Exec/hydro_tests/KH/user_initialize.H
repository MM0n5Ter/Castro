#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>

AMREX_INLINE
void user_initialize ()
{

    // Force a different pressure choice for problem 5

    if (problem::problem == 5) {
        problem::pressure = 10.0;
    }
}
#endif
