#ifndef user_problem_setup_H
#define user_problem_setup_H

#include <prob_parameters.H>
#include <eos.H>
#include <model_parser.H>
#include <ambient.H>

AMREX_INLINE
void user_problem_setup ()
{

    // Read initial model

    read_model_file(problem::model_file);

    if (NumAux != 2) {
        amrex::Error("NumAux in network is not equal to 2");
    }

    if (NumSpec != 1) {
        amrex::Error("NumSpec in network is not equal to 1");
    }

    problem::iye = network_aux_index("Ye");
    problem::iinvmu = network_aux_index("invmu");

}
#endif
