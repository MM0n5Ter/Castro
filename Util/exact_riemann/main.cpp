#include <iostream>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <vector>

#include <AMReX_ParmParse.H>

#include <extern_parameters.H>
#include <prob_parameters.H>
#include <eos.H>
#include <network.H>

#include <exact_riemann.H>

#include <hydrosim_params.H>
#include <params_type.H>

#include <struct_params.H>

int main(int argc, char *argv[]) {

    amrex::Initialize(argc, argv);

    std::cout << "starting the exact Riemann solver..." << std::endl;


    amrex::ParmParse pp("hydrosim");
#include <hydrosim_queries.H>


    init_prob_parameters();

    init_extern_parameters();

#ifdef REACTIONS
    network_init();
#endif

    eos_init(hydrosim::small_temp, hydrosim::small_dens);

    exact_riemann();

}
