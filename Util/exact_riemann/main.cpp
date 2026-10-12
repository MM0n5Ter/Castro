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

// a global struct to hold the params
#include <struct_params.H>

int main(int argc, char *argv[]) {

    amrex::Initialize(argc, argv);

    std::cout << "starting the exact Riemann solver..." << std::endl;

    // initialize the HydroSim runtime parameters

    amrex::ParmParse pp("hydrosim");
#include <hydrosim_queries.H>

    // initialize the external runtime parameters in C++

    init_prob_parameters();

    init_extern_parameters();

    // now initialize the C++ Microphysics
#ifdef REACTIONS
    network_init();
#endif

    eos_init(hydrosim::small_temp, hydrosim::small_dens);

    exact_riemann();

}
