#ifndef user_initialize_H
#define user_initialize_H

#include <prob_parameters.H>
#include <eos.H>
#include <network.H>

AMREX_INLINE
void user_initialize ()
{

    int ifuel1 = network_spec_index(problem::fuel1_name);
    int iash1 = network_spec_index(problem::ash1_name);

    int ifuel2 = network_spec_index(problem::fuel2_name);
    int iash2 = network_spec_index(problem::ash2_name);

    int ifuel3 = network_spec_index(problem::fuel3_name);
    int iash3 = network_spec_index(problem::ash3_name);

    int ifuel4 = network_spec_index(problem::fuel4_name);
    int iash4 = network_spec_index(problem::ash4_name);

    if (iash1 < 0 && iash2 < 0 && iash3 < 0 && iash4 < 0) {
        amrex::Error("no valid ash state defined");
    }


    for (int n = 0; n < NumSpec; n++) {
        problem::xn_fuel[n] = problem::smallx_init;
        problem::xn_ash[n] = problem::smallx_init;
    }

    if (ifuel1 >= 0) {
        problem::xn_fuel[ifuel1] = problem::X_fuel1;
    }

    if (ifuel2 >= 0) {
        problem::xn_fuel[ifuel2] = problem::X_fuel2;
    }

    if (ifuel3 >= 0) {
        problem::xn_fuel[ifuel3] = problem::X_fuel3;
    }

    if (ifuel4 >= 0) {
        problem::xn_fuel[ifuel4] = problem::X_fuel4;
    }

    if (iash1 >= 0) {
        problem::xn_ash[iash1] = problem::X_ash1;
    }

    if (iash2 >= 0) {
        problem::xn_ash[iash2] = problem::X_ash2;
    }

    if (iash3 >= 0) {
        problem::xn_ash[iash3] = problem::X_ash3;
    }

    if (iash4 >= 0) {
        problem::xn_ash[iash4] = problem::X_ash4;
    }


    Real sum_Xf = 0.0_rt;
    Real sum_Xa = 0.0_rt;
    for (int n = 0; n < NumSpec; n++) {
        sum_Xf += problem::xn_fuel[n];
        sum_Xa += problem::xn_ash[n];
    }

    if (std::abs(sum_Xf - 1.0) > 0.01) {
        std::cout << "sum_Xf = " << sum_Xf << std::endl;
        amrex::Error("fuel mass fractions don't sum to 1");
    }

    if (std::abs(sum_Xa - 1.0) > 0.01) {
        std::cout << "sum_Xa = " << sum_Xa << std::endl;
        amrex::Error("ash mass fractions don't sum to 1");
    }

    for (int n = 0; n < NumSpec; n++) {
        problem::xn_fuel[n] /= sum_Xf;
        problem::xn_ash[n] /= sum_Xa;
    }

    eos_t eos_state;
    eos_state.rho = problem::rho_fuel;
    eos_state.T = problem::T_fuel;
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = problem::xn_fuel[n];
    }

    eos(eos_input_rt, eos_state);

    problem::e_fuel = eos_state.e;
    problem::p_fuel = eos_state.p;


    eos_state.rho = problem::rho_fuel;
    eos_state.p = problem::p_fuel;
    eos_state.T = problem::T_ash;
    for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = problem::xn_ash[n];
    }

    eos(eos_input_tp, eos_state);

    problem::rho_ash = eos_state.rho;
    problem::e_ash = eos_state.e;


    problem::mass_flux = problem::rho_fuel * problem::v_inflow;


    if (hydrosim::sum_step_period > 0 && amrex::ParallelDescriptor::IOProcessor()) {

        HydroSim::user_run_logs.resize(1);

        HydroSim::user_run_logs[0] = std::make_unique<std::fstream>();
        HydroSim::user_run_logs[0]->open("toy_flame.log", std::ios::out | std::ios::app);
        if (!HydroSim::user_run_logs[0]->good()) {
            amrex::FileOpenFailed("toy_flame.log");
        }

    }

}
#endif
