#ifndef EXTERN_TYPE_H
#define EXTERN_TYPE_H

using namespace amrex::literals;

struct conductivity_param_t {
    amrex::Real const_conductivity{1.0e0_rt};
};

struct eos_param_t {
    bool eos_assume_neutral{true};
    amrex::Real eos_gamma{5.e0/3.e0_rt};
};

struct network_param_t {
    bool use_tables{false};
    bool use_c12ag_deboer17{false};
    amrex::Real small_x{1.e-30_rt};
};

struct extern_t {
    conductivity_param_t conductivity;
    eos_param_t eos;
    network_param_t network;
};

#endif
