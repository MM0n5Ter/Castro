#ifndef EXTERN_TYPE_H
#define EXTERN_TYPE_H

using namespace amrex::literals;

struct eos_param_t {
    bool eos_assume_neutral{true};
    amrex::Real eos_gamma{5.e0/3.e0_rt};
};

struct network_param_t {
    bool use_tables{false};
    bool use_c12ag_deboer17{false};
    amrex::Real small_x{1.e-30_rt};
};

struct opacity_param_t {
    amrex::Real const_kappa_p{-1.0e0_rt};
    amrex::Real kappa_p_exp_m{0.0e0_rt};
    amrex::Real kappa_p_exp_n{0.0e0_rt};
    amrex::Real kappa_p_exp_p{0.0e0_rt};
    amrex::Real const_kappa_r{-1.0e0_rt};
    amrex::Real kappa_r_exp_m{0.0e0_rt};
    amrex::Real kappa_r_exp_n{0.0e0_rt};
    amrex::Real kappa_r_exp_p{0.0e0_rt};
    amrex::Real const_scatter{0.0e0_rt};
    amrex::Real scatter_exp_m{0.0e0_rt};
    amrex::Real scatter_exp_n{0.0e0_rt};
    amrex::Real scatter_exp_p{0.0e0_rt};
    amrex::Real kappa_floor{0.0e0_rt};
    amrex::Real rad_temp_floor{1.e-10_rt};
};

struct extern_t {
    eos_param_t eos;
    network_param_t network;
    opacity_param_t opacity;
};

#endif
