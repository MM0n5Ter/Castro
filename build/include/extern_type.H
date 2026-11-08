#ifndef EXTERN_TYPE_H
#define EXTERN_TYPE_H

using namespace amrex::literals;

struct eos_param_t {
    bool use_eos_coulomb{true};
    bool eos_input_is_constant{true};
    amrex::Real eos_ttol{1.0e-8_rt};
    amrex::Real eos_dtol{1.0e-8_rt};
    amrex::Real prad_limiter_rho_c{-1.0e0_rt};
    amrex::Real prad_limiter_delta_rho{-1.0e0_rt};
};

struct integrator_param_t {
    bool call_eos_in_rhs{true};
    bool integrate_energy{true};
    bool burner_verbose{false};
    bool renormalize_abundances{false};
    bool use_jacobian_caching{true};
    bool use_burn_retry{false};
    bool retry_swap_jacobian{true};
    bool do_species_clip{false};
    bool do_corrector_validation{true};
    bool use_number_densities{false};
    bool scale_system{false};
    bool nse_include_enu_weak{true};
    bool linalg_do_pivoting{true};
    bool correct_jacobian_for_const_e{true};
    int ode_max_steps{150000};
    int nonaka_i{0};
    int nonaka_j{0};
    int nonaka_k{0};
    int nonaka_level{0};
    int subtract_internal_energy{0};
    int jacobian{1};
    amrex::Real X_reject_buffer{1.0_rt};
    amrex::Real SMALL_X_SAFE{1.0e-30_rt};
    amrex::Real MAX_TEMP{1.0e11_rt};
    amrex::Real react_boost{-1.e0_rt};
    amrex::Real ode_max_dt{1.e30_rt};
    amrex::Real retry_rtol_spec{-1_rt};
    amrex::Real retry_rtol_enuc{-1_rt};
    amrex::Real retry_atol_spec{-1_rt};
    amrex::Real retry_atol_enuc{-1_rt};
    amrex::Real nse_deriv_dt_factor{0.05_rt};
    amrex::Real species_failure_tolerance{1.e-2_rt};
    amrex::Real atol_spec{1.0e-12_rt};
    amrex::Real rtol_spec{1.0e-12_rt};
    amrex::Real atol_enuc{1.0e-8_rt};
    amrex::Real rtol_enuc{1.0e-6_rt};
    std::string nonaka_file{"nonaka_plot.dat"};
};

struct network_param_t {
    bool use_tables{false};
    bool use_c12ag_deboer17{false};
    amrex::Real small_x{1.e-30_rt};
};

struct screening_param_t {
    bool enable_chabrier1998_quantum_corr{false};
    bool enable_debye_huckel_skip{false};
    amrex::Real debye_huckel_skip_threshold{1.0e-2_rt};
};

struct extern_t {
    eos_param_t eos;
    integrator_param_t integrator;
    network_param_t network;
    screening_param_t screening;
};

#endif
