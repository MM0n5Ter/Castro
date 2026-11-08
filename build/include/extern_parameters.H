
#ifndef EXTERNAL_PARAMETERS_H
#define EXTERNAL_PARAMETERS_H
#include <AMReX_BLFort.H>
#include <AMReX_REAL.H>

#include <extern_type.H>

  extern_t init_extern_parameters();

  namespace eos_rp {
    extern AMREX_GPU_MANAGED amrex::Real eos_gamma;
    extern AMREX_GPU_MANAGED bool eos_assume_neutral;
  }
  namespace network_rp {
    extern AMREX_GPU_MANAGED amrex::Real small_x;
    extern AMREX_GPU_MANAGED bool use_tables;
    extern AMREX_GPU_MANAGED bool use_c12ag_deboer17;
  }
  namespace opacity_rp {
    extern AMREX_GPU_MANAGED amrex::Real const_kappa_p;
    extern AMREX_GPU_MANAGED amrex::Real kappa_p_exp_m;
    extern AMREX_GPU_MANAGED amrex::Real kappa_p_exp_n;
    extern AMREX_GPU_MANAGED amrex::Real kappa_p_exp_p;
    extern AMREX_GPU_MANAGED amrex::Real const_kappa_r;
    extern AMREX_GPU_MANAGED amrex::Real kappa_r_exp_m;
    extern AMREX_GPU_MANAGED amrex::Real kappa_r_exp_n;
    extern AMREX_GPU_MANAGED amrex::Real kappa_r_exp_p;
    extern AMREX_GPU_MANAGED amrex::Real const_scatter;
    extern AMREX_GPU_MANAGED amrex::Real scatter_exp_m;
    extern AMREX_GPU_MANAGED amrex::Real scatter_exp_n;
    extern AMREX_GPU_MANAGED amrex::Real scatter_exp_p;
    extern AMREX_GPU_MANAGED amrex::Real kappa_floor;
    extern AMREX_GPU_MANAGED amrex::Real rad_temp_floor;
  }

#endif
