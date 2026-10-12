
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

#endif
