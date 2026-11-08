
#ifndef PROBLEM_PARAMETERS_H
#define PROBLEM_PARAMETERS_H
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <network_properties.H>

  void init_prob_parameters();

  namespace problem {

  extern AMREX_GPU_MANAGED amrex::Real center[3];

  extern AMREX_GPU_MANAGED amrex::Real diff_coeff;

  extern AMREX_GPU_MANAGED amrex::Real T1;

  extern AMREX_GPU_MANAGED amrex::Real T2;

  extern AMREX_GPU_MANAGED amrex::Real rho0;

  extern AMREX_GPU_MANAGED amrex::Real t_0;

  extern AMREX_GPU_MANAGED bool gaussian_init;

  }


#endif
