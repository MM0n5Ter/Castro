
#ifndef PROBLEM_PARAMETERS_H
#define PROBLEM_PARAMETERS_H
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <network_properties.H>

  void init_prob_parameters();

  namespace problem {

  extern AMREX_GPU_MANAGED amrex::Real center[3];

  extern AMREX_GPU_MANAGED amrex::Real rhocv;

  extern AMREX_GPU_MANAGED amrex::Real T0;

  extern AMREX_GPU_MANAGED amrex::Real Eexp;

  extern AMREX_GPU_MANAGED amrex::Real rexp;

  }


#endif
