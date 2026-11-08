
#ifndef PROBLEM_PARAMETERS_H
#define PROBLEM_PARAMETERS_H
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <network_properties.H>

  void init_prob_parameters();

  namespace problem {

  extern AMREX_GPU_MANAGED amrex::Real center[3];

  extern AMREX_GPU_MANAGED amrex::Real rho0;

  extern AMREX_GPU_MANAGED amrex::Real T0;

  extern AMREX_GPU_MANAGED amrex::Real dp_fact;

  extern AMREX_GPU_MANAGED amrex::Real L_pert;

  extern AMREX_GPU_MANAGED amrex::Real s0;

  extern AMREX_GPU_MANAGED amrex::Real p0;

  extern AMREX_GPU_MANAGED amrex::Real xn_zone[NumSpec];

  extern std::string spec1_name;

  extern std::string spec2_name;

  extern std::string spec3_name;

  extern std::string spec4_name;

  extern std::string spec5_name;

  extern std::string spec6_name;

  extern AMREX_GPU_MANAGED amrex::Real spec1_frac;

  extern AMREX_GPU_MANAGED amrex::Real spec2_frac;

  extern AMREX_GPU_MANAGED amrex::Real spec3_frac;

  extern AMREX_GPU_MANAGED amrex::Real spec4_frac;

  extern AMREX_GPU_MANAGED amrex::Real spec5_frac;

  extern AMREX_GPU_MANAGED amrex::Real spec6_frac;

  }


#endif
