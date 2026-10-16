
#ifndef PROBLEM_PARAMETERS_H
#define PROBLEM_PARAMETERS_H
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <network_properties.H>

  void init_prob_parameters();

  namespace problem {

  extern AMREX_GPU_MANAGED amrex::Real center[3];

  extern AMREX_GPU_MANAGED amrex::Real p_ambient;

  extern AMREX_GPU_MANAGED amrex::Real dens_ambient;

  extern AMREX_GPU_MANAGED amrex::Real exp_energy;

  extern AMREX_GPU_MANAGED amrex::Real temp_ambient;

  extern AMREX_GPU_MANAGED amrex::Real e_ambient;

  extern AMREX_GPU_MANAGED amrex::Real r_init;

  extern AMREX_GPU_MANAGED int nsub;

  extern AMREX_GPU_MANAGED amrex::Real e_exp;

  }


#endif
