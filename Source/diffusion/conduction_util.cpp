#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>

#include <state_indices.H>

#include <hydrosim_params.H>
#include <eos.H>
#include <conductivity.H>

using namespace amrex;

void
fill_cell_conductivity(const Box& bx,
               Array4<Real const> const& field_arr,
               Array4<Real> const& cell_conductivity_arr) {

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    eos_t eos_state;
    eos_state.rho  = field_arr(i,j,k,URHO);
    Real rhoinv = 1.0_rt/eos_state.rho;

    eos_state.T = field_arr(i,j,k,UTEMP);   // needed as an initial guess
    eos_state.e = field_arr(i,j,k,UEINT) * rhoinv;
    for (int n = 0; n < NumSpec; n++) {
      eos_state.xn[n] = field_arr(i,j,k,UFS+n) * rhoinv;
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
      eos_state.aux[n] = field_arr(i,j,k,UFX+n) * rhoinv;
    }
#endif

    if (eos_state.e < 0.0_rt) {
      eos_state.T = hydrosim::small_temp;
      eos(eos_input_rt, eos_state);
    } else {
      eos(eos_input_re, eos_state);
    }


    if (eos_state.rho > hydrosim::conduction_cutoff_density) {
      conductivity(eos_state);

      if (eos_state.rho < hydrosim::conduction_cutoff_density_hi) {
        Real multiplier = (eos_state.rho - hydrosim::conduction_cutoff_density) /
          (hydrosim::conduction_cutoff_density_hi - hydrosim::conduction_cutoff_density);
        eos_state.conductivity = eos_state.conductivity * multiplier;
      }
    } else {
      eos_state.conductivity = 0.0_rt;
    }
    cell_conductivity_arr(i,j,k) = hydrosim::conduction_scale_factor * eos_state.conductivity;

  });
}


void
fill_conduction_coeff(const Box& bx,
                     Array4<Real const> const& field_arr,
                     Array4<Real> const& cell_conductivity_arr) {

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    eos_t eos_state;
    eos_state.rho  = field_arr(i,j,k,URHO);
    Real rhoinv = 1.0_rt/eos_state.rho;

    eos_state.T = field_arr(i,j,k,UTEMP);   // needed as an initial guess
    eos_state.e = field_arr(i,j,k,UEINT) * rhoinv;
    for (int n = 0; n < NumSpec; n++) {
      eos_state.xn[n] = field_arr(i,j,k,UFS+n) * rhoinv;
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
      eos_state.aux[n] = field_arr(i,j,k,UFX+n) * rhoinv;
    }
#endif

    if (eos_state.e < 0.0_rt) {
      eos_state.T = hydrosim::small_temp;
      eos(eos_input_rt, eos_state);
    } else {
      eos(eos_input_re, eos_state);
    }


    if (eos_state.rho > hydrosim::conduction_cutoff_density) {
      conductivity(eos_state);

      if (eos_state.rho < hydrosim::conduction_cutoff_density_hi) {
        Real multiplier = (eos_state.rho - hydrosim::conduction_cutoff_density) /
          (hydrosim::conduction_cutoff_density_hi - hydrosim::conduction_cutoff_density);
        eos_state.conductivity = eos_state.conductivity * multiplier;
      }
    } else {
      eos_state.conductivity = 0.0_rt;
    }
    cell_conductivity_arr(i,j,k) = hydrosim::conduction_scale_factor * eos_state.conductivity *
      rhoinv / eos_state.cv;

  });
}

