#include <HydroSim.H>
#include <HydroSim_helpers.H>

#include <mhd_helpers.H>

using namespace amrex;

void
HydroSim::PrimitiveToConserved(const Box& bx,
                   Array4<Real const> const& prim_arr,
                   Array4<Real> const& state_view) {

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    state_view(i,j,k,URHO) = prim_arr(i,j,k,QRHO);
    state_view(i,j,k,UMX) = prim_arr(i,j,k,QRHO)*prim_arr(i,j,k,QU);
    state_view(i,j,k,UMY) = prim_arr(i,j,k,QRHO)*prim_arr(i,j,k,QV);
    state_view(i,j,k,UMZ) = prim_arr(i,j,k,QRHO)*prim_arr(i,j,k,QW);

    eos_t eos_state;
    eos_state.rho = prim_arr(i,j,k,QRHO);
    eos_state.p = prim_arr(i,j,k,QPRES);
    eos_state.T = 100.0_rt;
    for (int n = 0; n < NumSpec; n++) {
      eos_state.xn[n] = prim_arr(i,j,k,QFS+n);
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
      eos_state.aux[n] = prim_arr(i,j,k,QFX+n);
    }
#endif

    eos(eos_input_rp, eos_state);

    state_view(i,j,k,UEDEN) = eos_state.rho * eos_state.e +
      + 0.5_rt * prim_arr(i,j,k,QRHO) * (prim_arr(i,j,k,QU) * prim_arr(i,j,k,QU) +
                                      prim_arr(i,j,k,QV) * prim_arr(i,j,k,QV) +
                                      prim_arr(i,j,k,QW) * prim_arr(i,j,k,QW)) +
      + 0.5_rt * (prim_arr(i,j,k,QBFIELDX) * prim_arr(i,j,k,QBFIELDX) +
                  prim_arr(i,j,k,QBFIELDY) * prim_arr(i,j,k,QBFIELDY) +
                  prim_arr(i,j,k,QBFIELDZ) * prim_arr(i,j,k,QBFIELDZ));

    state_view(i,j,k,UEINT) = eos_state.rho * eos_state.e;
    state_view(i,j,k,UTEMP) = eos_state.T;

    state_view(i,j,k,UBX) = prim_arr(i,j,k,QBFIELDX);
    state_view(i,j,k,UBY) = prim_arr(i,j,k,QBFIELDY);
    state_view(i,j,k,UBZ) = prim_arr(i,j,k,QBFIELDZ);

    for (int n = 0; n < NumSpec; n++) {
      state_view(i,j,k,UFS+n) = prim_arr(i,j,k,QRHO) * prim_arr(i,j,k,QFS+n);
    }

  });
}

void
HydroSim::primitive_half_update(const Box& bx,
                  Array4<Real> const& q2D,
                  Array4<Real const> const& prim_arr,
                  Array4<Real const> const& flxx,
                  Array4<Real const> const& flxy,
                  Array4<Real const> const& flxz,
                  const Real dt) {

  auto dx = geom.CellSizeArray();

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real div_terms[NUM_STATE+3];
    Real prim_flux_div[NQ];
    Real zone_prim[NQ];

    for (int n = 0; n < NUM_STATE+3; n++) {
      div_terms[n] = (flxx(i+1,j,k,n) - flxx(i,j,k,n)) / dx[0];
#if AMREX_SPACEDIM >= 2
      div_terms[n] += (flxy(i,j+1,k,n) - flxy(i,j,k,n)) / dx[1];
#endif
#if AMREX_SPACEDIM == 3
      div_terms[n] += (flxz(i,j,k+1,n) - flxz(i,j,k,n)) / dx[2];
#endif
    }

    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j,k,n);
    }

    to_prim_flux(prim_flux_div, div_terms, zone_prim);

    for (int n = 0; n < NQ; n++) {
      q2D(i,j,k,n) = prim_arr(i,j,k,n) - 0.5_rt * dt * prim_flux_div[n];
    }
  });
}
