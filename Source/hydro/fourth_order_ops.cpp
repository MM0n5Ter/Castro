#include <HydroSim.H>
#include <fourth_order_stencils.H>
#ifdef DIFFUSION
#include <conductivity.H>
#endif

using namespace amrex;

void
HydroSim::build_center_state(const Box& bx,
                         Array4<Real const> const& U,
                         Array4<Real> const& centered_state,
                         GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {


  [[maybe_unused]] const auto U_lower = lbound(U);
  [[maybe_unused]] const auto U_upper = ubound(U);

  [[maybe_unused]] const auto *lo = bx.loVect();
  [[maybe_unused]] const auto *hi = bx.hiVect();

  AMREX_ASSERT(U_lower.x <= lo[0]-1 && U_upper.x >= hi[0]+1);
#if AMREX_SPACEDIM >= 2
  AMREX_ASSERT(U_lower.y <= lo[1]-1 && U_upper.y >= hi[1]+1);
#endif
#if AMREX_SPACEDIM == 3
  AMREX_ASSERT(U_lower.z <= lo[2]-1 && U_upper.z >= hi[2]+1);
#endif

  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  GpuArray<bool, AMREX_SPACEDIM> lo_wrap;
  GpuArray<bool, AMREX_SPACEDIM> hi_wrap;
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    lo_wrap[idir] = lo_bc[idir] == amrex::PhysBCType::interior;
    hi_wrap[idir] = hi_bc[idir] == amrex::PhysBCType::interior;
  }

  amrex::ParallelFor(bx, U.nComp(),
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {
    Real lap = laplacian_at(i, j, k, n, U,
                                 lo_wrap, hi_wrap, domlo, domhi);

    centered_state(i,j,k,n) = U(i,j,k,n) - (1.0_rt/24.0_rt) * lap;

  });
}

void
HydroSim::build_center_state_in_place(const Box& bx,
                                  Array4<Real> const& U,
                                  Array4<Real> const& tmp,
                                  GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {



  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  GpuArray<bool, AMREX_SPACEDIM> lo_wrap;
  GpuArray<bool, AMREX_SPACEDIM> hi_wrap;
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    lo_wrap[idir] = lo_bc[idir] == amrex::PhysBCType::interior;
    hi_wrap[idir] = hi_bc[idir] == amrex::PhysBCType::interior;
  }

  for (int n = 0; n < U.nComp(); n++) {

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      tmp(i,j,k) = laplacian_at(i, j, k, n, U,
                                     lo_wrap, hi_wrap, domlo, domhi);
    });

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      U(i,j,k,n) = U(i,j,k,n) - (1.0_rt/24.0_rt) * tmp(i,j,k);
    });
  }
}


void
HydroSim::laplacian_of_state(const Box& bx,
                         Array4<Real const> const& U,
                         Array4<Real> const& lap, const int ncomp,
                         GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {


  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  GpuArray<bool, AMREX_SPACEDIM> lo_wrap;
  GpuArray<bool, AMREX_SPACEDIM> hi_wrap;
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    lo_wrap[idir] = lo_bc[idir] == amrex::PhysBCType::interior;
    hi_wrap[idir] = hi_bc[idir] == amrex::PhysBCType::interior;
  }

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    lap(i,j,k) = (1.0_rt/24.0_rt) *
      laplacian_at(i, j, k, ncomp, U,
                        lo_wrap, hi_wrap, domlo, domhi);
  });

}


void
HydroSim::build_fourth_average(const Box& bx,
                            Array4<Real> const& prim_arr,
                            Array4<Real const> const& q_mean_arr,
                            GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {


  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  GpuArray<bool, AMREX_SPACEDIM> lo_wrap;
  GpuArray<bool, AMREX_SPACEDIM> hi_wrap;
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    lo_wrap[idir] = lo_bc[idir] == amrex::PhysBCType::interior;
    hi_wrap[idir] = hi_bc[idir] == amrex::PhysBCType::interior;
  }

  amrex::ParallelFor(bx, prim_arr.nComp(),
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {
    Real lap = laplacian_at(i, j, k, n, q_mean_arr,
                                 lo_wrap, hi_wrap, domlo, domhi);

    prim_arr(i,j,k,n) += (1.0_rt/24.0_rt) * lap;
  });
}


void
HydroSim::build_fourth_in_place(const Box& bx,
                             Array4<Real> const& prim_arr,
                             Array4<Real> const& tmp,
                             GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {


  for (int n = 0; n < prim_arr.nComp(); n++) {
    build_fourth_in_place_n(bx, prim_arr, n, tmp, domlo, domhi);
  }
}


void
HydroSim::build_fourth_in_place_n(const Box& bx,
                               Array4<Real> const& prim_arr, const int ncomp,
                               Array4<Real> const& tmp,
                               GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi) {


  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  GpuArray<bool, AMREX_SPACEDIM> lo_wrap;
  GpuArray<bool, AMREX_SPACEDIM> hi_wrap;
  for (int idir = 0; idir < AMREX_SPACEDIM; idir++) {
    lo_wrap[idir] = lo_bc[idir] == amrex::PhysBCType::interior;
    hi_wrap[idir] = hi_bc[idir] == amrex::PhysBCType::interior;
  }

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    tmp(i,j,k) = laplacian_at(i, j, k, ncomp, prim_arr,
                                   lo_wrap, hi_wrap, domlo, domhi);
  });

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    prim_arr(i,j,k,ncomp) += (1.0_rt/24.0_rt) * tmp(i,j,k);
  });

}

void
HydroSim::fourth_viscosity(const Box& bx,
                     Array4<Real const> const& prim_arr,
                     Array4<Real const> const& qthermo_arr,
                     Array4<Real> const& viscosity_field,
                     const int idir) {


  constexpr Real beta = 0.3_rt;

  const auto dx = geom.CellSizeArray();

  Real dxinv = 1.0_rt / dx[0];

#if AMREX_SPACEDIM >= 2
  Real dyinv = 1.0_rt / dx[1];
#else
  Real dyinv = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
  Real dzinv = 1.0_rt / dx[2];
#else
  Real dzinv = 0.0_rt;
#endif

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real cmin;

    if (idir == 0) {


      viscosity_field(i,j,k) = (prim_arr(i,j,k,QU) - prim_arr(i-1,j,k,QU)) * dxinv;
#if AMREX_SPACEDIM >= 2
      viscosity_field(i,j,k) = viscosity_field(i,j,k) + 0.25_rt*(prim_arr(i,j+1,k,QV) - prim_arr(i,j-1,k,QV) +
                                           prim_arr(i-1,j+1,k,QV) - prim_arr(i-1,j-1,k,QV)) * dyinv;
#endif
#if AMREX_SPACEDIM >= 3
      viscosity_field(i,j,k) = viscosity_field(i,j,k) + 0.25_rt*(prim_arr(i,j,k+1,QW) - prim_arr(i,j,k-1,QW) +
                                           prim_arr(i-1,j,k+1,QW) - prim_arr(i-1,j,k-1,QW)) * dzinv;
#endif

      cmin = amrex::min(qthermo_arr(i,j,k,QC), qthermo_arr(i-1,j,k,QC));

    } else if (idir == 1) {


      viscosity_field(i,j,k) = (prim_arr(i,j,k,QV) - prim_arr(i,j-1,k,QV)) * dyinv;

      viscosity_field(i,j,k) = viscosity_field(i,j,k) + 0.25_rt*(prim_arr(i+1,j,k,QU) - prim_arr(i-1,j,k,QU) +
                                           prim_arr(i+1,j-1,k,QU) - prim_arr(i-1,j-1,k,QU)) * dxinv;

#if AMREX_SPACEDIM >= 3
      viscosity_field(i,j,k) = viscosity_field(i,j,k) + 0.25_rt*(prim_arr(i,j,k+1,QW) - prim_arr(i,j,k-1,QW) +
                                           prim_arr(i,j-1,k+1,QW) - prim_arr(i,j-1,k-1,QW)) * dzinv;
#endif

      cmin = amrex::min(qthermo_arr(i,j,k,QC), qthermo_arr(i,j-1,k,QC));

    } else {


      viscosity_field(i,j,k) = (prim_arr(i,j,k,QW) - prim_arr(i,j,k-1,QW)) * dzinv;

      viscosity_field(i,j,k) = viscosity_field(i,j,k) + 0.25_rt*(prim_arr(i,j+1,k,QV) - prim_arr(i,j-1,k,QV) +
                                           prim_arr(i,j+1,k-1,QV) - prim_arr(i,j-1,k-1,QV)) * dyinv;

      viscosity_field(i,j,k) = viscosity_field(i,j,k) + 0.25_rt*(prim_arr(i+1,j,k,QU) - prim_arr(i-1,j,k,QU) +
                                           prim_arr(i+1,j,k-1,QU) - prim_arr(i-1,j,k-1,QU)) * dxinv;

      cmin = amrex::min(qthermo_arr(i,j,k,QC), qthermo_arr(i,j,k-1,QC));

    }

    Real coeff = amrex::min(1.0_rt, (dx[idir] * viscosity_field(i,j,k)) * (dx[idir] * viscosity_field(i,j,k)) /
                                    (beta * (cmin * cmin)));

    if (viscosity_field(i,j,k) < 0.0_rt) {
      viscosity_field(i,j,k) = dx[idir] * viscosity_field(i,j,k) * coeff;
    } else {
      viscosity_field(i,j,k) = 0.0_rt;
    }

  });

}

#ifdef DIFFUSION
void
HydroSim::fourth_accumulate_diffusion(const Box& bx,
                                  Array4<Real const> const& prim_arr,
                                  const int temperature_index,
                                  Array4<Real const> const& interface_state,
                                  Array4<Real> const& F,
                                  const int idir, const bool holds_cell_average) {


  const auto dx = geom.CellSizeArray();

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    eos_t eos_state;

    eos_state.rho = interface_state(i,j,k,QRHO);
    eos_state.T = prim_arr(i,j,k,temperature_index);
    eos_state.e = interface_state(i,j,k,QEINTDENS) / interface_state(i,j,k,QRHO);
    for (int n = 0; n < NumSpec; n++) {
      eos_state.xn[n] = interface_state(i,j,k,QFS+n);
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
      eos_state.aux[n] = interface_state(i,j,k,QFX+n);
    }
#endif

    eos(eos_input_re, eos_state);
    conductivity(eos_state);

    Real dTdx;

    if (idir == 0) {

      if (holds_cell_average) {
        dTdx = (-prim_arr(i+1,j,k,temperature_index) + 15*prim_arr(i,j,k,temperature_index) -
                15*prim_arr(i-1,j,k,temperature_index) + prim_arr(i-2,j,k,temperature_index)) / (12.0_rt * dx[0]);

      } else {
        dTdx = (-prim_arr(i+1,j,k,temperature_index) + 27*prim_arr(i,j,k,temperature_index) -
                27*prim_arr(i-1,j,k,temperature_index) + prim_arr(i-2,j,k,temperature_index)) / (24.0_rt * dx[0]);
      }

    } else if (idir == 1) {

      if (holds_cell_average) {
        dTdx = (-prim_arr(i,j+1,k,temperature_index) + 15*prim_arr(i,j,k,temperature_index) -
                15*prim_arr(i,j-1,k,temperature_index) + prim_arr(i,j-2,k,temperature_index)) / (12.0_rt * dx[1]);

      } else {
        dTdx = (-prim_arr(i,j+1,k,temperature_index) + 27*prim_arr(i,j,k,temperature_index) -
                27*prim_arr(i,j-1,k,temperature_index) + prim_arr(i,j-2,k,temperature_index)) / (24.0_rt * dx[1]);
      }

    } else {

      if (holds_cell_average) {
        dTdx = (-prim_arr(i,j,k+1,temperature_index) + 15*prim_arr(i,j,k,temperature_index) -
                15*prim_arr(i,j,k-1,temperature_index) + prim_arr(i,j,k-2,temperature_index)) / (12.0_rt * dx[2]);

      } else {
        dTdx = (-prim_arr(i,j,k+1,temperature_index) + 27*prim_arr(i,j,k,temperature_index) -
                27*prim_arr(i,j,k-1,temperature_index) + prim_arr(i,j,k-2,temperature_index)) / (24.0_rt * dx[2]);
      }

    }

    F(i,j,k,UEINT) -= eos_state.conductivity * dTdx;
    F(i,j,k,UEDEN) -= eos_state.conductivity * dTdx;

  });
}
#endif
