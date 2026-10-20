#include <HydroSim.H>

#ifdef ROTATION
#include <Rotation.H>
#endif

#include <HydroSim_helpers.H>
#include <primitive_conversion.H>

#ifdef HYBRID_MOMENTUM
#include <hybrid_momentum.H>
#endif

#ifdef RADIATION
#include <Radiation.H>
#include <flux_limiter.H>
#include <radiation_helpers.H>
#endif

#include <eos.H>

using namespace amrex;

void
HydroSim::ctoprim(const Box& bx,
                const Real time,
                Array4<Real const> const& uin,
#ifdef MHD
                Array4<Real const> const& Bx,
                Array4<Real const> const& By,
                Array4<Real const> const& Bz,
#endif
#ifdef RADIATION
                Array4<Real const> const& RadEnergyIn,
                Array4<Real const> const& lam,
#endif
                Array4<Real> const& prim_arr,
                Array4<Real> const& qthermo_arr) {

  amrex::ignore_unused(time);

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
      auto q = [&] (int n) -> Real& { return prim_arr(i,j,k,n); };
      auto qthermo = [&] (int n) -> Real& { return qthermo_arr(i,j,k,n); };

      hydro::cell_to_primitive(i, j, k, uin,
#ifdef MHD
                                       Bx, By, Bz,
#endif
#ifdef RADIATION
                                       RadEnergyIn, lam,
#endif
                                       q, qthermo, prim_arr.nComp() == NQ);
  });
}

void
HydroSim::clamp_state_density(const Box& bx,
                                   Array4<Real> const& state_arr,
                                   const int warning_level) {

#ifdef HYBRID_MOMENTUM
  GeometryData geomdata = geom.data();
#endif

  amrex::ignore_unused(warning_level);

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    if (state_arr(i,j,k,URHO) < small_dens) {

#ifndef AMREX_USE_GPU
      if (warning_level > 1 ||
          (warning_level > 0 && state_arr(i,j,k,URHO) > hydrosim::retry_density_cutoff)) {
        std::cout << " " << std::endl;
        if (state_arr(i,j,k,URHO) < 0.0_rt) {
          std::cout << ">>> RESETTING NEG.  DENSITY AT " << i << ", " << j << ", " << k << std::endl;
        }
        else if (state_arr(i,j,k,URHO) == 0.0_rt) {
          amrex::Error("Density exactly zero at " + std::to_string(i) + ", " +
                                                    std::to_string(j) + ", " +
                                                    std::to_string(k));
        }
        else {
          std::cout << ">>> RESETTING SMALL DENSITY AT " << i << ", " << j << ", " << k << std::endl;
        }
        std::cout << ">>> FROM " << state_arr(i,j,k,URHO) << " TO " << small_dens << std::endl;
        std::cout << ">>> IN GRID " << bx << std::endl;
        std::cout << " " << std::endl;
      }
#endif

      for (int passive_index = 0; passive_index < npassive; passive_index++) {
          const int n = passive_cons_index(passive_index);
          state_arr(i,j,k,n) *= (small_dens / state_arr(i,j,k,URHO));
      }

      eos_re_t eos_state;
      eos_state.rho = small_dens;
      eos_state.T = small_temp;
      for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = state_arr(i,j,k,UFS+n) / small_dens;
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
        eos_state.aux[n] = state_arr(i,j,k,UFX+n) / small_dens;
      }
#endif

      eos(eos_input_rt, eos_state);

      state_arr(i,j,k,URHO ) = eos_state.rho;
      state_arr(i,j,k,UTEMP) = eos_state.T;

      state_arr(i,j,k,UMX) = 0.0_rt;
      state_arr(i,j,k,UMY) = 0.0_rt;
      state_arr(i,j,k,UMZ) = 0.0_rt;

      state_arr(i,j,k,UEINT) = eos_state.rho * eos_state.e;
      state_arr(i,j,k,UEDEN) = state_arr(i,j,k,UEINT);

#ifdef HYBRID_MOMENTUM
      GpuArray<Real, 3> loc;

      position(i, j, k, geomdata, loc);

      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        loc[dir] -= problem::center[dir];
      }

      GpuArray<Real, 3> cartesian_vector;

      for (int dir = 0; dir < 3; ++dir) {
        cartesian_vector[dir] = state_arr(i,j,k,UMX+dir);
      }

      GpuArray<Real, 3> hybrid_vector;

      cartesian_to_hybrid(loc, cartesian_vector, hybrid_vector);

      for (int dir = 0; dir < 3; ++dir) {
        state_arr(i,j,k,UMR+dir) = hybrid_vector[dir];
      }
#endif
    }
  });
}

void
HydroSim::set_reflecting_boundary_states(const Box& bx, const int idir,
                               Array4<Real> const& qm,
                               Array4<Real> const& qp) {

    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();

    const auto domlo = geom.Domain().loVect3d();
    const auto domhi = geom.Domain().hiVect3d();

    bool at_lo_symmetry = lo_bc[idir] == amrex::PhysBCType::symmetry;
    bool at_hi_symmetry = hi_bc[idir] == amrex::PhysBCType::symmetry;

    const int QUN = QU + idir;

    if (at_lo_symmetry) {

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {

            if ((idir == 0 && i == domlo[0]) ||
                (idir == 1 && j == domlo[1]) ||
                (idir == 2 && k == domlo[2])) {
                for (int n = 0; n < NQ; n++) {
                    if (n == QUN) {
                        qm(i,j,k,QUN) = -qp(i,j,k,QUN);
                    } else {
                        qm(i,j,k,n) = qp(i,j,k,n);
                    }
                }
            }
        });
    }

    if (at_hi_symmetry) {

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {

            if ((idir == 0 && i == domhi[0]+1) ||
                (idir == 1 && j == domhi[1]+1) ||
                (idir == 2 && k == domhi[2]+1)) {
                for (int n = 0; n < NQ; n++) {
                    if (n == QUN) {
                        qp(i,j,k,QUN) = -qm(i,j,k,QUN);
                    } else {
                        qp(i,j,k,n) = qm(i,j,k,n);
                    }
                }
            }
        });

    }
}

#if AMREX_SPACEDIM == 3
static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real corner_averaged_difference(Array4<Real const> const& q, int i, int j, int k, int comp,
                                int d, int a, int b)
{
    const int di = (d == 0), dj = (d == 1), dk = (d == 2);
    const int ai = (a == 0), aj = (a == 1), ak = (a == 2);
    const int bi = (b == 0), bj = (b == 1), bk = (b == 2);

    return 0.25_rt * (q(i,j,k,comp) - q(i-di,j-dj,k-dk,comp) +
                      q(i-bi,j-bj,k-bk,comp) - q(i-bi-di,j-bj-dj,k-bk-dk,comp) +
                      q(i-ai,j-aj,k-ak,comp) - q(i-ai-di,j-aj-dj,k-ak-dk,comp) +
                      q(i-ai-bi,j-aj-bj,k-ak-bk,comp) - q(i-ai-bi-di,j-aj-bj-dj,k-ak-bk-dk,comp));
}
#endif

void
HydroSim::divu(const Box& bx,
             Array4<Real const> const& prim_arr,
             Array4<Real> const& div) {

  const auto dx = geom.CellSizeArray();

#if AMREX_SPACEDIM <= 2
  const int coord_type = geom.Coord();
  const auto problo = geom.ProbLoArray();
#endif

  Real dxinv = 1.0_rt / dx[0];
#if AMREX_SPACEDIM >= 2
  Real dyinv = 1.0_rt / dx[1];
#endif
#if AMREX_SPACEDIM == 3
  Real dzinv = 1.0_rt / dx[2];
#endif

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

#if AMREX_SPACEDIM == 1
    if (coord_type == 0) {
        div(i,j,k) = (prim_arr(i,j,k,QU) - prim_arr(i-1,j,k,QU)) * dxinv;

    } else if (coord_type == 1) {
        if (i == 0) {
            div(i,j,k) = 0.0_rt;
        } else {
            Real rl = (i - 0.5_rt) * dx[0] + problo[0];
            Real rr = (i + 0.5_rt) * dx[0] + problo[0];
            Real rc = (i) * dx[0] + problo[0];

            div(i,j,k) = (rr * prim_arr(i,j,k,QU) - rl * prim_arr(i-1,j,k,QU)) * dxinv / rc;
        }
    } else {
        if (i == 0) {
            div(i,j,k) = 0.0_rt;
        } else {
            Real rl = (i - 0.5_rt) * dx[0] + problo[0];
            Real rr = (i + 0.5_rt) * dx[0] + problo[0];
            Real rc = (i) * dx[0] + problo[0];

            div(i,j,k) = (rr * rr * prim_arr(i,j,k,QU) - rl * rl * prim_arr(i-1,j,k,QU)) * dxinv / (rc * rc);
        }
    }
#endif

#if AMREX_SPACEDIM == 2
    Real ux = 0.0_rt;
    Real vy = 0.0_rt;

    if (coord_type == 0) {

        ux = 0.5_rt * (prim_arr(i,j,k,QU) - prim_arr(i-1,j,k,QU) + prim_arr(i,j-1,k,QU) - prim_arr(i-1,j-1,k,QU)) * dxinv;
        vy = 0.5_rt * (prim_arr(i,j,k,QV) - prim_arr(i,j-1,k,QV) + prim_arr(i-1,j,k,QV) - prim_arr(i-1,j-1,k,QV)) * dyinv;

    } else if (coord_type == 1) {

        if (i == 0) {
            ux = 0.0_rt;
        } else {
            Real rl = (i - 0.5_rt) * dx[0] + problo[0];
            Real rr = (i + 0.5_rt) * dx[0] + problo[0];
            Real rc = (i) * dx[0] + problo[0];

            Real ul = 0.5_rt * (prim_arr(i-1,j,k,QU) + prim_arr(i-1,j-1,k,QU));
            Real ur = 0.5_rt * (prim_arr(i,j,k,QU) + prim_arr(i,j-1,k,QU));

            ux = (rr * ur - rl * ul) * dxinv / rc;
        }

        Real vb = 0.5_rt * (prim_arr(i,j-1,k,QV) + prim_arr(i-1,j-1,k,QV));
        Real vt = 0.5_rt * (prim_arr(i,j,k,QV) + prim_arr(i-1,j,k,QV));

        vy = (vt - vb) * dyinv;

    } else {

        Real rl = (i - 0.5_rt) * dx[0] + problo[0];
        Real rr = (i + 0.5_rt) * dx[0] + problo[0];
        Real rc = (i) * dx[0] + problo[0];

        Real sint = std::sin((j + 0.5_rt) * dx[1] + problo[1]);
        Real sinb = std::sin((j - 0.5_rt) * dx[1] + problo[1]);
        Real sinc = std::sin(j  * dx[1] + problo[1]);

        Real ul = 0.5_rt * (prim_arr(i-1,j,k,QU) + prim_arr(i-1,j-1,k,QU));
        Real ur = 0.5_rt * (prim_arr(i,j,k,QU) + prim_arr(i,j-1,k,QU));

        ux = (ur * rr * rr - ul * rl * rl) * dxinv / (rc * rc);

        if (sinc == 0.0_rt) {
            vy = 0.0_rt;
        } else {
            Real vb = 0.5_rt * (prim_arr(i,j-1,k,QV) + prim_arr(i-1,j-1,k,QV));
            Real vt = 0.5_rt * (prim_arr(i,j,k,QV) + prim_arr(i-1,j,k,QV));

            vy = (vt * sint - vb * sinb) * dyinv / (rc * sinc);
        }
    }

    div(i,j,k) = ux + vy;
#endif

#if AMREX_SPACEDIM == 3
    Real ux = corner_averaged_difference(prim_arr, i, j, k, QU, 0, 1, 2) * dxinv;

    Real vy = corner_averaged_difference(prim_arr, i, j, k, QV, 1, 0, 2) * dyinv;

    Real wz = corner_averaged_difference(prim_arr, i, j, k, QW, 2, 0, 1) * dzinv;

    div(i,j,k) = ux + vy + wz;
#endif

  });

}

static AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real face_averaged_divergence(Array4<Real const> const& div, int i, int j, int k, int idir)
{
    if (idir == 0) {
        return 0.25_rt * (div(i,j,k) + div(i,j+dg1,k) +
                          div(i,j,k+dg2) + div(i,j+dg1,k+dg2));
    } else if (idir == 1) {
        return 0.25_rt * (div(i,j,k) + div(i+1,j,k) +
                          div(i,j,k+dg2) + div(i+1,j,k+dg2));
    }
    return 0.25_rt * (div(i,j,k) + div(i+1,j,k) +
                      div(i,j+dg1,k) + div(i+1,j+dg1,k));
}

void
HydroSim::apply_viscosity(const Box& bx,
                 const int idir,
                 Array4<Real const> const& div,
                 Array4<Real const> const& uin,
                 Array4<Real> const& flux) {

  const auto dx = geom.CellSizeArray();
  const auto coord = geom.Coord();
  const auto problo = geom.ProbLoArray();

  Real diff_coeff = viscosity_coeff;

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
      Real dL = dx[idir];
      if (coord == 2 && idir == 1) {
          Real r = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
          dL *= r;
      }

      Real face_divergence = face_averaged_divergence(div, i, j, k, idir);

      face_divergence = diff_coeff * std::min(0.0_rt, face_divergence);

      for (int n = 0; n < NUM_STATE; ++n) {

          if (n == UTEMP) {
              continue;
          }
#ifdef SHOCK_VAR
          if (n == UFLAG_SHOCK) {
              continue;
          }
#endif
#ifdef NSE_NET
          if (n == UPROTON_MU || n == UNEUTRON_MU) {
              continue;
          }
#endif

          Real face_jump{};

          if (idir == 0) {
              face_jump = face_divergence * (uin(i,j,k,n) - uin(i-1,j,k,n));
          } else if (idir == 1) {
              face_jump = face_divergence * (uin(i,j,k,n) - uin(i,j-dg1,k,n));
          } else {
              face_jump = face_divergence * (uin(i,j,k,n) - uin(i,j,k-dg2,n));
          }

          flux(i,j,k,n) += dL * face_jump;
      }
  });
}

#ifdef RADIATION
void
HydroSim::apply_viscosity_radiative(const Box& bx,
                     const int idir,
                     Array4<Real const> const& div,
                     Array4<Real const> const& RadEnergyIn,
                     Array4<Real> const& radiative_flux) {

  const auto dx = geom.CellSizeArray();

  amrex::Real diff_coeff = viscosity_coeff;

  int ngroups = Radiation::nGroups;

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

      Real face_divergence = face_averaged_divergence(div, i, j, k, idir);

      face_divergence = diff_coeff * std::min(0.0_rt, face_divergence);

      for (int n = 0; n < ngroups; ++n) {

          Real face_jump{};

          if (idir == 0) {
              face_jump = face_divergence * (RadEnergyIn(i,j,k,n) - RadEnergyIn(i-1,j,k,n));
          } else if (idir == 1) {
              face_jump = face_divergence * (RadEnergyIn(i,j,k,n) - RadEnergyIn(i,j-dg1,k,n));
          } else {
              face_jump = face_divergence * (RadEnergyIn(i,j,k,n) - RadEnergyIn(i,j,k-dg2,n));
          }

          radiative_flux(i,j,k,n) += dx[idir] * face_jump;
      }
  });
}
#endif

void
HydroSim::shock(const Box& bx,
              Array4<Real const> const& prim_arr,
              Array4<Real const> const& U_source_arr,
              Array4<Real> const& shk) {

  const auto dx = geom.CellSizeArray();
  const auto problo = geom.ProbLoArray();
  const int coord_type = geom.Coord();

  Real dxinv = 1.0_rt / dx[0];
#if AMREX_SPACEDIM >= 2
  Real dyinv = 1.0_rt / dx[1];
#endif
#if AMREX_SPACEDIM == 3
  Real dzinv = 1.0_rt / dx[2];
#endif

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real div_vel = 0.0_rt;

    switch (coord_type) {
    case 0:
      div_vel += 0.5_rt * (prim_arr(i+1,j,k,QU) - prim_arr(i-1,j,k,QU)) * dxinv;
#if (AMREX_SPACEDIM >= 2)
      div_vel += 0.5_rt * (prim_arr(i,j+1,k,QV) - prim_arr(i,j-1,k,QV)) * dyinv;
#endif
#if (AMREX_SPACEDIM == 3)
      div_vel += 0.5_rt * (prim_arr(i,j,k+1,QW) - prim_arr(i,j,k-1,QW)) * dzinv;
#endif
      break;

#if AMREX_SPACEDIM <= 2
    case 1: {

     Real rc = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
     Real rm = problo[0] + (static_cast<Real>(i) - 0.5_rt) * dx[0];
     Real rp = problo[0] + (static_cast<Real>(i) + 1.5_rt) * dx[0];

#if (AMREX_SPACEDIM == 1)
     div_vel += 0.5_rt * (rp * prim_arr(i+1,j,k,QU) - rm * prim_arr(i-1,j,k,QU)) / (rc * dx[0]);
#endif
#if (AMREX_SPACEDIM == 2)
     div_vel += 0.5_rt * (rp * prim_arr(i+1,j,k,QU) - rm * prim_arr(i-1,j,k,QU)) / (rc * dx[0]) +
              0.5_rt * (prim_arr(i,j+1,k,QV) - prim_arr(i,j-1,k,QV)) * dyinv;
#endif
     break;
    }
#endif

    case 2: {

      Real rc = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
      Real rm = problo[0] + (static_cast<Real>(i) - 0.5_rt) * dx[0];
      Real rp = problo[0] + (static_cast<Real>(i) + 1.5_rt) * dx[0];

      div_vel += 0.5_rt * (rp * rp * prim_arr(i+1,j,k,QU) - rm * rm * prim_arr(i-1,j,k,QU)) / (rc * rc * dx[0]);
#if AMREX_SPACEDIM == 2

      Real polar_center = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1];
      Real theta_minus = problo[1] + (static_cast<Real>(j) - 0.5_rt) * dx[1];
      Real theta_plus = problo[1] + (static_cast<Real>(j) + 1.5_rt) * dx[1];

      div_vel += 0.5_rt * (std::sin(theta_plus) * prim_arr(i,j+1,k,QV) -
                         std::sin(theta_minus) * prim_arr(i,j-1,k,QV)) /
          (rc * std::sin(polar_center) * dx[1]);
#endif
      break;
    }

#ifndef AMREX_USE_GPU
    default:
      amrex::Error("ERROR: invalid coord_type in shock");
#endif
    }

    Real pressure_diff_x = 0.5_rt * (prim_arr(i+1,j,k,QPRES) - prim_arr(i-1,j,k,QPRES));
    if (shock_sensor_subtract_src == 1) {
        pressure_diff_x += -0.25_rt * dx[0] * (U_source_arr(i+1,j,k,UMX) + 2.0_rt * U_source_arr(i,j,k,UMX) + U_source_arr(i-1,j,k,UMX));
    }
    Real pressure_diff_y = 0.0_rt;
    Real pressure_diff_z = 0.0_rt;
#if AMREX_SPACEDIM >= 2
    pressure_diff_y = 0.5_rt * (prim_arr(i,j+1,k,QPRES) - prim_arr(i,j-1,k,QPRES));
    if (shock_sensor_subtract_src == 1) {
        Real dy{dx[1]};
        if (coord_type == 2) {
            Real rc = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
            dy *= rc;
        }
        pressure_diff_y += -0.25_rt * dy * (U_source_arr(i,j+1,k,UMY) + 2.0_rt * U_source_arr(i,j,k,UMY) + U_source_arr(i,j-1,k,UMY));
    }
#endif
#if AMREX_SPACEDIM == 3
    pressure_diff_z = 0.5_rt * (prim_arr(i,j,k+1,QPRES) - prim_arr(i,j,k-1,QPRES));
    if (shock_sensor_subtract_src == 1) {
        pressure_diff_z += -0.25_rt * dx[2] * (U_source_arr(i,j,k+1,UMZ) + 2.0_rt * U_source_arr(i,j,k,UMZ) + U_source_arr(i,j,k-1,UMZ));
    }
#endif

    Real vel = std::sqrt(prim_arr(i,j,k,QU) * prim_arr(i,j,k,QU) +
                         prim_arr(i,j,k,QV) * prim_arr(i,j,k,QV) +
                         prim_arr(i,j,k,QW) * prim_arr(i,j,k,QW));

    Real relative_pressure_grad{0.0_rt};
    if (vel != 0.0) {
        relative_pressure_grad = std::abs(pressure_diff_x * prim_arr(i,j,k,QU) +
                                  pressure_diff_y * prim_arr(i,j,k,QV) +
                                  pressure_diff_z * prim_arr(i,j,k,QW)) / vel;
    }
    relative_pressure_grad /= prim_arr(i,j,k,QPRES);

    if (relative_pressure_grad > hydrosim::shock_sensor_threshold && div_vel < 0.0_rt) {
      shk(i,j,k) = 1.0_rt;
    } else {
      shk(i,j,k) = 0.0_rt;
    }
  });

}
