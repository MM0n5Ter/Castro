#include <HydroSim.H>

#ifdef DIFFUSION
#include <conductivity.H>
#endif

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
limit_fourth_order_edges(const Real am3, const Real am2, const Real am1, const Real a0,
                         const Real ap1, const Real ap2, const Real ap3,
                         const Real face_lo, const Real face_hi,
                         Real& lo_edge, Real& hi_edge) {

  constexpr Real C2 = 1.25_rt;
  constexpr Real C3 = 0.1_rt;

  Real jump_left = a0 - face_lo;
  Real jump_right = face_hi - a0;

  Real second_diff_face = 6.0_rt*(face_lo - 2.0_rt*a0 + face_hi);

  Real second_diff_m2 = am3 - 2.0_rt*am2 + am1;
  Real second_diff_m1 = am2 - 2.0_rt*am1 + a0;
  Real second_diff_0  = am1 - 2.0_rt*a0 + ap1;
  Real second_diff_p1 = a0 - 2.0_rt*ap1 + ap2;
  Real second_diff_p2 = ap1 - 2.0_rt*ap2 + ap3;

  if (jump_left * jump_right <= 0.0_rt ||
      (a0 - am2)*(ap2 - a0) <= 0.0_rt) {

    Real s = std::copysign(1.0_rt, second_diff_0);

    Real limited_second_diff;

    if (s == std::copysign(1.0_rt, second_diff_m1) &&
        s == std::copysign(1.0_rt, second_diff_p1) &&
        s == std::copysign(1.0_rt, second_diff_face)) {
      limited_second_diff = s * amrex::min(std::abs(second_diff_face), C2*std::abs(second_diff_m1),
                               C2*std::abs(second_diff_0), C2*std::abs(second_diff_p1));
    } else {
      limited_second_diff = 0.0_rt;
    }

    Real rho;
    if (std::abs(second_diff_face) <= 1.e-12_rt *
        amrex::max(std::abs(am2), std::abs(am1),
                   std::abs(a0), std::abs(ap1),
                   std::abs(ap2))) {
      rho = 0.0_rt;
    } else {
      rho = limited_second_diff/second_diff_face;
    }

    if (rho < 1.0_rt - 1.e-12_rt) {
      Real third_diff_m1 = second_diff_m1 - second_diff_m2;
      Real third_diff_0  = second_diff_0 - second_diff_m1;
      Real third_diff_p1 = second_diff_p1 - second_diff_0;
      Real third_diff_p2 = second_diff_p2 - second_diff_p1;

      Real third_diff_min = amrex::min(third_diff_m1, third_diff_0, third_diff_p1, third_diff_p2);
      Real third_diff_max = amrex::max(third_diff_m1, third_diff_0, third_diff_p1, third_diff_p2);

      if (C3 * std::max(std::abs(third_diff_min), std::abs(third_diff_max)) <=
          (third_diff_max - third_diff_min)) {
        if (jump_left*jump_right < 0.0_rt) {
          lo_edge = a0 - rho*jump_left;
          hi_edge = a0 + rho*jump_right;
        } else if (std::abs(jump_left) >= 2.0_rt*std::abs(jump_right)) {
          lo_edge = a0 - 2.0_rt*(1.0_rt - rho)*jump_right - rho*jump_left;
        } else if (std::abs(jump_right) >= 2.0_rt*std::abs(jump_left)) {
          hi_edge = a0 + 2.0_rt*(1.0_rt - rho)*jump_left + rho*jump_right;
        }

      }
    }

  } else {
    if (std::abs(jump_left) >= 2.0_rt*std::abs(jump_right)) {
      lo_edge = a0 - 2.0_rt*jump_right;
    }
    if (std::abs(jump_right) >= 2.0_rt*std::abs(jump_left)) {
      hi_edge = a0 + 2.0_rt*jump_left;
    }
  }
}

void
HydroSim::fourth_face_states(const Box& bx,
                          const int idir, const int ncomp,
                          Array4<Real const> const& a,
                          Array4<Real> const& face_values) {

  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  auto domlo = geom.Domain().loVect3d();
  auto domhi = geom.Domain().hiVect3d();

  const int di = (idir == 0);
  const int dj = (idir == 1);
  const int dk = (idir == 2);

  const bool lo_symmetric = lo_bc[idir] == amrex::PhysBCType::symmetry;
  const bool hi_symmetric = hi_bc[idir] == amrex::PhysBCType::symmetry;
  const int dlo = domlo[idir];
  const int dhi = domhi[idir];

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    const int cell = idir == 0 ? i : (idir == 1 ? j : k);

    auto at = [&] (int m) {
      return a(i+m*di, j+m*dj, k+m*dk, ncomp);
    };

    if (cell == dlo+1 && lo_symmetric) {
      face_values(i,j,k) = (1.0_rt/12.0_rt)*(3.0_rt*at(-1) + 13.0_rt*at(0) -
                                       5.0_rt*at(1) + at(2));

    } else if (cell == dlo && lo_symmetric) {
      face_values(i,j,k) = (1.0_rt/12.0_rt)*(25.0_rt*at(0) - 23.0_rt*at(1) +
                                       13.0_rt*at(2) - 3.0_rt*at(3));

    } else if (cell == dhi && hi_symmetric) {
      face_values(i,j,k) = (1.0_rt/12.0_rt)*(3.0_rt*at(0) + 13.0_rt*at(-1) -
                                       5.0_rt*at(-2) + at(-3));

    } else if (cell == dhi+1 && hi_symmetric) {
      face_values(i,j,k) = (1.0_rt/12.0_rt)*(25.0_rt*at(-1) - 23.0_rt*at(-2) +
                                       13.0_rt*at(-3) - 3.0_rt*at(-4));

    } else {
      face_values(i,j,k) = (7.0_rt/12.0_rt)*(at(-1) + at(0)) -
                     (1.0_rt/12.0_rt)*(at(-2) + at(1));
    }
  });
}

void
HydroSim::states(const Box& bx,
               const int idir, const int ncomp,
               Array4<Real const> const& a,
               Array4<Real const> const& face_values,
               Array4<Real const> const& flatten_factor,
               Array4<Real> const& al,
               Array4<Real> const& ar) {

  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  auto domlo = geom.Domain().loVect3d();
  auto domhi = geom.Domain().hiVect3d();

  if (idir == 0) {

    if (clamp_fourth_order == 0) {

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        al(i+1,j,k,ncomp) = face_values(i+1,j,k);
        ar(i,j,k,ncomp) = face_values(i,j,k);
      });

    } else {

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        al(i+1,j,k,ncomp) = face_values(i+1,j,k);
        ar(i,j,k,ncomp) = face_values(i,j,k);

        limit_fourth_order_edges(a(i-3,j,k,ncomp), a(i-2,j,k,ncomp), a(i-1,j,k,ncomp), a(i,j,k,ncomp),
                                 a(i+1,j,k,ncomp), a(i+2,j,k,ncomp), a(i+3,j,k,ncomp), face_values(i,j,k), face_values(i+1,j,k),
                                 ar(i,j,k,ncomp), al(i+1,j,k,ncomp));

        al(i+1,j,k,ncomp) = flatten_factor(i,j,k)*al(i+1,j,k,ncomp) + (1.0_rt - flatten_factor(i,j,k))*a(i,j,k,ncomp);
        ar(i,j,k,ncomp) = flatten_factor(i,j,k)*ar(i,j,k,ncomp) + (1.0_rt - flatten_factor(i,j,k))*a(i,j,k,ncomp);

        if (i == domlo[0]) {

            if (lo_bc[0] == amrex::PhysBCType::outflow) {

            } else if (lo_bc[0] == amrex::PhysBCType::symmetry) {
                if (ncomp == QU) {
                    al(domlo[0],j,k,QU) = -ar(domlo[0],j,k,QU);
                } else {
                    al(domlo[0],j,k,ncomp) = ar(domlo[0],j,k,ncomp);
                }

            } else if (lo_bc[0] == amrex::PhysBCType::interior) {

            } else {
                amrex::Error("ERROR: boundary conditions not supported for -X in states");
            }

        }

        if (i == domhi[0]+1) {

            if (hi_bc[0] == amrex::PhysBCType::outflow) {

            } else if (hi_bc[0] == amrex::PhysBCType::symmetry) {
                if (ncomp == QU) {
                    ar(domhi[0]+1,j,k,QU) = -al(domhi[0]+1,j,k,QU);
                } else {
                    ar(domhi[0]+1,j,k,ncomp) = al(domhi[0]+1,j,k,ncomp);
                }

            } else if (hi_bc[0] == amrex::PhysBCType::interior) {

            } else {
                amrex::Error("ERROR: boundary conditions not supported for +X in states");
            }

        }

      });

    }

#if AMREX_SPACEDIM >= 2
  } else if (idir == 1) {

    if (clamp_fourth_order == 0) {

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        al(i,j+1,k,ncomp) = face_values(i,j+1,k);
        ar(i,j,k,ncomp) = face_values(i,j,k);
      });

    } else {

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        al(i,j+1,k,ncomp) = face_values(i,j+1,k);
        ar(i,j,k,ncomp) = face_values(i,j,k);

        limit_fourth_order_edges(a(i,j-3,k,ncomp), a(i,j-2,k,ncomp), a(i,j-1,k,ncomp), a(i,j,k,ncomp),
                                 a(i,j+1,k,ncomp), a(i,j+2,k,ncomp), a(i,j+3,k,ncomp), face_values(i,j,k), face_values(i,j+1,k),
                                 ar(i,j,k,ncomp), al(i,j+1,k,ncomp));

        al(i,j+1,k,ncomp) = flatten_factor(i,j,k)*al(i,j+1,k,ncomp) + (1.0_rt - flatten_factor(i,j,k))*a(i,j,k,ncomp);
        ar(i,j,k,ncomp) = flatten_factor(i,j,k)*ar(i,j,k,ncomp) + (1.0_rt - flatten_factor(i,j,k))*a(i,j,k,ncomp);

        if (j == domlo[1]) {

            if (lo_bc[1] == amrex::PhysBCType::outflow) {

            } else if (lo_bc[1] == amrex::PhysBCType::symmetry) {
                if (ncomp == QV) {
                    al(i,domlo[1],k,QV) = -ar(i,domlo[1],k,QV);
                } else {
                    al(i,domlo[1],k,ncomp) = ar(i,domlo[1],k,ncomp);
                }

            } else if (lo_bc[1] == amrex::PhysBCType::interior) {

            } else {
                amrex::Error("ERROR: boundary conditions not supported for -Y in states");
            }

        }

        if (j == domhi[1]+1) {

            if (hi_bc[1] == amrex::PhysBCType::outflow) {

            } else if (hi_bc[1] == amrex::PhysBCType::symmetry) {
                if (ncomp == QV) {
                    ar(i,domhi[1]+1,k,QV) = -al(i,domhi[1]+1,k,QV);
                } else {
                    ar(i,domhi[1]+1,k,ncomp) = al(i,domhi[1]+1,k,ncomp);
                }

            } else if (hi_bc[1] == amrex::PhysBCType::interior) {

            } else {
                amrex::Error("ERROR: boundary conditions not supported for +Y in states");
            }

        }

      });

    }
#endif

#if AMREX_SPACEDIM == 3
  } else if (idir == 2) {

    if (clamp_fourth_order == 0) {

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        al(i,j,k+1,ncomp) = face_values(i,j,k+1);
        ar(i,j,k,ncomp) = face_values(i,j,k);
      });

    } else {

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {

        al(i,j,k+1,ncomp) = face_values(i,j,k+1);
        ar(i,j,k,ncomp) = face_values(i,j,k);

        limit_fourth_order_edges(a(i,j,k-3,ncomp), a(i,j,k-2,ncomp), a(i,j,k-1,ncomp), a(i,j,k,ncomp),
                                 a(i,j,k+1,ncomp), a(i,j,k+2,ncomp), a(i,j,k+3,ncomp), face_values(i,j,k), face_values(i,j,k+1),
                                 ar(i,j,k,ncomp), al(i,j,k+1,ncomp));

        al(i,j,k+1,ncomp) = flatten_factor(i,j,k)*al(i,j,k+1,ncomp) + (1.0_rt - flatten_factor(i,j,k))*a(i,j,k,ncomp);
        ar(i,j,k,ncomp) = flatten_factor(i,j,k)*ar(i,j,k,ncomp) + (1.0_rt - flatten_factor(i,j,k))*a(i,j,k,ncomp);

        if (k == domlo[2]) {

            if (lo_bc[2] == amrex::PhysBCType::outflow) {

            } else if (lo_bc[2] == amrex::PhysBCType::symmetry) {
                if (ncomp == QW) {
                    al(i,j,domlo[2],QW) = -ar(i,j,domlo[2],QW);
                } else {
                    al(i,j,domlo[2],ncomp) = ar(i,j,domlo[2],ncomp);
                }

            } else if (lo_bc[2] == amrex::PhysBCType::interior) {

            } else {
                amrex::Error("ERROR: boundary conditions not supported for -Z in states");
            }

        }

        if (k == domhi[2]+1) {

            if (hi_bc[2] == amrex::PhysBCType::outflow) {

            } else if (hi_bc[2] == amrex::PhysBCType::symmetry) {
                if (ncomp == QW) {
                    ar(i,j,domhi[2]+1,QW) = -al(i,j,domhi[2]+1,QW);
                } else {
                    ar(i,j,domhi[2]+1,ncomp) = al(i,j,domhi[2]+1,ncomp);
                }

            } else if (hi_bc[2] == amrex::PhysBCType::interior) {

            } else {
                amrex::Error("ERROR: boundary conditions not supported for +Z in states");
            }

        }

      });

    }
#endif

  }

}
