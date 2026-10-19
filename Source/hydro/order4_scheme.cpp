#include <HydroSim.H>

#ifdef DIFFUSION
#include <conductivity.H>
#endif

void
HydroSim::fourth_face_states(const Box& bx,
                          const int idir, const int ncomp,
                          Array4<Real const> const& a,
                          Array4<Real> const& face_values) {



  const int* lo_bc = phys_bc.lo();
  const int* hi_bc = phys_bc.hi();

  auto domlo = geom.Domain().loVect3d();
  auto domhi = geom.Domain().hiVect3d();

  if (idir == 0) {


    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {


      if (i == domlo[0]+1 && lo_bc[0] == amrex::PhysBCType::symmetry) {
        face_values(i,j,k) = (1.0_rt/12.0_rt)*(3.0_rt*a(i-1,j,k,ncomp) + 13.0_rt*a(i,j,k,ncomp) -
                                         5.0_rt*a(i+1,j,k,ncomp) + a(i+2,j,k,ncomp));

      } else if (i == domlo[0] && lo_bc[0] == amrex::PhysBCType::symmetry) {
        face_values(i,j,k) = (1.0_rt/12.0_rt)*(25.0_rt*a(i,j,k,ncomp) - 23.0_rt*a(i+1,j,k,ncomp) +
                                         13.0_rt*a(i+2,j,k,ncomp) - 3.0_rt*a(i+3,j,k,ncomp));

      } else if (i == domhi[0] && hi_bc[0] == amrex::PhysBCType::symmetry) {
        face_values(i,j,k) = (1.0_rt/12.0_rt)*(3.0_rt*a(i,j,k,ncomp) + 13.0_rt*a(i-1,j,k,ncomp) -
                                         5.0_rt*a(i-2,j,k,ncomp) + a(i-3,j,k,ncomp));

      } else if (i == domhi[0]+1 && hi_bc[0] == amrex::PhysBCType::symmetry) {
        face_values(i,j,k) = (1.0_rt/12.0_rt)*(25.0_rt*a(i-1,j,k,ncomp) - 23.0_rt*a(i-2,j,k,ncomp) +
                                         13.0_rt*a(i-3,j,k,ncomp) - 3.0_rt*a(i-4,j,k,ncomp));

      } else {
        face_values(i,j,k) = (7.0_rt/12.0_rt)*(a(i-1,j,k,ncomp) + a(i,j,k,ncomp)) -
                       (1.0_rt/12.0_rt)*(a(i-2,j,k,ncomp) + a(i+1,j,k,ncomp));
      }
    });

#if AMREX_SPACEDIM >= 2
  } else if (idir == 1) {


    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {


      if (j == domlo[1]+1 && lo_bc[1] == amrex::PhysBCType::symmetry) {
        face_values(i,j,k) = (1.0_rt/12.0_rt)*(3.0_rt*a(i,j-1,k,ncomp) + 13.0_rt*a(i,j,k,ncomp) -
                                         5.0_rt*a(i,j+1,k,ncomp) + a(i,j+2,k,ncomp));

      } else if (j == domlo[1] && lo_bc[1] == amrex::PhysBCType::symmetry) {
        face_values(i,j,k) = (1.0_rt/12.0_rt)*(25.0_rt*a(i,j,k,ncomp) - 23.0_rt*a(i,j+1,k,ncomp) +
                                         13.0_rt*a(i,j+2,k,ncomp) - 3.0_rt*a(i,j+3,k,ncomp));

      } else if (j == domhi[1] && hi_bc[1] == amrex::PhysBCType::symmetry) {
        face_values(i,j,k) = (1.0_rt/12.0_rt)*(3.0_rt*a(i,j,k,ncomp) + 13.0_rt*a(i,j-1,k,ncomp) -
                                         5.0_rt*a(i,j-2,k,ncomp) + a(i,j-3,k,ncomp));

      } else if (j == domhi[1]+1 && hi_bc[1] == amrex::PhysBCType::symmetry) {
        face_values(i,j,k) = (1.0_rt/12.0_rt)*(25.0_rt*a(i,j-1,k,ncomp) - 23.0_rt*a(i,j-2,k,ncomp) +
                                         13.0_rt*a(i,j-3,k,ncomp) - 3.0_rt*a(i,j-4,k,ncomp));

      } else {
        face_values(i,j,k) = (7.0_rt/12.0_rt)*(a(i,j-1,k,ncomp) + a(i,j,k,ncomp)) -
                       (1.0_rt/12.0_rt)*(a(i,j-2,k,ncomp) + a(i,j+1,k,ncomp));
      }

    });
#endif

#if AMREX_SPACEDIM == 3
  } else if (idir == 2) {


    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {


      if (k == domlo[2]+1 && lo_bc[2] == amrex::PhysBCType::symmetry) {
        face_values(i,j,k) = (1.0_rt/12.0_rt)*(3.0_rt*a(i,j,k-1,ncomp) + 13.0_rt*a(i,j,k,ncomp) -
                                         5.0_rt*a(i,j,k+1,ncomp) + a(i,j,k+2,ncomp));

      } else if (k == domlo[2] && lo_bc[2] == amrex::PhysBCType::symmetry) {
        face_values(i,j,k) = (1.0_rt/12.0_rt)*(25.0_rt*a(i,j,k,ncomp) - 23.0_rt*a(i,j,k+1,ncomp) +
                                         13.0_rt*a(i,j,k+2,ncomp) - 3.0_rt*a(i,j,k+3,ncomp));

      } else if (k == domhi[2] && hi_bc[2] == amrex::PhysBCType::symmetry) {
        face_values(i,j,k) = (1.0_rt/12.0_rt)*(3.0_rt*a(i,j,k,ncomp) + 13.0_rt*a(i,j,k-1,ncomp) -
                                         5.0_rt*a(i,j,k-2,ncomp) + a(i,j,k-3,ncomp));

      } else if (k == domhi[2]+1 && hi_bc[2] == amrex::PhysBCType::symmetry) {
        face_values(i,j,k) = (1.0_rt/12.0_rt)*(25.0_rt*a(i,j,k-1,ncomp) - 23.0_rt*a(i,j,k-2,ncomp) +
                                         13.0_rt*a(i,j,k-3,ncomp) - 3.0_rt*a(i,j,k-4,ncomp));

      } else {
        face_values(i,j,k) = (7.0_rt/12.0_rt)*(a(i,j,k-1,ncomp) + a(i,j,k,ncomp)) -
                       (1.0_rt/12.0_rt)*(a(i,j,k-2,ncomp) + a(i,j,k+1,ncomp));
      }

    });
#endif

  }
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

  constexpr Real C2 = 1.25_rt;
  constexpr Real C3 = 0.1_rt;

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

        Real jump_left = a(i,j,k,ncomp) - face_values(i,j,k);
        Real jump_right = face_values(i+1,j,k) - a(i,j,k,ncomp);

        Real second_diff_face = 6.0_rt*(face_values(i,j,k) - 2.0_rt*a(i,j,k,ncomp) + face_values(i+1,j,k));

        Real second_diff_m2 = a(i-3,j,k,ncomp) - 2.0_rt*a(i-2,j,k,ncomp) + a(i-1,j,k,ncomp);
        Real second_diff_m1 = a(i-2,j,k,ncomp) - 2.0_rt*a(i-1,j,k,ncomp) + a(i,j,k,ncomp);
        Real second_diff_0  = a(i-1,j,k,ncomp) - 2.0_rt*a(i,j,k,ncomp) + a(i+1,j,k,ncomp);
        Real second_diff_p1 = a(i,j,k,ncomp) - 2.0_rt*a(i+1,j,k,ncomp) + a(i+2,j,k,ncomp);
        Real second_diff_p2 = a(i+1,j,k,ncomp) - 2.0_rt*a(i+2,j,k,ncomp) + a(i+3,j,k,ncomp);

        if (jump_left * jump_right <= 0.0_rt ||
            (a(i,j,k,ncomp) - a(i-2,j,k,ncomp))*(a(i+2,j,k,ncomp) - a(i,j,k,ncomp)) <= 0.0_rt) {


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
              amrex::max(std::abs(a(i-2,j,k,ncomp)), std::abs(a(i-1,j,k,ncomp)),
                         std::abs(a(i,j,k,ncomp)), std::abs(a(i+1,j,k,ncomp)),
                         std::abs(a(i+2,j,k,ncomp)))) {
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
                ar(i,j,k,ncomp) = a(i,j,k,ncomp) - rho*jump_left;
                al(i+1,j,k,ncomp) = a(i,j,k,ncomp) + rho*jump_right;
              } else if (std::abs(jump_left) >= 2.0_rt*std::abs(jump_right)) {
                ar(i,j,k,ncomp) = a(i,j,k,ncomp) - 2.0_rt*(1.0_rt - rho)*jump_right - rho*jump_left;
              } else if (std::abs(jump_right) >= 2.0_rt*std::abs(jump_left)) {
                al(i+1,j,k,ncomp) = a(i,j,k,ncomp) + 2.0_rt*(1.0_rt - rho)*jump_left + rho*jump_right;
              }

            }
          }

        } else {
          if (std::abs(jump_left) >= 2.0_rt*std::abs(jump_right)) {
            ar(i,j,k,ncomp) = a(i,j,k,ncomp) - 2.0_rt*jump_right;
          }
          if (std::abs(jump_right) >= 2.0_rt*std::abs(jump_left)) {
            al(i+1,j,k,ncomp) = a(i,j,k,ncomp) + 2.0_rt*jump_left;
          }
        }


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

        Real jump_left = a(i,j,k,ncomp) - face_values(i,j,k);
        Real jump_right = face_values(i,j+1,k) - a(i,j,k,ncomp);

        Real second_diff_face = 6.0_rt*(face_values(i,j,k) - 2.0_rt*a(i,j,k,ncomp) + face_values(i,j+1,k));

        Real second_diff_m2 = a(i,j-3,k,ncomp) - 2.0_rt*a(i,j-2,k,ncomp) + a(i,j-1,k,ncomp);
        Real second_diff_m1 = a(i,j-2,k,ncomp) - 2.0_rt*a(i,j-1,k,ncomp) + a(i,j,k,ncomp);
        Real second_diff_0  = a(i,j-1,k,ncomp) - 2.0_rt*a(i,j,k,ncomp) + a(i,j+1,k,ncomp);
        Real second_diff_p1 = a(i,j,k,ncomp) - 2.0_rt*a(i,j+1,k,ncomp) + a(i,j+2,k,ncomp);
        Real second_diff_p2 = a(i,j+1,k,ncomp) - 2.0_rt*a(i,j+2,k,ncomp) + a(i,j+3,k,ncomp);

        if (jump_left * jump_right <= 0.0_rt ||
            (a(i,j,k,ncomp) - a(i,j-2,k,ncomp))*(a(i,j+2,k,ncomp) - a(i,j,k,ncomp)) <= 0.0_rt) {


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
              amrex::max(std::abs(a(i,j-2,k,ncomp)), std::abs(a(i,j-1,k,ncomp)),
                         std::abs(a(i,j,k,ncomp)), std::abs(a(i,j+1,k,ncomp)),
                         std::abs(a(i,j+2,k,ncomp)))) {
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
                ar(i,j,k,ncomp) = a(i,j,k,ncomp) - rho*jump_left;
                al(i,j+1,k,ncomp) = a(i,j,k,ncomp) + rho*jump_right;
              } else if (std::abs(jump_left) >= 2.0_rt*std::abs(jump_right)) {
                ar(i,j,k,ncomp) = a(i,j,k,ncomp) - 2.0_rt*(1.0_rt - rho)*jump_right - rho*jump_left;
              } else if (std::abs(jump_right) >= 2.0_rt*std::abs(jump_left)) {
                al(i,j+1,k,ncomp) = a(i,j,k,ncomp) + 2.0_rt*(1.0_rt - rho)*jump_left + rho*jump_right;
              }

            }
          }

        } else {
          if (std::abs(jump_left) >= 2.0_rt*std::abs(jump_right)) {
            ar(i,j,k,ncomp) = a(i,j,k,ncomp) - 2.0_rt*jump_right;
          }
          if (std::abs(jump_right) >= 2.0_rt*std::abs(jump_left)) {
            al(i,j+1,k,ncomp) = a(i,j,k,ncomp) + 2.0_rt*jump_left;
          }
        }


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

        Real jump_left = a(i,j,k,ncomp) - face_values(i,j,k);
        Real jump_right = face_values(i,j,k+1) - a(i,j,k,ncomp);

        Real second_diff_face = 6.0_rt*(face_values(i,j,k) - 2.0_rt*a(i,j,k,ncomp) + face_values(i,j,k+1));

        Real second_diff_m2 = a(i,j,k-3,ncomp) - 2.0_rt*a(i,j,k-2,ncomp) + a(i,j,k-1,ncomp);
        Real second_diff_m1 = a(i,j,k-2,ncomp) - 2.0_rt*a(i,j,k-1,ncomp) + a(i,j,k,ncomp);
        Real second_diff_0  = a(i,j,k-1,ncomp) - 2.0_rt*a(i,j,k,ncomp) + a(i,j,k+1,ncomp);
        Real second_diff_p1 = a(i,j,k,ncomp) - 2.0_rt*a(i,j,k+1,ncomp) + a(i,j,k+2,ncomp);
        Real second_diff_p2 = a(i,j,k+1,ncomp) - 2.0_rt*a(i,j,k+2,ncomp) + a(i,j,k+3,ncomp);

        if (jump_left * jump_right <= 0.0_rt ||
            (a(i,j,k,ncomp) - a(i,j,k-2,ncomp))*(a(i,j,k+2,ncomp) - a(i,j,k,ncomp)) <= 0.0_rt) {


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
              amrex::max(std::abs(a(i,j,k-2,ncomp)), std::abs(a(i,j,k-1,ncomp)),
                         std::abs(a(i,j,k,ncomp)), std::abs(a(i,j,k+1,ncomp)),
                         std::abs(a(i,j,k+2,ncomp)))) {
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
                ar(i,j,k,ncomp) = a(i,j,k,ncomp) - rho*jump_left;
                al(i,j,k+1,ncomp) = a(i,j,k,ncomp) + rho*jump_right;
              } else if (std::abs(jump_left) >= 2.0_rt*std::abs(jump_right)) {
                ar(i,j,k,ncomp) = a(i,j,k,ncomp) - 2.0_rt*(1.0_rt - rho)*jump_right - rho*jump_left;
              } else if (std::abs(jump_right) >= 2.0_rt*std::abs(jump_left)) {
                al(i,j,k+1,ncomp) = a(i,j,k,ncomp) + 2.0_rt*(1.0_rt - rho)*jump_left + rho*jump_right;
              }

            }
          }

        } else {
          if (std::abs(jump_left) >= 2.0_rt*std::abs(jump_right)) {
            ar(i,j,k,ncomp) = a(i,j,k,ncomp) - 2.0_rt*jump_right;
          }
          if (std::abs(jump_right) >= 2.0_rt*std::abs(jump_left)) {
            al(i,j,k+1,ncomp) = a(i,j,k,ncomp) + 2.0_rt*jump_left;
          }
        }

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
