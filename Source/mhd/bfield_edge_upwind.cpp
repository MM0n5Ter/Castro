#include <HydroSim.H>
#include <HydroSim_helpers.H>

#include <mhd_helpers.H>

using namespace amrex;

void
HydroSim::corner_coupling(const Box& bx,
                      Array4<Real> const& right_state_out,
                      Array4<Real> const& left_state_out,
                      Array4<Real const> const& ur,
                      Array4<Real const> const& ul,
                      Array4<Real const> const& flux_dir2,
                      Array4<Real const> const& Ed1,
                      Array4<Real const> const& Ed3,
                      const int d1, const int d2, const int d3,
                      const Real dt) {


  GpuArray<Real, 3> dx;
  for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      dx[i] = geom.CellSizeArray()[i];
  }
  for (int i = AMREX_SPACEDIM; i < 3; ++i) {
      dx[i] = 0.0_rt;
  }


  int cl[3] = {};
  int cr[3] = {};


  int b[3] = {};


  int err[3] = {};
  int elr[3] = {};
  int erl[3] = {};
  int ell[3] = {};



  cr[d2] = 1;

  b[d2] = 1;

  err[d2] = 1;
  err[d3] = 1;

  elr[d3] = 1;

  erl[d2] = 1;


  int sgn = levi_civita_sign(d1, d2, d3);
  Real third_dt_dx = dt/(3.0_rt * dx[d1]);

  int UBDIR1 = UBX + d1;
  int UBDIR2 = UBX + d2;
  int UBDIR3 = UBX + d3;

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {



    Real utmp[NUM_STATE+3];
    Real qtmp[NQ];

    for (int n = 0; n < NUM_STATE; n++) {
      if (n != UTEMP) {
        utmp[n] = ur(i,j,k,n) -
          third_dt_dx*(flux_dir2(i+cr[0],j+cr[1],k+cr[2],n) - flux_dir2(i,j,k,n));

      } else {
        utmp[n] = 0.0_rt;
      }
    }




    utmp[UBDIR1] = ur(i,j,k,UBDIR1) - sgn * third_dt_dx *
      (Ed3(i+b[0],j+b[1],k+b[2]) - Ed3(i,j,k));


    utmp[UBDIR3] = ur(i,j,k,UBDIR3) + sgn * 0.5_rt * third_dt_dx *
      ((Ed1(i+err[0],j+err[1],k+err[2]) - Ed1(i+elr[0],j+elr[1],k+elr[2])) +
       (Ed1(i+erl[0],j+erl[1],k+erl[2]) - Ed1(i,j,k)));

    utmp[UBDIR2] = ur(i,j,k,UBDIR2);

    ConservedToPrimitive(qtmp, utmp);

    for (int n = 0; n < NQ; n++) {
      right_state_out(i,j,k,n) = qtmp[n];
    }
  });



  cl[d1] -= 1;
  cr[d1] -= 1;


  err[d1] -= 1;
  elr[d1] -= 1;
  erl[d1] -= 1;
  ell[d1] -= 1;

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {


    Real utmp[NUM_STATE+3];
    Real qtmp[NQ];

    for (int n = 0; n < NUM_STATE; n++) {
      if (n != UTEMP) {
        utmp[n] = ul(i,j,k,n) -
          third_dt_dx*(flux_dir2(i+cr[0],j+cr[1],k+cr[2],n) -
                 flux_dir2(i+cl[0],j+cl[1],k+cl[2],n));

      } else {
        utmp[n] = 0.0_rt;
      }
    }


    utmp[UBDIR1] = ul(i,j,k,UBDIR1) - sgn * third_dt_dx *
      (Ed3(i+b[0],j+b[1],k+b[2]) - Ed3(i,j,k));

    utmp[UBDIR3] = ul(i,j,k,UBDIR3) + sgn * 0.5_rt * third_dt_dx *
      ((Ed1(i+err[0],j+err[1],k+err[2]) - Ed1(i+elr[0],j+elr[1],k+elr[2])) +
       (Ed1(i+erl[0],j+erl[1],k+erl[2]) - Ed1(i+ell[0],j+ell[1],k+ell[2])));

    utmp[UBDIR2] = ul(i,j,k,UBDIR2);

    ConservedToPrimitive(qtmp, utmp);

    for (int n = 0; n < NQ; n++) {
      left_state_out(i,j,k,n) = qtmp[n];
    }
  });
}


void
HydroSim::predict_half_step(const Box& bx,
                  Array4<Real> const& right_state_out,
                  Array4<Real> const& left_state_out,
                  Array4<Real const> const& ur,
                  Array4<Real const> const& ul,
                  Array4<Real const> const& flux_dir1,
                  Array4<Real const> const& flux_dir2,
                  Array4<Real const> const& Ed,
                  Array4<Real const> const& Ed1,
                  Array4<Real const> const& Ed2,
                  const int d, const int d1, const int d2, const Real dt) {


  GpuArray<Real, 3> dx;
  for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      dx[i] = geom.CellSizeArray()[i];
  }
  for (int i = AMREX_SPACEDIM; i < 3; ++i) {
      dx[i] = 0.0_rt;
  }


  int c1l[3] = {};
  int c1r[3] = {};
  int c2l[3] = {};
  int c2r[3] = {};

  int a1[3] = {};
  int a2[3] = {};

  int err[3] = {};
  int erl[3] = {};
  int elr[3] = {};
  int ell[3] = {};

  int edge1_right_right[3] = {};
  int edge1_right_left[3] = {};
  int edge1_left_right[3] = {};

  int edge2_right_right[3] = {};
  int edge2_right_left[3] = {};
  int edge2_left_right[3] = {};

  Real hdtdx = 0.5_rt * dt / dx[d];
  int sgn = -1 * levi_civita_sign(d, d1, d2);

  int UBDIR = UBX + d;
  int UBDIR1 = UBX + d1;
  int UBDIR2 = UBX + d2;


  c1r[d1] = 1;
  c2r[d2] = 1;

  err[d1] = 1;
  err[d2] = 1;

  erl[d1] = 1;

  elr[d2] = 1;

  edge1_right_right[d] = 1;
  edge1_right_right[d2] = 1;

  edge1_left_right[d2] = 1;

  edge1_right_left[d] = 1;

  edge2_right_right[d] = 1;
  edge2_right_right[d1] = 1;

  edge2_left_right[d1] = 1;

  edge2_right_left[d] = 1;

  a1[d2] = 1;
  a2[d1] = 1;


  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {



    Real utmp[NUM_STATE+3];
    Real qtmp[NQ];

    for (int n = 0; n < NUM_STATE; n++) {
      if (n != UTEMP) {
        utmp[n] = ur(i,j,k,n) -
          hdtdx * (flux_dir1(i+c1r[0],j+c1r[1],k+c1r[2],n) - flux_dir1(i,j,k,n)) -
          hdtdx * (flux_dir2(i+c2r[0],j+c2r[1],k+c2r[2],n) - flux_dir2(i,j,k,n));

      } else {
        utmp[n] = 0.0_rt;
      }
    }




    utmp[UBDIR] = ur(i,j,k,UBDIR) - sgn * hdtdx *
      ((Ed1(i+a1[0],j+a1[1],k+a1[2]) - Ed1(i,j,k)) -
       (Ed2(i+a2[0],j+a2[1],k+a2[2]) - Ed2(i,j,k)));


    utmp[UBDIR1] = ur(i,j,k,UBDIR1) + sgn * 0.5_rt * hdtdx *
      ((Ed(i+err[0],j+err[1],k+err[2]) - Ed(i+erl[0],j+erl[1],k+erl[2])) +
       (Ed(i+elr[0],j+elr[1],k+elr[2]) - Ed(i+ell[0],j+ell[1],k+ell[2])) -
       (Ed2(i+edge2_right_right[0],j+edge2_right_right[1],k+edge2_right_right[2]) - Ed2(i+edge2_left_right[0],j+edge2_left_right[1],k+edge2_left_right[2])) -
       (Ed2(i+edge2_right_left[0],j+edge2_right_left[1],k+edge2_right_left[2]) - Ed2(i+ell[0],j+ell[1],k+ell[2])));


    utmp[UBDIR2] = ur(i,j,k,UBDIR2) - sgn * 0.5_rt * hdtdx *
      ((Ed(i+err[0],j+err[1],k+err[2]) - Ed(i+elr[0],j+elr[1],k+elr[2])) +
       (Ed(i+erl[0],j+erl[1],k+erl[2]) - Ed(i+ell[0],j+ell[1],k+ell[2])) -
       (Ed1(i+edge1_right_right[0],j+edge1_right_right[1],k+edge1_right_right[2]) - Ed1(i+edge1_left_right[0],j+edge1_left_right[1],k+edge1_left_right[2])) -
       (Ed1(i+edge1_right_left[0],j+edge1_right_left[1],k+edge1_right_left[2]) - Ed1(i+ell[0],j+ell[1],k+ell[2])));


    ConservedToPrimitive(qtmp, utmp);

    for (int n = 0; n < NQ; n++) {
      right_state_out(i,j,k,n) = qtmp[n];
    }
  });



    c1r[d] -= 1;
    c1l[d] -= 1;
    c2r[d] -= 1;
    c2l[d] -= 1;



    err[d] -= 1;
    erl[d] -= 1;
    elr[d] -= 1;
    ell[d] -= 1;

    edge1_right_right[d] -= 1;
    edge1_right_left[d] -= 1;
    edge1_left_right[d] -= 1;

    edge2_right_right[d] -= 1;
    edge2_right_left[d] -= 1;
    edge2_left_right[d] -= 1;


  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {


    Real utmp[NUM_STATE+3];
    Real qtmp[NQ];


    for (int n = 0; n < NUM_STATE; n++) {
      if (n != UTEMP) {
        utmp[n] = ul(i,j,k,n) -
          hdtdx * (flux_dir1(i+c1r[0],j+c1r[1],k+c1r[2],n) -
                   flux_dir1(i+c1l[0],j+c1l[1],k+c1l[2],n)) -
          hdtdx * (flux_dir2(i+c2r[0],j+c2r[1],k+c2r[2],n) -
                   flux_dir2(i+c2l[0],j+c2l[1],k+c2l[2],n));

      } else {
        utmp[n] = 0.0_rt;
      }
    }



    utmp[UBDIR] = ul(i,j,k,UBDIR) - sgn * hdtdx *
      ((Ed1(i+a1[0],j+a1[1],k+a1[2]) - Ed1(i,j,k)) -
       (Ed2(i+a2[0],j+a2[1],k+a2[2]) - Ed2(i,j,k)));


    utmp[UBDIR1] = ul(i,j,k,UBDIR1) + sgn * 0.5_rt * hdtdx *
      ((Ed(i+err[0],j+err[1],k+err[2]) - Ed(i+erl[0],j+erl[1],k+erl[2])) +
       (Ed(i+elr[0],j+elr[1],k+elr[2]) - Ed(i+ell[0],j+ell[1],k+ell[2])) -
       (Ed2(i+edge2_right_right[0],j+edge2_right_right[1],k+edge2_right_right[2]) - Ed2(i+edge2_left_right[0],j+edge2_left_right[1],k+edge2_left_right[2])) -
       (Ed2(i+edge2_right_left[0],j+edge2_right_left[1],k+edge2_right_left[2]) - Ed2(i+ell[0],j+ell[1],k+ell[2])));


    utmp[UBDIR2] = ul(i,j,k,UBDIR2) - sgn * 0.5_rt * hdtdx *
      ((Ed(i+err[0],j+err[1],k+err[2]) - Ed(i+elr[0],j+elr[1],k+elr[2])) +
       (Ed(i+erl[0],j+erl[1],k+erl[2]) - Ed(i+ell[0],j+ell[1],k+ell[2])) -
       (Ed1(i+edge1_right_right[0],j+edge1_right_right[1],k+edge1_right_right[2]) - Ed1(i+edge1_left_right[0],j+edge1_left_right[1],k+edge1_left_right[2])) -
       (Ed1(i+edge1_right_left[0],j+edge1_right_left[1],k+edge1_right_left[2]) - Ed1(i+ell[0],j+ell[1],k+ell[2])));


    ConservedToPrimitive(qtmp, utmp);

    for (int n = 0; n < NQ; n++) {
      left_state_out(i,j,k,n) = qtmp[n];
    }
  });

}
