#include <HydroSim.H>

#include <mhd_helpers.H>
#include <HydroSim_helpers.H>

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

void
HydroSim::edge_emf_z(const Box& bx,
                        Array4<Real const> const& prim_arr,
                        Array4<Real> const& E,
                        Array4<Real const> const& flxx,
                        Array4<Real const> const& flxy) {


  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real zone_prim[NQ];




    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i-1,j-1,k,n);
    }
    Real EmfCenter = 0.0_rt;
    electric(zone_prim, EmfCenter, 2);
    Real a = 2.0_rt * (-flxx(i,j-1,k,UBY) - EmfCenter);


    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i-1,j,k,n);
    }
    electric(zone_prim, EmfCenter, 2);
    Real b = 2.0_rt * (-flxx(i,j,k,UBY) - EmfCenter);


    Real d1 = 0.0_rt;
    if ( flxy(i-1,j,k,URHO) > 0.0_rt) {
      d1 = a;
    } else if (flxy(i-1,j,k,URHO) < 0.0_rt) {
      d1 = b;
    } else {
      d1 = 0.50_rt * (a + b);
    }



    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j-1,k,n);
    }
    electric(zone_prim, EmfCenter, 2);
    a = 2.0_rt * (EmfCenter + flxx(i,j-1,k,UBY));


    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j,k,n);
    }
    electric(zone_prim, EmfCenter, 2);
    b = 2.0_rt * (EmfCenter + flxx(i,j,k,UBY));


    Real d2 = 0.0_rt;
    if (flxy(i,j,k,URHO) > 0.0_rt) {
      d2 = a;
    } else if (flxy(i,j,k,URHO) < 0.0_rt) {
      d2 = b;
    } else {
      d2 = 0.5_rt * (a + b);
    }


    Real dd1 = 0.125_rt * (d1 - d2);




    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i-1,j-1,k,n);
    }
    electric(zone_prim, EmfCenter, 2);
    a = 2.0_rt * (flxy(i-1,j,k,UBX) - EmfCenter);


    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j-1,k,n);
    }
    electric(zone_prim, EmfCenter, 2);
    b = 2.0_rt * (flxy(i,j,k,UBX) - EmfCenter);


    if (flxx(i,j-1,k,URHO) > 0.0_rt) {
      d1 = a;
    } else if (flxx(i,j-1,k,URHO) < 0.0_rt) {
      d1 = b;
    } else {
      d1 = 0.5_rt * (a + b);
    }



    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i-1,j,k,n);
    }
    electric(zone_prim, EmfCenter, 2);
    a = 2.0_rt * (EmfCenter - flxy(i-1,j,k,UBX));


    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j,k,n);
    }
    electric(zone_prim, EmfCenter, 2);
    b = 2.0_rt * (EmfCenter - flxy(i,j,k,UBX));


    if (flxx(i,j,k,URHO) > 0.0_rt) {
      d2 = a;
    } else if (flxx(i,j,k,URHO) < 0.0_rt) {
      d2 = b;
    } else {
      d2 = 0.5_rt * (a + b);
    }


    Real dd2 = 0.125_rt * (d1 - d2);


    E(i,j,k) = 0.25_rt * (-flxx(i,j,k,UBY) - flxx(i,j-1,k,UBY) +
                          flxy(i-1,j,k,UBX) + flxy(i,j,k,UBX)) + dd1 + dd2;

  });
}

void
HydroSim::edge_emf_y(const Box& bx,
                        Array4<Real const> const& prim_arr,
                        Array4<Real> const& E,
                        Array4<Real const> const& flxx,
                        Array4<Real const> const& flxz) {


  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real zone_prim[NQ];




    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i-1,j,k-1,n);
    }
    Real EmfCenter = 0.0_rt;
    electric(zone_prim, EmfCenter, 1);
    Real a = 2.0_rt * (-flxz(i-1,j,k,UBX) - EmfCenter);


    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j,k-1,n);
    }
    electric(zone_prim, EmfCenter, 1);
    Real b = 2.0_rt * (-flxz(i,j,k,UBX) - EmfCenter);


    Real d1 = 0.0;
    if (flxx(i,j,k-1,URHO) > 0.0_rt) {
      d1 = a;
    } else if (flxx(i,j,k-1,URHO) < 0.0_rt) {
      d1 = b;
    } else {
      d1 = 0.5_rt * (a + b);
    }



    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i-1,j,k,n);
    }
    electric(zone_prim, EmfCenter, 1);
    a = 2.0_rt * (EmfCenter + flxz(i-1,j,k,UBX));


    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j,k,n);
    }
    electric(zone_prim, EmfCenter, 1);
    b = 2.0_rt * (EmfCenter + flxz(i,j,k,UBX));


    Real d2 = 0.0;
    if (flxx(i,j,k,URHO) > 0.0_rt) {
      d2 = a;
    } else if (flxx(i,j,k,URHO) < 0.0_rt) {
      d2 = b;
    } else {
      d2 = 0.5_rt * (a + b);
    }


    Real dd1 = 0.125_rt * (d1 - d2);




    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i-1,j,k-1,n);
    }
    electric(zone_prim, EmfCenter, 1);
    a = 2.0_rt * (flxx(i,j,k-1,UBZ) - EmfCenter);


    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i-1,j,k,n);
    }
    electric(zone_prim, EmfCenter, 1);
    b = 2.0_rt * (flxx(i,j,k,UBZ) - EmfCenter);


    if (flxz(i-1,j,k,URHO) > 0.0_rt) {
      d1 = a;
    } else if (flxz(i-1,j,k,URHO) < 0.0_rt) {
      d1 = b;
    } else {
      d1 = 0.5_rt * (a + b);
    }



    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j,k-1,n);
    }
    electric(zone_prim, EmfCenter, 1);
    a = 2.0_rt * (EmfCenter - flxx(i,j,k-1,UBZ));


    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j,k,n);
    }
    electric(zone_prim, EmfCenter, 1);
    b = 2.0_rt * (EmfCenter - flxx(i,j,k,UBZ));


    if (flxz(i,j,k,URHO) > 0.0_rt) {
      d2 = a;
    } else if (flxz(i,j,k,URHO) < 0.0_rt) {
      d2 = b;
    } else {
      d2 = 0.5_rt * (a + b);
    }


    Real dd2 = 0.125_rt * (d1 - d2);


    E(i,j,k) = 0.25_rt * (-flxz(i,j,k,UBX) - flxz(i-1,j,k,UBX) +
                          flxx(i,j,k-1,UBZ) + flxx(i,j,k,UBZ)) + dd1 + dd2;

  });
}

void
HydroSim::edge_emf_x(const Box& bx,
                        Array4<Real const> const& prim_arr,
                        Array4<Real> const& E,
                        Array4<Real const> const& flxy,
                        Array4<Real const> const& flxz) {


  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real zone_prim[NQ];




    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j-1,k-1,n);
    }
    Real EmfCenter = 0.0_rt;
    electric(zone_prim, EmfCenter, 0);
    Real a = 2.0_rt * (-flxy(i,j,k-1,UBZ) - EmfCenter);


    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j-1,k,n);
    }
    electric(zone_prim, EmfCenter, 0);
    Real b = 2.0_rt *(-flxy(i,j,k,UBZ) - EmfCenter);


    Real d1 = 0.0;
    if (flxz(i,j-1,k,URHO) > 0.0_rt) {
      d1 = a;
    } else if (flxz(i,j-1,k,URHO) < 0.0_rt) {
      d1 = b;
    } else {
      d1 = 0.5_rt * (a + b);
    }



    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j,k-1,n);
    }
    electric(zone_prim, EmfCenter, 0);
    a = 2.0_rt * (EmfCenter + flxy(i,j,k-1,UBZ));


    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j,k,n);
    }
    electric(zone_prim, EmfCenter, 0);
    b = 2.0_rt * (EmfCenter + flxy(i,j,k,UBZ));


    Real d2 = 0.0;
    if (flxz(i,j,k,URHO) > 0.0_rt) {
      d2 = a;
    } else if (flxz(i,j,k,URHO) < 0.0_rt) {
      d2 = b;
    } else {
      d2 = 0.5_rt * (a + b);
    }


    Real dd1 = 0.125_rt * (d1 - d2);




    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j-1,k-1,n);
    }
    electric(zone_prim, EmfCenter, 0);
    a = 2.0_rt * (flxz(i,j-1,k,UBY) - EmfCenter);


    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j,k-1,n);
    }
    electric(zone_prim, EmfCenter, 0);
    b = 2.0_rt * (flxz(i,j,k,UBY) - EmfCenter);


    if (flxy(i,j,k-1,URHO) > 0.0_rt) {
      d1 = a;
    } else if (flxy(i,j,k-1,URHO) < 0.0_rt) {
      d1 = b;
    } else {
      d1 = 0.5_rt * (a + b);
    }



    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j-1,k,n);
    }
    electric(zone_prim, EmfCenter, 0);
    a = 2.0_rt * (EmfCenter - flxz(i,j-1,k,UBY));


    for (int n = 0; n < NQ; n++) {
      zone_prim[n] = prim_arr(i,j,k,n);
    }
    electric(zone_prim, EmfCenter, 0);
    b = 2.0_rt * (EmfCenter - flxz(i,j,k,UBY));


    if (flxy(i,j,k,URHO) > 0.0_rt) {
      d2 = a;
    } else if (flxy(i,j,k,URHO) < 0.0_rt) {
      d2 = b;
    } else {
      d2 = 0.5_rt * (a + b);
    }


    Real dd2 = 0.125_rt * (d1 - d2);


    E(i,j,k) = 0.25_rt * (-flxy(i,j,k,UBZ) - flxy(i,j,k-1,UBZ) +
                          flxz(i,j-1,k,UBY) + flxz(i,j,k,UBY)) + dd1 + dd2;

  });
}
