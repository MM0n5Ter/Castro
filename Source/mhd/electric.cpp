#include <HydroSim.H>

#include <mhd_helpers.H>

using namespace amrex;

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

