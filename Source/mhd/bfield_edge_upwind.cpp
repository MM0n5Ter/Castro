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

  // take conservative interface states ul and ur and update them
  // with with the transverse flux difference (corner coupling) to
  // produce ul_out and ur_out
  //
  // This implements step 3 of the CTU algorithm.
  // the normal direction (for the interface states) is d1
  // the transverse direction (for the flux difference) is d2

  GpuArray<Real, 3> dx;
  for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      dx[i] = geom.CellSizeArray()[i];
  }
  for (int i = AMREX_SPACEDIM; i < 3; ++i) {
      dx[i] = 0.0_rt;
  }

  // cl and cr are the offsets to the indices for the conserved state fluxes
  // they will be offset in d2 to capture the flux difference

  int cl[3] = {};
  int cr[3] = {};

  // b is for indexing into the magnetic field that is in the d1
  // (normal) direction

  int b[3] = {};

  // for indexing the electric field

  int err[3] = {};
  int elr[3] = {};
  int erl[3] = {};
  int ell[3] = {};

  // update the state on interface direction d1 with the input flux in direction d2

  // for the flux difference, F_r - F_l, we need to shift the indices in the first flux (F_r)
  // in d2 to get a difference across the interface.  We also need to shift by a zone in d1
  // for the left interface.  cr(:) and cl(:) will hold these shifts.

  // the first term of the flux_dir2 subtraction is shifted by 1 on the direction d2
  cr[d2] = 1;

  // for the normal B component
  b[d2] = 1;

  // err will capture the right state in both transverse directions
  // (e.g. Ez_{i+1/2,j+1/2,k})
  err[d2] = 1;
  err[d3] = 1;

  // elr will capture the right state in the second transverse direction
  // (e.g. Ez_{i-1/2,j+1/2,k})
  elr[d3] = 1;

  // erl will capture the right state in the first transverse direction
  // (e.g. Ez_{i+1/2,j-1/2,k})
  erl[d2] = 1;

  // ell is the lower-left E in both directions
  // (e.g. Ez_{i-1/2,j-1/2,k})

  int sgn = levi_civita_sign(d1, d2, d3);
  Real third_dt_dx = dt/(3.0_rt * dx[d1]);

  // UBDIR1 corresponds to d1, and UBDIR2 to d2, UBDIR3 to d3
  int UBDIR1 = UBX + d1;
  int UBDIR2 = UBX + d2;
  int UBDIR3 = UBX + d3;

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    // first the conserved state

    // right interface (e.g. U_{i-1/2,j,k,R} or the "-" state)

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

    // now magnetic fields

    // right state on the interface (e.g. B_{i-1/2,j,k,R} or `-` notation)

    // d1 -- this is perpendicular to the face
    // e.g., for d1 = x and
    // d2 = y, this gets updated as
    //
    // Bx|y_{i-1/2,j,k,R} = Bx_{i-1/2,j,k) -
    //     dt/3dx (Ez_{i-1/2,j+1/2,k) - Ez_{i-1/2,j-1/2,k})
    //
    // we use b[] to captured the j+1/2 indexing into Ez

    utmp[UBDIR1] = ur(i,j,k,UBDIR1) - sgn * third_dt_dx *
      (Ed3(i+b[0],j+b[1],k+b[2]) - Ed3(i,j,k));

    // d3 -- this is in the plane of the face
    // e.g.,g for d1 = x, and d2 = y, this gets updated as
    //
    // Bz|y_{i-1/2,j,k,R} = Bz_{i-1/2,j,k} + 1/2 dt/3dx
    //     (Ex_{i,j+1/2,k+1/2} - Ex_{i,j-1/2,k+1/2} +
    //      Ex_{i,j+1/2,k-1/2} - Ex_{i,j-1/2,k-1/2})
    //
    // we use err(:) for the first E term, elr(:) for the
    // second, and erl(:) for the third

    utmp[UBDIR3] = ur(i,j,k,UBDIR3) + sgn * 0.5_rt * third_dt_dx *
      ((Ed1(i+err[0],j+err[1],k+err[2]) - Ed1(i+elr[0],j+elr[1],k+elr[2])) +
       (Ed1(i+erl[0],j+erl[1],k+erl[2]) - Ed1(i,j,k)));

    // the component pointing in the transverse update direction, d2, is unchanged
    utmp[UBDIR2] = ur(i,j,k,UBDIR2);

    ConservedToPrimitive(qtmp, utmp);

    for (int n = 0; n < NQ; n++) {
      right_state_out(i,j,k,n) = qtmp[n];
    }
  });


  // left interface (e.g., U_{i-1/2,j,k,L} or the "+" state)
  // note: this uses information one zone to the left in d1

  cl[d1] -= 1;
  cr[d1] -= 1;

  // The in-plane B component at B_{i-1/2,j,k,L} uses the information one zone to the left
  // in direction d1

  err[d1] -= 1;
  elr[d1] -= 1;
  erl[d1] -= 1;
  ell[d1] -= 1;

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    // conservative state

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

    // left state on the interface (e.g. B_{i-1/2,j,k,L} or `+`)

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

  // Final transverse flux corrections to the conservative state

  GpuArray<Real, 3> dx;
  for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      dx[i] = geom.CellSizeArray()[i];
  }
  for (int i = AMREX_SPACEDIM; i < 3; ++i) {
      dx[i] = 0.0_rt;
  }

  // c1l, c1r are for indexing flux_dir1 offsets, c2l, c2r are for flux_dir2

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

  // UBDIR1 corresponds to d1, and UBDIR2 to d2, UBDIR3 to d3
  int UBDIR = UBX + d;
  int UBDIR1 = UBX + d1;
  int UBDIR2 = UBX + d2;


  c1r[d1] = 1;  // add +1 to the d1 direction in the first flux_dir1 term of the subtraction
  c2r[d2] = 1;  // add +1 to the d2 direction in the first flux_dir2 term of the subtraction

  // for Ed
  err[d1] = 1;
  err[d2] = 1;

  erl[d1] = 1;

  elr[d2] = 1;

  // for Ed1
  edge1_right_right[d] = 1;
  edge1_right_right[d2] = 1;

  edge1_left_right[d2] = 1;

  edge1_right_left[d] = 1;

  // for Ed2
  edge2_right_right[d] = 1;
  edge2_right_right[d1] = 1;

  edge2_left_right[d1] = 1;

  edge2_right_left[d] = 1;

  // for the normal component of B
  a1[d2] = 1;  // shift on first term of Ed1 subtraction, in d2 direction
  a2[d1] = 1;  // shift on first term of Ed2 subtraction, in d1 direction


  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    // first the conservative state

    // right interface (e.g. U_{i-1/2,j,k,R} or the "-" state)

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

    // now the magnetic field

    // right state on the interface (e.g. B_{i-1/2,j,k,R} or `-` notation)

    // Bd -- this is perpendicular to the face.

    utmp[UBDIR] = ur(i,j,k,UBDIR) - sgn * hdtdx *
      ((Ed1(i+a1[0],j+a1[1],k+a1[2]) - Ed1(i,j,k)) -
       (Ed2(i+a2[0],j+a2[1],k+a2[2]) - Ed2(i,j,k)));

    // Bd1 -- this is one of the components of B in the plane of the face d

    utmp[UBDIR1] = ur(i,j,k,UBDIR1) + sgn * 0.5_rt * hdtdx *
      ((Ed(i+err[0],j+err[1],k+err[2]) - Ed(i+erl[0],j+erl[1],k+erl[2])) +
       (Ed(i+elr[0],j+elr[1],k+elr[2]) - Ed(i+ell[0],j+ell[1],k+ell[2])) -
       (Ed2(i+edge2_right_right[0],j+edge2_right_right[1],k+edge2_right_right[2]) - Ed2(i+edge2_left_right[0],j+edge2_left_right[1],k+edge2_left_right[2])) -
       (Ed2(i+edge2_right_left[0],j+edge2_right_left[1],k+edge2_right_left[2]) - Ed2(i+ell[0],j+ell[1],k+ell[2])));

    // Bd2 -- this is the other component of B in the plane of the face d

    utmp[UBDIR2] = ur(i,j,k,UBDIR2) - sgn * 0.5_rt * hdtdx *
      ((Ed(i+err[0],j+err[1],k+err[2]) - Ed(i+elr[0],j+elr[1],k+elr[2])) +
       (Ed(i+erl[0],j+erl[1],k+erl[2]) - Ed(i+ell[0],j+ell[1],k+ell[2])) -
       (Ed1(i+edge1_right_right[0],j+edge1_right_right[1],k+edge1_right_right[2]) - Ed1(i+edge1_left_right[0],j+edge1_left_right[1],k+edge1_left_right[2])) -
       (Ed1(i+edge1_right_left[0],j+edge1_right_left[1],k+edge1_right_left[2]) - Ed1(i+ell[0],j+ell[1],k+ell[2])));

    // convert to primitive

    ConservedToPrimitive(qtmp, utmp);

    for (int n = 0; n < NQ; n++) {
      right_state_out(i,j,k,n) = qtmp[n];
    }
  });


  // for the left state U components on the face dir, the flux
  // difference is in the zone to the left

    c1r[d] -= 1;
    c1l[d] -= 1;
    c2r[d] -= 1;
    c2l[d] -= 1;

    // left state on the interface (e.g., B_{i-1/2,j,k,L} or `+`)

    // The in-plane B component at B_{i-1/2,j,k,L} uses the information one zone to the left
    // in direction d1

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

    // left interface (e.g., U_{i+1/2,j,k,L} or the "+" state)

    Real utmp[NUM_STATE+3];
    Real qtmp[NQ];

    // first the conserved state

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

    // now the B fields

    // Bd -- this is perpendicular to the face (with sign fix)

    // this is the same face as the right state, so the update the identical
    utmp[UBDIR] = ul(i,j,k,UBDIR) - sgn * hdtdx *
      ((Ed1(i+a1[0],j+a1[1],k+a1[2]) - Ed1(i,j,k)) -
       (Ed2(i+a2[0],j+a2[1],k+a2[2]) - Ed2(i,j,k)));

    // Bd1 -- first component on face d

    utmp[UBDIR1] = ul(i,j,k,UBDIR1) + sgn * 0.5_rt * hdtdx *
      ((Ed(i+err[0],j+err[1],k+err[2]) - Ed(i+erl[0],j+erl[1],k+erl[2])) +
       (Ed(i+elr[0],j+elr[1],k+elr[2]) - Ed(i+ell[0],j+ell[1],k+ell[2])) -
       (Ed2(i+edge2_right_right[0],j+edge2_right_right[1],k+edge2_right_right[2]) - Ed2(i+edge2_left_right[0],j+edge2_left_right[1],k+edge2_left_right[2])) -
       (Ed2(i+edge2_right_left[0],j+edge2_right_left[1],k+edge2_right_left[2]) - Ed2(i+ell[0],j+ell[1],k+ell[2])));

    // Bd2 -- second component on face d

    utmp[UBDIR2] = ul(i,j,k,UBDIR2) - sgn * 0.5_rt * hdtdx *
      ((Ed(i+err[0],j+err[1],k+err[2]) - Ed(i+elr[0],j+elr[1],k+elr[2])) +
       (Ed(i+erl[0],j+erl[1],k+erl[2]) - Ed(i+ell[0],j+ell[1],k+ell[2])) -
       (Ed1(i+edge1_right_right[0],j+edge1_right_right[1],k+edge1_right_right[2]) - Ed1(i+edge1_left_right[0],j+edge1_left_right[1],k+edge1_left_right[2])) -
       (Ed1(i+edge1_right_left[0],j+edge1_right_left[1],k+edge1_right_left[2]) - Ed1(i+ell[0],j+ell[1],k+ell[2])));

    // convert to primitive

    ConservedToPrimitive(qtmp, utmp);

    for (int n = 0; n < NQ; n++) {
      left_state_out(i,j,k,n) = qtmp[n];
    }
  });

}
