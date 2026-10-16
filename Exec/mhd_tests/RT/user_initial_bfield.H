#ifndef user_initial_bfield_H
#define user_initial_bfield_H

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void user_initial_bfield (int i, int j, int k,
                                  Array4<Real> const& B, const int ibdir,
                                  const GeometryData& geomdata)
{


    if (ibdir == 0) {
        B(i,j,k) = problem::B_0;
    } else {
        B(i,j,k) = 0.0_rt;
    }

}
#endif
