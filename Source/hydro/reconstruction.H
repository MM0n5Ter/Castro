#ifndef HYDROSIM_RECONSTRUCTION_H
#define HYDROSIM_RECONSTRUCTION_H

#include <AMReX_Array.H>

#include <state_indices.H>

namespace reconstruction {
   enum slope_indices : std::uint8_t {
       im2 = 0,
       im1,
       i0,
       ip1,
       ip2,
       nslp
   };

   // the zone values i-2 .. i+2 along one direction, indexed by slope_indices
   using ZoneStencil = amrex::GpuArray<amrex::Real, nslp>;

   // the stencils used to subtract hydrostatic equilibrium off of the
   // pressure before limiting: density, pressure, and the source in the
   // normal velocity equation (e.g. g)
   struct HydrostaticStencils {
       ZoneStencil rho;
       ZoneStencil p;
       ZoneStencil src;
   };
}

// Return the five-point stencil centered on (i,j,k) along direction
// idir, taking the zone values from value_at(ii,jj,kk).

template <typename ValueAt>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
reconstruction::ZoneStencil
fill_stencil(ValueAt value_at, const int idir,
             const int i, const int j, const int k) {

    using namespace reconstruction;

    ZoneStencil s;

    if (idir == 0) {
        s[im2] = value_at(i-2,j,k);
        s[im1] = value_at(i-1,j,k);
        s[i0]  = value_at(i,j,k);
        s[ip1] = value_at(i+1,j,k);
        s[ip2] = value_at(i+2,j,k);

    } else if (idir == 1) {
        s[im2] = value_at(i,j-2,k);
        s[im1] = value_at(i,j-1,k);
        s[i0]  = value_at(i,j,k);
        s[ip1] = value_at(i,j+1,k);
        s[ip2] = value_at(i,j+2,k);

    } else {
        s[im2] = value_at(i,j,k-2);
        s[im1] = value_at(i,j,k-1);
        s[i0]  = value_at(i,j,k);
        s[ip1] = value_at(i,j,k+1);
        s[ip2] = value_at(i,j,k+2);

    }

    return s;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
reconstruction::ZoneStencil
load_stencil(amrex::Array4<amrex::Real const> const& q_arr, const int idir,
             const int i, const int j, const int k, const int ncomp) {

    return fill_stencil([&] (int ii, int jj, int kk) { return q_arr(ii,jj,kk,ncomp); },
                        idir, i, j, k);

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
reconstruction::ZoneStencil
load_passive_stencil(amrex::Array4<amrex::Real const> const& U_arr,
                     amrex::Array4<amrex::Real const> const& rho_inv_arr,
                     const int idir,
                     const int i, const int j, const int k, const int ncomp) {

    // the passive is stored conserved; convert to primitive zone by zone

    return fill_stencil([&] (int ii, int jj, int kk) { return U_arr(ii,jj,kk,ncomp) * rho_inv_arr(ii,jj,kk); },
                        idir, i, j, k);

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
add_geometric_rho_source(amrex::Array4<amrex::Real const> const& q_arr,
                         amrex::Array4<amrex::Real const> const& dloga,
                         const int i, const int j, const int k,
                         const int ncomp, reconstruction::ZoneStencil& s) {

    using namespace reconstruction;

    // For idir == 0, i.e. r-direction:
    // this takes the form: -alpha rho u / r
    // where alpha = 1 for cylindrical and 2 for spherical
    // note dloga(idir==0) = alpha/r

    // For idir == 1,
    // i.e. theta-direction for spherical and z-direction for cylindrical
    // this takes the form: -rho v cot(theta) / r for spherical and 0 for cylindrical
    // note: dloga(idir==1) = cot(theta)/r for spherical and 0 for cylindrical.

    // ncomp should be QU for idir == 0 and QV for idir == 1

    if (ncomp == QU) {

        s[im2] += -dloga(i-2,j,k) * q_arr(i-2,j,k,QRHO) * q_arr(i-2,j,k,ncomp);
        s[im1] += -dloga(i-1,j,k) * q_arr(i-1,j,k,QRHO) * q_arr(i-1,j,k,ncomp);
        s[i0] += -dloga(i,j,k) * q_arr(i,j,k,QRHO) * q_arr(i,j,k,ncomp);
        s[ip1] += -dloga(i+1,j,k) * q_arr(i+1,j,k,QRHO) * q_arr(i+1,j,k,ncomp);
        s[ip2] += -dloga(i+2,j,k) * q_arr(i+2,j,k,QRHO) * q_arr(i+2,j,k,ncomp);

    } else if (ncomp == QV) {

        s[im2] += -dloga(i,j-2,k) * q_arr(i,j-2,k,QRHO) * q_arr(i,j-2,k,ncomp);
        s[im1] += -dloga(i,j-1,k) * q_arr(i,j-1,k,QRHO) * q_arr(i,j-1,k,ncomp);
        s[i0] += -dloga(i,j,k) * q_arr(i,j,k,QRHO) * q_arr(i,j,k,ncomp);
        s[ip1] += -dloga(i,j+1,k) * q_arr(i,j+1,k,QRHO) * q_arr(i,j+1,k,ncomp);
        s[ip2] += -dloga(i,j+2,k) * q_arr(i,j+2,k,QRHO) * q_arr(i,j+2,k,ncomp);

    }

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
add_geometric_rhoe_source(amrex::Array4<amrex::Real const> const& q_arr,
                          amrex::Array4<amrex::Real const> const& dloga,
                          const int i, const int j, const int k,
                          const int ncomp, reconstruction::ZoneStencil& s) {

    using namespace reconstruction;

    // For idir == 0, i.e. r-direction:
    // this takes the form: -alpha (rho e + p ) u / r
    // where alpha = 1 for cylindrical and 2 for spherical
    // note dloga(idir==0) = alpha/r

    // For idir == 1,
    // i.e. theta-direction for spherical and z-direction for cylindrical
    // this takes the form: - (rho e + p) v cot(theta) / r for spherical and 0 for cylindrical
    // note: dloga(idir==1) = cot(theta)/r for spherical and 0 for cylindrical.

    // ncomp should be QU for idir == 0 and QV for idir == 1

    if (ncomp == QU) {

        s[im2] += -dloga(i-2,j,k) * (q_arr(i-2,j,k,QRHOEI) + q_arr(i-2,j,k,QPRES)) * q_arr(i-2,j,k,ncomp);
        s[im1] += -dloga(i-1,j,k) * (q_arr(i-1,j,k,QRHOEI) + q_arr(i-1,j,k,QPRES)) * q_arr(i-1,j,k,ncomp);
        s[i0] += -dloga(i,j,k) * (q_arr(i,j,k,QRHOEI) + q_arr(i,j,k,QPRES)) * q_arr(i,j,k,ncomp);
        s[ip1] += -dloga(i+1,j,k) * (q_arr(i+1,j,k,QRHOEI) + q_arr(i+1,j,k,QPRES)) * q_arr(i+1,j,k,ncomp);
        s[ip2] += -dloga(i+2,j,k) * (q_arr(i+2,j,k,QRHOEI) + q_arr(i+2,j,k,QPRES)) * q_arr(i+2,j,k,ncomp);

    } else if (ncomp == QV) {

        s[im2] += -dloga(i,j-2,k) * (q_arr(i,j-2,k,QRHOEI) + q_arr(i,j-2,k,QPRES)) * q_arr(i,j-2,k,ncomp);
        s[im1] += -dloga(i,j-1,k) * (q_arr(i,j-1,k,QRHOEI) + q_arr(i,j-1,k,QPRES)) * q_arr(i,j-1,k,ncomp);
        s[i0] += -dloga(i,j,k) * (q_arr(i,j,k,QRHOEI) + q_arr(i,j,k,QPRES)) * q_arr(i,j,k,ncomp);
        s[ip1] += -dloga(i,j+1,k) * (q_arr(i,j+1,k,QRHOEI) + q_arr(i,j+1,k,QPRES)) * q_arr(i,j+1,k,ncomp);
        s[ip2] += -dloga(i,j+2,k) * (q_arr(i,j+2,k,QRHOEI) + q_arr(i,j+2,k,QPRES)) * q_arr(i,j+2,k,ncomp);

    }

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
add_geometric_p_source(amrex::Array4<amrex::Real const> const& q_arr,
                       amrex::Array4<amrex::Real const> const& qaux_arr,
                       amrex::Array4<amrex::Real const> const& dloga,
                       const int i, const int j, const int k,
                       const int ncomp, reconstruction::ZoneStencil& s) {

    using namespace reconstruction;

    // For idir == 0, i.e. r-direction:
    // this takes the form: -alpha Gamma1 p u / r
    // where alpha = 1 for cylindrical and 2 for spherical
    // note: dloga(idir==0) = alpha/r

    // For idir == 1,
    // i.e. theta-direction for spherical and z-direction for cylindrical
    // this takes the form: - Gamma1 p v cot(theta) / r for spherical and 0 for cylindrical
    // note: dloga(idir==1) = cot(theta)/r for spherical and 0 for cylindrical.

    // ncomp should be QU for idir == 0 and QV for idir == 1


    if (ncomp == QU) {

        s[im2] += -dloga(i-2,j,k) * q_arr(i-2,j,k,QPRES) * qaux_arr(i-2,j,k,QGAM1) * q_arr(i-2,j,k,ncomp);
        s[im1] += -dloga(i-1,j,k) * q_arr(i-1,j,k,QPRES) * qaux_arr(i-1,j,k,QGAM1) * q_arr(i-1,j,k,ncomp);
        s[i0] += -dloga(i,j,k) * q_arr(i,j,k,QPRES) * qaux_arr(i,j,k,QGAM1) * q_arr(i,j,k,ncomp);
        s[ip1] += -dloga(i+1,j,k) * q_arr(i+1,j,k,QPRES) * qaux_arr(i+1,j,k,QGAM1) * q_arr(i+1,j,k,ncomp);
        s[ip2] += -dloga(i+2,j,k) * q_arr(i+2,j,k,QPRES) * qaux_arr(i+2,j,k,QGAM1) * q_arr(i+2,j,k,ncomp);

    } else if (ncomp == QV) {

        s[im2] += -dloga(i,j-2,k) * q_arr(i,j-2,k,QPRES) * qaux_arr(i,j-2,k,QGAM1) * q_arr(i,j-2,k,ncomp);
        s[im1] += -dloga(i,j-1,k) * q_arr(i,j-1,k,QPRES) * qaux_arr(i,j-1,k,QGAM1) * q_arr(i,j-1,k,ncomp);
        s[i0] += -dloga(i,j,k) * q_arr(i,j,k,QPRES) * qaux_arr(i,j,k,QGAM1) * q_arr(i,j,k,ncomp);
        s[ip1] += -dloga(i,j+1,k) * q_arr(i,j+1,k,QPRES) * qaux_arr(i,j+1,k,QGAM1) * q_arr(i,j+1,k,ncomp);
        s[ip2] += -dloga(i,j+2,k) * q_arr(i,j+2,k,QPRES) * qaux_arr(i,j+2,k,QGAM1) * q_arr(i,j+2,k,ncomp);
    }

}


#endif
