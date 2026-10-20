#ifndef HYDROSIM_RECONSTRUCTION_H
#define HYDROSIM_RECONSTRUCTION_H

#include <state_indices.H>
#include <cmath>

#include <AMReX_REAL.H>

#include <reconstruction.H>

namespace reconstruction {
   enum stencil_position : std::uint8_t {
       im2 = 0,
       im1,
       i0,
       ip1,
       ip2,
       nslp
   };
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
gather_stencil(amrex::Array4<amrex::Real const> const& prim_arr, const int idir,
             const int i, const int j, const int k, const int ncomp,
             amrex::Real* s) {

    using namespace reconstruction;

    if (idir == 0) {
        s[im2] = prim_arr(i-2,j,k,ncomp);
        s[im1] = prim_arr(i-1,j,k,ncomp);
        s[i0]  = prim_arr(i,j,k,ncomp);
        s[ip1] = prim_arr(i+1,j,k,ncomp);
        s[ip2] = prim_arr(i+2,j,k,ncomp);

      } else if (idir == 1) {
        s[im2] = prim_arr(i,j-2,k,ncomp);
        s[im1] = prim_arr(i,j-1,k,ncomp);
        s[i0]  = prim_arr(i,j,k,ncomp);
        s[ip1] = prim_arr(i,j+1,k,ncomp);
        s[ip2] = prim_arr(i,j+2,k,ncomp);

      } else {
        s[im2] = prim_arr(i,j,k-2,ncomp);
        s[im1] = prim_arr(i,j,k-1,ncomp);
        s[i0]  = prim_arr(i,j,k,ncomp);
        s[ip1] = prim_arr(i,j,k+1,ncomp);
        s[ip2] = prim_arr(i,j,k+2,ncomp);

      }

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
fill_passive_stencil(amrex::Array4<amrex::Real const> const& field_arr,
                     amrex::Array4<amrex::Real const> const& inv_density_arr,
                     const int idir,
                     const int i, const int j, const int k, const int ncomp,
                     amrex::Real* s) {

    using namespace reconstruction;

    if (idir == 0) {
        s[im2] = field_arr(i-2,j,k,ncomp) * inv_density_arr(i-2,j,k);
        s[im1] = field_arr(i-1,j,k,ncomp) * inv_density_arr(i-1,j,k);
        s[i0]  = field_arr(i,j,k,ncomp) * inv_density_arr(i,j,k);
        s[ip1] = field_arr(i+1,j,k,ncomp) * inv_density_arr(i+1,j,k);
        s[ip2] = field_arr(i+2,j,k,ncomp) * inv_density_arr(i+2,j,k);

      } else if (idir == 1) {
        s[im2] = field_arr(i,j-2,k,ncomp) * inv_density_arr(i,j-2,k);
        s[im1] = field_arr(i,j-1,k,ncomp) * inv_density_arr(i,j-1,k);
        s[i0]  = field_arr(i,j,k,ncomp) * inv_density_arr(i,j,k);
        s[ip1] = field_arr(i,j+1,k,ncomp) * inv_density_arr(i,j+1,k);
        s[ip2] = field_arr(i,j+2,k,ncomp) * inv_density_arr(i,j+2,k);

      } else {
        s[im2] = field_arr(i,j,k-2,ncomp) * inv_density_arr(i,j,k-2);
        s[im1] = field_arr(i,j,k-1,ncomp) * inv_density_arr(i,j,k-1);
        s[i0]  = field_arr(i,j,k,ncomp) * inv_density_arr(i,j,k);
        s[ip1] = field_arr(i,j,k+1,ncomp) * inv_density_arr(i,j,k+1);
        s[ip2] = field_arr(i,j,k+2,ncomp) * inv_density_arr(i,j,k+2);

      }

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
add_geometric_density_source(amrex::Array4<amrex::Real const> const& prim_arr,
                         amrex::Array4<amrex::Real const> const& dloga,
                         const int i, const int j, const int k,
                         const int ncomp, amrex::Real* s) {

    using namespace reconstruction;




    if (ncomp == QU) {

        s[im2] += -dloga(i-2,j,k) * prim_arr(i-2,j,k,QRHO) * prim_arr(i-2,j,k,ncomp);
        s[im1] += -dloga(i-1,j,k) * prim_arr(i-1,j,k,QRHO) * prim_arr(i-1,j,k,ncomp);
        s[i0] += -dloga(i,j,k) * prim_arr(i,j,k,QRHO) * prim_arr(i,j,k,ncomp);
        s[ip1] += -dloga(i+1,j,k) * prim_arr(i+1,j,k,QRHO) * prim_arr(i+1,j,k,ncomp);
        s[ip2] += -dloga(i+2,j,k) * prim_arr(i+2,j,k,QRHO) * prim_arr(i+2,j,k,ncomp);

    } else if (ncomp == QV) {

        s[im2] += -dloga(i,j-2,k) * prim_arr(i,j-2,k,QRHO) * prim_arr(i,j-2,k,ncomp);
        s[im1] += -dloga(i,j-1,k) * prim_arr(i,j-1,k,QRHO) * prim_arr(i,j-1,k,ncomp);
        s[i0] += -dloga(i,j,k) * prim_arr(i,j,k,QRHO) * prim_arr(i,j,k,ncomp);
        s[ip1] += -dloga(i,j+1,k) * prim_arr(i,j+1,k,QRHO) * prim_arr(i,j+1,k,ncomp);
        s[ip2] += -dloga(i,j+2,k) * prim_arr(i,j+2,k,QRHO) * prim_arr(i,j+2,k,ncomp);

    }

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
add_geometric_energy_source(amrex::Array4<amrex::Real const> const& prim_arr,
                          amrex::Array4<amrex::Real const> const& dloga,
                          const int i, const int j, const int k,
                          const int ncomp, amrex::Real* s) {

    using namespace reconstruction;




    if (ncomp == QU) {

        s[im2] += -dloga(i-2,j,k) * (prim_arr(i-2,j,k,QEINTDENS) + prim_arr(i-2,j,k,QPRES)) * prim_arr(i-2,j,k,ncomp);
        s[im1] += -dloga(i-1,j,k) * (prim_arr(i-1,j,k,QEINTDENS) + prim_arr(i-1,j,k,QPRES)) * prim_arr(i-1,j,k,ncomp);
        s[i0] += -dloga(i,j,k) * (prim_arr(i,j,k,QEINTDENS) + prim_arr(i,j,k,QPRES)) * prim_arr(i,j,k,ncomp);
        s[ip1] += -dloga(i+1,j,k) * (prim_arr(i+1,j,k,QEINTDENS) + prim_arr(i+1,j,k,QPRES)) * prim_arr(i+1,j,k,ncomp);
        s[ip2] += -dloga(i+2,j,k) * (prim_arr(i+2,j,k,QEINTDENS) + prim_arr(i+2,j,k,QPRES)) * prim_arr(i+2,j,k,ncomp);

    } else if (ncomp == QV) {

        s[im2] += -dloga(i,j-2,k) * (prim_arr(i,j-2,k,QEINTDENS) + prim_arr(i,j-2,k,QPRES)) * prim_arr(i,j-2,k,ncomp);
        s[im1] += -dloga(i,j-1,k) * (prim_arr(i,j-1,k,QEINTDENS) + prim_arr(i,j-1,k,QPRES)) * prim_arr(i,j-1,k,ncomp);
        s[i0] += -dloga(i,j,k) * (prim_arr(i,j,k,QEINTDENS) + prim_arr(i,j,k,QPRES)) * prim_arr(i,j,k,ncomp);
        s[ip1] += -dloga(i,j+1,k) * (prim_arr(i,j+1,k,QEINTDENS) + prim_arr(i,j+1,k,QPRES)) * prim_arr(i,j+1,k,ncomp);
        s[ip2] += -dloga(i,j+2,k) * (prim_arr(i,j+2,k,QEINTDENS) + prim_arr(i,j+2,k,QPRES)) * prim_arr(i,j+2,k,ncomp);

    }

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
add_geometric_pressure_source(amrex::Array4<amrex::Real const> const& prim_arr,
                       amrex::Array4<amrex::Real const> const& qthermo_arr,
                       amrex::Array4<amrex::Real const> const& dloga,
                       const int i, const int j, const int k,
                       const int ncomp, amrex::Real* s) {

    using namespace reconstruction;





    if (ncomp == QU) {

        s[im2] += -dloga(i-2,j,k) * prim_arr(i-2,j,k,QPRES) * qthermo_arr(i-2,j,k,QGAMMA1) * prim_arr(i-2,j,k,ncomp);
        s[im1] += -dloga(i-1,j,k) * prim_arr(i-1,j,k,QPRES) * qthermo_arr(i-1,j,k,QGAMMA1) * prim_arr(i-1,j,k,ncomp);
        s[i0] += -dloga(i,j,k) * prim_arr(i,j,k,QPRES) * qthermo_arr(i,j,k,QGAMMA1) * prim_arr(i,j,k,ncomp);
        s[ip1] += -dloga(i+1,j,k) * prim_arr(i+1,j,k,QPRES) * qthermo_arr(i+1,j,k,QGAMMA1) * prim_arr(i+1,j,k,ncomp);
        s[ip2] += -dloga(i+2,j,k) * prim_arr(i+2,j,k,QPRES) * qthermo_arr(i+2,j,k,QGAMMA1) * prim_arr(i+2,j,k,ncomp);

    } else if (ncomp == QV) {

        s[im2] += -dloga(i,j-2,k) * prim_arr(i,j-2,k,QPRES) * qthermo_arr(i,j-2,k,QGAMMA1) * prim_arr(i,j-2,k,ncomp);
        s[im1] += -dloga(i,j-1,k) * prim_arr(i,j-1,k,QPRES) * qthermo_arr(i,j-1,k,QGAMMA1) * prim_arr(i,j-1,k,ncomp);
        s[i0] += -dloga(i,j,k) * prim_arr(i,j,k,QPRES) * qthermo_arr(i,j,k,QGAMMA1) * prim_arr(i,j,k,ncomp);
        s[ip1] += -dloga(i,j+1,k) * prim_arr(i,j+1,k,QPRES) * qthermo_arr(i,j+1,k,QGAMMA1) * prim_arr(i,j+1,k,ncomp);
        s[ip2] += -dloga(i,j+2,k) * prim_arr(i,j+2,k,QPRES) * qthermo_arr(i,j+2,k,QGAMMA1) * prim_arr(i,j+2,k,ncomp);
    }

}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int
source_trace_needed(amrex::Array4<amrex::Real const> const& prim_source, const int idir,
                   const int i, const int j, const int k, const int ncomp) {

    int trace_enabled = 0;
    if (idir == 0) {
        for (int b = i-2; b <= i+2; b++) {
            if (std::abs(prim_source(b,j,k,ncomp)) > 0.0_rt) {
                trace_enabled = 1;
                break;
            }
        }
    } else if (idir == 1) {
        for (int b = j-2; b <= j+2; b++) {
            if (std::abs(prim_source(i,b,k,ncomp)) > 0.0_rt) {
                trace_enabled = 1;
                break;
            }
        }
    } else {
        for (int b = k-2; b <= k+2; b++) {
            if (std::abs(prim_source(i,j,b,ncomp)) > 0.0_rt) {
                trace_enabled = 1;
                break;
            }
        }
    }

    return trace_enabled;
}


#endif
