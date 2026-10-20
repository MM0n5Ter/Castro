#ifndef _ELLIPTIC_OPS_H_
#define _ELLIPTIC_OPS_H_

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

#include <radiation_helpers.H>

using namespace amrex;

namespace ELLIPTIC_OPS
{
    struct BoundaryFaceStencil {
        Real h;
        int icp, jcp, kcp;
        int iep, jep, kep;
    };

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    BoundaryFaceStencil boundary_face_stencil (int cdir, const Real* dx)
    {
        BoundaryFaceStencil nb{};

#if AMREX_SPACEDIM == 1
        if (cdir == 0) {
            nb.h = dx[0];
            nb.icp = -1;
        }
        else if (cdir == 1) {
            nb.h = dx[0];
            nb.icp = 1;
            nb.iep = 1;
        }
#elif AMREX_SPACEDIM == 2
        if (cdir == 0) {
            nb.h = dx[0];
            nb.icp = -1;
        }
        else if (cdir == 2) {
            nb.h = dx[0];
            nb.icp = 1;
            nb.iep = 1;
        }
        else if (cdir == 1) {
            nb.h = dx[1];
            nb.jcp = -1;
        }
        else if (cdir == 3) {
            nb.h = dx[1];
            nb.jcp = 1;
            nb.jep = 1;
        }
#else
        if (cdir == 0) {
            nb.h = dx[0];
            nb.icp = -1;
        }
        else if (cdir == 3) {
            nb.h = dx[0];
            nb.icp = 1;
            nb.iep = 1;
        }
        else if (cdir == 1) {
            nb.h = dx[1];
            nb.jcp = -1;
        }
        else if (cdir == 4) {
            nb.h = dx[1];
            nb.jcp = 1;
            nb.jep = 1;
        }
        else if (cdir == 2) {
            nb.h = dx[2];
            nb.kcp = -1;
        }
        else if (cdir == 5) {
            nb.h = dx[2];
            nb.kcp = 1;
            nb.kep = 1;
        }
#endif

        return nb;
    }

    AMREX_INLINE
    void face_bc_term (Array4<Real> const bc_term,
                 Array4<Real> const er,
                 const Box& reg,
                 int cdir, int bct, Real bcl,
                 Array4<Real const> const bcval,
                 Array4<int const> const mask,
                 Array4<Real const> const d,
                 const Real* dx)
    {
        const BoundaryFaceStencil nb = boundary_face_stencil(cdir, dx);

        amrex::LoopOnCpu(reg, [=] (int i, int j, int k) noexcept
        {
            if (mask.contains(i+nb.icp,j+nb.jcp,k+nb.kcp)) {
                if (mask(i+nb.icp,j+nb.jcp,k+nb.kcp) > 0) {
                    if (bct == AMREX_LO_DIRICHLET) {
                        Real d_sign = 1.0_rt;
                        if (nb.iep != 0 || nb.jep != 0 || nb.kep != 0) {
                            d_sign = -1.0_rt;
                        }
                        bc_term(i+nb.iep,j+nb.jep,k+nb.kep) = d(i+nb.iep,j+nb.jep,k+nb.kep) * d_sign * (er(i,j,k) - bcval(i+nb.icp,j+nb.jcp,k+nb.kcp)) / (0.5_rt * nb.h + bcl);
                    }
                }
            }
        });
    }

    AMREX_INLINE
    void face_bc_term_3d (Array4<Real> const bc_term,
                  Array4<Real const> const er,
                  const Box& reg,
                  int cdir, int bctype,
                  Array4<int const> const tf,
                  Real bcl,
                  Array4<Real const> const bcval,
                  Array4<int const> const mask,
                  Array4<Real const> const d,
                  const Real* const dx)
    {
        const BoundaryFaceStencil nb = boundary_face_stencil(cdir, dx);

        amrex::LoopOnCpu(reg, [=] (int i, int j, int k) noexcept
        {
            if (!mask.contains(i+nb.icp,j+nb.jcp,k+nb.kcp) || mask(i+nb.icp,j+nb.jcp,k+nb.kcp) <= 0) {
                return;
            }

            int bct;
            if (bctype == -1) {
                bct = tf(i+nb.icp,j+nb.jcp,k+nb.kcp);
            }
            else {
                bct = bctype;
            }
            if (bct == AMREX_LO_DIRICHLET) {
                Real d_sign = 1.0_rt;
                if (nb.iep != 0 || nb.jep != 0 || nb.kep != 0) {
                    d_sign = -1.0_rt;
                }
                bc_term(i+nb.iep,j+nb.jep,k+nb.kep) = d(i+nb.iep,j+nb.jep,k+nb.kep) * d_sign * (er(i,j,k) - bcval(i+nb.icp,j+nb.jcp,k+nb.kcp)) / (0.5_rt * nb.h + bcl);
            }
            else if (bct == AMREX_LO_NEUMANN && bcval(i+nb.icp,j+nb.jcp,k+nb.kcp) == 0.0_rt) {
                bc_term(i+nb.iep,j+nb.jep,k+nb.kep) = 0.0_rt;
            }
        });
    }

    AMREX_INLINE
    void face_flux (Array4<Real> const flux,
                Array4<Real> const er,
                const Box& reg,
                int cdir, int bct, int bho, Real bcl,
                Array4<Real const> const bcval,
                Array4<int const> const mask,
                Array4<Real const> const b,
                Real beta, const Real* dx, int bc_variant)
    {
        Real h;

        auto face_xlo = false;
        bool face_xhi = false;
        auto face_ylo = false;
        bool face_yhi = false;
        bool z_left = false;
        bool z_right = false;

#if AMREX_SPACEDIM == 1
        if (cdir == 0) {
            h = dx[0];
            face_xlo = true;
        }
        else if (cdir == 1) {
            h = dx[0];
            face_xhi = true;
        }
#elif AMREX_SPACEDIM == 2
        if (cdir == 0) {
            h = dx[0];
            face_xlo = true;
        }
        else if (cdir == 2) {
            h = dx[0];
            face_xhi = true;
        }
        else if (cdir == 1) {
            h = dx[1];
            face_ylo = true;
        }
        else if (cdir == 3) {
            h = dx[1];
            face_yhi = true;
        }
#else
        if (cdir == 0) {
            h = dx[0];
            face_xlo = true;
        }
        else if (cdir == 3) {
            h = dx[0];
            face_xhi = true;
        }
        else if (cdir == 1) {
            h = dx[1];
            face_ylo = true;
        }
        else if (cdir == 4) {
            h = dx[1];
            face_yhi = true;
        }
        else if (cdir == 2) {
            h = dx[2];
            z_left = true;
        }
        else if (cdir == 5) {
            h = dx[2];
            z_right = true;
        }
#endif

        Real bfv, bfm, stencil_weight_second;

        if (bct == AMREX_LO_DIRICHLET) {
            if (!(bho >= 1)) {
                bfv = beta / (0.5e0_rt * h + bcl);
                bfm = bfv;
            } else {
                Real h2 = 0.5e0_rt * h;
                Real th2 = 3.e0_rt * h2;
                bfv = 2.e0_rt * beta * h / ((bcl + h2) * (bcl + th2));
                bfm = (beta / h) * (th2 - bcl) / (bcl + h2);
                stencil_weight_second = (beta / h) * (bcl - h2) / (bcl + th2);
            }
        }
        else {
            amrex::Error("face_flux: unsupported boundary type");
        }

        if (bc_variant == 0) {
            bfv = 0.e0_rt;
        }

        int region_ilo = reg.loVect3d()[0];
        int region_ihi = reg.hiVect3d()[0];
        int region_jlo = reg.loVect3d()[1];
        int region_jhi = reg.hiVect3d()[1];
        int region_klo = reg.loVect3d()[2];
        int region_khi = reg.hiVect3d()[2];

        if (face_xlo) {
            int i = region_ilo;
            for (int k = region_klo; k <= region_khi; ++k) {
                for (int j = region_jlo; j <= region_jhi; ++j) {
                    if (mask(i-1,j,k) > 0) {
                        flux(i,j,k) = b(i,j,k) * (bfv * bcval(i-1,j,k) - bfm * er(i,j,k));
                        if (bho >= 1) {
                            flux(i,j,k) = flux(i,j,k) - b(i,j,k) * stencil_weight_second * er(i+1,j,k);
                        }
                    }
                }
            }
        }
        else if (face_xhi) {
            int i = region_ihi;
            for (int k = region_klo; k <= region_khi; ++k) {
                for (int j = region_jlo; j <= region_jhi; ++j) {
                    if (mask(i+1,j,k) > 0) {
                        flux(i+1,j,k) = -b(i+1,j,k) * (bfv * bcval(i+1,j,k) - bfm * er(i,j,k));
                        if (bho >= 1) {
                            flux(i+1,j,k) = flux(i+1,j,k) + b(i+1,j,k) * stencil_weight_second * er(i-1,j,k);
                        }
                    }
                }
            }
        }
        else if (face_ylo) {
            int j = region_jlo;
            for (int k = region_klo; k <= region_khi; ++k) {
                for (int i = region_ilo; i <= region_ihi; ++i) {
                    if (mask(i,j-1,k) > 0) {
                        flux(i,j,k) = b(i,j,k) * (bfv * bcval(i,j-1,k) - bfm * er(i,j,k));
                        if (bho >= 1) {
                            flux(i,j,k) = flux(i,j,k) - b(i,j,k) * stencil_weight_second * er(i,j+1,k);
                        }
                    }
                }
            }
        }
        else if (face_yhi) {
            int j = region_jhi;
            for (int k = region_klo; k <= region_khi; ++k) {
                for (int i = region_ilo; i <= region_ihi; ++i) {
                    if (mask(i,j+1,k) > 0) {
                        flux(i,j+1,k) = -b(i,j+1,k) * (bfv * bcval(i,j+1,k) - bfm * er(i,j,k));
                        if (bho >= 1) {
                            flux(i,j+1,k) = flux(i,j+1,k) + b(i,j+1,k) * stencil_weight_second * er(i,j-1,k);
                        }
                    }
                }
            }
        }
        else if (z_left) {
            int k = region_klo;
            for (int j = region_jlo; j <= region_jhi; ++j) {
                for (int i = region_ilo; i <= region_ihi; ++i) {
                    if (mask(i,j,k-1) > 0) {
                        flux(i,j,k) = b(i,j,k) * (bfv * bcval(i,j,k-1) - bfm * er(i,j,k));
                        if (bho >= 1) {
                            flux(i,j,k) = flux(i,j,k) - b(i,j,k) * stencil_weight_second * er(i,j,k+1);
                        }
                    }
                }
            }
        }
        else if (z_right) {
            int k = region_khi;
            for (int j = region_jlo; j <= region_jhi; ++j) {
                for (int i = region_ilo; i <= region_ihi; ++i) {
                    if (mask(i,j,k+1) > 0) {
                        flux(i,j,k+1) = -b(i,j,k+1) * (bfv * bcval(i,j,k+1) - bfm * er(i,j,k));
                        if (bho >= 1) {
                            flux(i,j,k+1) = flux(i,j,k+1) + b(i,j,k+1) * stencil_weight_second * er(i,j,k-1);
                        }
                    }
                }
            }
        }
        else {
            std::cout << "face_flux: impossible face orientation" << std::endl;
        }
    }

    AMREX_INLINE
    void face_flux_3d (Array4<Real> const flux,
                 Array4<Real const> const er,
                 const Box& reg,
                 int cdir, int bctype,
                 Array4<int const> const tf,
                 int bho, Real bcl,
                 Array4<Real const> const bcval,
                 Array4<int const> const mask,
                 Array4<Real const> const b,
                 Real beta, const Real* const dx, Real c,
                 const Orientation& ori,
                 const GeometryData& geomdata, int bc_variant,
                 Array4<Real const> const spa)
    {
        const BoundaryFaceStencil nb = boundary_face_stencil(cdir, dx);

        amrex::LoopOnCpu(reg, [=] (int i, int j, int k) noexcept
        {
            Real r;
            face_geometry_factors(i, j, k, reg.loVect()[0], reg.hiVect()[0], geomdata, ori.coordDir(), ori.isLow(), r);

            Real bfv, bfm, stencil_weight_second, h2, th2;
            int bct;

            if (!mask.contains(i+nb.icp,j+nb.jcp,k+nb.kcp) || mask(i+nb.icp,j+nb.jcp,k+nb.kcp) <= 0) {
                return;
            }

            if (bctype == -1) {
                bct = tf(i+nb.icp,j+nb.jcp,k+nb.kcp);
            }
            else {
                bct = bctype;
            }
            if (bct == AMREX_LO_DIRICHLET) {
                if (bho >= 1) {
                    h2 = 0.5e0_rt * nb.h;
                    th2 = 3.e0_rt * h2;
                    bfv = 2.e0_rt * beta * nb.h / ((bcl + h2) * (bcl + th2)) * b(i,j,k);
                    bfm = (beta / nb.h) * (th2 - bcl) / (bcl + h2)  * b(i,j,k);
                    stencil_weight_second = (beta / nb.h) * (bcl - h2) / (bcl + th2) * b(i,j,k);
                }
                else {
                    bfv = beta / (0.5e0_rt * nb.h + bcl) * b(i,j,k);
                    bfm = bfv;
                }
            }
            else if (bct == AMREX_LO_NEUMANN) {
                bfv  = beta * r;
                bfm  = 0.e0_rt;
                stencil_weight_second = 0.e0_rt;
            }
            else if (bct == AMREX_LO_MARSHAK) {
                bfv = 2.e0_rt * beta * r;
                if (bho >= 1) {
                    bfm  =  0.375e0_rt * c * bfv;
                    stencil_weight_second = -0.125e0_rt * c * bfv;
                }
                else {
                    bfm = 0.25e0_rt * c * bfv;
                }
            }
            else if (bct == AMREX_LO_SANCHEZ_POMRANING) {
                bfv = 2.e0_rt * beta * r;
                if (bho >= 1) {
                    bfm =   1.5e0_rt * spa(i,j,k) * c * bfv;
                    stencil_weight_second = -0.5e0_rt * spa(i,j,k) * c * bfv;
                }
                else {
                    bfm = spa(i,j,k) * c * bfv;
                }
            }
            else {
                amrex::Error("face_flux_3d: unsupported boundary type");
            }
            if (bc_variant == 0) {
                bfv = 0.e0_rt;
            }

            Real face_normal_sign = 1.0_rt;
            if (nb.iep != 0 || nb.jep != 0 || nb.kep != 0) {
                face_normal_sign = -1.0_rt;
            }

            flux(i+nb.iep,j+nb.jep,k+nb.kep) = face_normal_sign * (bfv * bcval(i+nb.icp,j+nb.jcp,k+nb.kcp) - bfm * er(i,j,k));

            if (bho >= 1) {
                Real df = -stencil_weight_second * er(i-nb.icp,j-nb.jcp,k-nb.kcp);
                flux(i+nb.iep,j+nb.jep,k+nb.kep) -= face_normal_sign * df;
            }
        });
    }
}

#endif
