#ifndef HDIFFOP_KERNELS_H
#define HDIFFOP_KERNELS_H

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

#include <rad_util.H>

using namespace amrex;

namespace HDIFFOP
{
    // hdop is Hypre abec, where abec is the form of the linear equation
    // we are solving:
    //
    // alpha*phi - div(beta*grad phi) + div(\vec{c}*phi)

    AMREX_INLINE
    void bc_face_flux (Array4<Real> const flux,
                Array4<Real> const er,
                const Box& reg,
                int cdir, int bct, int bho, Real bcl,
                Array4<Real const> const bcval,
                Array4<int const> const mask,
                Array4<Real const> const b,
                Real beta, const GpuArray<Real, AMREX_SPACEDIM>& dx, int inhom)
    {
        Real h;

        bool x_left = false;
        bool x_right = false;
        bool y_left = false;
        auto y_right = false;
        bool z_left = false;
        bool z_right = false;

#if AMREX_SPACEDIM == 1
        if (cdir == 0) {
            h = dx[0];
            x_left = true;
        }
        else if (cdir == 1) {
            h = dx[0];
            x_right = true;
        }
#elif AMREX_SPACEDIM == 2
        if (cdir == 0) {
            h = dx[0];
            x_left = true;
        }
        else if (cdir == 2) {
            h = dx[0];
            x_right = true;
        }
        else if (cdir == 1) {
            h = dx[1];
            y_left = true;
        }
        else if (cdir == 3) {
            h = dx[1];
            y_right = true;
        }
#else
        if (cdir == 0) {
            h = dx[0];
            x_left = true;
        }
        else if (cdir == 3) {
            h = dx[0];
            x_right = true;
        }
        else if (cdir == 1) {
            h = dx[1];
            y_left = true;
        }
        else if (cdir == 4) {
            h = dx[1];
            y_right = true;
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

        Real bc_coef_v, bc_coef_m, bc_coef_m2;

        if (bct == AMREX_LO_DIRICHLET) {
            if (bho >= 1) {
                Real h2 = 0.5e0_rt * h;
                Real th2 = 3.e0_rt * h2;
                bc_coef_v = 2.e0_rt * beta * h / ((bcl + h2) * (bcl + th2));
                bc_coef_m = (beta / h) * (th2 - bcl) / (bcl + h2);
                bc_coef_m2 = (beta / h) * (bcl - h2) / (bcl + th2);
            }
            else {
                bc_coef_v = beta / (0.5e0_rt * h + bcl);
                bc_coef_m = bc_coef_v;
            }
        }
        else {
            amrex::Error("bc_face_flux: unsupported boundary type");
        }

        if (inhom == 0) {
            bc_coef_v = 0.e0_rt;
        }

        int reg_ilo = reg.loVect3d()[0];
        int reg_ihi = reg.hiVect3d()[0];
        int reg_jlo = reg.loVect3d()[1];
        int reg_jhi = reg.hiVect3d()[1];
        int reg_klo = reg.loVect3d()[2];
        int reg_khi = reg.hiVect3d()[2];

        if (x_left) {
            int i = reg_ilo;
            for (int k = reg_klo; k <= reg_khi; ++k) {
                for (int j = reg_jlo; j <= reg_jhi; ++j) {
                    if (mask(i-1,j,k) > 0) {
                        flux(i,j,k) = b(i,j,k) * (bc_coef_v * bcval(i-1,j,k) - bc_coef_m * er(i,j,k));
                        if (bho >= 1) {
                            flux(i,j,k) = flux(i,j,k) - b(i,j,k) * bc_coef_m2 * er(i+1,j,k);
                        }
                    }
                }
            }
        }
        else if (x_right) {
            int i = reg_ihi;
            for (int k = reg_klo; k <= reg_khi; ++k) {
                for (int j = reg_jlo; j <= reg_jhi; ++j) {
                    if (mask(i+1,j,k) > 0) {
                        flux(i+1,j,k) = -b(i+1,j,k) * (bc_coef_v * bcval(i+1,j,k) - bc_coef_m * er(i,j,k));
                        if (bho >= 1) {
                            flux(i+1,j,k) = flux(i+1,j,k) + b(i+1,j,k) * bc_coef_m2 * er(i-1,j,k);
                        }
                    }
                }
            }
        }
        else if (y_left) {
            int j = reg_jlo;
            for (int k = reg_klo; k <= reg_khi; ++k) {
                for (int i = reg_ilo; i <= reg_ihi; ++i) {
                    if (mask(i,j-1,k) > 0) {
                        flux(i,j,k) = b(i,j,k) * (bc_coef_v * bcval(i,j-1,k) - bc_coef_m * er(i,j,k));
                        if (bho >= 1) {
                            flux(i,j,k) = flux(i,j,k) - b(i,j,k) * bc_coef_m2 * er(i,j+1,k);
                        }
                    }
                }
            }
        }
        else if (y_right) {
            int j = reg_jhi;
            for (int k = reg_klo; k <= reg_khi; ++k) {
                for (int i = reg_ilo; i <= reg_ihi; ++i) {
                    if (mask(i,j+1,k) > 0) {
                        flux(i,j+1,k) = -b(i,j+1,k) * (bc_coef_v * bcval(i,j+1,k) - bc_coef_m * er(i,j,k));
                        if (bho >= 1) {
                            flux(i,j+1,k) = flux(i,j+1,k) + b(i,j+1,k) * bc_coef_m2 * er(i,j-1,k);
                        }
                    }
                }
            }
        }
        else if (z_left) {
            int k = reg_klo;
            for (int j = reg_jlo; j <= reg_jhi; ++j) {
                for (int i = reg_ilo; i <= reg_ihi; ++i) {
                    if (mask(i,j,k-1) > 0) {
                        flux(i,j,k) = b(i,j,k) * (bc_coef_v * bcval(i,j,k-1) - bc_coef_m * er(i,j,k));
                        if (bho >= 1) {
                            flux(i,j,k) = flux(i,j,k) - b(i,j,k) * bc_coef_m2 * er(i,j,k+1);
                        }
                    }
                }
            }
        }
        else if (!(z_right)) {
            std::cout << "bc_face_flux: impossible face orientation" << std::endl;
        } else {
            int k = reg_khi;
            for (int j = reg_jlo; j <= reg_jhi; ++j) {
                for (int i = reg_ilo; i <= reg_ihi; ++i) {
                    if (mask(i,j,k+1) > 0) {
                        flux(i,j,k+1) = -b(i,j,k+1) * (bc_coef_v * bcval(i,j,k+1) - bc_coef_m * er(i,j,k));
                        if (bho >= 1) {
                            flux(i,j,k+1) = flux(i,j,k+1) + b(i,j,k+1) * bc_coef_m2 * er(i,j,k-1);
                        }
                    }
                }
            }
        }
    }

    AMREX_INLINE
    void bc_face_flux_typed (Array4<Real> const flux,
                 Array4<Real const> const er,
                 const Box& reg,
                 int cdir, int bctype,
                 Array4<int const> const tf,
                 int bho, Real bcl,
                 Array4<Real const> const bcval,
                 Array4<int const> const mask,
                 Array4<Real const> const b,
                 Real beta, const GpuArray<Real, AMREX_SPACEDIM>& dx, Real c,
                 const Orientation& ori,
                 const GeometryData& geomdata, int inhom,
                 Array4<Real const> const spa)
    {
        Real h;

        // Index shift for whichever edge we're checking.
        // Negative means we're looking at the lo edge,
        // positive means we're looking at the hi edge.
        // The first group is for cell centers, the second
        // group is for cell edges.

        int icp = 0;
        int jcp = 0;
        int kcp = 0;

        int iep = 0;
        int jep = 0;
        int kep = 0;

#if AMREX_SPACEDIM == 1
        if (cdir == 0) {
            h = dx[0];
            icp = -1;
        }
        else if (cdir == 1) {
            h = dx[0];
            icp = 1;
            iep = 1;
        }
#elif AMREX_SPACEDIM == 2
        if (cdir == 0) {
            h = dx[0];
            icp = -1;
        }
        else if (cdir == 2) {
            h = dx[0];
            icp = 1;
            iep = 1;
        }
        else if (cdir == 1) {
            h = dx[1];
            jcp = -1;
        }
        else if (cdir == 3) {
            h = dx[1];
            jcp = 1;
            jep = 1;
        }
#else
        if (cdir == 0) {
            h = dx[0];
            icp = -1;
        }
        else if (cdir == 3) {
            h = dx[0];
            icp = 1;
            iep = 1;
        }
        else if (cdir == 1) {
            h = dx[1];
            jcp = -1;
        }
        else if (cdir == 4) {
            h = dx[1];
            jcp = 1;
            jep = 1;
        }
        else if (cdir == 2) {
            h = dx[2];
            kcp = -1;
        }
        else if (cdir == 5) {
            h = dx[2];
            kcp = 1;
            kep = 1;
        }
#endif

        amrex::LoopOnCpu(reg, [=] (int i, int j, int k) noexcept
        {
            const Real r = face_metric(i, j, k, reg.loVect()[0], reg.hiVect()[0], geomdata, ori.coordDir(), ori.isLow());

            Real bc_coef_v, bc_coef_m, bc_coef_m2, h2, th2;
            int bct;

            if (mask.contains(i+icp,j+jcp,k+kcp)) {
                if (mask(i+icp,j+jcp,k+kcp) > 0) {
                    if (bctype == -1) {
                        bct = tf(i+icp,j+jcp,k+kcp);
                    }
                    else {
                        bct = bctype;
                    }
                    if (bct == AMREX_LO_DIRICHLET) {
                        if (bho >= 1) {
                            h2 = 0.5e0_rt * h;
                            th2 = 3.e0_rt * h2;
                            bc_coef_v = 2.e0_rt * beta * h / ((bcl + h2) * (bcl + th2)) * b(i,j,k);
                            bc_coef_m = (beta / h) * (th2 - bcl) / (bcl + h2)  * b(i,j,k);
                            bc_coef_m2 = (beta / h) * (bcl - h2) / (bcl + th2) * b(i,j,k);
                        }
                        else {
                            bc_coef_v = beta / (0.5e0_rt * h + bcl) * b(i,j,k);
                            bc_coef_m = bc_coef_v;
                        }
                    }
                    else if (bct == AMREX_LO_NEUMANN) {
                        bc_coef_v  = beta * r;
                        bc_coef_m  = 0.e0_rt;
                        bc_coef_m2 = 0.e0_rt;
                    }
                    else if (bct == AMREX_LO_MARSHAK) {
                        bc_coef_v = 2.e0_rt * beta * r;
                        if (bho >= 1) {
                            bc_coef_m  =  0.375e0_rt * c * bc_coef_v;
                            bc_coef_m2 = -0.125e0_rt * c * bc_coef_v;
                        }
                        else {
                            bc_coef_m = 0.25e0_rt * c * bc_coef_v;
                        }
                    }
                    else if (bct == AMREX_LO_SANCHEZ_POMRANING) {
                        bc_coef_v = 2.e0_rt * beta * r;
                        if (bho >= 1) {
                            bc_coef_m =   1.5e0_rt * spa(i,j,k) * c * bc_coef_v;
                            bc_coef_m2 = -0.5e0_rt * spa(i,j,k) * c * bc_coef_v;
                        }
                        else {
                            bc_coef_m = spa(i,j,k) * c * bc_coef_v;
                        }
                    }
                    else {
                        amrex::Error("bc_face_flux_typed: unsupported boundary type");
                    }
                    if (inhom == 0) {
                        bc_coef_v = 0.e0_rt;
                    }

                    Real flux_sign = 1.0_rt;
                    if (iep != 0 || jep != 0 || kep != 0) {
                        // right edge
                        flux_sign = -1.0_rt;
                    }

                    flux(i+iep,j+jep,k+kep) = flux_sign * (bc_coef_v * bcval(i+icp,j+jcp,k+kcp) - bc_coef_m * er(i,j,k));

                    if (bho >= 1) {
                        Real df = -bc_coef_m2 * er(i-icp,j-jcp,k-kcp);
                        flux(i+iep,j+jep,k+kep) -= flux_sign * df;
                    }
                }
            }
        });
    }

    AMREX_INLINE
    void bc_face_dterm (Array4<Real> const dterm,
                 Array4<Real> const er,
                 const Box& reg,
                 int cdir, int bct, Real bcl,
                 Array4<Real const> const bcval,
                 Array4<int const> const mask,
                 Array4<Real const> const d,
                 const GpuArray<Real, AMREX_SPACEDIM>& dx)
    {
        Real h;

        // Index shift for whichever edge we're checking.
        // Negative means we're looking at the lo edge,
        // positive means we're looking at the hi edge.
        // The first group is for cell centers, the second
        // group is for cell edges.

        int icp = 0;
        int jcp = 0;
        int kcp = 0;

        int iep = 0;
        int jep = 0;
        int kep = 0;

#if AMREX_SPACEDIM == 1
        if (cdir == 0) {
            h = dx[0];
            icp = -1;
        }
        else if (cdir == 1) {
            h = dx[0];
            icp = 1;
            iep = 1;
        }
#elif AMREX_SPACEDIM == 2
        if (cdir == 0) {
            h = dx[0];
            icp = -1;
        }
        else if (cdir == 2) {
            h = dx[0];
            icp = 1;
            iep = 1;
        }
        else if (cdir == 1) {
            h = dx[1];
            jcp = -1;
        }
        else if (cdir == 3) {
            h = dx[1];
            jcp = 1;
            jep = 1;
        }
#else
        if (cdir == 0) {
            h = dx[0];
            icp = -1;
        }
        else if (cdir == 3) {
            h = dx[0];
            icp = 1;
            iep = 1;
        }
        else if (cdir == 1) {
            h = dx[1];
            jcp = -1;
        }
        else if (cdir == 4) {
            h = dx[1];
            jcp = 1;
            jep = 1;
        }
        else if (cdir == 2) {
            h = dx[2];
            kcp = -1;
        }
        else if (cdir == 5) {
            h = dx[2];
            kcp = 1;
            kep = 1;
        }
#endif

        amrex::LoopOnCpu(reg, [=] (int i, int j, int k) noexcept
        {
            if (mask.contains(i+icp,j+jcp,k+kcp)) {
                if (mask(i+icp,j+jcp,k+kcp) > 0) {
                    if (bct == AMREX_LO_DIRICHLET) {
                        Real d_sign = 1.0_rt;
                        if (iep != 0 || jep != 0 || kep != 0) {
                            // right edge
                            d_sign = -1.0_rt;
                        }
                        dterm(i+iep,j+jep,k+kep) = d(i+iep,j+jep,k+kep) * d_sign * (er(i,j,k) - bcval(i+icp,j+jcp,k+kcp)) / (0.5_rt * h + bcl);
                    }
                }
            }
        });
    }

    AMREX_INLINE
    void bc_face_dterm_typed (Array4<Real> const dterm,
                  Array4<Real const> const er,
                  const Box& reg,
                  int cdir, int bctype,
                  Array4<int const> const tf,
                  Real bcl,
                  Array4<Real const> const bcval,
                  Array4<int const> const mask,
                  Array4<Real const> const d,
                  const GpuArray<Real, AMREX_SPACEDIM>& dx)
    {
        Real h;

        // Index shift for whichever edge we're checking.
        // Negative means we're looking at the lo edge,
        // positive means we're looking at the hi edge.
        // The first group is for cell centers, the second
        // group is for cell edges.

        int icp = 0;
        int jcp = 0;
        int kcp = 0;

        int iep = 0;
        int jep = 0;
        int kep = 0;

#if AMREX_SPACEDIM == 1
        if (cdir == 0) {
            h = dx[0];
            icp = -1;
        }
        else if (cdir == 1) {
            h = dx[0];
            icp = 1;
            iep = 1;
        }
#elif AMREX_SPACEDIM == 2
        if (cdir == 0) {
            h = dx[0];
            icp = -1;
        }
        else if (cdir == 2) {
            h = dx[0];
            icp = 1;
            iep = 1;
        }
        else if (cdir == 1) {
            h = dx[1];
            jcp = -1;
        }
        else if (cdir == 3) {
            h = dx[1];
            jcp = 1;
            jep = 1;
        }
#else
        if (cdir == 0) {
            h = dx[0];
            icp = -1;
        }
        else if (cdir == 3) {
            h = dx[0];
            icp = 1;
            iep = 1;
        }
        else if (cdir == 1) {
            h = dx[1];
            jcp = -1;
        }
        else if (cdir == 4) {
            h = dx[1];
            jcp = 1;
            jep = 1;
        }
        else if (cdir == 2) {
            h = dx[2];
            kcp = -1;
        }
        else if (cdir == 5) {
            h = dx[2];
            kcp = 1;
            kep = 1;
        }
#endif

        amrex::LoopOnCpu(reg, [=] (int i, int j, int k) noexcept
        {
            if (mask.contains(i+icp,j+jcp,k+kcp)) {
                if (mask(i+icp,j+jcp,k+kcp) > 0) {
                    int bct;
                    if (bctype == -1) {
                        bct = tf(i+icp,j+jcp,k+kcp);
                    }
                    else {
                        bct = bctype;
                    }
                    if (bct == AMREX_LO_DIRICHLET) {
                        Real d_sign = 1.0_rt;
                        if (iep != 0 || jep != 0 || kep != 0) {
                            // right edge
                            d_sign = -1.0_rt;
                        }
                        dterm(i+iep,j+jep,k+kep) = d(i+iep,j+jep,k+kep) * d_sign * (er(i,j,k) - bcval(i+icp,j+jcp,k+kcp)) / (0.5_rt * h + bcl);
                    }
                    else if (bct == AMREX_LO_NEUMANN && bcval(i+icp,j+jcp,k+kcp) == 0.0_rt) {
                        dterm(i+iep,j+jep,k+kep) = 0.0_rt;
                    }
                }
            }
        });
    }
} // namespace HDIFFOP

#endif
