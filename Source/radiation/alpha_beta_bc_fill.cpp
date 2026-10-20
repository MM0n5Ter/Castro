#include <AMReX_ParmParse.H>
#include <AMReX_LO_BCTYPES.H>

#include <HypreSolvers.H>
#include <elliptic_face_ops.H>
#include <radiation_helpers.H>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <_hypre_struct_mv.h>

using namespace amrex;

void AlphaBetaSolver::fillBcMat (const Box& bx,
                       int cdir, int bct, Real bcl,
                       Array4<int const> const& mask,
                       Array4<Real const> const& b,
                       Real beta, const Real* dx,
                       Array4<GpuArray<Real, AMREX_SPACEDIM+1>> const& mat)
{
    bool xlo = false;
    bool ylo = false;
    bool zlo = false;

    bool xhi = false;
    bool yhi = false;
    bool zhi = false;

    Real h;

    if (AMREX_SPACEDIM == 1) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            xhi = true;
            h = dx[0];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else if (AMREX_SPACEDIM == 2) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 2) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 3) {
            yhi = true;
            h = dx[1];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 3) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 4) {
            yhi = true;
            h = dx[1];
        }
        else if (cdir == 2) {
            zlo = true;
            h = dx[2];
        }
        else if (cdir == 5) {
            zhi = true;
            h = dx[2];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }

    const Real fac = beta / (h * h);;

    Real bfv, bfm;

    if (bct == AMREX_LO_DIRICHLET) {
        bfv = fac * h / (0.5e0_rt * h + bcl);
        bfm = bfv - fac;
    }
    else if (bct == AMREX_LO_NEUMANN) {
        bfv = beta / h;
        bfm = -fac;
    }
    else {
        amrex::Error("fillBcMat: unsupported boundary type");
    }

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        if (mask.contains(i-1,j,k)) {

            if (xlo && mask(i-1,j,k) > 0) {
                mat(i,j,k)[AMREX_SPACEDIM] += bfm * b(i,j,k);
                mat(i,j,k)[0] = 0.e0_rt;
            }

        }
        else if (mask.contains(i+1,j,k)) {

            if (xhi && mask(i+1,j,k) > 0) {
                mat(i,j,k)[AMREX_SPACEDIM] += bfm * b(i+1,j,k);
            }

        }
        else if (mask.contains(i,j-1,k)) {

            if (ylo && mask(i,j-1,k) > 0) {
                mat(i,j,k)[AMREX_SPACEDIM] += bfm * b(i,j,k);
                mat(i,j,k)[1] = 0.e0_rt;
            }

        }
        else if (mask.contains(i,j+1,k)) {

            if (yhi && mask(i,j+1,k) > 0) {
                mat(i,j,k)[AMREX_SPACEDIM] += bfm * b(i,j+1,k);
            }

        }
        else if (mask.contains(i,j,k-1)) {

            if (zlo && mask(i,j,k-1) > 0) {
                mat(i,j,k)[AMREX_SPACEDIM] += bfm * b(i,j,k);
                mat(i,j,k)[2] = 0.e0_rt;
            }

        }
        else if (mask.contains(i,j,k+1)) {

            if (zhi && mask(i,j,k+1) > 0) {
                mat(i,j,k)[3] += bfm * b(i,j,k+1);
            }

        }
    });

    Gpu::synchronize();
}

static AMREX_GPU_HOST_DEVICE AMREX_INLINE
Real boundary_diagonal_term (int bct, Real fac, Real h, Real bcl, Real beta, Real r, Real c,
                             Real b_face, Array4<Real const> const& spa, int i, int j, int k)
{
    Real bfm, bfv;

    if (bct == AMREX_LO_DIRICHLET) {
        bfv = fac * h / (0.5e0_rt * h + bcl);
        bfm = bfv * b_face;
    }
    else if (bct == AMREX_LO_NEUMANN) {
        bfm = 0.e0_rt;
    }
    else if (bct == AMREX_LO_MARSHAK) {
        bfv = 2.e0_rt * beta * r / h;
        bfm = 0.25e0_rt * c * bfv;
    }
    else if (bct == AMREX_LO_SANCHEZ_POMRANING) {
        bfv = 2.e0_rt * beta * r / h;
        bfm = spa(i,j,k) * c * bfv;
    }
#ifndef AMREX_USE_GPU
    else {
        amrex::Error("fillBcMat3: unsupported boundary type");
    }
#endif

    return bfm;
}

void AlphaBetaSolver::fillBcMat3 (const Box& bx,
                        int side_is_low, int idir,
                        int cdir, int bctype,
                        Array4<int const> const& tf,
                        Real bcl,
                        Array4<int const> const& mask,
                        Array4<Real const> const& b,
                        Real beta, const GeometryData& geomdata,
                        Real c,
                        Array4<Real const> const& spa,
                        Array4<GpuArray<Real, AMREX_SPACEDIM+1>> const& mat)
{
    bool xlo = false;
    bool ylo = false;
    bool zlo = false;

    bool xhi = false;
    bool yhi = false;
    bool zhi = false;

    const auto dx = geomdata.CellSize();

    Real h;

    if (AMREX_SPACEDIM == 1) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            xhi = true;
            h = dx[0];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else if (AMREX_SPACEDIM == 2) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 2) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 3) {
            yhi = true;
            h = dx[1];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 3) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 4) {
            yhi = true;
            h = dx[1];
        }
        else if (cdir == 2) {
            zlo = true;
            h = dx[2];
        }
        else if (cdir == 5) {
            zhi = true;
            h = dx[2];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }

    const Real fac = beta / (h * h);


    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        Real r;
        face_geometry_factors(i, j, k, bx.loVect()[0], bx.hiVect()[0], geomdata, idir, side_is_low, r);

        int bct;
        Real bfm;

        if (mask.contains(i-1,j,k)) {

            if (xlo && mask(i-1,j,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i-1,j,k);
                }
                else {
                    bct = bctype;
                }

                bfm = boundary_diagonal_term(bct, fac, h, bcl, beta, r, c, b(i,j,k), spa, i, j, k);

                mat(i,j,k)[AMREX_SPACEDIM] += bfm - fac * b(i,j,k);
                mat(i,j,k)[0] = 0.e0_rt;

            }

        }
        else if (mask.contains(i+1,j,k)) {

            if (xhi && mask(i+1,j,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i+1,j,k);
                }
                else {
                    bct = bctype;
                }

                bfm = boundary_diagonal_term(bct, fac, h, bcl, beta, r, c, b(i+1,j,k), spa, i, j, k);

                mat(i,j,k)[AMREX_SPACEDIM] += bfm - fac * b(i+1,j,k);

            }

        }
        else if (mask.contains(i,j-1,k)) {

            if (ylo && mask(i,j-1,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j-1,k);
                }
                else {
                    bct = bctype;
                }

                bfm = boundary_diagonal_term(bct, fac, h, bcl, beta, r, c, b(i,j,k), spa, i, j, k);

                mat(i,j,k)[AMREX_SPACEDIM] += bfm - fac * b(i,j,k);
                mat(i,j,k)[1] = 0.e0_rt;

            }

        }
        else if (mask.contains(i,j+1,k)) {

            if (yhi && mask(i,j+1,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j+1,k);
                }
                else {
                    bct = bctype;
                }

                bfm = boundary_diagonal_term(bct, fac, h, bcl, beta, r, c, b(i,j+1,k), spa, i, j, k);

                mat(i,j,k)[AMREX_SPACEDIM] += bfm - fac * b(i,j+1,k);

            }


        }
        else if (mask.contains(i,j,k-1)) {

            if (zlo && mask(i,j,k-1) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j,k-1);
                }
                else {
                    bct = bctype;
                }

                bfm = boundary_diagonal_term(bct, fac, h, bcl, beta, r, c, b(i,j,k), spa, i, j, k);

                mat(i,j,k)[AMREX_SPACEDIM] += bfm - fac * b(i,j,k);
                mat(i,j,k)[2] = 0.e0_rt;

            }

        }
        else if (mask.contains(i,j,k+1)) {

            if (zhi && mask(i,j,k+1) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j,k+1);
                }
                else {
                    bct = bctype;
                }

                bfm = boundary_diagonal_term(bct, fac, h, bcl, beta, r, c, b(i,j,k+1), spa, i, j, k);

                mat(i,j,k)[AMREX_SPACEDIM] += bfm - fac * b(i,j,k+1);

            }

        }
    });

    Gpu::synchronize();
}

void AlphaBetaSolver::fillBcVec (const Box& bx,
                       Array4<Real> const& vec,
                       int cdir, int bct, int bho, Real bcl,
                       Array4<Real const> const& bcval,
                       Array4<int const> const& mask,
                       Array4<Real const> const& b,
                       Real beta, const Real* dx)
{
    bool xlo = false;
    bool ylo = false;
    bool zlo = false;

    bool xhi = false;
    bool yhi = false;
    bool zhi = false;

    Real h, bfv, h2, th2;

    if (AMREX_SPACEDIM == 1) {
        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            xhi = true;
            h = dx[0];
        }
        else {
            amrex::Error("Unknown cdir");
        }
    }
    else if (AMREX_SPACEDIM == 2) {
        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 2) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 3) {
            yhi = true;
            h = dx[1];
        }
        else {
            amrex::Error("Unknown cdir");
        }
    }
    else {
        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 3) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 4) {
            yhi = true;
            h = dx[1];
        }
        else if (cdir == 2) {
            zlo = true;
            h = dx[2];
        }
        else if (cdir == 5) {
            zhi = true;
            h = dx[2];
        }
        else {
            amrex::Error("Unknown cdir");
        }
    }

    if (bct == AMREX_LO_DIRICHLET) {
        if (bho >= 1) {
            h2 = 0.5e0_rt * h;
            th2 = 3.e0_rt * h2;
            bfv = 2.e0_rt * beta / ((bcl + h2) * (bcl + th2));
        }
        else {
            bfv = (beta / h) / (0.5e0_rt * h + bcl);
        }
    }
    else if (bct == AMREX_LO_NEUMANN) {
        bfv = beta / h;
    }
    else {
        amrex::Error("fillBcVec: unsupported boundary type");
    }

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        if (mask.contains(i-1,j,k)) {

            if (xlo && mask(i-1,j,k) > 0) {

                vec(i,j,k) = vec(i,j,k) + bfv * b(i,j,k) * bcval(i-1,j,k);

            }

        }
        else if (mask.contains(i+1,j,k)) {

            if (xhi && mask(i+1,j,k) > 0) {

                vec(i,j,k) = vec(i,j,k) + bfv * b(i+1,j,k) * bcval(i+1,j,k);

            }

        }
        else if (mask.contains(i,j-1,k)) {

            if (ylo && mask(i,j-1,k) > 0) {

                vec(i,j,k) = vec(i,j,k) + bfv * b(i,j,k) * bcval(i,j-1,k);

            }

        }
        else if (mask.contains(i,j+1,k)) {

            if (yhi && mask(i,j+1,k) > 0) {

                vec(i,j,k) = vec(i,j,k) + bfv * b(i,j+1,k) * bcval(i,j+1,k);

            }

        }
        else if (mask.contains(i,j,k-1)) {

            if (zlo && mask(i,j,k-1) > 0) {

                vec(i,j,k) = vec(i,j,k) + bfv * b(i,j,k) * bcval(i,j,k-1);

            }

        }
        else if (mask.contains(i,j,k+1)) {

            if (zhi && mask(i,j,k+1) > 0) {

                vec(i,j,k) = vec(i,j,k) + bfv * b(i,j,k+1) * bcval(i,j,k+1);

            }

        }

    });

    Gpu::synchronize();
}

void AlphaBetaSolver::fillBcVec3 (const Box& bx,
                        int side_is_low, int idir,
                        Array4<Real> const& vec,
                        int cdir, int bctype,
                        Array4<int const> const& tf,
                        int bho, Real bcl,
                        Array4<Real const> const& bcval,
                        Array4<int const> const& mask,
                        Array4<Real const> const& b,
                        Real beta, const GeometryData& geomdata)
{
    bool xlo = false;
    bool ylo = false;
    bool zlo = false;

    bool xhi = false;
    bool yhi = false;
    bool zhi = false;

    Real h;

    const auto dx = geomdata.CellSize();

    if (AMREX_SPACEDIM == 1) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            xhi = true;
            h = dx[0];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else if (AMREX_SPACEDIM == 2) {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 2) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 3) {
            yhi = true;
            h = dx[1];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }
    else {

        if (cdir == 0) {
            xlo = true;
            h = dx[0];
        }
        else if (cdir == 3) {
            xhi = true;
            h = dx[0];
        }
        else if (cdir == 1) {
            ylo = true;
            h = dx[1];
        }
        else if (cdir == 4) {
            yhi = true;
            h = dx[1];
        }
        else if (cdir == 2) {
            zlo = true;
            h = dx[2];
        }
        else if (cdir == 5) {
            zhi = true;
            h = dx[2];
        }
        else {
            amrex::Error("Unknown cdir");
        }

    }

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        Real r;
        face_geometry_factors(i, j, k, bx.loVect()[0], bx.hiVect()[0], geomdata, idir, side_is_low, r);

        int bct;

        auto boundary_rhs_coef = [&] (int bct_kind, Real b_face) {
            Real bfv, h2, th2;

            if (bct_kind == AMREX_LO_DIRICHLET) {
                if (bho >= 1) {
                    h2 = 0.5e0_rt * h;
                    th2 = 3.e0_rt * h2;
                    bfv = 2.e0_rt * beta / ((bcl + h2) * (bcl + th2));
                }
                else {
                    bfv = (beta / h) / (0.5e0_rt * h + bcl);
                }

                bfv = bfv * b_face;
            }
            else if (bct_kind == AMREX_LO_NEUMANN) {
                bfv = beta * r / h;
            }
            else if (bct_kind == AMREX_LO_MARSHAK || bct_kind == AMREX_LO_SANCHEZ_POMRANING) {
                bfv = 2.e0_rt * beta * r / h;
            }
#ifndef AMREX_USE_GPU
            else {
                amrex::Error("fillBcVec3: unsupported boundary type");
            }
#endif

            return bfv;
        };

        if (mask.contains(i-1,j,k)) {

            if (xlo && mask(i-1,j,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i-1,j,k);
                }
                else {
                    bct = bctype;
                }

                vec(i,j,k) = vec(i,j,k) + boundary_rhs_coef(bct, b(i,j,k)) * bcval(i-1,j,k);

            }

        }
        else if (mask.contains(i+1,j,k)) {

            if (xhi && mask(i+1,j,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i+1,j,k);
                }
                else {
                    bct = bctype;
                }

                vec(i,j,k) = vec(i,j,k) + boundary_rhs_coef(bct, b(i+1,j,k)) * bcval(i+1,j,k);

            }

        }
        else if (mask.contains(i,j-1,k)) {

            if (ylo && mask(i,j-1,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j-1,k);
                }
                else {
                    bct = bctype;
                }

                vec(i,j,k) = vec(i,j,k) + boundary_rhs_coef(bct, b(i,j,k)) * bcval(i,j-1,k);

            }

        }
        else if (mask.contains(i,j+1,k)) {

            if (yhi && mask(i,j+1,k) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j+1,k);
                }
                else {
                    bct = bctype;
                }

                vec(i,j,k) = vec(i,j,k) + boundary_rhs_coef(bct, b(i,j+1,k)) * bcval(i,j+1,k);

            }

        }
        else if (mask.contains(i,j,k-1)) {

            if (zlo && mask(i,j,k-1) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j,k-1);
                }
                else {
                    bct = bctype;
                }

                vec(i,j,k) = vec(i,j,k) + boundary_rhs_coef(bct, b(i,j,k)) * bcval(i,j,k-1);

            }

        }
        else if (mask.contains(i,j,k+1)) {

            if (zhi && mask(i,j,k+1) > 0) {

                if (bctype == -1) {
                    bct = tf(i,j,k+1);
                }
                else {
                    bct = bctype;
                }

                vec(i,j,k) = vec(i,j,k) + boundary_rhs_coef(bct, b(i,j,k+1)) * bcval(i,j,k+1);

            }

        }
    });

    Gpu::synchronize();
}

void AlphaBetaSolver::fillFaceMat (const Box& bx,
                        Array4<Real const> const& b,
                        Real beta, const Real* dx,
                        int idir,
                        Array4<GpuArray<Real, AMREX_SPACEDIM+1>> const& mat)
{

    if (idir == 0) {

        const Real fac = beta / (dx[0] * dx[0]);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            mat(i,j,k)[0] = -fac * b(i,j,k);
            mat(i,j,k)[AMREX_SPACEDIM] += fac * (b(i,j,k) + b(i+1,j,k));
        });

    }
    else if (idir == 1) {

        const Real fac = beta / (dx[1] * dx[1]);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            mat(i,j,k)[1] = -fac * b(i,j,k);
            mat(i,j,k)[AMREX_SPACEDIM] += fac * (b(i,j,k) + b(i,j+1,k));
        });

    }
    else {

        const Real fac = beta / (dx[2] * dx[2]);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            mat(i,j,k)[2] = -fac * b(i,j,k);
            mat(i,j,k)[AMREX_SPACEDIM] += fac * (b(i,j,k) + b(i,j,k+1));
        });

    }

    Gpu::synchronize();
}

void AlphaBetaSolver::fillMassMat (const Box& bx,
                        Array4<Real const> const& a,
                        Real alpha,
                        Array4<GpuArray<Real, AMREX_SPACEDIM+1>> const& mat)
{
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        if (alpha == 0.e0_rt) {
            mat(i,j,k)[AMREX_SPACEDIM] = 0.e0_rt;
        }
        else {
            mat(i,j,k)[AMREX_SPACEDIM] = alpha * a(i,j,k);
        }
    });

    Gpu::synchronize();
}

void AlphaBetaSolver::faceGeometryFactors(Vector<Real>& r,
                              const Box& reg,
                              const Orientation& ori,
                              const Geometry& geom)
{
  if (ori.coordDir() == 0) {
    if (geom.IsCartesian()) {
      r.resize(1, 1.0);
    }
    else {
      r.resize(1);
      if (!(ori.isLow())) {
        r[0] = geom.HiEdge(reg.bigEnd(0), 0);
      } else {
        r[0] = geom.LoEdge(reg.smallEnd(0), 0);
      }
      if (geom.IsSPHERICAL()) {
        r[0] *= r[0];
      }
    }
  }
  else {
    if (geom.IsCartesian()) {
      r.resize(reg.length(0), 1.0);
    }
    else {
      BL_ASSERT(geom.IsRZ());
      geom.GetCellLoc(r, reg, 0);
    }
  }
}

void AlphaBetaSolver::faceFluxAtBoundary(MultiFab* Flux, MultiFab& sol_field, int icomp,
                             BoundaryMode bc_variant)
{
    BL_PROFILE("AlphaBetaSolver::faceFluxAtBoundary");

    const BoxArray &grids = sol_field.boxArray();

    const BaseBoundary& bd = getBoundaryData();
    const Box& domain = bd.getDomain();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {

        for (MFIter si(sol_field); si.isValid(); ++si) {
            int i = si.index();
            const Box &reg = grids[i];
            for (OrientationIter oitr; oitr; oitr++) {
                int cdir(oitr());
                int idim = oitr().coordDir();
                const RadBoundaryCondition &bct = bd.bndryConds(oitr())[i];
                const Real      &bcl = bd.bndryLocs(oitr())[i];
                const FArrayBox       &fs  = bd.bndryValues(oitr())[si];
                const Mask      &msk = bd.bndryMasks(oitr(),i);

                if (reg[oitr()] == domain[oitr()]) {
                    int bctype = bct;
                    Array4<int const> bc_type_arr;
                    if (bd.isMixedFace(oitr())) {
                        const BaseFab<int> &tf = *(bd.boundaryTypes(oitr())[i]);
                        bc_type_arr = tf.array();
                        bctype = -1;
                    }
                    Array4<Real const> bc_alpha_arr;
                    if (SPa != 0) {
                        bc_alpha_arr = (*SPa)[si].array();
                    }
                    ELLIPTIC_OPS::face_flux_3d(Flux[idim][si].array(),
                                  sol_field[si].array(icomp),
                                  reg,
                                  cdir, bctype,
                                  bc_type_arr,
                                  bho, bcl,
                                  fs.array(bc_component),
                                  msk.array(),
                                  (*bcoefs[idim])[si].array(),
                                  beta, dx, flux_scale,
                                  oitr(), geom.data(), bc_variant,
                                  bc_alpha_arr);
                }
                else {
                    ELLIPTIC_OPS::face_flux(Flux[idim][si].array(),
                                 sol_field[si].array(icomp),
                                 reg,
                                 cdir, bct, bho, bcl,
                                 fs.array(bc_component),
                                 msk.array(),
                                 (*bcoefs[idim])[si].array(),
                                 beta, geom.CellSizeArray(), bc_variant);
                }
            }
        }
    }
}
