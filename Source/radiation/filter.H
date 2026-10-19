#ifndef smoothing_H
#define smoothing_H

namespace filter
{
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real ff1 (int i)
    {
        if (i == 0) {
            return 0.50_rt;
        }
        else if (i == 1) {
            return 0.25_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real edge_weight_1 (int i)
    {
        if (i == 0) {
            return 0.75_rt;
        }
        else if (i == 1) {
            return 0.25_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real ff2 (int i, int j)
    {
        if (i == 0 && j == 0) {
            return 0.625_rt;
        }
        else if (i == 1 && j == 0) {
            return 0.25_rt;
        }
        else if (i == 2 && j == 0) {
            return -0.0625_rt;
        }
        else if (i == 0 && j == 1) {
            return 0.375_rt;
        }
        else if (i == 1 && j == 1) {
            return 0.25_rt;
        }
        else if (i == 2 && j == 1) {
            return 0.0625_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real edge_weight_2_0 (int i)
    {
        if (i == 0) {
            return 17.0_rt / 16.0_rt;
        }
        else if (i == 1) {
            return -2.0_rt / 16.0_rt;
        }
        else if (i == 2) {
            return 1.0_rt / 16.0_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real edge_weight_2_1 (int i)
    {
        if (i == -1) {
            return -2.0_rt / 16.0_rt;
        }
        else if (i == 0) {
            return 21.0_rt / 16.0_rt;
        }
        else if (i == 1) {
            return -4.0_rt / 16.0_rt;
        }
        else if (i == 2) {
            return 1.0_rt / 16.0_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real ff3 (int i, int j)
    {
        if (i == 0 && j == 0) {
            return 44.0_rt / 64.0_rt;
        }
        else if (i == 1 && j == 0) {
            return 15.0_rt / 64.0_rt;
        }
        else if (i == 2 && j == 0) {
            return -6.0_rt / 64.0_rt;
        }
        else if (i == 3 && j == 0) {
            return 1.0_rt / 64.0_rt;
        }
        else if (i == 0 && j == 1) {
            return 32.0_rt / 64.0_rt;
        }
        else if (i == 1 && j == 1) {
            return 18.0_rt / 64.0_rt;
        }
        else if (i == 2 && j == 1) {
            return 0.e0_rt;
        }
        else if (i == 3 && j == 1) {
            return -2.0_rt / 64.0_rt;
        }
        else if (i == 0 && j == 2) {
            return 20.0_rt / 64.0_rt;
        }
        else if (i == 1 && j == 2) {
            return 15.0_rt / 64.0_rt;
        }
        else if (i == 2 && j == 2) {
            return 6.0_rt / 64.0_rt;
        }
        else if (i == 3 && j == 2) {
            return 1.0_rt / 64.0_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real edge_weight_3_0 (int i)
    {
        if (i == 0) {
            return 63.0_rt / 64.0_rt;
        }
        else if (i == 1) {
            return 3.0_rt / 64.0_rt;
        }
        else if (i == 2) {
            return -3.0_rt / 64.0_rt;
        }
        else if (i == 3) {
            return 1.0_rt / 64.0_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real edge_weight_3_1 (int i)
    {
        if (i == -1) {
            return 3.0_rt / 64.0_rt;
        }
        else if (i == 0) {
            return 54.0_rt / 64.0_rt;
        }
        else if (i == 1) {
            return 12.0_rt / 64.0_rt;
        }
        else if (i == 2) {
            return -6.e0_rt/64.e0_rt;
        }
        else if (i == 3) {
            return 1.0_rt / 64.0_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real edge_weight_3_2 (int i)
    {
        if (i == -2) {
            return -3.0_rt / 64.0_rt;
        }
        else if (i == -1) {
            return 12.0_rt / 64.0_rt;
        }
        else if (i == 0) {
            return 45.0_rt / 64.0_rt;
        }
        else if (i == 1) {
            return 15.0_rt / 64.0_rt;
        }
        else if (i == 2) {
            return -6.0_rt / 64.0_rt;
        }
        else if (i == 3) {
            return 1.0_rt / 64.0_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real ff4 (int i, int j)
    {
        if (i == 0 && j == 0) {
            return 186.0_rt / 256.0_rt;
        }
        else if (i == 1 && j == 0) {
            return 56.0_rt / 256.0_rt;
        }
        else if (i == 2 && j == 0) {
            return -28.0_rt / 256.0_rt;
        }
        else if (i == 3 && j == 0) {
            return 8.0_rt / 256.0_rt;
        }
        else if (i == 4 && j == 0) {
            return -1.0_rt / 256.0_rt;
        }
        else if (i == 0 && j == 1) {
            return 146.0_rt / 256.0_rt;
        }
        else if (i == 1 && j == 1) {
            return 72.0_rt / 256.0_rt;
        }
        else if (i == 2 && j == 1) {
            return -12.0_rt / 256.0_rt;
        }
        else if (i == 3 && j == 1) {
            return -8.0_rt / 256.0_rt;
        }
        else if (i == 4 && j == 1) {
            return 3.0_rt / 256.0_rt;
        }
        else if (i == 0 && j == 2) {
            return 110.0_rt / 256.0_rt;
        }
        else if (i == 1 && j == 2) {
            return 72.0_rt / 256.0_rt;
        }
        else if (i == 2 && j == 2) {
            return 12.0_rt / 256.0_rt;
        }
        else if (i == 3 && j == 2) {
            return -8.0_rt / 256.0_rt;
        }
        else if (i == 4 && j == 2) {
            return -3.0_rt / 256.0_rt;
        }
        else if (i == 0 && j == 3) {
            return 70.0_rt / 256.0_rt;
        }
        else if (i == 1 && j == 3) {
            return 56.0_rt / 256.0_rt;
        }
        else if (i == 2 && j == 3) {
            return 28.0_rt / 256.0_rt;
        }
        else if (i == 3 && j == 3) {
            return 8.0_rt / 256.0_rt;
        }
        else if (i == 4 && j == 3) {
            return 1.0_rt / 256.0_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real edge_weight_4_0 (int i)
    {
        if (i == 0) {
            return 257.0_rt / 256.0_rt;
        }
        else if (i == 1) {
            return -4.0_rt / 256.0_rt;
        }
        else if (i == 2) {
            return 6.0_rt / 256.0_rt;
        }
        else if (i == 3) {
            return -4.0_rt / 256.0_rt;
        }
        else if (i == 4) {
            return 1.0_rt / 256.0_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real edge_weight_4_1 (int i)
    {
        if (i == -1) {
            return -4.0_rt / 256.0_rt;
        }
        else if (i == 0) {
            return 273.0_rt / 256.0_rt;
        }
        else if (i == 1) {
            return -28.0_rt / 256.0_rt;
        }
        else if (i == 2) {
            return 22.0_rt / 256.0_rt;
        }
        else if (i == 3) {
            return -8.0_rt / 256.0_rt;
        }
        else if (i == 4) {
            return 1.0_rt / 256.0_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real edge_weight_4_2 (int i)
    {
        if (i == -2) {
            return 6.0_rt / 256.0_rt;
        }
        else if (i == -1) {
            return -28.0_rt / 256.0_rt;
        }
        else if (i == 0) {
            return 309.0_rt / 256.0_rt;
        }
        else if (i == 1) {
            return -52.0_rt / 256.0_rt;
        }
        else if (i == 2) {
            return 28.0_rt / 256.0_rt;
        }
        else if (i == 3) {
            return -8.0_rt / 256.0_rt;
        }
        else if (i == 4) {
            return 1.0_rt / 256.0_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real edge_weight_4_3 (int i)
    {
        if (i == -3) {
            return -4.0_rt / 256.0_rt;
        }
        else if (i == -2) {
            return 22.0_rt / 256.0_rt;
        }
        else if (i == -1) {
            return -52.0_rt / 256.0_rt;
        }
        else if (i == 0) {
            return 325.0_rt / 256.0_rt;
        }
        else if (i == 1) {
            return -56.0_rt / 256.0_rt;
        }
        else if (i == 2) {
            return 28.0_rt / 256.0_rt;
        }
        else if (i == 3) {
            return -8.0_rt / 256.0_rt;
        }
        else if (i == 4) {
            return 1.0_rt / 256.0_rt;
        }
        else {
            return 0.0_rt;
        }
    }

    template<int width>
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void load_smoothing_stencil(Array4<Real const> const input,
                             Array1D<Real, -width, width>& stencil,
                             int dir, int i, int j, int k, int g)
    {
        stencil(0) = input(i,j,k,g);

        if constexpr (width >= 1) {
            if (dir == 0) {
                stencil(-1) = input(i-1,j,k,g);
                stencil( 1) = input(i+1,j,k,g);
            }
            else if (dir == 1) {
                stencil(-1) = input(i,j-1,k,g);
                stencil( 1) = input(i,j+1,k,g);
            }
            else {
                stencil(-1) = input(i,j,k-1,g);
                stencil( 1) = input(i,j,k+1,g);
            }
        }

        if constexpr (width >= 2) {
            if (dir == 0) {
                stencil(-2) = input(i-2,j,k,g);
                stencil( 2) = input(i+2,j,k,g);
            }
            else if (dir == 1) {
                stencil(-2) = input(i,j-2,k,g);
                stencil( 2) = input(i,j+2,k,g);
            }
            else {
                stencil(-2) = input(i,j,k-2,g);
                stencil( 2) = input(i,j,k+2,g);
            }
        }

        if constexpr (width >= 3) {
            if (dir == 0) {
                stencil(-3) = input(i-3,j,k,g);
                stencil( 3) = input(i+3,j,k,g);
            }
            else if (dir == 1) {
                stencil(-3) = input(i,j-3,k,g);
                stencil( 3) = input(i,j+3,k,g);
            }
            else {
                stencil(-3) = input(i,j,k-3,g);
                stencil( 3) = input(i,j,k+3,g);
            }
        }

        if constexpr (width >= 4) {
            if (dir == 0) {
                stencil(-4) = input(i-4,j,k,g);
                stencil( 4) = input(i+4,j,k,g);
            }
            else if (dir == 1) {
                stencil(-4) = input(i,j-4,k,g);
                stencil( 4) = input(i,j+4,k,g);
            }
            else {
                stencil(-4) = input(i,j,k-4,g);
                stencil( 4) = input(i,j,k+4,g);
            }
        }
    }

    template<int width>
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real apply_filter (Array4<Real const> const Er,
                       Array4<Real const> const lam,
                       int dir, int S, int i, int j, int k, int g)
    {
        Real smoothed_limiter;


        Array1D<Real, -width, width> Rad_energy_stencil, limiter_stencil;

        load_smoothing_stencil<width>(Er, Rad_energy_stencil, dir, i, j, k, g);
        load_smoothing_stencil<width>(lam, limiter_stencil, dir, i, j, k, g);

        if constexpr (width == 1) {
            if (Rad_energy_stencil(-1) == -1.0_rt) {
                smoothed_limiter = edge_weight_1(0) * limiter_stencil(0) +
                         edge_weight_1(1) * limiter_stencil(1);
            }
            else if (Rad_energy_stencil(1) == -1.0_rt) {
                smoothed_limiter = edge_weight_1(1) * limiter_stencil(-1) +
                         edge_weight_1(0) * limiter_stencil( 0);
            }
            else {
                smoothed_limiter = ff1(0) * limiter_stencil(0) +
                         ff1(1) * (limiter_stencil(-1) + limiter_stencil(1));
            }
        }
        else if constexpr (width == 2) {
            if (Rad_energy_stencil(-1) == -1.0_rt) {
                smoothed_limiter = edge_weight_2_0(0) * limiter_stencil(0) +
                         edge_weight_2_0(1) * limiter_stencil(1) +
                         edge_weight_2_0(2) * limiter_stencil(2);
            }
            else if (Rad_energy_stencil(-2) == -1.0_rt) {
                smoothed_limiter = edge_weight_2_1(-1) * limiter_stencil(-1) +
                         edge_weight_2_1( 0) * limiter_stencil( 0) +
                         edge_weight_2_1( 1) * limiter_stencil( 1) +
                         edge_weight_2_1( 2) * limiter_stencil( 2);
            }
            else if (Rad_energy_stencil(1) == -1.0_rt) {
                smoothed_limiter = edge_weight_2_0(2) * limiter_stencil(-2) +
                         edge_weight_2_0(1) * limiter_stencil(-1) +
                         edge_weight_2_0(0) * limiter_stencil( 0);
            }
            else if (Rad_energy_stencil(2) == -1.0_rt) {
                smoothed_limiter = edge_weight_2_1( 2) * limiter_stencil(-2) +
                         edge_weight_2_1( 1) * limiter_stencil(-1) +
                         edge_weight_2_1( 0) * limiter_stencil( 0) +
                         edge_weight_2_1(-1) * limiter_stencil( 1);
            }
            else {
                smoothed_limiter = ff2(0,S) * limiter_stencil(0) +
                         ff2(1,S) * (limiter_stencil(-1) + limiter_stencil(1)) +
                         ff2(2,S) * (limiter_stencil(-2) + limiter_stencil(2));
            }
        }
        else if constexpr (width == 3) {
            if (Rad_energy_stencil(-1) == -1.0_rt) {
                smoothed_limiter = edge_weight_3_0(0) * limiter_stencil(0) +
                         edge_weight_3_0(1) * limiter_stencil(1) +
                         edge_weight_3_0(2) * limiter_stencil(2) +
                         edge_weight_3_0(3) * limiter_stencil(3);
            }
            else if (Rad_energy_stencil(-2) == -1.0_rt) {
                smoothed_limiter = edge_weight_3_1(-1) * limiter_stencil(-1) +
                         edge_weight_3_1( 0) * limiter_stencil( 0) +
                         edge_weight_3_1( 1) * limiter_stencil( 1) +
                         edge_weight_3_1( 2) * limiter_stencil( 2) +
                         edge_weight_3_1( 3) * limiter_stencil( 3);
            }
            else if (Rad_energy_stencil(-3) == -1.0_rt) {
                smoothed_limiter = edge_weight_3_2(-2) * limiter_stencil(-2) +
                         edge_weight_3_2(-1) * limiter_stencil(-1) +
                         edge_weight_3_2( 0) * limiter_stencil( 0) +
                         edge_weight_3_2( 1) * limiter_stencil( 1) +
                         edge_weight_3_2( 2) * limiter_stencil( 2) +
                         edge_weight_3_2( 3) * limiter_stencil( 3);
            }
            else if (Rad_energy_stencil(1) == -1.0_rt) {
                smoothed_limiter = edge_weight_3_0(3) * limiter_stencil(-3) +
                         edge_weight_3_0(2) * limiter_stencil(-2) +
                         edge_weight_3_0(1) * limiter_stencil(-1) +
                         edge_weight_3_0(0) * limiter_stencil( 0);
            }
            else if (Rad_energy_stencil(2) == -1.0_rt) {
                smoothed_limiter = edge_weight_3_1( 3) * limiter_stencil(-3) +
                         edge_weight_3_1( 2) * limiter_stencil(-2) +
                         edge_weight_3_1( 1) * limiter_stencil(-1) +
                         edge_weight_3_1( 0) * limiter_stencil( 0) +
                         edge_weight_3_1(-1) * limiter_stencil( 1);
            }
            else if (Rad_energy_stencil(3) == -1.0_rt) {
                smoothed_limiter = edge_weight_3_2( 3) * limiter_stencil(-3) +
                         edge_weight_3_2( 2) * limiter_stencil(-2) +
                         edge_weight_3_2( 1) * limiter_stencil(-1) +
                         edge_weight_3_2( 0) * limiter_stencil( 0) +
                         edge_weight_3_2(-1) * limiter_stencil( 1) +
                         edge_weight_3_2(-2) * limiter_stencil( 2);
            }
            else {
                smoothed_limiter = ff3(0,S) * limiter_stencil(0) +
                         ff3(1,S) * (limiter_stencil(-1) + limiter_stencil(1)) +
                         ff3(2,S) * (limiter_stencil(-2) + limiter_stencil(2)) +
                         ff3(3,S) * (limiter_stencil(-3) + limiter_stencil(3));
            }
        }
        else if constexpr (width == 4) {
            if (Rad_energy_stencil(-1) == -1.0_rt) {
                smoothed_limiter = edge_weight_4_0(0) * limiter_stencil(0) +
                         edge_weight_4_0(1) * limiter_stencil(1) +
                         edge_weight_4_0(2) * limiter_stencil(2) +
                         edge_weight_4_0(3) * limiter_stencil(3) +
                         edge_weight_4_0(4) * limiter_stencil(4);
            }
            else if (Rad_energy_stencil(-2) == -1.0_rt) {
                smoothed_limiter = edge_weight_4_1(-1) * limiter_stencil(-1) +
                         edge_weight_4_1( 0) * limiter_stencil( 0) +
                         edge_weight_4_1( 1) * limiter_stencil( 1) +
                         edge_weight_4_1( 2) * limiter_stencil( 2) +
                         edge_weight_4_1( 3) * limiter_stencil( 3) +
                         edge_weight_4_1( 4) * limiter_stencil( 4);
            }
            else if (Rad_energy_stencil(-3) == -1.0_rt) {
                smoothed_limiter = edge_weight_4_2(-2) * limiter_stencil(-2) +
                         edge_weight_4_2(-1) * limiter_stencil(-1) +
                         edge_weight_4_2( 0) * limiter_stencil( 0) +
                         edge_weight_4_2( 1) * limiter_stencil( 1) +
                         edge_weight_4_2( 2) * limiter_stencil( 2) +
                         edge_weight_4_2( 3) * limiter_stencil( 3) +
                         edge_weight_4_2( 4) * limiter_stencil( 4);
            }
            else if (Rad_energy_stencil(-4) == -1.0_rt) {
                smoothed_limiter = edge_weight_4_3(-3) * limiter_stencil(-3) +
                         edge_weight_4_3(-2) * limiter_stencil(-2) +
                         edge_weight_4_3(-1) * limiter_stencil(-1) +
                         edge_weight_4_3( 0) * limiter_stencil( 0) +
                         edge_weight_4_3( 1) * limiter_stencil( 1) +
                         edge_weight_4_3( 2) * limiter_stencil( 2) +
                         edge_weight_4_3( 3) * limiter_stencil( 3) +
                         edge_weight_4_3( 4) * limiter_stencil( 4);
            }
            else if (Rad_energy_stencil(1) == -1.0_rt) {
                smoothed_limiter = edge_weight_4_0(4) * limiter_stencil(-4) +
                         edge_weight_4_0(3) * limiter_stencil(-3) +
                         edge_weight_4_0(2) * limiter_stencil(-2) +
                         edge_weight_4_0(1) * limiter_stencil(-1) +
                         edge_weight_4_0(0) * limiter_stencil( 0);
            }
            else if (Rad_energy_stencil(2) == -1.0_rt) {
                smoothed_limiter = edge_weight_4_1( 4) * limiter_stencil(-4) +
                         edge_weight_4_1( 3) * limiter_stencil(-3) +
                         edge_weight_4_1( 2) * limiter_stencil(-2) +
                         edge_weight_4_1( 1) * limiter_stencil(-1) +
                         edge_weight_4_1( 0) * limiter_stencil( 0) +
                         edge_weight_4_1(-1) * limiter_stencil( 1);
            }
            else if (Rad_energy_stencil(3) == -1.0_rt) {
                smoothed_limiter = edge_weight_4_2( 4) * limiter_stencil(-4) +
                         edge_weight_4_2( 3) * limiter_stencil(-3) +
                         edge_weight_4_2( 2) * limiter_stencil(-2) +
                         edge_weight_4_2( 1) * limiter_stencil(-1) +
                         edge_weight_4_2( 0) * limiter_stencil( 0) +
                         edge_weight_4_2(-1) * limiter_stencil( 1) +
                         edge_weight_4_2(-2) * limiter_stencil( 2);
            }
            else if (Rad_energy_stencil(4) == -1.0_rt) {
                smoothed_limiter = edge_weight_4_3( 4) * limiter_stencil(-4) +
                         edge_weight_4_3( 3) * limiter_stencil(-3) +
                         edge_weight_4_3( 2) * limiter_stencil(-2) +
                         edge_weight_4_3( 1) * limiter_stencil(-1) +
                         edge_weight_4_3( 0) * limiter_stencil( 0) +
                         edge_weight_4_3(-1) * limiter_stencil( 1) +
                         edge_weight_4_3(-2) * limiter_stencil( 2) +
                         edge_weight_4_3(-3) * limiter_stencil( 3);
            }
            else {
                smoothed_limiter = ff4(0,S) * limiter_stencil(0) +
                         ff4(1,S) * (limiter_stencil(-1) + limiter_stencil(1)) +
                         ff4(2,S) * (limiter_stencil(-2) + limiter_stencil(2)) +
                         ff4(3,S) * (limiter_stencil(-3) + limiter_stencil(3)) +
                         ff4(4,S) * (limiter_stencil(-4) + limiter_stencil(4));
            }
        }

        smoothed_limiter = std::min(1.0_rt / 3.0_rt, std::max(1.0e-25_rt, smoothed_limiter));

        return smoothed_limiter;
    }
}

#endif
