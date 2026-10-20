#ifndef HYDROSIM_FILTER_WEIGHTS_H
#define HYDROSIM_FILTER_WEIGHTS_H

namespace filter
{
    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real ff1 (int i)
    {
        if (i == 0) {
            return 0.50_rt;
        }
        else if (!(i == 1)) {
            return 0.0_rt;
        } else {
            return 0.25_rt;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real ff2 (int i, int j)
    {
        constexpr Real w[2][3] = {
            {0.625_rt, 0.25_rt, -0.0625_rt},
            {0.375_rt, 0.25_rt, 0.0625_rt}
        };

        if (i < 0 || i >= 3 || j < 0 || j >= 2) {
            return 0.0_rt;
        }

        return w[j][i];
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real ff3 (int i, int j)
    {
        constexpr Real w[3][4] = {
            {44.0_rt / 64.0_rt, 15.0_rt / 64.0_rt, -6.0_rt / 64.0_rt, 1.0_rt / 64.0_rt},
            {32.0_rt / 64.0_rt, 18.0_rt / 64.0_rt, 0.e0_rt, -2.0_rt / 64.0_rt},
            {20.0_rt / 64.0_rt, 15.0_rt / 64.0_rt, 6.0_rt / 64.0_rt, 1.0_rt / 64.0_rt}
        };

        if (i < 0 || i >= 4 || j < 0 || j >= 3) {
            return 0.0_rt;
        }

        return w[j][i];
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real ff4 (int i, int j)
    {
        constexpr Real w[4][5] = {
            {186.0_rt / 256.0_rt, 56.0_rt / 256.0_rt, -28.0_rt / 256.0_rt, 8.0_rt / 256.0_rt, -1.0_rt / 256.0_rt},
            {146.0_rt / 256.0_rt, 72.0_rt / 256.0_rt, -12.0_rt / 256.0_rt, -8.0_rt / 256.0_rt, 3.0_rt / 256.0_rt},
            {110.0_rt / 256.0_rt, 72.0_rt / 256.0_rt, 12.0_rt / 256.0_rt, -8.0_rt / 256.0_rt, -3.0_rt / 256.0_rt},
            {70.0_rt / 256.0_rt, 56.0_rt / 256.0_rt, 28.0_rt / 256.0_rt, 8.0_rt / 256.0_rt, 1.0_rt / 256.0_rt}
        };

        if (i < 0 || i >= 5 || j < 0 || j >= 4) {
            return 0.0_rt;
        }

        return w[j][i];
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real edge_weight_1 (int i)
    {
        if (i == 0) {
            return 0.75_rt;
        }
        else if (!(i == 1)) {
            return 0.0_rt;
        } else {
            return 0.25_rt;
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
        else if (!(i == 2)) {
            return 0.0_rt;
        } else {
            return 1.0_rt / 16.0_rt;
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
        else if (!(i == 3)) {
            return 0.0_rt;
        } else {
            return 1.0_rt / 64.0_rt;
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
        else if (!(i == 3)) {
            return 0.0_rt;
        } else {
            return 1.0_rt / 64.0_rt;
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
        else if (!(i == 4)) {
            return 0.0_rt;
        } else {
            return 1.0_rt / 256.0_rt;
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
        else if (!(i == 4)) {
            return 0.0_rt;
        } else {
            return 1.0_rt / 256.0_rt;
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
        else if (!(i == 4)) {
            return 0.0_rt;
        } else {
            return 1.0_rt / 256.0_rt;
        }
    }
}

#endif
