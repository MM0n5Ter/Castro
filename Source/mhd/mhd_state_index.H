#ifndef HYDROSIM_MHD_STATE_INDEX_H
#define HYDROSIM_MHD_STATE_INDEX_H

constexpr int UBX = NUM_STATE;
constexpr int UBY = NUM_STATE+1;
constexpr int UBZ = NUM_STATE+2;

constexpr int NUM_EIGVEC = 7;

constexpr int EIGVEC_RHO = 0;
constexpr int EIGVEC_U = 1;
constexpr int EIGVEC_V = 2;
constexpr int EIGVEC_W = 3;
constexpr int EIGVEC_PRESSURE = 4;

constexpr int EIGVEC_BPERP1 = 5;
constexpr int EIGVEC_BPERP2 = 6;

#endif
