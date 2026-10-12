#ifndef HYDROSIM_MHD_SIZES_H
#define HYDROSIM_MHD_SIZES_H

constexpr int UMAGX = NUM_STATE;
constexpr int UMAGY = NUM_STATE+1;
constexpr int UMAGZ = NUM_STATE+2;

constexpr int NEVEC = 7;

constexpr int IEVEC_RHO = 0;
constexpr int IEVEC_U = 1;
constexpr int IEVEC_V = 2;
constexpr int IEVEC_W = 3;
constexpr int IEVEC_P = 4;

// perpendicular magnetic field components
constexpr int IEVEC_BT = 5;
constexpr int IEVEC_BTT = 6;

#endif
