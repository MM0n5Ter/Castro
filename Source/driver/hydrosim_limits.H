#ifndef HYDROSIM_LIMITS_H
#define HYDROSIM_LIMITS_H

constexpr int LEVEL_LIMIT{15};

#ifdef MHD
constexpr int NUM_GROW{6};
#else
constexpr int NUM_GROW{4};
#endif

#ifdef MHD
constexpr int SRC_GHOST_WIDTH{6};
#elif TRUE_SDC
constexpr int SRC_GHOST_WIDTH{NUM_GROW};
#else
constexpr int SRC_GHOST_WIDTH{3};
#endif

#endif
