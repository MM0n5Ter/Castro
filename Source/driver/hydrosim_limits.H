#ifndef HYDROSIM_LIMITS_H
#define HYDROSIM_LIMITS_H

constexpr int LEVEL_LIMIT{15};

constexpr int NUM_GROW{4};

#if TRUE_SDC
constexpr int SRC_GHOST_WIDTH{NUM_GROW};
#else
constexpr int SRC_GHOST_WIDTH{3};
#endif

#endif
