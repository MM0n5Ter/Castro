#ifndef HYDROSIM_LIMITS_H
#define HYDROSIM_LIMITS_H

constexpr int LEVEL_LIMIT{15};

// NUM_GROW is the number of ghost cells needed for the hyperbolic
// portions -- note that this includes the flattening, which
// generally requires 4 ghost cells
#ifdef MHD
constexpr int NUM_GROW{6};
#else
constexpr int NUM_GROW{4};
#endif

// SRC_GHOST_WIDTH is for quantities that will be reconstructed, but
// don't need the full stencil required for flattening
#ifdef MHD
constexpr int SRC_GHOST_WIDTH{6};
#elif TRUE_SDC
constexpr int SRC_GHOST_WIDTH{NUM_GROW};
#else
constexpr int SRC_GHOST_WIDTH{3};
#endif

#endif
