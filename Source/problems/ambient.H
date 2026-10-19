#ifndef ambient_region_H
#define ambient_region_H

#include <AMReX_REAL.H>
#include <state_indices.H>

namespace ambient {
    extern AMREX_GPU_MANAGED amrex::Real ambient_reference_state[NUM_STATE];
}

#endif
