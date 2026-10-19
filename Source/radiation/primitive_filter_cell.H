#ifndef primitive_filter_H
#define primitive_filter_H

#include <AMReX.H>
#include <filter.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void primitive_filter_cell (int i, int j, int k,
                Array4<Real> const& filtered_state,
                Array4<Real> const& Snew,
                int temp_filter_index, int S,
                const GeometryData& geomdata,
                Real time)
{
    // this is a stub -- a problem can override this in its own directory
    // to implement filtering
}

#endif
