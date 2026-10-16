#ifndef HYDROSIM_SDC_H
#define HYDROSIM_SDC_H

#include <AMReX_REAL.H>

#ifndef AMREX_USE_GPU
void sdc_node_update(int m_start, int m_end, amrex::Real dt);
#endif

#ifdef REACTIONS
///
/// What the zone values of a conserved state represent: cell
/// averages or cell-center values.
///
enum class ZoneValues : std::uint8_t {
    Average = 0,
    Center
};

/// Take an input conserved state U_state, evaluate the instantaneous
/// reaction rates, and store the result in R_source/
///
/// Special consideration is needed for fourth_order accuracy:
///
///  * If we come in with U_state representing an average
///    (input = ZoneValues::Average), then, for fourth_order, we will
///    convert U_state to cell-centers (consuming one ghost cell),
///    burn on the cell-centers (including one ghost cell) and then
///    average back to cell-centers (consuming another ghosst cell).
///    This requires >= 2 ghost cells for U_state, which we assume
///    are valid.
///
///  * If we come in with U_state representing cell centers
///    (input = ZoneValues::Center), then we just return R_source
///    at the cell-centers.
///
void assemble_old_react_source(amrex::MultiFab& U_state,
                                amrex::MultiFab& R_source,
                                const ZoneValues input);
#endif

#endif
