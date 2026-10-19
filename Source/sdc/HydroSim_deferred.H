#ifndef HYDROSIM_DEFERRED_H
#define HYDROSIM_DEFERRED_H

#include <AMReX_REAL.H>

#ifndef AMREX_USE_GPU
void do_sdc_correct(int m_start, int m_end, amrex::Real dt);
#endif

#ifdef REACTIONS
/// Take an input conserved state cell_cons, evaluate the instantaneous
/// reaction rates, and store the result in React_source/
///
/// Special consideration is needed for order4_scheme accuracy:
///
///  * If we come in with cell_cons representing an average
///    (input_is_mean = true), then, for order4_scheme, we will
///    convert cell_cons to cell-centers (consuming one ghost cell),
///    burn on the cell-centers (including one ghost cell) and then
///    average back to cell-centers (consuming another ghosst cell).
///    This requires >= 2 ghost cells for cell_cons, which we assume
///    are valid.
///
///  * If we come in with cell_cons representing cell centers
///    (input_is_mean = false), then we just return React_source
///    at the cell-centers.
///
void assemble_old_react_source(amrex::MultiFab& cell_cons,
                                amrex::MultiFab& React_source,
                                const bool input_is_mean);
#endif

static
void hs_deferred_update_advection_o2_lobatto(const amrex::Box& bx,
                                        amrex::Real substep_dt, amrex::Real dt,
                                        amrex::Array4<const amrex::Real> const& k_m,
                                        amrex::Array4<amrex::Real> const& k_n,
                                        amrex::Array4<const amrex::Real> const& A_m,
                                        amrex::Array4<const amrex::Real> const& Advect_0_old,
                                        amrex::Array4<const amrex::Real> const& Advect_1_old,
                                        int m_start);

static
void hs_deferred_update_advection_o2_radau(const amrex::Box& bx,
                                      amrex::Real substep_dt, amrex::Real dt,
                                      amrex::Array4<const amrex::Real> const& k_m,
                                      amrex::Array4<amrex::Real> const& k_n,
                                      amrex::Array4<const amrex::Real> const& A_m,
                                      amrex::Array4<const amrex::Real> const& Advect_0_old,
                                      amrex::Array4<const amrex::Real> const& Advect_1_old,
                                      amrex::Array4<const amrex::Real> const& Advect_2_old,
                                      int m_start);

static
void hs_deferred_update_advection_o4_lobatto(const amrex::Box& bx,
                                        amrex::Real substep_dt, amrex::Real dt,
                                        amrex::Array4<const amrex::Real> const& k_m,
                                        amrex::Array4<amrex::Real> const& k_n,
                                        amrex::Array4<const amrex::Real> const& A_m,
                                        amrex::Array4<const amrex::Real> const& Advect_0_old,
                                        amrex::Array4<const amrex::Real> const& Advect_1_old,
                                        amrex::Array4<const amrex::Real> const& Advect_2_old,
                                        int m_start);

static
void hs_deferred_update_advection_o4_radau(const amrex::Box& bx,
                                      amrex::Real substep_dt, amrex::Real dt,
                                      amrex::Array4<const amrex::Real> const& k_m,
                                      amrex::Array4<amrex::Real> const& k_n,
                                      amrex::Array4<const amrex::Real> const& A_m,
                                      amrex::Array4<const amrex::Real> const& Advect_0_old,
                                      amrex::Array4<const amrex::Real> const& Advect_1_old,
                                      amrex::Array4<const amrex::Real> const& Advect_2_old,
                                      amrex::Array4<const amrex::Real> const& Advect_3_old,
                                      int m_start);
#ifdef REACTIONS
static
void hs_deferred_compute_correction_o2_lobatto(const amrex::Box& bx,
                               amrex::Real substep_dt, amrex::Real dt,
                               amrex::Array4<const amrex::Real> const& A_m,
                               amrex::Array4<const amrex::Real> const& Advect_0_old,
                               amrex::Array4<const amrex::Real> const& Advect_1_old,
                               amrex::Array4<const amrex::Real> const& React_0_old,
                               amrex::Array4<const amrex::Real> const& React_1_old,
                               amrex::Array4<amrex::Real> const& C,
                               int m_start);

static
void hs_deferred_compute_correction_o2_radau(const amrex::Box& bx,
                             amrex::Real substep_dt, amrex::Real dt,
                             amrex::Array4<const amrex::Real> const& A_m,
                             amrex::Array4<const amrex::Real> const& Advect_0_old,
                             amrex::Array4<const amrex::Real> const& Advect_1_old,
                             amrex::Array4<const amrex::Real> const& Advect_2_old,
                             amrex::Array4<const amrex::Real> const& React_0_old,
                             amrex::Array4<const amrex::Real> const& React_1_old,
                             amrex::Array4<const amrex::Real> const& React_2_old,
                             amrex::Array4<amrex::Real> const& C,
                             int m_start);

static
void hs_deferred_compute_correction_o4_lobatto(const amrex::Box& bx,
                               amrex::Real substep_dt, amrex::Real dt,
                               amrex::Array4<const amrex::Real> const& A_m,
                               amrex::Array4<const amrex::Real> const& Advect_0_old,
                               amrex::Array4<const amrex::Real> const& Advect_1_old,
                               amrex::Array4<const amrex::Real> const& Advect_2_old,
                               amrex::Array4<const amrex::Real> const& React_0_old,
                               amrex::Array4<const amrex::Real> const& React_1_old,
                               amrex::Array4<const amrex::Real> const& React_2_old,
                               amrex::Array4<amrex::Real> const& C,
                               int m_start);

static
void hs_deferred_compute_correction_o4_radau(const amrex::Box& bx,
                             amrex::Real substep_dt, amrex::Real dt,
                             amrex::Array4<const amrex::Real> const& A_m,
                             amrex::Array4<const amrex::Real> const& Advect_0_old,
                             amrex::Array4<const amrex::Real> const& Advect_1_old,
                             amrex::Array4<const amrex::Real> const& Advect_2_old,
                             amrex::Array4<const amrex::Real> const& Advect_3_old,
                             amrex::Array4<const amrex::Real> const& React_0_old,
                             amrex::Array4<const amrex::Real> const& React_1_old,
                             amrex::Array4<const amrex::Real> const& React_2_old,
                             amrex::Array4<const amrex::Real> const& React_3_old,
                             amrex::Array4<amrex::Real> const& C,
                             int m_start);

static
void hs_deferred_conservative_update(const amrex::Box& bx, amrex::Real const substep_dt,
                                amrex::Array4<const amrex::Real> const& Conserved_old,
                                amrex::Array4<amrex::Real> const& cons_new,
                                amrex::Array4<const amrex::Real> const& C,
                                amrex::Array4<const amrex::Real> const& react_new);
#endif

static
void hs_deferred_compute_initial_guess(const amrex::Box& bx,
                                  amrex::Array4<const amrex::Real> const& Conserved_old,
                                  amrex::Array4<const amrex::Real> const& cons_new,
                                  amrex::Array4<const amrex::Real> const& Advection_old,
                                  amrex::Array4<const amrex::Real> const& Reaction_old,
                                  amrex::Array4<amrex::Real> const& State_guess,
                                  amrex::Real const substep_dt, int const sdc_iter);

#ifdef REACTIONS
static
void hs_store_reaction_state(const amrex::Box& bx,
                             amrex::Array4<const amrex::Real> const& Reaction_old,
                             amrex::Array4<amrex::Real> const& React_store);
#endif

#endif
