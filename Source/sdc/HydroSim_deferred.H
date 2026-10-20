#ifndef HYDROSIM_DEFERRED_H
#define HYDROSIM_DEFERRED_H

#include <AMReX_REAL.H>

#ifndef AMREX_USE_GPU
void do_sdc_correct(int m_start, int m_end, amrex::Real dt);
#endif

#ifdef REACTIONS
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
struct QuadratureNodeTerms {
    amrex::GpuArray<amrex::Array4<const amrex::Real>, 4> advect, react;
};

static
void hs_deferred_compute_correction_o2_lobatto(const amrex::Box& bx,
                               amrex::Real substep_dt, amrex::Real dt,
                               amrex::Array4<const amrex::Real> const& A_m,
                               const QuadratureNodeTerms& nodes,
                               amrex::Array4<amrex::Real> const& C,
                               int m_start);

static
void hs_deferred_compute_correction_o2_radau(const amrex::Box& bx,
                             amrex::Real substep_dt, amrex::Real dt,
                             amrex::Array4<const amrex::Real> const& A_m,
                             const QuadratureNodeTerms& nodes,
                             amrex::Array4<amrex::Real> const& C,
                             int m_start);

static
void hs_deferred_compute_correction_o4_lobatto(const amrex::Box& bx,
                               amrex::Real substep_dt, amrex::Real dt,
                               amrex::Array4<const amrex::Real> const& A_m,
                               const QuadratureNodeTerms& nodes,
                               amrex::Array4<amrex::Real> const& C,
                               int m_start);

static
void hs_deferred_compute_correction_o4_radau(const amrex::Box& bx,
                             amrex::Real substep_dt, amrex::Real dt,
                             amrex::Array4<const amrex::Real> const& A_m,
                             const QuadratureNodeTerms& nodes,
                             amrex::Array4<amrex::Real> const& C,
                             int m_start);
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
