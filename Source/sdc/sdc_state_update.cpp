#include <HydroSim.H>

using namespace amrex;

#ifdef REACTIONS
void HydroSim::hs_store_reaction_state(const Box& bx,
                                     Array4<const Real> const& Reaction_old,
                                     Array4<Real> const& React_store)
{


    if (keep_omegadot) {
        amrex::ParallelFor(bx, NumSpec,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            React_store(i,j,k,1+n) = Reaction_old(i,j,k,UFS+n);
        });

#if NAUX_NET > 0
        amrex::ParallelFor(bx, NumAux,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            React_store(i,j,k,1+NumSpec+n) = Reaction_old(i,j,k,UFX+n);
        });
#endif
    }

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        React_store(i,j,k,0) = Reaction_old(i,j,k,UEDEN);
    });
}
#endif

void HydroSim::hs_deferred_compute_initial_guess(const Box& bx,
                                          Array4<const Real> const& Conserved_old,
                                          Array4<const Real> const& cons_new,
                                          Array4<const Real> const& Advection_old,
                                          Array4<const Real> const& Reaction_old,
                                          Array4<Real> const& State_guess,
                                          Real const substep_dt, int const sdc_iter)
{

    if (sdc_iter == 0)
    {
        amrex::ParallelFor(bx, State_guess.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            State_guess(i,j,k,n) = Conserved_old(i,j,k,n) + substep_dt * Advection_old(i,j,k,n) + substep_dt * Reaction_old(i,j,k,n);
        });
    }
    else
    {
        amrex::ParallelFor(bx, State_guess.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            State_guess(i,j,k,n) = cons_new(i,j,k,n);
        });
    }

}
