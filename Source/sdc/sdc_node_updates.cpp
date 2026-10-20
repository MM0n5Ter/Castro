#include <HydroSim.H>

using namespace amrex;

void
HydroSim::hs_deferred_update_advection_o2_lobatto(const Box& bx,
                                           Real substep_dt, Real dt,
                                           Array4<const Real> const& k_m,
                                           Array4<Real> const& k_n,
                                           Array4<const Real> const& A_m,
                                           Array4<const Real> const& Advect_0_old,
                                           Array4<const Real> const& Advect_1_old,
                                           int m_start)
{

    amrex::ignore_unused(substep_dt);
    amrex::ignore_unused(A_m);
    amrex::ignore_unused(m_start);

    amrex::ParallelFor(bx, k_n.nComp(),
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        k_n(i,j,k,n) = k_m(i,j,k,n) + 0.5_rt * dt * (Advect_0_old(i,j,k,n) + Advect_1_old(i,j,k,n));
    });
}

#ifdef REACTIONS
void
HydroSim::hs_deferred_compute_correction_o2_lobatto(const Box& bx,
                                  Real substep_dt, Real dt,
                                  Array4<const Real> const& A_m,
                                  const QuadratureNodeTerms& nodes,
                                  Array4<Real> const& C,
                                  int m_start)
{

    amrex::ignore_unused(substep_dt);
    amrex::ignore_unused(dt);
    amrex::ignore_unused(A_m);
    amrex::ignore_unused(m_start);

    amrex::ParallelFor(bx, C.nComp(),
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        C(i,j,k,n) = -nodes.react[1](i,j,k,n) +
            0.5_rt * (nodes.advect[0](i,j,k,n) + nodes.advect[1](i,j,k,n)) +
            0.5_rt * (nodes.react[0](i,j,k,n) + nodes.react[1](i,j,k,n));
    });

}
#endif

void
HydroSim::hs_deferred_update_advection_o2_radau(const Box& bx,
                                         Real substep_dt, Real dt,
                                         Array4<const Real> const& k_m,
                                         Array4<Real> const& k_n,
                                         Array4<const Real> const& A_m,
                                         Array4<const Real> const& Advect_0_old,
                                         Array4<const Real> const& Advect_1_old,
                                         Array4<const Real> const& Advect_2_old,
                                         int m_start)
{

    if (m_start == 0)
    {
        amrex::ParallelFor(bx, k_n.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            k_n(i,j,k,n) = k_m(i,j,k,n) +
                substep_dt * (A_m(i,j,k,n) - Advect_0_old(i,j,k,n)) +
                dt/12.0_rt * (5.0_rt*Advect_1_old(i,j,k,n) - Advect_2_old(i,j,k,n));
        });
    }
    else if (m_start == 1)
    {
        amrex::ParallelFor(bx, k_n.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            k_n(i,j,k,n) = k_m(i,j,k,n) +
                substep_dt * (A_m(i,j,k,n) - Advect_1_old(i,j,k,n)) +
                dt/3.0_rt * (Advect_1_old(i,j,k,n) + Advect_2_old(i,j,k,n));
        });
    }
}

#ifdef REACTIONS
void
HydroSim::hs_deferred_compute_correction_o2_radau(const Box& bx,
                                Real substep_dt, Real dt,
                                Array4<const Real> const& A_m,
                                const QuadratureNodeTerms& nodes,
                                Array4<Real> const& C,
                                int m_start)
{

    if (m_start == 0)
    {
        amrex::ParallelFor(bx, C.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            C(i,j,k,n) = -nodes.react[1](i,j,k,n) +
                (A_m(i,j,k,n) - nodes.advect[0](i,j,k,n)) +
                (dt/substep_dt) * (1.0_rt/12.0_rt) *
                (5.0_rt*(nodes.advect[1](i,j,k,n) + nodes.react[1](i,j,k,n)) -
                        (nodes.advect[2](i,j,k,n) + nodes.react[2](i,j,k,n)));
        });
    }
    else if (m_start == 1)
    {
        amrex::ParallelFor(bx, C.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            C(i,j,k,n) = -nodes.react[2](i,j,k,n) +
                (A_m(i,j,k,n) - nodes.advect[1](i,j,k,n)) +
                (dt/substep_dt) * (1.0_rt/3.0_rt) *
                ((nodes.advect[1](i,j,k,n) + nodes.react[1](i,j,k,n)) +
                 (nodes.advect[2](i,j,k,n) + nodes.react[2](i,j,k,n)));
        });
    }
}
#endif

void
HydroSim::hs_deferred_update_advection_o4_lobatto(const Box& bx,
                                           Real substep_dt, Real dt,
                                           Array4<const Real> const& k_m,
                                           Array4<Real> const& k_n,
                                           Array4<const Real> const& A_m,
                                           Array4<const Real> const& Advect_0_old,
                                           Array4<const Real> const& Advect_1_old,
                                           Array4<const Real> const& Advect_2_old,
                                           int m_start)
{

    if (m_start == 0)
    {
        amrex::ParallelFor(bx, k_n.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            k_n(i,j,k,n) = k_m(i,j,k,n) +
                substep_dt * (A_m(i,j,k,n) - Advect_0_old(i,j,k,n)) +
                dt/24.0_rt * (5.0_rt*Advect_0_old(i,j,k,n) + 8.0_rt*Advect_1_old(i,j,k,n) - Advect_2_old(i,j,k,n));
        });
    }
    else if (m_start == 1)
    {
        amrex::ParallelFor(bx, k_n.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            k_n(i,j,k,n) = k_m(i,j,k,n) +
                substep_dt * (A_m(i,j,k,n) - Advect_1_old(i,j,k,n)) +
                dt/24.0_rt * (-Advect_0_old(i,j,k,n) + 8.0_rt*Advect_1_old(i,j,k,n) + 5.0_rt*Advect_2_old(i,j,k,n));
        });
    }
    else
    {
        amrex::Abort("error in hs_deferred_update_advection_o4_lobatto -- should not be here");
    }
}

#ifdef REACTIONS
void
HydroSim::hs_deferred_compute_correction_o4_lobatto(const Box& bx,
                                  Real substep_dt, Real dt,
                                  Array4<const Real> const& A_m,
                                  const QuadratureNodeTerms& nodes,
                                  Array4<Real> const& C,
                                  int m_start)
{

    amrex::ignore_unused(A_m);
    amrex::ignore_unused(substep_dt);
    amrex::ignore_unused(dt);

    if (m_start == 0)
    {
        amrex::ParallelFor(bx, C.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real integral = 1.0_rt/12.0_rt * (5.0_rt*(nodes.advect[0](i,j,k,n) + nodes.react[0](i,j,k,n)) +
                                              8.0_rt*(nodes.advect[1](i,j,k,n) + nodes.react[1](i,j,k,n)) -
                                              (nodes.advect[2](i,j,k,n) + nodes.react[2](i,j,k,n)));

            C(i,j,k,n) = (A_m(i,j,k,n) - nodes.advect[0](i,j,k,n)) - nodes.react[1](i,j,k,n) + integral;
        });
    }
    else if (m_start == 1)
    {
        amrex::ParallelFor(bx, C.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real integral = 1.0_rt/12.0_rt * (-(nodes.advect[0](i,j,k,n) + nodes.react[0](i,j,k,n)) +
                                              8.0_rt*(nodes.advect[1](i,j,k,n) + nodes.react[1](i,j,k,n)) +
                                              5.0_rt*(nodes.advect[2](i,j,k,n) + nodes.react[2](i,j,k,n)));

            C(i,j,k,n) = (A_m(i,j,k,n) - nodes.advect[1](i,j,k,n)) - nodes.react[2](i,j,k,n) + integral;
        });
    }
    else
    {
        amrex::Abort("error in ca_sdc_compute_C4 -- should not be here");
    }

}
#endif

void
HydroSim::hs_deferred_update_advection_o4_radau(const Box& bx,
                                         Real substep_dt, Real dt,
                                         Array4<const Real> const& k_m,
                                         Array4<Real> const& k_n,
                                         Array4<const Real> const& A_m,
                                         Array4<const Real> const& Advect_0_old,
                                         Array4<const Real> const& Advect_1_old,
                                         Array4<const Real> const& Advect_2_old,
                                         Array4<const Real> const& Advect_3_old,
                                         int m_start)
{

    if (m_start == 0)
    {
        amrex::ParallelFor(bx, k_n.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            k_n(i,j,k,n) = k_m(i,j,k,n) +
                substep_dt * (A_m(i,j,k,n) - Advect_0_old(i,j,k,n)) +
                dt/1800.0_rt * ((-35.0_rt*std::sqrt(6.0_rt) + 440.0_rt)*Advect_1_old(i,j,k,n) +
                                (-169.0_rt*std::sqrt(6.0_rt) + 296.0_rt)*Advect_2_old(i,j,k,n) +
                                (-16.0_rt + 24.0_rt*std::sqrt(6.0_rt))*Advect_3_old(i,j,k,n));
        });
    }
    else if (m_start == 1)
    {
        amrex::ParallelFor(bx, k_n.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            k_n(i,j,k,n) = k_m(i,j,k,n) +
                substep_dt * (A_m(i,j,k,n) - Advect_1_old(i,j,k,n)) +
                dt/150.0_rt * ((-12.0_rt + 17.0_rt*std::sqrt(6.0_rt))*Advect_1_old(i,j,k,n) +
                               (12.0_rt + 17.0_rt*std::sqrt(6.0_rt))*Advect_2_old(i,j,k,n) +
                               (-4.0_rt*std::sqrt(6.0_rt))*Advect_3_old(i,j,k,n));
        });
    }
    else if (m_start == 2)
    {
        amrex::ParallelFor(bx, k_n.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            k_n(i,j,k,n) = k_m(i,j,k,n) +
                substep_dt * (A_m(i,j,k,n) - Advect_2_old(i,j,k,n)) +
                dt/600.0_rt * ((168.0_rt - 73.0_rt*std::sqrt(6.0_rt))*Advect_1_old(i,j,k,n) +
                               (120.0_rt + 5.0_rt*std::sqrt(6.0_rt))*Advect_2_old(i,j,k,n) +
                               (72.0_rt + 8.0_rt*std::sqrt(6.0_rt))*Advect_3_old(i,j,k,n));
        });
    }
    else
    {
        amrex::Abort("error in hs_deferred_update_advection_o4_radau -- should not be here");
    }
}

#ifdef REACTIONS
void
HydroSim::hs_deferred_compute_correction_o4_radau(const Box& bx,
                                Real substep_dt, Real dt,
                                Array4<const Real> const& A_m,
                                const QuadratureNodeTerms& nodes,
                                Array4<Real> const& C,
                                int m_start)
{

    if (m_start == 0)
    {
        amrex::ParallelFor(bx, C.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real integral = (dt/substep_dt) * (1.0_rt/1800.0_rt) *
                ((-35.0_rt*std::sqrt(6.0_rt) + 440.0_rt)*(nodes.advect[1](i,j,k,n) + nodes.react[1](i,j,k,n)) +
                 (-169.0_rt*std::sqrt(6.0_rt) + 296.0_rt)*(nodes.advect[2](i,j,k,n) + nodes.react[2](i,j,k,n)) +
                 (-16.0_rt + 24.0_rt*std::sqrt(6.0_rt))*(nodes.advect[3](i,j,k,n) + nodes.react[3](i,j,k,n)));

            C(i,j,k,n) = (A_m(i,j,k,n) - nodes.advect[0](i,j,k,n)) - nodes.react[1](i,j,k,n) + integral;
        });
    }
    else if (m_start == 1)
    {
        amrex::ParallelFor(bx, C.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real integral = (dt/substep_dt) * (1.0_rt/150.0_rt) *
                ((-12.0_rt + 17.0_rt*std::sqrt(6.0_rt))*(nodes.advect[1](i,j,k,n) + nodes.react[1](i,j,k,n)) +
                 (12.0_rt + 17.0_rt*std::sqrt(6.0_rt))*(nodes.advect[2](i,j,k,n) + nodes.react[2](i,j,k,n)) +
                 (-4.0_rt*std::sqrt(6.0_rt))*(nodes.advect[3](i,j,k,n) + nodes.react[3](i,j,k,n)));

            C(i,j,k,n) = (A_m(i,j,k,n) - nodes.advect[1](i,j,k,n)) - nodes.react[2](i,j,k,n) + integral;
        });
    }
    else if (m_start == 2)
    {
        amrex::ParallelFor(bx, C.nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real integral = (dt/substep_dt) * (1.0_rt/600.0_rt) *
                ((168.0_rt - 73.0_rt*std::sqrt(6.0_rt))*(nodes.advect[1](i,j,k,n) + nodes.react[1](i,j,k,n)) +
                 (120.0_rt + 5.0_rt*std::sqrt(6.0_rt))*(nodes.advect[2](i,j,k,n) + nodes.react[2](i,j,k,n)) +
                 (72.0_rt + 8.0_rt*std::sqrt(6.0_rt))*(nodes.advect[3](i,j,k,n) + nodes.react[3](i,j,k,n)));

            C(i,j,k,n) = (A_m(i,j,k,n) - nodes.advect[2](i,j,k,n)) - nodes.react[3](i,j,k,n) + integral;
        });
    }
    else
    {
        amrex::Abort("error in ca_sdc_compute_C4 -- should not be here");
    }
}
#endif
