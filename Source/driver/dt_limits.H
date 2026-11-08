#ifndef DTLIMIT_H
#define DTLIMIT_H

#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <AMReX_MultiFab.H>

#include <eos.H>




#ifdef REACTIONS
#include <actual_network.H>
#ifdef NEW_NETWORK_IMPLEMENTATION
#include <rhs.H>
#else
#include <actual_rhs.H>
#endif
#include <burner.H>
#endif

namespace timestep {

    template <typename EosT>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void load_eos_state(Array4<Real const> const& u, int i, int j, int k, Real rhoInv, EosT& eos_state)
    {
        eos_state.rho = u(i,j,k,URHO);
        eos_state.T = u(i,j,k,UTEMP);
        eos_state.e = u(i,j,k,UEINT) * rhoInv;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n] = u(i,j,k,UFS+n) * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; n++) {
            eos_state.aux[n] = u(i,j,k,UFX+n) * rhoInv;
        }
#endif
    }

    enum dtlimit_type : std::uint8_t { hydro = 0,
                                     mhd,
                                     burning,
                                     radiation,
                                     diffusion
    };


    using dt_and_cell = amrex::ValLocPair<amrex::Real, amrex::IntVect>;

    template <dtlimit_type T, typename... MFs>
    AMREX_INLINE
    dt_and_cell
    estdt (const amrex::GeometryData& geomdata,
           const amrex::MultiFab& coverMask,
           const amrex::MultiFab& activeState,
           const MFs&... mfs)
    {
        amrex::ignore_unused(coverMask, mfs...);

        constexpr std::size_t count_multifabs = sizeof...(MFs);

        auto const& ua = activeState.const_arrays();

        bool exclude_covered_zones = coverMask.isDefined();

        MultiArray4<Real const> null_view{};
        const auto& ma = exclude_covered_zones ? coverMask.const_arrays() : null_view;

        if constexpr (T == hydro) {

            static_assert(count_multifabs == 0);


            auto r = amrex::ParReduce(TypeList<ReduceOpMin>{}, TypeList<dt_and_cell>{}, activeState,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> GpuTuple<dt_and_cell>
            {
                Array4<Real const> const& u = ua[box_no];

                IntVect idx(AMREX_D_DECL(i,j,k));

                const auto* dx = geomdata.CellSize();
#if AMREX_SPACEDIM >= 2
                const auto* problo = geomdata.ProbLo();
                const auto coord = geomdata.Coord();
#endif

                Real rhoInv = 1.0_rt / u(i,j,k,URHO);

                eos_rep_t eos_state;
                load_eos_state(u, i, j, k, rhoInv, eos_state);

                eos(eos_input_re, eos_state);


                Real ux = u(i,j,k,UMX) * rhoInv;
#if AMREX_SPACEDIM >= 2
                Real uy = u(i,j,k,UMY) * rhoInv;
#endif
#if AMREX_SPACEDIM == 3
                Real uz = u(i,j,k,UMZ) * rhoInv;
#endif

                Real c = eos_state.cs;

                Real dt1 = dx[0]/(c + std::abs(ux));

                Real dt2;
#if AMREX_SPACEDIM >= 2
                dt2 = dx[1]/(c + std::abs(uy));
                if (coord == 2) {
                    dt2 *= problo[0] + 0.5_rt * dx[0];
                }
#else
                dt2 = dt1;
#endif

                Real dt3;
#if AMREX_SPACEDIM == 3
                dt3 = dx[2]/(c + std::abs(uz));
#else
                dt3 = dt1;
#endif

                if (hydrosim::time_scheme == 0 || hydrosim::time_scheme == 3) {
                    Real dt = amrex::min(dt1, dt2, dt3);
                    return {dt_and_cell{dt, idx}};

                } else {
                    Real dt_tmp = 1.0_rt/dt1;
#if AMREX_SPACEDIM >= 2
                    dt_tmp += 1.0_rt/dt2;
#endif
#if AMREX_SPACEDIM == 3
                    dt_tmp += 1.0_rt/dt3;
#endif

                    return {dt_and_cell{1.0_rt/dt_tmp, idx}};
                }
            });

            return r;

        }
#ifdef REACTIONS
        else if constexpr (T == burning) {

            if (hydrosim::burn_dt_energy > 1.e199_rt && hydrosim::burn_dt_massfrac > 1.e199_rt) {
                IntVect idx(AMREX_D_DECL(0,0,0));
                return {dt_and_cell{1.e200_rt, idx}};
            }

            auto r = amrex::ParReduce(TypeList<ReduceOpMin>{}, TypeList<dt_and_cell>{}, activeState,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> GpuTuple<dt_and_cell>
            {
                Real dt_tmp = 1.e200_rt;

                Array4<Real const> const& S = ua[box_no];
                Array4<Real const> const& mask = ma[box_no];

                IntVect idx(AMREX_D_DECL(i,j,k));

                const auto* dx = geomdata.CellSize();
#if AMREX_SPACEDIM >= 2
                const auto* problo = geomdata.ProbLo();
                const auto coord = geomdata.Coord();
#endif


                if (exclude_covered_zones && mask.contains(i,j,k)) {
                    if (mask(i,j,k) == 0.0_rt) {
                        return {dt_and_cell{dt_tmp, idx}};
                    }
                }


                const Real derivative_min_magnitude = 1.e-50_rt;


                Real rhoInv = 1.0_rt / S(i,j,k,URHO);

                burn_t burn_state;

#if AMREX_SPACEDIM == 1
                burn_state.dx = dx[0];
#else
                Real dx1 = dx[1];
#if AMREX_SPACEDIM >= 2
                if (coord == 2) {
                    dx1 *= problo[0] + 0.5_rt * dx[0];
                }
#endif
                burn_state.dx = amrex::min(AMREX_D_DECL(dx[0], dx1, dx[2]));
#endif

                burn_state.rho = S(i,j,k,URHO);
                burn_state.T   = S(i,j,k,UTEMP);
                burn_state.e   = S(i,j,k,UEINT) * rhoInv;
                for (int n = 0; n < NumSpec; ++n) {
                    burn_state.xn[n] = S(i,j,k,UFS+n) * rhoInv;
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; ++n) {
                    burn_state.aux[n] = S(i,j,k,UFX+n) * rhoInv;
                }
#endif


                if (burn_state.T < hydrosim::burn_temp_min || burn_state.T > hydrosim::burn_temp_max ||
                    burn_state.rho < hydrosim::burn_rho_min || burn_state.rho > hydrosim::burn_rho_max) {
                    return {dt_and_cell{dt_tmp, idx}};
                }

#ifdef SHOCK_VAR
                if (S(i,j,k,UFLAG_SHOCK) > 0.0_rt && hydrosim::skip_burning_in_shocks == 1) {
                    return {dt_and_cell{dt_tmp, idx}};
                }
#endif

                Real e = burn_state.e;
                amrex::GpuArray<Real, NumSpec> X;
                for (int n = 0; n < NumSpec; ++n) {
                    X[n] = amrex::max(burn_state.xn[n], small_x);
                }

                eos(eos_input_rt, burn_state);

                Array1D<Real, 1, neqs> ydot;
                actual_rhs(burn_state, ydot);

                Real dedt = ydot(net_ienuc);
                amrex::GpuArray<Real, NumSpec> dXdt;
                for (int n = 0; n < NumSpec; ++n) {
                    dXdt[n] = ydot(n+1) * aion[n];
                }


                dedt = amrex::max(std::abs(dedt), derivative_min_magnitude);

                for (int n = 0; n < NumSpec; ++n) {
                    if (X[n] >= hydrosim::nuc_dt_abund_floor) {
                        dXdt[n] = amrex::max(std::abs(dXdt[n]), derivative_min_magnitude);
                    } else {
                        dXdt[n] = derivative_min_magnitude;
                    }
                }

                    dt_tmp = hydrosim::burn_dt_energy * e / dedt;
                for (int n = 0; n < NumSpec; ++n) {
                    dt_tmp = amrex::min(dt_tmp, hydrosim::burn_dt_massfrac * (X[n] / dXdt[n]));
                }

                return {dt_and_cell{dt_tmp, idx}};
            });

            return r;

        }
#endif

    }


}

#endif
