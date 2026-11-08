#ifndef DTLIMIT_H
#define DTLIMIT_H

#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <AMReX_MultiFab.H>

#include <eos.H>

#ifdef DIFFUSION
#include <conductivity.H>
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
#ifdef DIFFUSION
        else if constexpr (T == diffusion) {


            const Real diffusion_min_density = hydrosim::conduction_cutoff_density;
            const Real timestep_ceiling = hydrosim::max_dt;
            const Real cfl_limit = hydrosim::cfl;

            auto r = amrex::ParReduce(TypeList<ReduceOpMin>{}, TypeList<dt_and_cell>{}, activeState,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> GpuTuple<dt_and_cell>
            {
                Array4<Real const> const& state_fab_view = ua[box_no];

                IntVect idx(AMREX_D_DECL(i,j,k));

                const auto* dx = geomdata.CellSize();
#if AMREX_SPACEDIM >= 2
                const auto* problo = geomdata.ProbLo();
                const auto coord = geomdata.Coord();
#endif

                if (state_fab_view(i,j,k,URHO) > diffusion_min_density) {

                    Real rho_inv = 1.0_rt/state_fab_view(i,j,k,URHO);

                    eos_t eos_state;

                    load_eos_state(state_fab_view, i, j, k, rho_inv, eos_state);

                    eos(eos_input_re, eos_state);

                    conductivity(eos_state);

                    Real D = eos_state.conductivity * rho_inv / eos_state.cv;

                    Real dt1 = 0.5_rt * dx[0]*dx[0] / D;

                    Real dt2;
#if AMREX_SPACEDIM >= 2
                    dt2 = 0.5_rt * dx[1]*dx[1] / D;
                    if (coord == 2) {
                        Real rc = problo[0] + 0.5_rt * dx[0];
                        dt2 *= rc * rc;
                    }
#else
                    dt2 = dt1;
#endif

                    Real dt3;
#if AMREX_SPACEDIM >= 3
                    dt3 = 0.5_rt * dx[2]*dx[2] / D;
#else
                    dt3 = dt1;
#endif

                    return {dt_and_cell{amrex::min(dt1, dt2, dt3), idx}};

                } else {
                    return {dt_and_cell{timestep_ceiling/cfl_limit, idx}};
                }
            });

            return r;

        }
#endif

    }


}

#endif
