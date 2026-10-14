#ifndef TIMESTEP_H
#define TIMESTEP_H

#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <AMReX_MultiFab.H>

#include <eos.H>





namespace timestep {

    enum estdt_type : std::uint8_t { hydro = 0,
                                     mhd,
                                     burning,
                                     radiation,
                                     diffusion
    };

    // Estimate timestep given various constraints

    template <estdt_type T, typename... MFs>
    AMREX_INLINE
    amrex::ValLocPair<amrex::Real, amrex::IntVect>
    estdt (const amrex::GeometryData& geomdata,
           const amrex::MultiFab& maskMF,
           const amrex::MultiFab& stateMF,
           const MFs&... mfs)
    {
        amrex::ignore_unused(maskMF, mfs...);

        constexpr std::size_t N_mfs = sizeof...(MFs);

        auto const& ua = stateMF.const_arrays();

        bool mask_covered_zones = maskMF.isDefined();

        MultiArray4<Real const> empty_arr{};
        const auto& ma = mask_covered_zones ? maskMF.const_arrays() : empty_arr;

        if constexpr (T == hydro) {

            // Validate we don't have any extra MultiFabs.
            static_assert(N_mfs == 0);

            // Courant-condition limited timestep

            auto r = amrex::ParReduce(TypeList<ReduceOpMin>{}, TypeList<ValLocPair<Real, IntVect>>{}, stateMF,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> GpuTuple<ValLocPair<Real, IntVect>>
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
                eos_state.rho = u(i,j,k,URHO);
                eos_state.T = u(i,j,k,UTEMP);
                eos_state.e = u(i,j,k,UEINT) * rhoInv;
                for (int n = 0; n < NumSpec; n++) {
                    eos_state.xn[n] = u(i,j,k,USPEC+n) * rhoInv;
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                    eos_state.aux[n] = u(i,j,k,UAUXV+n) * rhoInv;
                }
#endif

                eos(eos_input_re, eos_state);

                // Compute velocity and then calculate CFL timestep.

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
                    // dx[1] in Spherical2D is just dtheta, need rdtheta for physical length
                    // so just multiply by the smallest r
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

                // The CTU method has a less restrictive timestep than MOL-based
                // schemes (including the true SDC).  Since the simplified SDC
                // solver is based on CTU, we can use its timestep.
                if (hydrosim::time_integration_method == 0 || hydrosim::time_integration_method == 3) {
                    Real dt = amrex::min(dt1, dt2, dt3);
                    return {ValLocPair<Real, IntVect>{dt, idx}};

                } else {
                    // method of lines-style constraint is tougher
                    Real dt_tmp = 1.0_rt/dt1;
#if AMREX_SPACEDIM >= 2
                    dt_tmp += 1.0_rt/dt2;
#endif
#if AMREX_SPACEDIM == 3
                    dt_tmp += 1.0_rt/dt3;
#endif

                    return {ValLocPair<Real, IntVect>{1.0_rt/dt_tmp, idx}};
                }
            });

            return r;

        } // hydro

    }

#ifdef RADIATION
    ///
    /// Radiation hydro timestep
    ///
    amrex::Real estdt_rad (const amrex::MultiFab& stateMF, const amrex::MultiFab& radMF,
                           const amrex::GeometryData& geomdata);
#endif

} // namespace timestep

#endif // TIMESTEP_H
