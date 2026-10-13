#ifndef TIMESTEP_H
#define TIMESTEP_H

#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Reduce.H>
#include <AMReX_MultiFab.H>

#include <eos.H>


#ifdef MHD
#include <mhd_util.H>
#endif



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
                    eos_state.xn[n] = u(i,j,k,UFS+n) * rhoInv;
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                    eos_state.aux[n] = u(i,j,k,UFX+n) * rhoInv;
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
                if (castro::time_integration_method == 0 || castro::time_integration_method == 3) {
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
#ifdef MHD
        else if constexpr (T == mhd) {

            // MHD timestep limiter

            // Validate we have three extra MultiFabs (Bx, By, and Bz).

            static_assert(N_mfs == 3);

            auto const& [bx, by, bz] = std::tie(mfs...);

            auto const& bxa = bx.const_arrays();
            auto const& bya = by.const_arrays();
            auto const& bza = bz.const_arrays();

            auto r = amrex::ParReduce(TypeList<ReduceOpMin>{}, TypeList<ValLocPair<Real, IntVect>>{}, stateMF,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> GpuTuple<ValLocPair<Real, IntVect>>
            {

                Array4<Real const> const& u_arr = ua[box_no];

                Array4<Real const> const& bx_arr = bxa[box_no];
                Array4<Real const> const& by_arr = bya[box_no];
                Array4<Real const> const& bz_arr = bza[box_no];

                IntVect idx(AMREX_D_DECL(i,j,k));

                const auto* dx = geomdata.CellSize();
#if AMREX_SPACEDIM >= 2
                const auto* problo = geomdata.ProbLo();
                const auto coord = geomdata.Coord();
#endif

                Real rhoInv = 1.0_rt / u_arr(i,j,k,URHO);
                Real bcx = 0.5_rt * (bx_arr(i+1,j,k) + bx_arr(i,j,k));
                Real bcy = 0.5_rt * (by_arr(i,j+1,k) + by_arr(i,j,k));
                Real bcz = 0.5_rt * (bz_arr(i,j,k+1) + bz_arr(i,j,k));

                Real ux = u_arr(i,j,k,UMX) * rhoInv;
                Real uy = u_arr(i,j,k,UMY) * rhoInv;
                Real uz = u_arr(i,j,k,UMZ) * rhoInv;

                eos_rep_t eos_state;
                eos_state.rho = u_arr(i,j,k,URHO);
                eos_state.e = u_arr(i,j,k,UEINT) * rhoInv;
                eos_state.T = u_arr(i,j,k,UTEMP);
                for (int n = 0; n < NumSpec; n++) {
                  eos_state.xn[n] = u_arr(i,j,k,UFS+n) * rhoInv;
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                  eos_state.aux[n] = u_arr(i,j,k,UFX+n) * rhoInv;
                }
#endif

                eos(eos_input_re, eos_state);

                Real e  = u_arr(i,j,k,UEINT) * rhoInv;

                Real as = eos_state.gam1 * eos_state.p * rhoInv;
                Real ca = (bcx*bcx + bcy*bcy + bcz*bcz) * rhoInv;

                Real cx;
                Real cy;
                Real cz;

                if (e > 0_rt) {
                  Real cad = bcx*bcx * rhoInv;
                  eos_soundspeed_mhd(cx, as, ca, cad);

                  cad = bcy*bcy * rhoInv;
                  eos_soundspeed_mhd(cy, as, ca, cad);

                  cad = bcz*bcz * rhoInv;
                  eos_soundspeed_mhd(cz, as, ca, cad);

                } else {
                  cx = 0.0_rt;
                  cy = 0.0_rt;
                  cz = 0.0_rt;
                }

                Real dt1 = dx[0]/(cx + std::abs(ux));

                Real dt2;
#if AMREX_SPACEDIM >= 2
                dt2 = dx[1]/(cy + std::abs(uy));
                if (coord == 2) {
                    dt2 *= problo[0] + 0.5_rt * dx[0];
                }
#else
                dt2 = dt1;
#endif

                Real dt3;
#if AMREX_SPACEDIM == 3
                dt3 = dx[2]/(cz + std::abs(uz));
#else
                dt3 = dt1;
#endif

                return {ValLocPair<Real, IntVect>{amrex::min(dt1, dt2, dt3), idx}};

            });

            return r;

        } // mhd
#endif // MHD

    }


} // namespace timestep

#endif // TIMESTEP_H
