#ifndef TIMESTEP_H
#define TIMESTEP_H

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
#ifdef REACTIONS
        else if constexpr (T == burning) {

            if (hydrosim::dtnuc_e > 1.e199_rt && hydrosim::dtnuc_X > 1.e199_rt) {
                IntVect idx(AMREX_D_DECL(0,0,0));
                return {ValLocPair<Real, IntVect>{1.e200_rt, idx}};
            }

            auto r = amrex::ParReduce(TypeList<ReduceOpMin>{}, TypeList<ValLocPair<Real, IntVect>>{}, stateMF,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) -> GpuTuple<ValLocPair<Real, IntVect>>
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

                // Don't consider zones that are masked out.

                if (mask_covered_zones && mask.contains(i,j,k)) {
                    if (mask(i,j,k) == 0.0_rt) {
                        return {ValLocPair<Real, IntVect>{dt_tmp, idx}};
                    }
                }

                // Set a floor on the minimum size of a derivative. This floor
                // is small enough such that it will result in no timestep limiting.

                const Real derivative_floor = 1.e-50_rt;

                // We want to limit the timestep so that it is not larger than
                // dtnuc_e * (e / (de/dt)).  If the timestep factor dtnuc is
                // equal to 1, this says that we don't want the
                // internal energy to change by any more than its current
                // magnitude in the next timestep.
                //
                // If dtnuc is less than one, it controls the fraction we will
                // allow the internal energy to change in this timestep due to
                //  nuclear burning, provided that our instantaneous estimate
                // of the energy release is representative of the full timestep.
                //
                // We also do the same thing for the species, using a timestep
                // limiter dtnuc_X * (X_k / (dX_k/dt)). To prevent changes
                // due to trace isotopes that we probably are not interested in,
                // only apply the limiter to species with an abundance greater
                // than a user-specified threshold.
                //
                // To estimate de/dt and dX/dt, we are going to call the RHS of the
                // burner given the current state data. We need to do an EOS
                // call before we do the RHS call so that we have accurate
                // values for the thermodynamic data like abar, zbar, etc.
                // But we will call in (rho, T) mode, which is inexpensive.

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
                    burn_state.xn[n] = S(i,j,k,USPEC+n) * rhoInv;
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; ++n) {
                    burn_state.aux[n] = S(i,j,k,UAUXV+n) * rhoInv;
                }
#endif

                // Don't consider zones that are outside our burning criteria.

                if (burn_state.T < hydrosim::react_T_min || burn_state.T > hydrosim::react_T_max ||
                    burn_state.rho < hydrosim::react_rho_min || burn_state.rho > hydrosim::react_rho_max) {
                    return {ValLocPair<Real, IntVect>{dt_tmp, idx}};
                }

                // Don't consider zones with shocks if we disabled
                // burning in shocks
#ifdef SHOCK_VAR
                if (S(i,j,k,USHOCK) > 0.0_rt && hydrosim::disable_shock_burning == 1) {
                    return {ValLocPair<Real, IntVect>{dt_tmp, idx}};
                }
#endif

                Real e = burn_state.e;
                Real X[NumSpec];
                for (int n = 0; n < NumSpec; ++n) {
                    X[n] = amrex::max(burn_state.xn[n], small_x);
                }

                eos(eos_input_rt, burn_state);

                Array1D<Real, 1, neqs> ydot;
                actual_rhs(burn_state, ydot);

                Real dedt = ydot(net_ienuc);
                Real dXdt[NumSpec];
                for (int n = 0; n < NumSpec; ++n) {
                    dXdt[n] = ydot(n+1) * aion[n];
                }

                // Apply a floor to the derivatives. This ensures that we don't
                // divide by zero; it also gives us a quick method to disable
                // the timestep limiting, because the floor is small enough
                // that the implied timestep will be very large, and thus
                // ignored compared to other limiters.

                dedt = amrex::max(std::abs(dedt), derivative_floor);

                for (int n = 0; n < NumSpec; ++n) {
                    if (X[n] >= hydrosim::dtnuc_X_threshold) {
                        dXdt[n] = amrex::max(std::abs(dXdt[n]), derivative_floor);
                    } else {
                        dXdt[n] = derivative_floor;
                    }
                }

                    dt_tmp = hydrosim::dtnuc_e * e / dedt;
                for (int n = 0; n < NumSpec; ++n) {
                    dt_tmp = amrex::min(dt_tmp, hydrosim::dtnuc_X * (X[n] / dXdt[n]));
                }

                return {ValLocPair<Real, IntVect>{dt_tmp, idx}};
            });

            return r;

        } // burning
#endif // REACTIONS

    }


} // namespace timestep

#endif // TIMESTEP_H
