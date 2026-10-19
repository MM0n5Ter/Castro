#ifndef primitive_filter_H
#define primitive_filter_H

#include <AMReX.H>
#include <filter.H>

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void primitive_filter_cell (int i, int j, int k,
                Array4<Real> const& filtered_state,
                Array4<Real> const& Snew,
                int temp_filter_index, int S,
                const GeometryData& geomdata,
                Real time)
{
#ifndef AMREX_USE_GPU
    AMREX_ALWAYS_ASSERT(AMREX_SPACEDIM == 1);
#endif

    using namespace filter;

    bool filtered = false;

    if (time < problem::filter_timemax || Snew(i,j,k,URHO) < problem::filter_rhomax) {
        if (temp_filter_index == 1) {
            filtered = true;
            Snew(i,j,k,URHO ) = ff1(0) *  filtered_state(i  ,j,k,URHO ) +
                                ff1(1) * (filtered_state(i-1,j,k,URHO ) + filtered_state(i+1,j,k,URHO ));
            Snew(i,j,k,UMX  ) = ff1(0) *  filtered_state(i  ,j,k,UMX  ) +
                                ff1(1) * (filtered_state(i-1,j,k,UMX  ) + filtered_state(i+1,j,k,UMX  ));
            Snew(i,j,k,UTEMP) = ff1(0) *  filtered_state(i  ,j,k,UTEMP) +
                                ff1(1) * (filtered_state(i-1,j,k,UTEMP) + filtered_state(i+1,j,k,UTEMP));
        }
        else if (temp_filter_index == 2) {
            filtered = true;
            Snew(i,j,k,URHO ) = ff2(0,S) *  filtered_state(i  ,j,k,URHO ) +
                                ff2(1,S) * (filtered_state(i-1,j,k,URHO ) + filtered_state(i+1,j,k,URHO )) +
                                ff2(2,S) * (filtered_state(i-2,j,k,URHO ) + filtered_state(i+2,j,k,URHO ));
            Snew(i,j,k,UMX  ) = ff2(0,S) *  filtered_state(i  ,j,k,UMX  ) +
                                ff2(1,S) * (filtered_state(i-1,j,k,UMX  ) + filtered_state(i+1,j,k,UMX  )) +
                                ff2(2,S) * (filtered_state(i-2,j,k,UMX  ) + filtered_state(i+2,j,k,UMX  ));
            Snew(i,j,k,UTEMP) = ff2(0,S) *  filtered_state(i  ,j,k,UTEMP) +
                                ff2(1,S) * (filtered_state(i-1,j,k,UTEMP) + filtered_state(i+1,j,k,UTEMP)) +
                                ff2(2,S) * (filtered_state(i-2,j,k,UTEMP) + filtered_state(i+2,j,k,UTEMP));
        }
        else if (temp_filter_index == 3) {
            filtered = true;
            Snew(i,j,k,URHO ) = ff3(0,S) *  filtered_state(i  ,j,k,URHO ) +
                                ff3(1,S) * (filtered_state(i-1,j,k,URHO ) + filtered_state(i+1,j,k,URHO )) +
                                ff3(2,S) * (filtered_state(i-2,j,k,URHO ) + filtered_state(i+2,j,k,URHO )) +
                                ff3(3,S) * (filtered_state(i-3,j,k,URHO ) + filtered_state(i+3,j,k,URHO ));
            Snew(i,j,k,UMX  ) = ff3(0,S) *  filtered_state(i  ,j,k,UMX  ) +
                                ff3(1,S) * (filtered_state(i-1,j,k,UMX  ) + filtered_state(i+1,j,k,UMX  )) +
                                ff3(2,S) * (filtered_state(i-2,j,k,UMX  ) + filtered_state(i+2,j,k,UMX  )) +
                                ff3(3,S) * (filtered_state(i-3,j,k,UMX  ) + filtered_state(i+3,j,k,UMX  ));
            Snew(i,j,k,UTEMP) = ff3(0,S) *  filtered_state(i  ,j,k,UTEMP) +
                                ff3(1,S) * (filtered_state(i-1,j,k,UTEMP) + filtered_state(i+1,j,k,UTEMP)) +
                                ff3(2,S) * (filtered_state(i-2,j,k,UTEMP) + filtered_state(i+2,j,k,UTEMP)) +
                                ff3(3,S) * (filtered_state(i-3,j,k,UTEMP) + filtered_state(i+3,j,k,UTEMP));
        }
        else if (temp_filter_index == 4) {
            filtered = true;
            Snew(i,j,k,URHO ) = ff4(0,S) *  filtered_state(i  ,j,k,URHO ) +
                                ff4(1,S) * (filtered_state(i-1,j,k,URHO ) + filtered_state(i+1,j,k,URHO )) +
                                ff4(2,S) * (filtered_state(i-2,j,k,URHO ) + filtered_state(i+2,j,k,URHO )) +
                                ff4(3,S) * (filtered_state(i-3,j,k,URHO ) + filtered_state(i+3,j,k,URHO )) +
                                ff4(4,S) * (filtered_state(i-4,j,k,URHO ) + filtered_state(i+4,j,k,URHO ));
            Snew(i,j,k,UMX  ) = ff4(0,S) *  filtered_state(i  ,j,k,UMX  ) +
                                ff4(1,S) * (filtered_state(i-1,j,k,UMX  ) + filtered_state(i+1,j,k,UMX  )) +
                                ff4(2,S) * (filtered_state(i-2,j,k,UMX  ) + filtered_state(i+2,j,k,UMX  )) +
                                ff4(3,S) * (filtered_state(i-3,j,k,UMX  ) + filtered_state(i+3,j,k,UMX  )) +
                                ff4(4,S) * (filtered_state(i-4,j,k,UMX  ) + filtered_state(i+4,j,k,UMX  ));
            Snew(i,j,k,UTEMP) = ff4(0,S) *  filtered_state(i  ,j,k,UTEMP) +
                                ff4(1,S) * (filtered_state(i-1,j,k,UTEMP) + filtered_state(i+1,j,k,UTEMP)) +
                                ff4(2,S) * (filtered_state(i-2,j,k,UTEMP) + filtered_state(i+2,j,k,UTEMP)) +
                                ff4(3,S) * (filtered_state(i-3,j,k,UTEMP) + filtered_state(i+3,j,k,UTEMP)) +
                                ff4(4,S) * (filtered_state(i-4,j,k,UTEMP) + filtered_state(i+4,j,k,UTEMP));
        }
    }

    if (filtered) {

        Snew(i,j,k,URHO ) = amrex::max(Snew(i,j,k,URHO ), hydrosim::small_dens);
        Snew(i,j,k,UTEMP) = amrex::max(Snew(i,j,k,UTEMP), hydrosim::small_temp);

        eos_t eos_state;

        eos_state.rho = Snew(i,j,k,URHO);
        eos_state.T = Snew(i,j,k,UTEMP);

        Real rhotmpInv = 1.0_rt / filtered_state(i,j,k,URHO);
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = filtered_state(i,j,k,UFS+n) * rhotmpInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; ++n) {
            eos_state.aux[n] = filtered_state(i,j,k,UFX+n) * rhotmpInv;
        }
#endif

        eos(eos_input_rt, eos_state);

        Snew(i,j,k,UEINT) = Snew(i,j,k,URHO) * eos_state.e;
        Snew(i,j,k,UEDEN) = Snew(i,j,k,UEINT) + 0.5_rt *
                                                (Snew(i,j,k,UMX) * Snew(i,j,k,UMX)) /
                                                 Snew(i,j,k,URHO);
        for (int n = 0; n < NumAdvected; ++n) {
            Snew(i,j,k,UFA+n) = filtered_state(i,j,k,UFA+n) * rhotmpInv * Snew(i,j,k,URHO);
        }
        for (int n = 0; n < NumSpec; ++n) {
            Snew(i,j,k,UFS+n) = eos_state.xn[n] * Snew(i,j,k,URHO);
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; ++n) {
            Snew(i,j,k,UFX+n) = eos_state.aux[n] * Snew(i,j,k,URHO);
        }
#endif

    }
}

#endif
