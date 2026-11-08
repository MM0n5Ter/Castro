#ifndef transport_util_H
#define transport_util_H

#include <HydroSim_helpers.H>



namespace hydro {

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void source_to_primitive(int i, int j, int k, const Real dt,
                 Array4<Real const> const& field_arr,
                 Array4<Real const> const& prim_arr,
                 Array4<Real const> const& previous_source,
                 Array4<Real const> const& source_correction,
                 Array4<Real> const& prim_source)
{

    amrex::ignore_unused(dt);

    for (int n = 0; n < NQSOURCE; ++n) {
      prim_source(i,j,k,n) = 0.0_rt;
    }


    Real sourceTerms[NUM_SOURCE] = {0.0_rt};

    for (int n = 0; n < NUM_SOURCE; n++) {

        if (time_scheme == CornerTrace && source_half_step == 1) {
            if (n == UMX || n == UMY || n == UMZ) {
                sourceTerms[n] += 0.5 * dt * source_correction(i,j,k,n);
            }
        } else if (time_scheme == SimplifiedIterCorrection  && source_half_step == 1) {
            sourceTerms[n] += source_correction(i,j,k,n);
        }

        sourceTerms[n] += previous_source(i,j,k,n);
    }

    Real rhoinv = 1.0_rt / prim_arr(i,j,k,QRHO);

    eos_rep_t eos_state;
    eos_state.T = prim_arr(i,j,k,QTEMP);
    eos_state.rho = prim_arr(i,j,k,QRHO);
    eos_state.e = prim_arr(i,j,k,QEINTDENS) * rhoinv;
    for (int n = 0; n < NumSpec; n++) {
      eos_state.xn[n]  = field_arr(i,j,k,UFS+n) * rhoinv;
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
      eos_state.aux[n] = field_arr(i,j,k,UFX+n) * rhoinv;
    }
#endif

    eos(eos_input_re, eos_state);

    prim_source(i,j,k,QRHO) = sourceTerms[URHO];
    prim_source(i,j,k,QU) = (sourceTerms[UMX] - prim_arr(i,j,k,QU) * prim_source(i,j,k,QRHO)) * rhoinv;
    prim_source(i,j,k,QV) = (sourceTerms[UMY] - prim_arr(i,j,k,QV) * prim_source(i,j,k,QRHO)) * rhoinv;
    prim_source(i,j,k,QW) = (sourceTerms[UMZ] - prim_arr(i,j,k,QW) * prim_source(i,j,k,QRHO)) * rhoinv;
    prim_source(i,j,k,QEINTDENS) = sourceTerms[UEINT];
    prim_source(i,j,k,QPRES ) = eos_state.dpde *
      (prim_source(i,j,k,QEINTDENS) - prim_arr(i,j,k,QEINTDENS) * prim_source(i,j,k,QRHO)*rhoinv) *
      rhoinv + eos_state.dpdr_e * prim_source(i,j,k,QRHO);

}


template<class T, class U>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void cell_to_primitive(int i, int j, int k,
                               Array4<Real const> const& uin,
                               T& q, U& qthermo, const bool include_passives)
{
#ifndef AMREX_USE_GPU
    if (uin(i,j,k,URHO) <= 0.0_rt) {
        std::cout << std::endl;
        std::cout << ">>> Error: cell_to_primitive " << i << " " << j << " " << k << std::endl;
        std::cout << ">>> ... negative density " << uin(i,j,k,URHO) << std::endl;
        amrex::Error("Error:: hydro::cell_to_primitive()");
    } else if (uin(i,j,k,URHO) < hydrosim::small_dens) {
        std::cout << std::endl;
        std::cout << ">>> Error: cell_to_primitive " << i << " " << j << " " << k << std::endl;
        std::cout << ">>> ... small density " << uin(i,j,k,URHO) << std::endl;
        amrex::Error("Error:: hydro::cell_to_primitive()");
    }
#endif

    q(QRHO) = uin(i,j,k,URHO);
    Real rhoinv = 1.0_rt / q(QRHO);

    q(QU) = uin(i,j,k,UMX) * rhoinv;
    q(QV) = uin(i,j,k,UMY) * rhoinv;
    q(QW) = uin(i,j,k,UMZ) * rhoinv;



    Real kineng = 0.5_rt * q(QRHO) * (q(QU) * q(QU) + q(QV) * q(QV) + q(QW) * q(QW));

    if ((uin(i,j,k,UEDEN) - kineng) > hydrosim::dual_energy_switch_hi*uin(i,j,k,UEDEN)) {
        q(QEINTDENS) = (uin(i,j,k,UEDEN) - kineng) * rhoinv;
    } else {
        q(QEINTDENS) = uin(i,j,k,UEINT) * rhoinv;
    }

    q(QTEMP) = uin(i,j,k,UTEMP);

    if (include_passives) {
        for (int passive_index = 0; passive_index < npassive; passive_index++) {
            int n  = passive_cons_index(passive_index);
            int iq = passive_prim_index(passive_index);
            q(iq) = uin(i,j,k,n) * rhoinv;
        }
    }

    eos_rep_t eos_state;
    eos_state.T = q(QTEMP);
    eos_state.rho = q(QRHO);
    eos_state.e = q(QEINTDENS);
    for (int n = 0; n < NumSpec; n++) {
      eos_state.xn[n]  = uin(i,j,k,UFS+n) * rhoinv;
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
        eos_state.aux[n] = uin(i,j,k,UFX+n) * rhoinv;
    }
#endif

    eos(eos_input_re, eos_state);

    q(QTEMP) = eos_state.T;
    q(QEINTDENS) = eos_state.e * q(QRHO);
    q(QPRES) = eos_state.p;


    qthermo(QGAMMA1) = eos_state.gam1;
    qthermo(QC) = eos_state.cs;
}

}

#endif
