#ifndef HYDROSIM_RIEMANN_H
#define HYDROSIM_RIEMANN_H

#include <AMReX_REAL.H>

using namespace amrex::literals;

struct RiemannState
{
    amrex::Real rho;
    amrex::Real p;
    amrex::Real rhoe;
    amrex::Real gamc;
    amrex::Real un;
    amrex::Real ut;
    amrex::Real utt;


};


struct RiemannAux
{
    amrex::Real csmall;
    amrex::Real cavg;
    amrex::Real bnd_fac;
};


AMREX_INLINE
std::ostream& operator<< (std::ostream& o, RiemannState const& rs)
{
    o << "rho  = " << rs.rho << std::endl;
    o << "p    = " << rs.p << std::endl;
    o << "rhoe = " << rs.rhoe << std::endl;
    o << "gamc = " << rs.gamc << std::endl;
    o << "un   = " << rs.un << std::endl;
    o << "ut   = " << rs.ut << std::endl;
    o << "utt  = " << rs.utt << std::endl;
    return o;
}


AMREX_INLINE
std::ostream& operator<< (std::ostream& o, RiemannAux const& ra)
{
    o << "csmall  = " << ra.csmall << std::endl;
    o << "cavg    = " << ra.cavg << std::endl;
    o << "bnd_fac = " << ra.bnd_fac << std::endl;
    return o;
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
load_input_states(const int i, const int j, const int k, const int idir,
                  amrex::Array4<amrex::Real const> const& qleft_arr,
                  amrex::Array4<amrex::Real const> const& qright_arr,
                  amrex::Array4<amrex::Real const> const& qaux_arr,
                  RiemannState& ql, RiemannState& qr, RiemannAux& raux) {

    const amrex::Real small = 1.e-8_rt;

    // fill the states directly from inputs

    ql.rho = amrex::max(qleft_arr(i,j,k,QRHO), small_dens);
    qr.rho = amrex::max(qright_arr(i,j,k,QRHO), small_dens);

    if (idir == 0) {
        ql.un = qleft_arr(i,j,k,QU);
        ql.ut = qleft_arr(i,j,k,QV);
        ql.utt = qleft_arr(i,j,k,QW);

        qr.un = qright_arr(i,j,k,QU);
        qr.ut = qright_arr(i,j,k,QV);
        qr.utt = qright_arr(i,j,k,QW);

        raux.csmall = amrex::max(small,
                                 small * amrex::max(qaux_arr(i,j,k,QC), qaux_arr(i-1,j,k,QC)));
        raux.cavg = 0.5_rt * (qaux_arr(i,j,k,QC) + qaux_arr(i-1,j,k,QC));

        ql.gamc = qaux_arr(i-1,j,k,QGAM1);
        qr.gamc = qaux_arr(i,j,k,QGAM1);


    } else if (idir == 1) {
        ql.un = qleft_arr(i,j,k,QV);
        ql.ut = qleft_arr(i,j,k,QU);
        ql.utt = qleft_arr(i,j,k,QW);

        qr.un = qright_arr(i,j,k,QV);
        qr.ut = qright_arr(i,j,k,QU);
        qr.utt = qright_arr(i,j,k,QW);

        raux.csmall = amrex::max(small,
                                 small * amrex::max(qaux_arr(i,j,k,QC), qaux_arr(i,j-1,k,QC)));
        raux.cavg = 0.5_rt * (qaux_arr(i,j,k,QC) + qaux_arr(i,j-1,k,QC));

        ql.gamc = qaux_arr(i,j-1,k,QGAM1);
        qr.gamc = qaux_arr(i,j,k,QGAM1);


    } else {
        ql.un = qleft_arr(i,j,k,QW);
        ql.ut = qleft_arr(i,j,k,QU);
        ql.utt = qleft_arr(i,j,k,QV);

        qr.un = qright_arr(i,j,k,QW);
        qr.ut = qright_arr(i,j,k,QU);
        qr.utt = qright_arr(i,j,k,QV);

        raux.csmall = amrex::max(small,
                                 small * amrex::max(qaux_arr(i,j,k,QC), qaux_arr(i,j,k-1,QC)));
        raux.cavg = 0.5_rt * (qaux_arr(i,j,k,QC) + qaux_arr(i,j,k-1,QC));

        ql.gamc = qaux_arr(i,j,k-1,QGAM1);
        qr.gamc = qaux_arr(i,j,k,QGAM1);


    }

    ql.p = qleft_arr(i,j,k,QPRES);
    qr.p = qright_arr(i,j,k,QPRES);

    ql.rhoe = qleft_arr(i,j,k,QRHOEI);
    qr.rhoe = qright_arr(i,j,k,QRHOEI);


    // cleaning


    // sometime we come in here with negative energy or pressure
    // note: reset both in either case, to remain thermo
    // consistent
    if (ql.rhoe <= 0.0_rt || ql.p < small_pres) {
#ifndef AMREX_USE_GPU
        std::cout <<  "WARNING: (rho e)_l < 0 or pl < small_pres in Riemann: " << ql.rhoe << " " << ql.p << " " << small_pres << std::endl;
#endif

        eos_rep_t eos_state;
        eos_state.T = small_temp;
        eos_state.rho = ql.rho;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n] = qleft_arr(i,j,k,QSPEC+n);
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; n++) {
            eos_state.aux[n] = qleft_arr(i,j,k,QAUXV+n);
        }
#endif

        eos(eos_input_rt, eos_state);

        ql.rhoe = ql.rho * eos_state.e;
        ql.p = eos_state.p;
        ql.gamc = eos_state.gam1;
    }

    if (qr.rhoe <= 0.0_rt || qr.p < small_pres) {
#ifndef AMREX_USE_GPU
        std::cout << "WARNING: (rho e)_r < 0 or pr < small_pres in Riemann: " << qr.rhoe << " " << qr.p << " " << small_pres << std::endl;
#endif

        eos_rep_t eos_state;
        eos_state.T = small_temp;
        eos_state.rho = qr.rho;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n] = qright_arr(i,j,k,QSPEC+n);
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; n++) {
            eos_state.aux[n] = qright_arr(i,j,k,QAUXV+n);
        }
#endif

        eos(eos_input_rt, eos_state);

        qr.rhoe = qr.rho * eos_state.e;
        qr.p = eos_state.p;
        qr.gamc = eos_state.gam1;
    }

}


#endif
