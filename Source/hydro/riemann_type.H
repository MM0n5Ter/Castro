#ifndef HYDROSIM_RIEMANN_H
#define HYDROSIM_RIEMANN_H

#include <AMReX_REAL.H>
#include <AMReX_Array.H>

using namespace amrex::literals;

struct RiemannVars
{
    // velocity components normal and transverse to the interface
    struct Velocity
    {
        amrex::Real un;
        amrex::Real ut;
        amrex::Real utt;
    };

#ifdef RADIATION
    struct RadiationFields
    {
        amrex::Real lam[NGROUPS];
        amrex::Real er[NGROUPS];
        amrex::Real p_g;
        amrex::Real rhoe_g;
        amrex::Real gamma1_gas;
    };
#endif

    amrex::Real rho;
    amrex::Real p;
    amrex::Real rhoe;
    amrex::Real gamma1;
    Velocity vel;

#ifdef RADIATION
    RadiationFields rad;
#endif

};


struct RiemannParams
{
    amrex::Real cs_floor;
    amrex::Real cs_mean;
    amrex::Real boundary_scale;
};


// the index extent of the domain, and whether its low and high faces
// in the solve direction are walls (where the normal velocity on the
// interface is zeroed)

struct DomainEdges
{
    amrex::GpuArray<int, 3> lo;
    amrex::GpuArray<int, 3> hi;
    bool wall_lo;
    bool wall_hi;
};


// the left and right states of one interface's Riemann problem,
// together with the auxiliary data the solvers need

struct RiemannProblem
{
    RiemannVars left;
    RiemannVars right;
    RiemannParams aux;
};


AMREX_INLINE
std::ostream& operator<< (std::ostream& o, RiemannVars const& rs)
{
    o << "rho  = " << rs.rho << std::endl;
    o << "p    = " << rs.p << std::endl;
    o << "rhoe = " << rs.rhoe << std::endl;
    o << "gamma1 = " << rs.gamma1 << std::endl;
    o << "un   = " << rs.vel.un << std::endl;
    o << "ut   = " << rs.vel.ut << std::endl;
    o << "utt  = " << rs.vel.utt << std::endl;
    return o;
}


AMREX_INLINE
std::ostream& operator<< (std::ostream& o, RiemannParams const& ra)
{
    o << "cs_floor  = " << ra.cs_floor << std::endl;
    o << "cs_mean    = " << ra.cs_mean << std::endl;
    o << "boundary_scale = " << ra.boundary_scale << std::endl;
    return o;
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RiemannProblem
load_input_states(const int i, const int j, const int k, const int idir,
                  amrex::Array4<amrex::Real const> const& qleft_arr,
                  amrex::Array4<amrex::Real const> const& qright_arr,
                  amrex::Array4<amrex::Real const> const& qaux_arr) {

    const amrex::Real small = 1.e-8_rt;

    RiemannProblem states;
    RiemannVars& ql = states.left;
    RiemannVars& qr = states.right;
    RiemannParams& riemann_aux = states.aux;

    // fill the states directly from inputs

    ql.rho = amrex::max(qleft_arr(i,j,k,QRHO), small_dens);
    qr.rho = amrex::max(qright_arr(i,j,k,QRHO), small_dens);

    if (idir == 0) {
        ql.vel.un = qleft_arr(i,j,k,QU);
        ql.vel.ut = qleft_arr(i,j,k,QV);
        ql.vel.utt = qleft_arr(i,j,k,QW);

        qr.vel.un = qright_arr(i,j,k,QU);
        qr.vel.ut = qright_arr(i,j,k,QV);
        qr.vel.utt = qright_arr(i,j,k,QW);

        riemann_aux.cs_floor = amrex::max(small,
                                 small * amrex::max(qaux_arr(i,j,k,QC), qaux_arr(i-1,j,k,QC)));
        riemann_aux.cs_mean = 0.5_rt * (qaux_arr(i,j,k,QC) + qaux_arr(i-1,j,k,QC));

        ql.gamma1 = qaux_arr(i-1,j,k,QGAM1);
        qr.gamma1 = qaux_arr(i,j,k,QGAM1);

#ifdef RADIATION
        ql.rad.gamma1_gas = qaux_arr(i-1,j,k,QGAM1GAS);
        qr.rad.gamma1_gas = qaux_arr(i,j,k,QGAM1GAS);

        for (int g = 0; g < NGROUPS; g++) {
            ql.rad.lam[g] = qaux_arr(i-1,j,k,QFLXLIM+g);
            qr.rad.lam[g] = qaux_arr(i,j,k,QFLXLIM+g);
        }
#endif

    } else if (idir == 1) {
        ql.vel.un = qleft_arr(i,j,k,QV);
        ql.vel.ut = qleft_arr(i,j,k,QU);
        ql.vel.utt = qleft_arr(i,j,k,QW);

        qr.vel.un = qright_arr(i,j,k,QV);
        qr.vel.ut = qright_arr(i,j,k,QU);
        qr.vel.utt = qright_arr(i,j,k,QW);

        riemann_aux.cs_floor = amrex::max(small,
                                 small * amrex::max(qaux_arr(i,j,k,QC), qaux_arr(i,j-1,k,QC)));
        riemann_aux.cs_mean = 0.5_rt * (qaux_arr(i,j,k,QC) + qaux_arr(i,j-1,k,QC));

        ql.gamma1 = qaux_arr(i,j-1,k,QGAM1);
        qr.gamma1 = qaux_arr(i,j,k,QGAM1);

#ifdef RADIATION
        ql.rad.gamma1_gas = qaux_arr(i,j-1,k,QGAM1GAS);
        qr.rad.gamma1_gas = qaux_arr(i,j,k,QGAM1GAS);

        for (int g = 0; g < NGROUPS; g++) {
            ql.rad.lam[g] = qaux_arr(i,j-1,k,QFLXLIM+g);
            qr.rad.lam[g] = qaux_arr(i,j,k,QFLXLIM+g);
        }
#endif

    } else {
        ql.vel.un = qleft_arr(i,j,k,QW);
        ql.vel.ut = qleft_arr(i,j,k,QU);
        ql.vel.utt = qleft_arr(i,j,k,QV);

        qr.vel.un = qright_arr(i,j,k,QW);
        qr.vel.ut = qright_arr(i,j,k,QU);
        qr.vel.utt = qright_arr(i,j,k,QV);

        riemann_aux.cs_floor = amrex::max(small,
                                 small * amrex::max(qaux_arr(i,j,k,QC), qaux_arr(i,j,k-1,QC)));
        riemann_aux.cs_mean = 0.5_rt * (qaux_arr(i,j,k,QC) + qaux_arr(i,j,k-1,QC));

        ql.gamma1 = qaux_arr(i,j,k-1,QGAM1);
        qr.gamma1 = qaux_arr(i,j,k,QGAM1);

#ifdef RADIATION
        ql.rad.gamma1_gas = qaux_arr(i,j,k-1,QGAM1GAS);
        qr.rad.gamma1_gas = qaux_arr(i,j,k,QGAM1GAS);

        for (int g = 0; g < NGROUPS; g++) {
            ql.rad.lam[g] = qaux_arr(i,j,k-1,QFLXLIM+g);
            qr.rad.lam[g] = qaux_arr(i,j,k,QFLXLIM+g);
        }
#endif

    }

#ifndef RADIATION
    ql.p = qleft_arr(i,j,k,QPRES);
    qr.p = qright_arr(i,j,k,QPRES);

    ql.rhoe = qleft_arr(i,j,k,QRHOEI);
    qr.rhoe = qright_arr(i,j,k,QRHOEI);

#else
    ql.p = qleft_arr(i,j,k,QPTOTAL);
    ql.rhoe = qleft_arr(i,j,k,QRHOEITOT);
    for (int g = 0; g < NGROUPS; g++) {
      ql.rad.er[g] = qleft_arr(i,j,k,QERAD+g);
    }
    ql.rad.p_g = qleft_arr(i,j,k,QPRES);
    ql.rad.rhoe_g = qleft_arr(i,j,k,QRHOEI);

    qr.p = qright_arr(i,j,k,QPTOTAL);
    qr.rhoe = qright_arr(i,j,k,QRHOEITOT);
    for (int g = 0; g < NGROUPS; g++) {
      qr.rad.er[g] = qright_arr(i,j,k,QERAD+g);
    }
    qr.rad.p_g = qright_arr(i,j,k,QPRES);
    qr.rad.rhoe_g = qright_arr(i,j,k,QRHOEI);
#endif

    // cleaning

#ifdef TRUE_SDC
    if (use_reconstructed_gamma1 == 1) {
        ql.gamma1 = qleft_arr(i,j,k,QGAM1P);
        qr.gamma1 = qright_arr(i,j,k,QGAM1P);
    }
#endif

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
        ql.gamma1 = eos_state.gam1;
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
        qr.gamma1 = eos_state.gam1;
    }

    return states;
}


#endif
