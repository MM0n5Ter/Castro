#ifndef HYDROSIM_RIEMANN_STATES_H
#define HYDROSIM_RIEMANN_STATES_H

#include <AMReX_REAL.H>

using namespace amrex::literals;

struct InterfaceState
{
    amrex::Real rho;
    amrex::Real p;
    amrex::Real rhoe;
    amrex::Real gamma1_cell;
    amrex::Real un;
    amrex::Real ut;
    amrex::Real utt;

#ifdef RADIATION
    amrex::Real lam[NGROUPS];
    amrex::Real er[NGROUPS];
    amrex::Real p_g;
    amrex::Real energy_density_gas;
    amrex::Real gamma1_gas;
#endif

};


struct InterfaceAux
{
    amrex::Real sound_floor;
    amrex::Real mean_sound_speed;
    amrex::Real boundary_factor;
};


AMREX_INLINE
std::ostream& operator<< (std::ostream& o, InterfaceState const& rs)
{
    o << "rho  = " << rs.rho << std::endl;
    o << "p    = " << rs.p << std::endl;
    o << "rhoe = " << rs.rhoe << std::endl;
    o << "gamma1_cell = " << rs.gamma1_cell << std::endl;
    o << "un   = " << rs.un << std::endl;
    o << "ut   = " << rs.ut << std::endl;
    o << "utt  = " << rs.utt << std::endl;
    return o;
}


AMREX_INLINE
std::ostream& operator<< (std::ostream& o, InterfaceAux const& ra)
{
    o << "sound_floor  = " << ra.sound_floor << std::endl;
    o << "mean_sound_speed    = " << ra.mean_sound_speed << std::endl;
    o << "boundary_factor = " << ra.boundary_factor << std::endl;
    return o;
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
fetch_interface_inputs(const int i, const int j, const int k, const int idir,
                  amrex::Array4<amrex::Real const> const& left_prim_arr,
                  amrex::Array4<amrex::Real const> const& right_prim_arr,
                  amrex::Array4<amrex::Real const> const& qthermo_arr,
                  InterfaceState& ql, InterfaceState& qr, InterfaceAux& interface_aux) {

    const amrex::Real small = 1.e-8_rt;


    ql.rho = amrex::max(left_prim_arr(i,j,k,QRHO), small_dens);
    qr.rho = amrex::max(right_prim_arr(i,j,k,QRHO), small_dens);

    if (idir == 0) {
        ql.un = left_prim_arr(i,j,k,QU);
        ql.ut = left_prim_arr(i,j,k,QV);
        ql.utt = left_prim_arr(i,j,k,QW);

        qr.un = right_prim_arr(i,j,k,QU);
        qr.ut = right_prim_arr(i,j,k,QV);
        qr.utt = right_prim_arr(i,j,k,QW);

        interface_aux.sound_floor = amrex::max(small,
                                 small * amrex::max(qthermo_arr(i,j,k,QC), qthermo_arr(i-1,j,k,QC)));
        interface_aux.mean_sound_speed = 0.5_rt * (qthermo_arr(i,j,k,QC) + qthermo_arr(i-1,j,k,QC));

        ql.gamma1_cell = qthermo_arr(i-1,j,k,QGAMMA1);
        qr.gamma1_cell = qthermo_arr(i,j,k,QGAMMA1);

#ifdef RADIATION
        ql.gamma1_gas = qthermo_arr(i-1,j,k,QGAMMA1G);
        qr.gamma1_gas = qthermo_arr(i,j,k,QGAMMA1G);

        for (int g = 0; g < NGROUPS; g++) {
            ql.lam[g] = qthermo_arr(i-1,j,k,QFLUXLIM+g);
            qr.lam[g] = qthermo_arr(i,j,k,QFLUXLIM+g);
        }
#endif

    } else if (idir == 1) {
        ql.un = left_prim_arr(i,j,k,QV);
        ql.ut = left_prim_arr(i,j,k,QU);
        ql.utt = left_prim_arr(i,j,k,QW);

        qr.un = right_prim_arr(i,j,k,QV);
        qr.ut = right_prim_arr(i,j,k,QU);
        qr.utt = right_prim_arr(i,j,k,QW);

        interface_aux.sound_floor = amrex::max(small,
                                 small * amrex::max(qthermo_arr(i,j,k,QC), qthermo_arr(i,j-1,k,QC)));
        interface_aux.mean_sound_speed = 0.5_rt * (qthermo_arr(i,j,k,QC) + qthermo_arr(i,j-1,k,QC));

        ql.gamma1_cell = qthermo_arr(i,j-1,k,QGAMMA1);
        qr.gamma1_cell = qthermo_arr(i,j,k,QGAMMA1);

#ifdef RADIATION
        ql.gamma1_gas = qthermo_arr(i,j-1,k,QGAMMA1G);
        qr.gamma1_gas = qthermo_arr(i,j,k,QGAMMA1G);

        for (int g = 0; g < NGROUPS; g++) {
            ql.lam[g] = qthermo_arr(i,j-1,k,QFLUXLIM+g);
            qr.lam[g] = qthermo_arr(i,j,k,QFLUXLIM+g);
        }
#endif

    } else {
        ql.un = left_prim_arr(i,j,k,QW);
        ql.ut = left_prim_arr(i,j,k,QU);
        ql.utt = left_prim_arr(i,j,k,QV);

        qr.un = right_prim_arr(i,j,k,QW);
        qr.ut = right_prim_arr(i,j,k,QU);
        qr.utt = right_prim_arr(i,j,k,QV);

        interface_aux.sound_floor = amrex::max(small,
                                 small * amrex::max(qthermo_arr(i,j,k,QC), qthermo_arr(i,j,k-1,QC)));
        interface_aux.mean_sound_speed = 0.5_rt * (qthermo_arr(i,j,k,QC) + qthermo_arr(i,j,k-1,QC));

        ql.gamma1_cell = qthermo_arr(i,j,k-1,QGAMMA1);
        qr.gamma1_cell = qthermo_arr(i,j,k,QGAMMA1);

#ifdef RADIATION
        ql.gamma1_gas = qthermo_arr(i,j,k-1,QGAMMA1G);
        qr.gamma1_gas = qthermo_arr(i,j,k,QGAMMA1G);

        for (int g = 0; g < NGROUPS; g++) {
            ql.lam[g] = qthermo_arr(i,j,k-1,QFLUXLIM+g);
            qr.lam[g] = qthermo_arr(i,j,k,QFLUXLIM+g);
        }
#endif

    }

#ifndef RADIATION
    ql.p = left_prim_arr(i,j,k,QPRES);
    qr.p = right_prim_arr(i,j,k,QPRES);

    ql.rhoe = left_prim_arr(i,j,k,QEINTDENS);
    qr.rhoe = right_prim_arr(i,j,k,QEINTDENS);

#else
    ql.p = left_prim_arr(i,j,k,QPTOTAL);
    ql.rhoe = left_prim_arr(i,j,k,QETOTDENS);
    for (int g = 0; g < NGROUPS; g++) {
      ql.er[g] = left_prim_arr(i,j,k,QRADENG+g);
    }
    ql.p_g = left_prim_arr(i,j,k,QPRES);
    ql.energy_density_gas = left_prim_arr(i,j,k,QEINTDENS);

    qr.p = right_prim_arr(i,j,k,QPTOTAL);
    qr.rhoe = right_prim_arr(i,j,k,QETOTDENS);
    for (int g = 0; g < NGROUPS; g++) {
      qr.er[g] = right_prim_arr(i,j,k,QRADENG+g);
    }
    qr.p_g = right_prim_arr(i,j,k,QPRES);
    qr.energy_density_gas = right_prim_arr(i,j,k,QEINTDENS);
#endif


#ifdef TRUE_SDC
    if (use_recon_gamma1 == 1) {
        ql.gamma1_cell = left_prim_arr(i,j,k,QGC);
        qr.gamma1_cell = right_prim_arr(i,j,k,QGC);
    }
#endif

    if (ql.rhoe <= 0.0_rt || ql.p < pressure_cutoff) {
#ifndef AMREX_USE_GPU
        std::cout <<  "WARNING: (rho e)_l < 0 or pl < pressure_cutoff in Riemann: " << ql.rhoe << " " << ql.p << " " << pressure_cutoff << std::endl;
#endif

        eos_rep_t eos_state;
        eos_state.T = small_temp;
        eos_state.rho = ql.rho;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n] = left_prim_arr(i,j,k,QFS+n);
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; n++) {
            eos_state.aux[n] = left_prim_arr(i,j,k,QFX+n);
        }
#endif

        eos(eos_input_rt, eos_state);

        ql.rhoe = ql.rho * eos_state.e;
        ql.p = eos_state.p;
        ql.gamma1_cell = eos_state.gam1;
    }

    if (qr.rhoe <= 0.0_rt || qr.p < pressure_cutoff) {
#ifndef AMREX_USE_GPU
        std::cout << "WARNING: (rho e)_r < 0 or pr < pressure_cutoff in Riemann: " << qr.rhoe << " " << qr.p << " " << pressure_cutoff << std::endl;
#endif

        eos_rep_t eos_state;
        eos_state.T = small_temp;
        eos_state.rho = qr.rho;
        for (int n = 0; n < NumSpec; n++) {
            eos_state.xn[n] = right_prim_arr(i,j,k,QFS+n);
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; n++) {
            eos_state.aux[n] = right_prim_arr(i,j,k,QFX+n);
        }
#endif

        eos(eos_input_rt, eos_state);

        qr.rhoe = qr.rho * eos_state.e;
        qr.p = eos_state.p;
        qr.gamma1_cell = eos_state.gam1;
    }

}


#endif
