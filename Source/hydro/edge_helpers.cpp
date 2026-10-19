#include <HydroSim.H>

using namespace amrex;

void
HydroSim::reset_edge_thermo(const Box& bx,
                                Array4<Real> const& edge_state)
{

    int apply_eos = transverse_eos_pressure;
    int reset_eint_flag = transverse_rebuild_rhoe;

    Real temperature_floor = small_temp;
    Real pressure_floor = pressure_cutoff;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

#ifdef RADIATION
        Real prior_pressure = edge_state(i,j,k,QPRES);
#endif

        eos_rep_t eos_state;

        if (reset_eint_flag == 1) {
            if (edge_state(i,j,k,QEINTDENS) < 0.0_rt) {

                eos_state.rho = edge_state(i,j,k,QRHO);
                eos_state.T = temperature_floor;
                for (int n = 0; n < NumSpec; ++n) {
                    eos_state.xn[n] = edge_state(i,j,k,QFS+n);
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; ++n) {
                    eos_state.aux[n] = edge_state(i,j,k,QFX+n);
                }
#endif

                eos(eos_input_rt, eos_state);

                edge_state(i,j,k,QEINTDENS) = edge_state(i,j,k,QRHO) * eos_state.e;
                edge_state(i,j,k,QPRES) = eos_state.p;
            }
        }

        if (apply_eos == 1) {
            eos_state.rho = edge_state(i,j,k,QRHO);
            eos_state.e   = edge_state(i,j,k,QEINTDENS) / edge_state(i,j,k,QRHO);
            eos_state.T   = temperature_floor;
            for (int n = 0; n < NumSpec; ++n) {
                eos_state.xn[n] = edge_state(i,j,k,QFS+n);
            }
#if NAUX_NET > 0
            for (int n = 0; n < NumAux; ++n) {
                eos_state.aux[n] = edge_state(i,j,k,QFX+n);
            }
#endif

            eos(eos_input_re, eos_state);

            edge_state(i,j,k,QEINTDENS) = eos_state.e * eos_state.rho;
            edge_state(i,j,k,QPRES) = amrex::max(eos_state.p, pressure_floor);
        }

#ifdef RADIATION
        edge_state(i,j,k,QPTOTAL) = edge_state(i,j,k,QPTOTAL) + (edge_state(i,j,k,QPRES) - prior_pressure);
#endif

    });

}



void
HydroSim::edge_temp_to_pressure(const Box& bx,
                                Array4<Real> const& qm,
                                Array4<Real> const& qp)
{


    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {


        eos_rep_t eos_state;

        eos_state.rho    = qp(i,j,k,QRHO);
        eos_state.T      = qp(i,j,k,QTEMP);
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = qp(i,j,k,QFS+n);
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; ++n) {
            eos_state.aux[n] = qp(i,j,k,QFX+n);
        }
#endif

        eos(eos_input_rt, eos_state);

        qp(i,j,k,QPRES) = eos_state.p;
        qp(i,j,k,QEINTDENS) = qp(i,j,k,QRHO) * eos_state.e;

        eos_state.rho    = qm(i,j,k,QRHO);
        eos_state.T      = qm(i,j,k,QTEMP);
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = qm(i,j,k,QFS+n);
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; ++n) {
            eos_state.aux[n] = qm(i,j,k,QFX+n);
        }
#endif

        eos(eos_input_rt, eos_state);

        qm(i,j,k,QPRES) = eos_state.p;
        qm(i,j,k,QEINTDENS) = qm(i,j,k,QRHO) * eos_state.e;

    });

}
