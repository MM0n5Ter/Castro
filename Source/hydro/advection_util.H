#ifndef advection_util_H
#define advection_util_H

#include <HydroSim_util.H>



namespace hydro {

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void src_to_prim(int i, int j, int k, const Real dt,
                 Array4<Real const> const& U_arr,
                 Array4<Real const> const& q_arr,
                 Array4<Real const> const& old_src,
                 Array4<Real const> const& src_corr,
                 Array4<Real> const& srcQ)
{

    amrex::ignore_unused(dt);

    for (int n = 0; n < NQSRC; ++n) {
      srcQ(i,j,k,n) = 0.0_rt;
    }

    // the conserved source may have a predictor that time-centers it

    Real srcU[NSRC] = {0.0_rt};

    for (int n = 0; n < NSRC; n++) {

        if (time_integration_method == CornerTransportUpwind && source_term_predictor == 1) {
            if (n == UMX || n == UMY || n == UMZ) {
                srcU[n] += 0.5 * dt * src_corr(i,j,k,n);
            }
        } else if (time_integration_method == SimplifiedSpectralDeferredCorrections  && source_term_predictor == 1) {
            srcU[n] += src_corr(i,j,k,n);
        }

        srcU[n] += old_src(i,j,k,n);
    }

    Real rhoinv = 1.0_rt / q_arr(i,j,k,QRHO);

    // get the needed derivatives
    eos_rep_t eos_state;
    eos_state.T = q_arr(i,j,k,QTEMP);
    eos_state.rho = q_arr(i,j,k,QRHO);
    eos_state.e = q_arr(i,j,k,QRHOEI) * rhoinv;
    for (int n = 0; n < NumSpec; n++) {
      eos_state.xn[n]  = U_arr(i,j,k,USPEC+n) * rhoinv;
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
      eos_state.aux[n] = U_arr(i,j,k,UAUXV+n) * rhoinv;
    }
#endif

    eos(eos_input_re, eos_state);

    srcQ(i,j,k,QRHO) = srcU[URHO];
    srcQ(i,j,k,QU) = (srcU[UMX] - q_arr(i,j,k,QU) * srcQ(i,j,k,QRHO)) * rhoinv;
    srcQ(i,j,k,QV) = (srcU[UMY] - q_arr(i,j,k,QV) * srcQ(i,j,k,QRHO)) * rhoinv;
    srcQ(i,j,k,QW) = (srcU[UMZ] - q_arr(i,j,k,QW) * srcQ(i,j,k,QRHO)) * rhoinv;
    srcQ(i,j,k,QRHOEI) = srcU[UEINT];
    srcQ(i,j,k,QPRES ) = eos_state.dpde *
      (srcQ(i,j,k,QRHOEI) - q_arr(i,j,k,QRHOEI) * srcQ(i,j,k,QRHO)*rhoinv) *
      rhoinv + eos_state.dpdr_e * srcQ(i,j,k,QRHO);

}

///
/// The fields of the other physics modules that the conversion to
/// primitive variables reads.  A member only exists when its module
/// is built.
///
/// Bx, By, Bz  components of the magnetic field (if USE_MHD=TRUE)
/// Erin        radiation energy (if USE_RAD=TRUE)
/// lam         radiation flux limiter (if USE_RAD=TRUE)
///
struct CoupledFields
{
};

///
/// Convert conservative variables to primitive variables. The method is
/// templated so that q/qaux can either be either Array4 or Array1D.
///
/// @param i         x index
/// @param j         y index
/// @param k         z index
/// @param uin       input conserved state
/// @param coupled   magnetic field and radiation fields, as built
/// @param q         output primitive state
/// @param qaux      output auxiliary quantities
///

template<class T, class U>
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void conservative_to_primitive(int i, int j, int k,
                               Array4<Real const> const& uin,
                               CoupledFields const& coupled,
                               T& q, U& qaux, const bool fill_passives)
{
#ifndef AMREX_USE_GPU
    if (uin(i,j,k,URHO) <= 0.0_rt) {
        std::cout << std::endl;
        std::cout << ">>> Error: conservative_to_primitive " << i << " " << j << " " << k << std::endl;
        std::cout << ">>> ... negative density " << uin(i,j,k,URHO) << std::endl;
        amrex::Error("Error:: hydro::conservative_to_primitive()");
    } else if (uin(i,j,k,URHO) < hydrosim::small_dens) {
        std::cout << std::endl;
        std::cout << ">>> Error: conservative_to_primitive " << i << " " << j << " " << k << std::endl;
        std::cout << ">>> ... small density " << uin(i,j,k,URHO) << std::endl;
        amrex::Error("Error:: hydro::conservative_to_primitive()");
    }
#endif

    q(QRHO) = uin(i,j,k,URHO);
    Real rhoinv = 1.0_rt / q(QRHO);

    q(QU) = uin(i,j,k,UMX) * rhoinv;
    q(QV) = uin(i,j,k,UMY) * rhoinv;
    q(QW) = uin(i,j,k,UMZ) * rhoinv;


    // Get the internal energy, which we'll use for
    // determining the pressure.  We use a dual energy
    // formalism. If (E - K) < eta1 and eta1 is suitably
    // small, then we risk serious numerical truncation error
    // in the internal energy.  Therefore we'll use the result
    // of the separately updated internal energy equation.
    // Otherwise, we'll set e = E - K.

    Real kineng = 0.5_rt * q(QRHO) * (q(QU) * q(QU) + q(QV) * q(QV) + q(QW) * q(QW));

    if ((uin(i,j,k,UEDEN) - kineng) > hydrosim::dual_energy_eta1*uin(i,j,k,UEDEN)) {
        q(QRHOEI) = (uin(i,j,k,UEDEN) - kineng) * rhoinv;
    } else {
        q(QRHOEI) = uin(i,j,k,UEINT) * rhoinv;
    }

    q(QTEMP) = uin(i,j,k,UTEMP);

    if (fill_passives) {
        // Load passively advected quantities into q
        for (int ipassive = 0; ipassive < npassive; ipassive++) {
            int n  = cons_passive_index(ipassive);
            int iq = prim_passive_index(ipassive);
            q(iq) = uin(i,j,k,n) * rhoinv;
        }
    }

    // get gamma1, p, T, c, csml using q state
    eos_rep_t eos_state;
    eos_state.T = q(QTEMP);
    eos_state.rho = q(QRHO);
    eos_state.e = q(QRHOEI);
    for (int n = 0; n < NumSpec; n++) {
      eos_state.xn[n]  = uin(i,j,k,USPEC+n) * rhoinv;
    }
#if NAUX_NET > 0
    for (int n = 0; n < NumAux; n++) {
        eos_state.aux[n] = uin(i,j,k,UAUXV+n) * rhoinv;
    }
#endif

    eos(eos_input_re, eos_state);

    q(QTEMP) = eos_state.T;
    q(QRHOEI) = eos_state.e * q(QRHO);
    q(QPRES) = eos_state.p;


    qaux(QGAM1) = eos_state.gam1;
    qaux(QC) = eos_state.cs;
}

///
/// the actual work routine that does the conversion of conserved to primitive
/// variables.
///
/// @param bx        the box to operate over
/// @param time      current time
/// @param uin       input conserved state
/// @param coupled   magnetic field and radiation fields, as built
/// @param q_arr     output primitive state
/// @param qaux_arr  output auxiliary quantities
///
void ctoprim(const Box& bx,
             const Real time,
             Array4<Real const> const& uin,
             CoupledFields const& coupled,
             Array4<Real> const& q_arr,
             Array4<Real> const& qaux_arr);

void store_face_state(const Box& bx,
                         Array4<Real const> const& q_riemann,
                         Array4<Real> const& qface);

void recompute_face_thermo(const Box& bx,
                             Array4<Real> const& qedge);

void face_pressure_from_temp(const Box& bx,
                             Array4<Real> const& qm,
                             Array4<Real> const& qp);

///
/// when using simplified-SDC, add the SDC source term predictor to
/// the interface states
///
/// @param bx          the box to operate over
/// @param idir        coordinate direction
/// @param dt          timestep
/// @param qleft       left state at the interface
/// @param qright      right state at the interface
/// @param sdc_source  primitive variable SDC source array
///
void add_sdc_source_to_states(const Box& bx, const int idir, const Real dt,
                              Array4<Real> const& qleft,
                              Array4<Real> const& qright,
                              Array4<Real const> const& sdc_src);

void normalize_species_fluxes(const Box& bx, Array4<Real> const& flux);

///
/// The fluxes the small-density limiter works on are either per unit
/// area and time (the limiter scales them by dt * area), or already
/// integrated over the face area and the timestep.
///
enum class FluxForm : std::uint8_t {
    PerUnitArea = 0,
    Integrated
};

void limit_fluxes_for_density_floor(const Box& bx,
                                      int idir,
                                      Array4<Real const> const& u,
                                      Array4<Real const> const& vol,
                                      Array4<Real> const& flux,
                                      Array4<Real const> const& area,
                                      Real dt,
                                      FluxForm form);

}

#endif
