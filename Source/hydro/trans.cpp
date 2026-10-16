#include <HydroSim.H>
#include <HydroSim_util.H>


using namespace amrex;

// Index offsets, relative to the zone of an interface state in direction
// idir_n, of the two faces in direction idir_t that bound the transverse
// flux difference: lo is the face on the low side, hi the one on the high
// side.  d = 0 is used for the plus state and d = -1 for the minus state,
// which sees the transverse flux of the zone below the interface.

static void
transverse_face_offsets (int idir_t, int idir_n, int d, Dim3& lo, Dim3& hi)
{
    lo = Dim3{0, 0, 0};
    hi = Dim3{0, 0, 0};

    if (idir_t == 0) {
        hi.x = 1;
    } else if (idir_t == 1) {
        hi.y = 1;
    } else {
        hi.z = 1;
    }

    if (idir_n == 0) {
        lo.x += d;
        hi.x += d;
    } else if (idir_n == 1) {
        lo.y += d;
        hi.y += d;
    } else {
        lo.z += d;
        hi.z += d;
    }
}

// add the transverse flux difference in direction idir_t to the
// interface states in direction idir_n

void
HydroSim::transverse_correct_single(const Box& bx,
                     int idir_t, int idir_n,
                     Array4<Real const> const& qm,
                     Array4<Real> const& qmo,
                     Array4<Real const> const& qp,
                     Array4<Real> const& qpo,
                     Array4<Real const> const& qaux_arr,
                     TransverseFaces const& faces_t,
                     Real hdt, Real coef_dtdx)
{
    // Evaluate the transverse terms for both
    // the minus and plus states.

    transverse_correct_single_impl(bx, idir_t, idir_n, -1,
                        qm, qmo,
                        qaux_arr,
                        faces_t,
                        hdt, coef_dtdx);

    transverse_correct_single_impl(bx, idir_t, idir_n, 0,
                        qp, qpo,
                        qaux_arr,
                        faces_t,
                        hdt, coef_dtdx);

}


void
HydroSim::transverse_correct_single_impl(const Box& bx,  // NOLINT(readability-convert-member-functions-to-static)
                            int idir_t, int idir_n, int d,
                            Array4<Real const> const& q_arr,
                            Array4<Real> const& qo_arr,
                            Array4<Real const> const& qaux_arr,
                            TransverseFaces const& faces_t,
                            Real hdt, Real coef_dtdx)
{

    //       qm|qp
    //         |
    // --------+--------
    //   i-1       i
    //        i-1/2
    //
    // the qp state will see the transverse flux in zone i
    // the qm state will see the transverse flux in zone i-1

    // we account for this with the 'd' variable:
    // d = 0 will do qp and d = -1 will do qm

    // idir_t is the transverse direction and we set il,jl,kl
    // and ir,jr,kr to be the face-centered indices needed for
    // the transverse flux difference

    amrex::ignore_unused(hdt);

    auto const& flux_t = faces_t.flux;
    auto const& q_t = faces_t.qface;
#if AMREX_SPACEDIM == 2
    auto const& area_t = faces_t.area;
    auto const& vol = faces_t.vol;
#endif

#if AMREX_SPACEDIM == 2
    int coord = geom.Coord();
    const auto dx = geom.CellSizeArray();
    const auto problo = geom.ProbLoArray();
#endif

    bool reset_density = transverse_reset_density;
    bool reset_rhoe = transverse_reset_rhoe;
    bool use_eos = transverse_use_eos;
    Real small_p = small_pres;
    Real temp_guess = T_guess;
    Real abundance_tol = abundance_failure_tolerance;

    Dim3 lo;
    Dim3 hi;
    transverse_face_offsets(idir_t, idir_n, d, lo, hi);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

        // We are handling the states at the interface of
        // (i, i+1) in the x-direction, and similarly for
        // the y- and z- directions.

        const int il = i + lo.x;
        const int jl = j + lo.y;  // NOLINT(misc-confusable-identifiers)
        const int kl = k + lo.z;

        const int ir = i + hi.x;
        const int jr = j + hi.y;
        const int kr = k + hi.z;

        // Update all of the passively-advected quantities with the
        // transverse term and convert back to the primitive quantity.

#if AMREX_SPACEDIM == 2
        const Real volinv = 1.0_rt / vol(il,jl,kl);
        const Real rrnew = q_arr(i,j,k,QRHO) - hdt * (area_t(ir,jr,kr) * flux_t(ir,jr,kr,URHO) -
                                                area_t(il,jl,kl) * flux_t(il,jl,kl,URHO)) * volinv;
#else
        const Real rrnew = q_arr(i,j,k,QRHO) - coef_dtdx * (flux_t(ir,jr,kr,URHO) - flux_t(il,jl,kl,URHO));
#endif

        for (int ipassive = 0; ipassive < npassive; ipassive++) {
            const int n = cons_passive_index(ipassive);
            const int nqp = prim_passive_index(ipassive);
#if AMREX_SPACEDIM == 2
            const Real rho_q_upd = q_arr(i,j,k,QRHO) * q_arr(i,j,k,nqp) -
                hdt * (area_t(ir,jr,kr) * flux_t(ir,jr,kr,n) -
                       area_t(il,jl,kl) * flux_t(il,jl,kl,n)) * volinv;
#else
            const Real rho_q_upd = q_arr(i,j,k,QRHO) * q_arr(i,j,k,nqp) -
                coef_dtdx * (flux_t(ir,jr,kr,n) - flux_t(il,jl,kl,n));
#endif
            qo_arr(i,j,k,nqp) = rho_q_upd / rrnew;
        }

        const Real p_face_p  = q_t(ir,jr,kr,IFCPRES);
        const Real p_face_m  = q_t(il,jl,kl,IFCPRES);
        const Real u_face_p  = q_t(ir,jr,kr,IFCU+idir_t);
        const Real u_face_m  = q_t(il,jl,kl,IFCU+idir_t);


        // we need to augment our conserved system with either a p
        // equation or gammae (if we have ppm_predict_gammae = 1) to
        // be able to deal with the general EOS

#if AMREX_SPACEDIM == 2
        Real dup = area_t(ir,jr,kr) * p_face_p * u_face_p - area_t(il,jl,kl) * p_face_m * u_face_m;
        Real du = area_t(ir,jr,kr) * u_face_p - area_t(il,jl,kl) * u_face_m;
#else
        Real dup = p_face_p * u_face_p - p_face_m * u_face_m;
        Real du = u_face_p - u_face_m;
#endif
        Real pav = 0.5_rt * (p_face_p + p_face_m);

        // this is the gas gamma_1
        Real gamma1 = qaux_arr(il,jl,kl,QGAM1);


        // Convert to conservation form
        Real rrn = q_arr(i,j,k,QRHO);
        Real run = rrn * q_arr(i,j,k,QU);
        Real rvn = rrn * q_arr(i,j,k,QV);
        Real rwn = rrn * q_arr(i,j,k,QW);
        Real ke_o = 0.5_rt * rrn * (q_arr(i,j,k,QU) * q_arr(i,j,k,QU) + q_arr(i,j,k,QV) * q_arr(i,j,k,QV) + q_arr(i,j,k,QW) * q_arr(i,j,k,QW));
        Real ren = q_arr(i,j,k,QRHOEI) + ke_o;

#if AMREX_SPACEDIM == 2
        // Add transverse predictor

        Real rho_tr = rrnew;

        // Note that pressure may be treated specially here, depending on
        // the geometry.  Our y-interface equation for (rho u) is:
        //
        //  d(rho u)/dt + d(rho u v)/dy = - 1/r d(r rho u u)/dr - dp/dr
        //
        // in cylindrical coords -- note that the p term is not
        // in a divergence for UMX in the x-direction, so there
        // are no area factors.
        //
        // Similarly for Spherical2D geometry, we have:
        // d(rho u)/dt + d(rho u v)/(rdtheta) = -1/r^2 d(r^2 rho u u)/dr - dp/dr
        // d(rho v)/dt + d(rho u v)/dr = -1/(r sin(theta)) d(sin(theta) rho v v)/dtheta - 1/r dp/dtheta
        //
        // For these non-cartesian geometries, we do not
        // include p in our definition of the flux in the
        // x-direction for Cylindrical2D or both x- and y-direction for spherical 2D
        // So we need to fix this now.

        Real rhou_tr = run - hdt * (area_t(ir,jr,kr) * flux_t(ir,jr,kr,UMX) -
                                   area_t(il,jl,kl) * flux_t(il,jl,kl,UMX)) * volinv;
        if (idir_t == 0 && !momentum_flux_has_pressure(0, idir_t, coord)) {
            rhou_tr = rhou_tr - coef_dtdx * (p_face_p - p_face_m);
        }
        Real rhov_tr = rvn - hdt * (area_t(ir,jr,kr) * flux_t(ir,jr,kr,UMY) -
                                   area_t(il,jl,kl) * flux_t(il,jl,kl,UMY)) * volinv;
        if (idir_t == 1 && !momentum_flux_has_pressure(1, idir_t, coord)) {
            Real r = problo[0] + static_cast<Real>(il + 0.5_rt) * dx[0];
            rhov_tr = rhov_tr - coef_dtdx / r * (p_face_p - p_face_m);
        }
        Real rhow_tr = rwn - hdt * (area_t(ir,jr,kr) * flux_t(ir,jr,kr,UMZ) -
                                   area_t(il,jl,kl) * flux_t(il,jl,kl,UMZ)) * volinv;
        Real rhoE_tr = ren - hdt * (area_t(ir,jr,kr) * flux_t(ir,jr,kr,UEDEN) -
                                   area_t(il,jl,kl) * flux_t(il,jl,kl,UEDEN)) * volinv;


#else
        // Add transverse predictor
        Real rho_tr = rrnew;
        Real rhou_tr = run - coef_dtdx * (flux_t(ir,jr,kr,UMX)   - flux_t(il,jl,kl,UMX));
        Real rhov_tr = rvn - coef_dtdx * (flux_t(ir,jr,kr,UMY)   - flux_t(il,jl,kl,UMY));
        Real rhow_tr = rwn - coef_dtdx * (flux_t(ir,jr,kr,UMZ)   - flux_t(il,jl,kl,UMZ));
        Real rhoE_tr = ren - coef_dtdx * (flux_t(ir,jr,kr,UEDEN) - flux_t(il,jl,kl,UEDEN));
#endif

        // Reset to original value if adding transverse terms made density negative
        bool reset_state = false;
        if (reset_density && rho_tr < 0.0_rt) {
            rho_tr = rrn;
            rhou_tr = run;
            rhov_tr = rvn;
            rhow_tr = rwn;
            rhoE_tr = ren;
            for (int ipassive = 0; ipassive < npassive; ++ipassive) {
                int nqp = prim_passive_index(ipassive);
                qo_arr(i,j,k,nqp) = q_arr(i,j,k,nqp);
            }
            reset_state = true;
        }

        // Reset to original value if adding transverse terms made any mass fraction invalid.

        for (int n = 0; n < NumSpec; ++n) {
            if (qo_arr(i,j,k,n+QSPEC) > 1.0_rt + abundance_tol ||
                qo_arr(i,j,k,n+QSPEC) < -abundance_tol) {
                rho_tr = rrn;
                rhou_tr = run;
                rhov_tr = rvn;
                rhow_tr = rwn;
                rhoE_tr = ren;
                for (int ipassive = 0; ipassive < npassive; ++ipassive) {
                    int nqp = prim_passive_index(ipassive);
                    qo_arr(i,j,k,nqp) = q_arr(i,j,k,nqp);
                }
                reset_state = true;
                break;
            }
        }

        // Convert back to primitive form
        qo_arr(i,j,k,QRHO) = rho_tr;
        const Real rhoinv = 1.0_rt / rho_tr;

        qo_arr(i,j,k,QU) = rhou_tr * rhoinv;
        qo_arr(i,j,k,QV) = rhov_tr * rhoinv;
        qo_arr(i,j,k,QW) = rhow_tr * rhoinv;

        // note: we run the risk of (rho e) being negative here
        Real ke_tr = 0.5_rt * (rhou_tr * rhou_tr + rhov_tr * rhov_tr + rhow_tr * rhow_tr) * rhoinv;
        qo_arr(i,j,k,QRHOEI) = rhoE_tr - ke_tr;

        if (!reset_state) {
            // do the transverse terms for p, gamma, and rhoe, as necessary

            if (reset_rhoe == 1 && qo_arr(i,j,k,QRHOEI) <= 0.0_rt) {
                // If it is negative, reset the internal energy by
                // using the discretized expression for updating (rho e).
#if AMREX_SPACEDIM == 2
                qo_arr(i,j,k,QRHOEI) = q_arr(i,j,k,QRHOEI) - hdt * (area_t(ir,jr,kr) * flux_t(ir,jr,kr,UEINT) -
                                                            area_t(il,jl,kl) * flux_t(il,jl,kl,UEINT) +
                                                            pav * du) * volinv;
#else
                qo_arr(i,j,k,QRHOEI) = q_arr(i,j,k,QRHOEI) - coef_dtdx * (flux_t(ir,jr,kr,UEINT) - flux_t(il,jl,kl,UEINT) + pav * du);
#endif
            }

            // If (rho e) is negative by this point,
            // set it back to the original interface state,
            // which turns off the transverse correction.

            if (qo_arr(i,j,k,QRHOEI) <= 0.0_rt) {
                qo_arr(i,j,k,QRHOEI) = q_arr(i,j,k,QRHOEI);
            }

            // Pretend QRHOEI has been fixed
            // We can get pressure update via eos or using p-evolution equation

            if (use_eos) {

                // With the EOS route:
                eos_rep_t eos_state;
                eos_state.rho = rho_tr;
                eos_state.e = qo_arr(i,j,k,QRHOEI) * rhoinv;
                eos_state.T = temp_guess;

                for (int n = 0; n < NumSpec; n++) {
                    eos_state.xn[n] = qo_arr(i,j,k,QSPEC+n);
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                    eos_state.aux[n] = qo_arr(i,j,k,QAUXV+n);
                }
#endif
                eos(eos_input_re, eos_state);

                Real pres_tr = eos_state.p;
                qo_arr(i,j,k,QPRES) = amrex::max(pres_tr, small_p);

            } else {

                // Add the transverse term to the p evolution eq here.
#if AMREX_SPACEDIM == 2
                // the divergences here, dup and du, already have area factors
                Real pres_tr = q_arr(i,j,k,QPRES) - hdt * (dup + pav * du * (gamma1 - 1.0_rt)) * volinv;
#else
                Real pres_tr = q_arr(i,j,k,QPRES) - coef_dtdx * (dup + pav * du * (gamma1 - 1.0_rt));
#endif
                qo_arr(i,j,k,QPRES) = amrex::max(pres_tr, small_p);
            }
        }
        else {
            qo_arr(i,j,k,QPRES) = q_arr(i,j,k,QPRES);
            qo_arr(i,j,k,QRHOEI) = q_arr(i,j,k,QRHOEI);
        }


    });

}



void
HydroSim::transverse_correct_final(const Box& bx,
                    int idir_n, int idir_t1, int idir_t2,
                    Array4<Real const> const& qm,
                    Array4<Real> const& qmo,
                    Array4<Real const> const& qp,
                    Array4<Real> const& qpo,
                    Array4<Real const> const& qaux_arr,
                    TransverseFaces const& faces_t1,
                    TransverseFaces const& faces_t2,
                    Real coef_dtdx_t1, Real coef_dtdx_t2)
{
    // Evaluate the transverse terms for both
    // the minus and plus states.

    transverse_correct_final_impl(bx, idir_n, idir_t1, idir_t2, -1,
                       qm, qmo,
                       qaux_arr,
                       faces_t1,
                       faces_t2,
                       coef_dtdx_t1, coef_dtdx_t2);

    transverse_correct_final_impl(bx, idir_n, idir_t1, idir_t2, 0,
                       qp, qpo,
                       qaux_arr,
                       faces_t1,
                       faces_t2,
                       coef_dtdx_t1, coef_dtdx_t2);

}



void
HydroSim::transverse_correct_final_impl(const Box& bx,  // NOLINT(readability-convert-member-functions-to-static)
                           int idir_n, int idir_t1, int idir_t2, int d,
                           Array4<Real const> const& q_arr,
                           Array4<Real> const& qo_arr,
                           Array4<Real const> const& qaux_arr,
                           TransverseFaces const& faces_t1,
                           TransverseFaces const& faces_t2,
                           Real coef_dtdx_t1, Real coef_dtdx_t2)
{

    auto const& flux_t1 = faces_t1.flux;
    auto const& q_t1 = faces_t1.qface;
    auto const& flux_t2 = faces_t2.flux;
    auto const& q_t2 = faces_t2.qface;

    bool reset_density = transverse_reset_density;
    bool reset_rhoe = transverse_reset_rhoe;
    bool use_eos = transverse_use_eos;
    Real small_p = small_pres;
    Real temp_guess = T_guess;
    Real abundance_tol = abundance_failure_tolerance;

    Dim3 lo_t1;
    Dim3 hi_t1;
    transverse_face_offsets(idir_t1, idir_n, d, lo_t1, hi_t1);

    Dim3 lo_t2;
    Dim3 hi_t2;
    transverse_face_offsets(idir_t2, idir_n, d, lo_t2, hi_t2);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

        // the normal state, offset for the plus/minus state
        const int iln = i + lo_t1.x;
        const int jln = j + lo_t1.y;
        const int kln = k + lo_t1.z;

        // the first transverse state
        const int il_t1 = i + lo_t1.x;
        const int jl_t1 = j + lo_t1.y;
        const int kl_t1 = k + lo_t1.z;

        const int ir_t1 = i + hi_t1.x;
        const int jr_t1 = j + hi_t1.y;
        const int kr_t1 = k + hi_t1.z;

        // the second transverse state
        const int il_t2 = i + lo_t2.x;
        const int jl_t2 = j + lo_t2.y;
        const int kl_t2 = k + lo_t2.z;

        const int ir_t2 = i + hi_t2.x;
        const int jr_t2 = j + hi_t2.y;
        const int kr_t2 = k + hi_t2.z;

        // Update all of the passively-advected quantities with the
        // transverse terms and convert back to the primitive quantity.

        const Real rrn = q_arr(i,j,k,QRHO);
        const Real rrnew = rrn - coef_dtdx_t1 * (flux_t1(ir_t1,jr_t1,kr_t1,URHO) -
                                        flux_t1(il_t1,jl_t1,kl_t1,URHO))
                          - coef_dtdx_t2 * (flux_t2(ir_t2,jr_t2,kr_t2,URHO) -
                                        flux_t2(il_t2,jl_t2,kl_t2,URHO));

        for (int ipassive = 0; ipassive < npassive; ++ipassive) {
            const int n = cons_passive_index(ipassive);
            const int nqp = prim_passive_index(ipassive);

            Real rho_q_o = rrn * q_arr(i,j,k,nqp);
            Real rho_q_tr = rho_q_o - coef_dtdx_t1 * (flux_t1(ir_t1,jr_t1,kr_t1,n) -
                                              flux_t1(il_t1,jl_t1,kl_t1,n))
                                - coef_dtdx_t2 * (flux_t2(ir_t2,jr_t2,kr_t2,n) -
                                              flux_t2(il_t2,jl_t2,kl_t2,n));

            qo_arr(i,j,k,nqp) = rho_q_tr / rrnew;
        }

        // Add the transverse differences to the normal states for the
        // fluid variables.

        const Real p_face_t1p  = q_t1(ir_t1,jr_t1,kr_t1,IFCPRES);
        const Real p_face_t1m  = q_t1(il_t1,jl_t1,kl_t1,IFCPRES);
        const Real u_face_t1p  = q_t1(ir_t1,jr_t1,kr_t1,IFCU+idir_t1);
        const Real u_face_t1m  = q_t1(il_t1,jl_t1,kl_t1,IFCU+idir_t1);

        const Real p_face_t2p  = q_t2(ir_t2,jr_t2,kr_t2,IFCPRES);
        const Real p_face_t2m  = q_t2(il_t2,jl_t2,kl_t2,IFCPRES);
        const Real u_face_t2p  = q_t2(ir_t2,jr_t2,kr_t2,IFCU+idir_t2);
        const Real u_face_t2m  = q_t2(il_t2,jl_t2,kl_t2,IFCU+idir_t2);

        const Real d_pu_t1 = p_face_t1p * u_face_t1p - p_face_t1m * u_face_t1m;
        const Real p_avg_t1 = 0.5_rt * (p_face_t1p + p_face_t1m);
        const Real du_t1 = u_face_t1p - u_face_t1m;
        const Real p_corr_t1 = coef_dtdx_t1 * (d_pu_t1 + p_avg_t1 * du_t1 * (qaux_arr(iln,jln,kln,QGAM1) - 1.0_rt));

        const Real d_pu_t2 = p_face_t2p * u_face_t2p - p_face_t2m * u_face_t2m;
        const Real p_avg_t2 = 0.5_rt * (p_face_t2p + p_face_t2m);
        const Real du_t2 = u_face_t2p - u_face_t2m;
        const Real p_corr_t2 = coef_dtdx_t2 * (d_pu_t2 + p_avg_t2 * du_t2 * (qaux_arr(iln,jln,kln,QGAM1) - 1.0_rt));


        // Convert to conservation form
        Real run = rrn * q_arr(i,j,k,QU);
        Real rvn = rrn * q_arr(i,j,k,QV);
        Real rwn = rrn * q_arr(i,j,k,QW);
        Real ke_o = 0.5_rt * rrn * (q_arr(i,j,k,QU) * q_arr(i,j,k,QU) + q_arr(i,j,k,QV) * q_arr(i,j,k,QV) + q_arr(i,j,k,QW) * q_arr(i,j,k,QW));
        Real ren = q_arr(i,j,k,QRHOEI) + ke_o;

        // Add transverse predictor
        Real rho_tr = rrnew;
        Real rhou_tr = run - coef_dtdx_t1 * (flux_t1(ir_t1,jr_t1,kr_t1,UMX) -
                                        flux_t1(il_t1,jl_t1,kl_t1,UMX))
                          - coef_dtdx_t2 * (flux_t2(ir_t2,jr_t2,kr_t2,UMX) -
                                        flux_t2(il_t2,jl_t2,kl_t2,UMX));
        Real rhov_tr = rvn - coef_dtdx_t1 * (flux_t1(ir_t1,jr_t1,kr_t1,UMY) -
                                        flux_t1(il_t1,jl_t1,kl_t1,UMY))
                          - coef_dtdx_t2 * (flux_t2(ir_t2,jr_t2,kr_t2,UMY) -
                                        flux_t2(il_t2,jl_t2,kl_t2,UMY));
        Real rhow_tr = rwn - coef_dtdx_t1 * (flux_t1(ir_t1,jr_t1,kr_t1,UMZ) -
                                        flux_t1(il_t1,jl_t1,kl_t1,UMZ))
                          - coef_dtdx_t2 * (flux_t2(ir_t2,jr_t2,kr_t2,UMZ) -
                                        flux_t2(il_t2,jl_t2,kl_t2,UMZ));
        Real rhoE_tr = ren - coef_dtdx_t1 * (flux_t1(ir_t1,jr_t1,kr_t1,UEDEN) -
                                        flux_t1(il_t1,jl_t1,kl_t1,UEDEN))
                          - coef_dtdx_t2 * (flux_t2(ir_t2,jr_t2,kr_t2,UEDEN) -
                                        flux_t2(il_t2,jl_t2,kl_t2,UEDEN));

        // Reset to original value if adding transverse terms
        // made density negative
        bool reset_state = false;
        if (reset_density && rho_tr < 0.0_rt) {
            rho_tr = rrn;
            rhou_tr = run;
            rhov_tr = rvn;
            rhow_tr = rwn;
            rhoE_tr = ren;
            for (int ipassive = 0; ipassive < npassive; ++ipassive) {
                int nqp = prim_passive_index(ipassive);
                qo_arr(i,j,k,nqp) = q_arr(i,j,k,nqp);
            }
            reset_state = true;
        }

        // Reset to original value if adding transverse terms made any mass fraction invalid.

        for (int n = 0; n < NumSpec; ++n) {
            if (qo_arr(i,j,k,n+QSPEC) > 1.0_rt + abundance_tol ||
                qo_arr(i,j,k,n+QSPEC) < -abundance_tol) {
                rho_tr = rrn;
                rhou_tr = run;
                rhov_tr = rvn;
                rhow_tr = rwn;
                rhoE_tr = ren;
                for (int ipassive = 0; ipassive < npassive; ++ipassive) {
                    int nqp = prim_passive_index(ipassive);
                    qo_arr(i,j,k,nqp) = q_arr(i,j,k,nqp);
                }
                reset_state = true;
                break;
            }
        }

        qo_arr(i,j,k,QRHO) = rho_tr;

        qo_arr(i,j,k,QU) = rhou_tr / rho_tr;
        qo_arr(i,j,k,QV) = rhov_tr / rho_tr;
        qo_arr(i,j,k,QW) = rhow_tr / rho_tr;

        // note: we run the risk of (rho e) being negative here
        const Real ke_tr = 0.5_rt * (rhou_tr * rhou_tr + rhov_tr * rhov_tr + rhow_tr * rhow_tr) / rho_tr;
        qo_arr(i,j,k,QRHOEI) = rhoE_tr - ke_tr;

        if (!reset_state) {
            if (reset_rhoe == 1 && qo_arr(i,j,k,QRHOEI) <= 0.0_rt) {
                // If it is negative, reset the internal energy by
                // using the discretized expression for updating
                // (rho e).
                qo_arr(i,j,k,QRHOEI) = q_arr(i,j,k,QRHOEI)
                                 - coef_dtdx_t1 * (flux_t1(ir_t1,jr_t1,kr_t1,UEINT) -
                                               flux_t1(il_t1,jl_t1,kl_t1,UEINT) + p_avg_t1 * du_t1)
                                 - coef_dtdx_t2 * (flux_t2(ir_t2,jr_t2,kr_t2,UEINT) -
                                               flux_t2(il_t2,jl_t2,kl_t2,UEINT) + p_avg_t2 * du_t2);
            }

            // If (rho e) is negative by this point,
            // set it back to the original interface state,
            // which turns off the transverse correction.

            if (qo_arr(i,j,k,QRHOEI) <= 0.0_rt) {
                qo_arr(i,j,k,QRHOEI) = q_arr(i,j,k,QRHOEI);
            }

            // Pretend QRHOEI has been fixed
            // We can get pressure update via eos or using p-evolution equation

            if (use_eos) {

                // With the EOS route:
                eos_rep_t eos_state;
                eos_state.rho = rho_tr;
                eos_state.e = qo_arr(i,j,k,QRHOEI) / rho_tr;
                eos_state.T = temp_guess;

                for (int n = 0; n < NumSpec; n++) {
                    eos_state.xn[n] = qo_arr(i,j,k,QSPEC+n);
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; n++) {
                    eos_state.aux[n] = qo_arr(i,j,k,QAUXV+n);
                }
#endif
                eos(eos_input_re, eos_state);
                qo_arr(i,j,k,QPRES) = eos_state.p;

            } else {

                // add the transverse term to the p evolution eq here
                Real pres_tr = q_arr(i,j,k,QPRES) - p_corr_t1 - p_corr_t2;
                qo_arr(i,j,k,QPRES) = pres_tr;
            }

        }
        else {
            qo_arr(i,j,k,QPRES) = q_arr(i,j,k,QPRES);
            qo_arr(i,j,k,QRHOEI) = q_arr(i,j,k,QRHOEI);
        }

        qo_arr(i,j,k,QPRES) = amrex::max(qo_arr(i,j,k,QPRES), small_p);


    });

}
