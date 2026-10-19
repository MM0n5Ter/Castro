
#ifdef TRUE_SDC
///
/// Calculate primitive variables from conserved variables (uses StateData)
///
/// @param time     current time
///
    void make_primitives(const amrex::Real time);
#endif

///
/// convert the conservative state cell averages to primitive cell
/// averages with 4th order accuracy
///
/// @param time     current time
///
    void make_primitives_fourth(const amrex::Real time);

///
/// Check to see if the CFL condition has been violated
///
/// @param State MultiFab of conserved variables
/// @param dt timestep
///
    void verify_courant_limit(const amrex::MultiFab& State, const amrex::Real dt);

///
/// this constructs the hydrodynamic source (essentially the flux
/// divergence) using the CTU framework for unsplit hydrodynamics
///
/// @param time     current time
/// @param dt       timestep
///
    step_result assemble_ctu_hydro_source(amrex::Real time, amrex::Real dt);

///
/// this constructs the hydrodynamic source (essentially the flux
/// divergence) using method of lines integration.  The output, is the
/// divergence of the fluxes, A = -div{F(U)}
///
/// @param time      current time
/// @param dt        timestep
/// @param A_divergence  divergence of the fluxes
///
    void assemble_mol_hydro_source(amrex::Real time, amrex::Real dt, amrex::MultiFab& A_divergence);


///
/// when using simplified-SDC, add the SDC source term predictor to
/// the interface states
///
/// @param bx          the box to operate over
/// @param idir        coordinate direction
/// @param dt          timestep
/// @param prim_left       left state at the interface
/// @param prim_right      right state at the interface
/// @param sdc_source  primitive variable SDC source array
///
    static void inject_deferred_source(const Box& bx, const int idir, const Real dt,
                                      Array4<Real> const& prim_left,
                                      Array4<Real> const& prim_right,
                                      Array4<Real const> const& deferred_source);

///
/// convert the conserved form of the source terms, S(U), into source terms
/// for the primitive variable equations, S(q).
///
/// @param bx      the box to operate over
/// @param dt      timestep
/// @param prim_arr   primitive variable state array
/// @param previous_source conserved source array
#ifndef TRUE_SDC
/// @param source_correction source correction array
#endif
/// @param prim_source    primitive variable source array
///
    void source_to_primitive(const amrex::Box& bx, const Real dt,
                     amrex::Array4<amrex::Real const> const& prim_arr,
                     amrex::Array4<amrex::Real const> const& previous_source,
#ifndef TRUE_SDC
                     amrex::Array4<amrex::Real const> const& source_correction,
#endif
                     amrex::Array4<amrex::Real> const& prim_source);

///
/// the actual work routine that does the conversion of conserved to primitive
/// variables.
///
/// @param bx        the box to operate over
/// @param time      current time
/// @param uin       input conserved state
/// @param Bx        x-component of magnetic field (if USE_MHD=TRUE)
/// @param By        y-component of magnetic field (if USE_MHD=TRUE)
/// @param Bz        z-component of magnetic field (if USE_MHD=TRUE)
/// @param RadEnergyIn      radiation energy (if USE_RAD=TRUE)
/// @param lam       radiation flux limiter (if USE_RAD=TRUE)
/// @param prim_arr     output primitive state
/// @param qthermo_arr  output auxiliary quantities
///
    static void ctoprim(const amrex::Box& bx,
                 const amrex::Real time,
                 amrex::Array4<amrex::Real const> const& uin,
#ifdef MHD
                 amrex::Array4<amrex::Real const> const& Bx,
                 amrex::Array4<amrex::Real const> const& By,
                 amrex::Array4<amrex::Real const> const& Bz,
#endif
#ifdef RADIATION
                 amrex::Array4<amrex::Real const> const& RadEnergyIn,
                 amrex::Array4<amrex::Real const> const& lam,
#endif
                 amrex::Array4<amrex::Real> const& prim_arr,
                 amrex::Array4<amrex::Real> const& qthermo_arr);


///
/// A multidimensional shock detection algorithm
///
/// @param bx         the box to operate over
/// @param prim_arr      the primitive variable state
/// @param U_scr_arr  the conservative state sources
/// @param shk        the shock flag (1 = shock, 0 = no shock)
///
    void shock(const amrex::Box& bx,
               amrex::Array4<amrex::Real const> const& prim_arr,
               amrex::Array4<amrex::Real const> const& U_source_arr,
               amrex::Array4<amrex::Real> const& shk);

///
/// Compute the node-centered velocity divergence (DU)_{i-1/2.j-1/2,k-1/2}
///
/// @param bx     the box to operate over
/// @param prim_arr  input primitive variable state
/// @param div    output velocity divergence
///
    void divu(const amrex::Box& bx,
              amrex::Array4<amrex::Real const> const& prim_arr,
              amrex::Array4<amrex::Real> const& div);

///
/// Update the flux with the artificial viscosity.  This is a 2nd order
/// accurate implementation.
///
/// @param bx    the box to operate over
/// @param idir  the coordinate direction (0 = x, 1 = y, 2 = z)
/// @param div   the node-centered velocity divergence
/// @param uin   the conserved state
/// @param flux  the flux in direction idir
///
    void apply_viscosity(const amrex::Box& bx,
                  const int idir,
                  amrex::Array4<amrex::Real const> const& div,
                  amrex::Array4<amrex::Real const> const& uin,
                  amrex::Array4<amrex::Real> const& flux);

#ifdef RADIATION
///
/// Update the radiation flux with the artificial viscosity.  This is a 2nd order
/// accurate implementation.
///
/// @param bx       the box to operate over
/// @param idir     the coordinate direction (0 = x, 1 = y, 2 = z)
/// @param div      the node-centered velocity divergence
/// @param RadEnergyIn     the radiation energy
/// @param radiative_flux  the radiation energy flux in direction idir
///
    void apply_viscosity_radiative(const amrex::Box& bx,
                      const int idir,
                      amrex::Array4<amrex::Real const> const& div,
                      amrex::Array4<amrex::Real const> const& RadEnergyIn,
                      amrex::Array4<amrex::Real> const& radiative_flux);


    void
    rad_cons_update(const amrex::Box& bx,
                   amrex::Array4<amrex::Real> const& update,
                   amrex::Array4<amrex::Real const> const& RadEnergyIn,
                   amrex::Array4<amrex::Real> const& RadEnergyOut,
                   amrex::Array4<amrex::Real const> const& radiative_flux1,
                   amrex::Array4<amrex::Real const> const& qx,
                   amrex::Array4<amrex::Real const> const& area1,
#if AMREX_SPACEDIM >= 2
                   amrex::Array4<amrex::Real const> const& radiative_flux2,
                   amrex::Array4<amrex::Real const> const& qy,
                   amrex::Array4<amrex::Real const> const& area2,
#endif
#if AMREX_SPACEDIM == 3
                   amrex::Array4<amrex::Real const> const& radiative_flux3,
                   amrex::Array4<amrex::Real const> const& qz,
                   amrex::Array4<amrex::Real const> const& face_area3,
#endif
                   int& rad_step_count,
                   amrex::Array4<amrex::Real const> const& vol,
                   const amrex::Real dt);

#endif

///
/// Compute the normal left and right primitive variable interface
/// states at each interface using the piecewise parabolic method with
/// characteristic tracing.  This implementation is for pure
/// hydrodynamics
///
/// @param bx       the box to operate over
/// @param vbx      the valid region box (excludes ghost cells)
/// @param field_arr    the conserved variable state
/// @param inv_density_arr the inverse density array
/// @param prim_arr    the primitive variable state
/// @param qthermo_arr the auxiliary state
/// @param prim_source     the source terms for the primitive variable equations
/// @param qxm      left interface state in x, q_{i-1/2,j,k,L}
/// @param qxp      right interface state in x, q_{i-1/2,j,k,R}
/// @param qym      left interface state in y, q_{i,j-1/2,k,L}
/// @param qyp      right interface state in y, q_{i,j-1/2,k,R}
/// @param qzm      left interface state in z, q_{i,j,k-1/2,L}
/// @param qzp      right interface state in z, q_{i,j,k-1/2,R}
/// @param logAreaRateX   the geometric factor d(log area) in x (r) dir
/// @param logAreaRateY   the geometric factor d(log area) in y (theta) dir
/// @param dt       timestep
///
    void corner_ppm_states(const amrex::Box& bx, const amrex::Box& vbx,
                        amrex::Array4<amrex::Real const> const& field_arr,
                        amrex::Array4<amrex::Real const> const& inv_density_arr,
                        amrex::Array4<amrex::Real const> const& prim_arr,
                        amrex::Array4<amrex::Real const> const& qthermo_arr,
                        amrex::Array4<amrex::Real const> const& prim_source,
                        amrex::Array4<amrex::Real> const& qxm,
                        amrex::Array4<amrex::Real> const& qxp,
#if AMREX_SPACEDIM >= 2
                        amrex::Array4<amrex::Real> const& qym,
                        amrex::Array4<amrex::Real> const& qyp,
#endif
#if AMREX_SPACEDIM == 3
                        amrex::Array4<amrex::Real> const& qzm,
                        amrex::Array4<amrex::Real> const& qzp,
#endif
#if AMREX_SPACEDIM < 3
                        amrex::Array4<amrex::Real const> const& logAreaRateX,
#endif
#if AMREX_SPACEDIM == 2
                        amrex::Array4<amrex::Real const> const& logAreaRateY,
#endif
                        const amrex::Real dt);

///
/// Compute the normal left and right primitive variable interface
/// states at each interface using piecewise linear reconstruction with
/// characteristic tracing.  This implementation is for pure
/// hydrodynamics
///
/// @param bx       the box to operate over
/// @param vbx      the valid region box (excludes ghost cells)
/// @param field_arr    the conserved variable state
/// @param inv_density_arr the inverse density array
/// @param prim_arr    the primitive variable state
/// @param qthermo_arr the auxiliary state
/// @param prim_source     the source terms for the primitive variable equations
/// @param qxm      left interface state in x, q_{i-1/2,j,k,L}
/// @param qxp      right interface state in x, q_{i-1/2,j,k,R}
/// @param qym      left interface state in y, q_{i,j-1/2,k,L}
/// @param qyp      right interface state in y, q_{i,j-1/2,k,R}
/// @param qzm      left interface state in z, q_{i,j,k-1/2,L}
/// @param qzp      right interface state in z, q_{i,j,k-1/2,R}
/// @param logAreaRateX   the geometric factor d(log area) in x (r) dir
/// @param logAreaRateY   the geometric factor d(log area) in y (theta) dir
/// @param dt       timestep
///
    void corner_plm_states(const amrex::Box& bx, const amrex::Box& vbx,
                        amrex::Array4<amrex::Real const> const& field_arr,
                        amrex::Array4<amrex::Real const> const& inv_density_arr,
                        amrex::Array4<amrex::Real const> const& prim_arr,
                        amrex::Array4<amrex::Real const> const& qthermo_arr,
                        amrex::Array4<amrex::Real const> const& prim_source,
                        amrex::Array4<amrex::Real> const& qxm,
                        amrex::Array4<amrex::Real> const& qxp,
#if AMREX_SPACEDIM >= 2
                        amrex::Array4<amrex::Real> const& qym,
                        amrex::Array4<amrex::Real> const& qyp,
#endif
#if AMREX_SPACEDIM == 3
                        amrex::Array4<amrex::Real> const& qzm,
                        amrex::Array4<amrex::Real> const& qzp,
#endif
#if AMREX_SPACEDIM < 3
                        amrex::Array4<amrex::Real const> const& logAreaRateX,
#endif
#if AMREX_SPACEDIM == 2
                        amrex::Array4<amrex::Real const> const& logAreaRateY,
#endif
                        const amrex::Real dt);

#ifdef RADIATION
///
/// Compute the normal left and right primitive variable interface
/// states at each interface using the piecewise parabolic method with
/// characteristic tracing.  This implementation is for the radiation
/// hydrodynamics.
///
/// @param bx       the box to operate over
/// @param vbx      the valid region box (excludes ghost cells)
/// @param field_arr    the conserved variable state
/// @param inv_density_arr the inverse density array
/// @param prim_arr    the primitive variable state
/// @param qthermo_arr the auxiliary state
/// @param prim_source     the source terms for the primitive variable equations
/// @param qxm      left interface state in x, q_{i-1/2,j,k,L}
/// @param qxp      right interface state in x, q_{i-1/2,j,k,R}
/// @param qym      left interface state in y, q_{i,j-1/2,k,L}
/// @param qyp      right interface state in y, q_{i,j-1/2,k,R}
/// @param qzm      left interface state in z, q_{i,j,k-1/2,L}
/// @param qzp      right interface state in z, q_{i,j,k-1/2,R}
/// @param logAreaRateX   the geometric factor d(log area) in x (r) dir
/// @param logAreaRateY   the geometric factor d(log area) in y (theta) dir
/// @param dt       timestep
///
    void corner_ppm_rad_states(const amrex::Box& bx, const amrex::Box& vbx,
                            amrex::Array4<amrex::Real const> const& field_arr,
                            amrex::Array4<amrex::Real const> const& inv_density_arr,
                            amrex::Array4<amrex::Real const> const& prim_arr,
                            amrex::Array4<amrex::Real const> const& qthermo_arr,
                            amrex::Array4<amrex::Real const> const& prim_source,
                            amrex::Array4<amrex::Real> const& qxm,
                            amrex::Array4<amrex::Real> const& qxp,
#if AMREX_SPACEDIM >= 2
                            amrex::Array4<amrex::Real> const& qym,
                            amrex::Array4<amrex::Real> const& qyp,
#endif
#if AMREX_SPACEDIM == 3
                            amrex::Array4<amrex::Real> const& qzm,
                            amrex::Array4<amrex::Real> const& qzp,
#endif
#if AMREX_SPACEDIM < 3
                            amrex::Array4<amrex::Real const> const& logAreaRateX,
#endif
#if AMREX_SPACEDIM == 2
                            amrex::Array4<amrex::Real const> const& logAreaRateY,
#endif
                            const amrex::Real dt);
#endif

///
/// Compute the left and right primitive variable interface states at
/// each interface using piecewise linear reconstruction for a
/// method-of-lines type of integration (i.e., no prediction to n+1/2
/// or characteristic tracing).  This implementation is for pure
/// hydrodynamics
///
/// @param bx          the box to operate over
/// @param idir        coordinate direction (0 = x, 1 = y, 2 = z)
/// @param prim_arr       the primitive variable state
/// @param flatten_arr   flattening coefficient
/// @param prim_source_arr   source terms for the primitive variable equations
/// @param dq          slope of the primitive variables
/// @param qm          left interface state, e.g., q_{i-1/2,j,k,L}
/// @param qp          right interface state, e.g., q_{i-1/2,j,k,R}
///
    void lines_plm_reconstruct(const amrex::Box& bx,
                             const int idir,
                             amrex::Array4<amrex::Real const> const& prim_arr,
                             amrex::Array4<amrex::Real const> const& flatten_arr,
                             amrex::Array4<amrex::Real const> const& prim_source_arr,
                             amrex::Array4<amrex::Real> const& dq,
                             amrex::Array4<amrex::Real> const& qm,
                             amrex::Array4<amrex::Real> const& qp);
///
/// Compute the left and right primitive variable interface states at
/// each interface using piecewise parabolic reconstruction for a
/// method-of-lines type of integration (i.e., no prediction to n+1/2
/// or characteristic tracing).  This implementation is for pure
/// hydrodynamics
///
/// @param bx          the box to operate over
/// @param idir        coordinate direction (0 = x, 1 = y, 2 = z)
/// @param prim_arr       the primitive variable state
/// @param flatten_arr   flattening coefficient
/// @param qm          left interface state, e.g., q_{i-1/2,j,k,L}
/// @param qp          right interface state, e.g., q_{i-1/2,j,k,R}
///
    void lines_ppm_reconstruct(const amrex::Box& bx,
                             const int idir,
                             amrex::Array4<amrex::Real const> const& prim_arr,
                             amrex::Array4<amrex::Real const> const& flatten_arr,
                             amrex::Array4<amrex::Real> const& qm,
                             amrex::Array4<amrex::Real> const& qp);

///
/// Compute the conservative update of the fluid state given the fluxes.
/// This returns a source term, update, of the form -div{F} + S(U) which
/// can then be added to the old state to update in time.  This version
/// is written for the method-of-lines / true SDC framework.
///
/// @param bx      the box to operate over
/// @param shk     the shock flag
/// @param sourceTerms    source terms for the conservative equations
/// @param update  the final conserved state update, -div{F} + S(U)
/// @param xface_flux   flux in the x-direction
/// @param flux1   flux in the y-direction
/// @param flux2   flux in the z-direction
/// @param face_area0   area of x faces
/// @param area1   area of y faces
/// @param area2   area of z faces
/// @param q0      Godunov state on x faces
/// @param q1      Godunov state on y faces
/// @param vol     cell volume
///
    void lines_cons_update(const amrex::Box& bx,
#ifdef SHOCK_VAR
                    amrex::Array4<amrex::Real const> const& shk,
#endif
                    amrex::Array4<amrex::Real const> const& sourceTerms,
                    amrex::Array4<amrex::Real> const& update,
                    const amrex::Real dt,
                    amrex::Array4<amrex::Real const> const& xface_flux,
#if AMREX_SPACEDIM >= 2
                    amrex::Array4<amrex::Real const> const& flux1,
#endif
#if AMREX_SPACEDIM == 3
                    amrex::Array4<amrex::Real const> const& flux2,
#endif
                    amrex::Array4<amrex::Real const> const& face_area0,
#if AMREX_SPACEDIM >= 2
                    amrex::Array4<amrex::Real const> const& area1,
#endif
#if AMREX_SPACEDIM == 3
                    amrex::Array4<amrex::Real const> const& area2,
#endif
#if AMREX_SPACEDIM <= 2
                    amrex::Array4<amrex::Real const> const& q0,
#if AMREX_SPACEDIM == 2
                    amrex::Array4<amrex::Real const> const& q1,
#endif
#endif
                    amrex::Array4<amrex::Real const> const& vol);

///
/// Compute the diffusive flux term and add it to the energy fluxes.
/// This implementation is second-order accurate.
///
/// @param bx    the box to operate over
/// @param idir  the coordinate direction (0 = x, 1 = y, 2 = z)
/// @param U     the conserved state
/// @param cond  thermal conductivity
/// @param flux  hydrodynamic flux in direction idir
///
     void
     lines_diffusion_flux(const amrex::Box& bx,
                        const int idir,
                        amrex::Array4<amrex::Real const> const& U,
                        amrex::Array4<amrex::Real const> const& cond,
                        amrex::Array4<amrex::Real> const& flux);

///
/// Add the transverse flux difference in idir_tan to the interface states in
/// direction idir_normal as part of the CTU hydrodynamics update.  This is the main driver.
///
/// @param bx        the box to operate over
/// @param idir_tan    direction for the transverse flux difference (0 = x, 1 = y, 2 = z)
/// @param idir_normal    direction of the interface states normal (0 = x, 1 = y, 2 = z)
/// @param qm        input left interface state
/// @param qmo       updated left interface state
/// @param qp        input right interface state
/// @param qpo       updated right interface state
/// @param qthermo      auxiliary state
/// @param flux_tan    flux in the idir_tan direction
/// @param radiative_flux_tan   radiation flux in the idir_tan direction
/// @param q_t       Godunov state in the idir_tan direction
/// @param area_t    face area in the idir_tan direction
/// @param vol       cell volume
/// @param hdt       1/2 * timestep
/// @param third_dt_dx     weighted timestep over cell width from the CTU algorithm
///
     void tangential_pass(const amrex::Box& bx,
                       int idir_tan, int idir_normal,
                       amrex::Array4<amrex::Real const> const& qm,
                       amrex::Array4<amrex::Real> const& qmo,
                       amrex::Array4<amrex::Real const> const& qp,
                       amrex::Array4<amrex::Real> const& qpo,
                       amrex::Array4<amrex::Real const> const& qthermo,
                       amrex::Array4<amrex::Real const> const& flux_tan,
#ifdef RADIATION
                       amrex::Array4<amrex::Real const> const& radiative_flux_tan,
#endif
                       amrex::Array4<amrex::Real const> const& q_t,
#if AMREX_SPACEDIM == 2
                       amrex::Array4<amrex::Real const> const& area_t,
                       amrex::Array4<amrex::Real const> const& vol,
#endif
                       amrex::Real hdt, amrex::Real third_dt_dx);

///
/// Add the transverse flux difference in idir_tan to the interface states in
/// direction idir_normal as part of the CTU hydrodynamics update.
///
/// @param bx        the box to operate over
/// @param idir_tan    direction for the transverse flux difference (0 = x, 1 = y, 2 = z)
/// @param idir_normal    direction of the interface states normal (0 = x, 1 = y, 2 = z)
/// @param d         a flag set by the calling driver that determines which interface
///                  we are updating
/// @param prim_arr     input interface state
/// @param q_out_arr    updated interface state
/// @param qthermo      auxiliary state
/// @param flux_tan    flux in the idir_tan direction
/// @param radiative_flux_tan   radiation flux in the idir_tan direction
/// @param q_t       Godunov state in the idir_tan direction
/// @param area_t    face area in the idir_tan direction
/// @param vol       cell volume
/// @param hdt       1/2 * timestep
/// @param third_dt_dx     weighted timestep over cell width from the CTU algorithm
///
     void apply_tangential_pass(const amrex::Box& bx,
                              int idir_tan, int idir_normal, int d,
                              amrex::Array4<amrex::Real const> const& prim_arr,
                              amrex::Array4<amrex::Real> const& q_out_arr,
                              amrex::Array4<amrex::Real const> const& qthermo,
                              amrex::Array4<amrex::Real const> const& flux_tan,
#ifdef RADIATION
                              amrex::Array4<amrex::Real const> const& radiative_flux_tan,
#endif
                              amrex::Array4<amrex::Real const> const& q_t,
#if AMREX_SPACEDIM == 2
                              amrex::Array4<amrex::Real const> const& area_t,
                              amrex::Array4<amrex::Real const> const& vol,
#endif
                              amrex::Real hdt, amrex::Real third_dt_dx);

///
/// The final transverse update where the flux difference in both
/// perpendicular directions are added to the normal flux.  This is
/// the main driver.
///
/// @param bx        the box to operate over
/// @param idir_normal    direction of the interface states normal (0 = x, 1 = y, 2 = z)
/// @param idir_tan1   direction for the first transverse flux difference (0 = x, 1 = y, 2 = z)
/// @param idir_tan2   direction for the second transverse flux difference (0 = x, 1 = y, 2 = z)
/// @param qm        input left interface state
/// @param qmo       updated left interface state
/// @param qp        input right interface state
/// @param qpo       updated right interface state
/// @param qthermo      auxiliary state
/// @param flux_tan1   flux in the idir_tan1 direction
/// @param radiative_flux_tan1  radiation flux in the idir_tan1 direction
/// @param flux_tan2   flux in the idir_tan2 direction
/// @param radiative_flux_tan2  radiation flux in the idir_tan2 direction
/// @param q_tan1      Godunov state in the idir_tan1 direction
/// @param q_tan2      Godunov state in the idir_tan2 direction
/// @param third_dt_dx_tan1  weight * timestep in idir_tan1 direction
/// @param third_dt_dx_tan2  weight * timestep in idir_tan2 direction
///
     static void tangential_correction(const amrex::Box& bx,
                      int idir_normal, int idir_tan1, int idir_tan2,
                      amrex::Array4<amrex::Real const> const& qm,
                      amrex::Array4<amrex::Real> const& qmo,
                      amrex::Array4<amrex::Real const> const& qp,
                      amrex::Array4<amrex::Real> const& qpo,
                      amrex::Array4<amrex::Real const> const& qthermo,
                      amrex::Array4<amrex::Real const> const& flux_tan1,
#ifdef RADIATION
                      amrex::Array4<amrex::Real const> const& radiative_flux_tan1,
#endif
                      amrex::Array4<amrex::Real const> const& flux_tan2,
#ifdef RADIATION
                      amrex::Array4<amrex::Real const> const& radiative_flux_tan2,
#endif
                      amrex::Array4<amrex::Real const> const& q_tan1,
                      amrex::Array4<amrex::Real const> const& q_tan2,
                      amrex::Real third_dt_dx_tan1, amrex::Real third_dt_dx_tan2);

///
/// The final transverse update where the flux difference in both
/// perpendicular directions are added to the normal flux.
///
/// @param bx        the box to operate over
/// @param idir_normal    direction of the interface states normal (0 = x, 1 = y, 2 = z)
/// @param idir_tan1   direction for the first transverse flux difference (0 = x, 1 = y, 2 = z)
/// @param idir_tan2   direction for the second transverse flux difference (0 = x, 1 = y, 2 = z)
/// @param d         a flag set by the driver to indicate which state we are operating on
/// @param prim_arr     input interface state
/// @param q_out_arr    updated interface state
/// @param qthermo      auxiliary state
/// @param flux_tan1   flux in the idir_tan1 direction
/// @param radiative_flux_tan1  radiation flux in the idir_tan1 direction
/// @param flux_tan2   flux in the idir_tan2 direction
/// @param radiative_flux_tan2  radiation flux in the idir_tan2 direction
/// @param q_tan1      Godunov state in the idir_tan1 direction
/// @param q_tan2      Godunov state in the idir_tan2 direction
/// @param third_dt_dx_tan1  weight * timestep in idir_tan1 direction
/// @param third_dt_dx_tan2  weight * timestep in idir_tan2 direction
///
     static void apply_tangential_correction(const amrex::Box& bx,
                             int idir_normal, int idir_tan1, int idir_tan2, int d,
                             amrex::Array4<amrex::Real const> const& prim_arr,
                             amrex::Array4<amrex::Real> const& q_out_arr,
                             amrex::Array4<amrex::Real const> const& qthermo,
                             amrex::Array4<amrex::Real const> const& flux_tan1,
#ifdef RADIATION
                             amrex::Array4<amrex::Real const> const& radiative_flux_tan1,
#endif
                             amrex::Array4<amrex::Real const> const& flux_tan2,
#ifdef RADIATION
                             amrex::Array4<amrex::Real const> const& radiative_flux_tan2,
#endif
                             amrex::Array4<amrex::Real const> const& q_tan1,
                             amrex::Array4<amrex::Real const> const& q_tan2,
                             amrex::Real third_dt_dx_tan1, amrex::Real third_dt_dx_tan2);

///
/// Reconstruct the primitive state as parabola, integrate under them,
/// and perform the characteristic tracing to get the interface states.
/// This is for the CTU hydrodynamics scheme.
///
/// @param bx        the box to operate over
/// @param idir      coordinate direction of the interface (0 = x, 1 = y, 2 = z)
/// @param field_arr     the conserved variable state
/// @param inv_density_arr the inverse density array
/// @param prim_arr     primitive variable state
/// @param qthermo_arr  the auxiliary state
/// @param prim_source      primitive variable equation source terms
/// @param qm        left interface state (e.g., q_{i-1/2,j,k,L})
/// @param qp        right interface state (e.g., q_{i-1/2,j,k,R})
/// @param dloga     geometric factor dlog(area)
/// @param vbx       the valid region box (excluding ghost cells)
/// @param dt        timestep
///
    void trace_piecewise_parabolic(const amrex::Box& bx,
                   const int idir,
                   amrex::Array4<amrex::Real const> const& field_arr,
                   amrex::Array4<amrex::Real const> const& inv_density_arr,
                   amrex::Array4<amrex::Real const> const& prim_arr,
                   amrex::Array4<amrex::Real const> const& qthermo_arr,
                   amrex::Array4<amrex::Real const> const& prim_source,
                   amrex::Array4<amrex::Real> const& qm,
                   amrex::Array4<amrex::Real> const& qp,
#if (AMREX_SPACEDIM < 3)
                   amrex::Array4<amrex::Real const> const& dloga,
#endif
                   const amrex::Box& vbx,
                   const amrex::Real dt);

///
/// Reconstruct the primitive state as piecewise linear, integrate under them,
/// and perform the characteristic tracing to get the interface states.
/// This is for the CTU hydrodynamics scheme.
///
/// @param bx           the box to operate over
/// @param idir         coordinate direction of the interface (0 = x, 1 = y, 2 = z)
/// @param field_arr        the conserved variable state
/// @param inv_density_arr  the inverse density array
/// @param prim_arr        primitive variable state
/// @param qthermo_arr     the auxiliary state
/// @param PrimitiveSource         primitive variable equation source terms
/// @param qm           left interface state (e.g., q_{i-1/2,j,k,L})
/// @param qp           right interface state (e.g., q_{i-1/2,j,k,R})
/// @param dloga        geometric factor dlog(area)
/// @param vbx          the valid region box (excluding ghost cells)
/// @param dt           timestep
///
    void trace_piecewise_linear(const amrex::Box& bx,
                   const int idir,
                   amrex::Array4<amrex::Real const> const& field_arr,
                   amrex::Array4<amrex::Real const> const& inv_density_arr,
                   amrex::Array4<amrex::Real const> const& prim_arr,
                   amrex::Array4<amrex::Real const> const& qthermo_arr,
                   amrex::Array4<amrex::Real> const& qm,
                   amrex::Array4<amrex::Real> const& qp,
#if (AMREX_SPACEDIM < 3)
                   amrex::Array4<amrex::Real const> const& dloga,
#endif
                   amrex::Array4<amrex::Real const> const& PrimitiveSource,
                   const amrex::Box& vbx,
                   const amrex::Real dt);

///
/// Reconstruct the primitive state as parabola, integrate under them,
/// and perform the characteristic tracing to get the interface states.
/// This is for the CTU radiation hydrodynamics scheme.
///
/// @param bx        the box to operate over
/// @param idir      coordinate direction of the interface (0 = x, 1 = y, 2 = z)
/// @param field_arr     the conserved variable state
/// @param inv_density_arr the inverse density array
/// @param prim_arr     primitive variable state
/// @param qthermo_arr  the auxiliary state
/// @param prim_source      primitive variable equation source terms
/// @param qm        left interface state (e.g., q_{i-1/2,j,k,L})
/// @param qp        right interface state (e.g., q_{i-1/2,j,k,R})
/// @param dloga     geometric factor dlog(area)
/// @param vbx       the valid region box (excluding ghost cells)
/// @param dt        timestep
///
    void trace_parabolic_radiative(const amrex::Box& bx,
                       const int idir,
                       amrex::Array4<amrex::Real const> const& field_arr,
                       amrex::Array4<amrex::Real const> const& inv_density_arr,
                       amrex::Array4<amrex::Real const> const& prim_arr,
                       amrex::Array4<amrex::Real const> const& qthermo_arr,
                       amrex::Array4<amrex::Real const> const& prim_source,
                       amrex::Array4<amrex::Real> const& qm,
                       amrex::Array4<amrex::Real> const& qp,
#if (AMREX_SPACEDIM < 3)
                       amrex::Array4<amrex::Real const> const& dloga,
#endif
                       const amrex::Box& vbx,
                       const amrex::Real dt);

///
/// Compute the conservative update of the hydrodynamics state and store it
/// in update so it can be applied later to advance the state
///
/// @param bx      the box to operate over
/// @param cons_new   the new hydrodynamics conserved state
/// @param xface_flux   flux in the x direction
/// @param qx      Godunov state in the x direction
/// @param flux1   flux in the y direction
/// @param qy      Godunov state in the y direction
/// @param flux2   flux in the z direction
/// @param qz      Godunov state in the z direction
/// @param dt      timestep
///
    void cons_update_hydro(const amrex::Box& bx,
                      amrex::Array4<amrex::Real> const& cons_new,
                      amrex::Array4<amrex::Real> const& xface_flux,
                      amrex::Array4<amrex::Real const> const& qx,
#if AMREX_SPACEDIM >= 2
                      amrex::Array4<amrex::Real> const& flux1,
                      amrex::Array4<amrex::Real const> const& qy,
#endif
#if AMREX_SPACEDIM == 3
                      amrex::Array4<amrex::Real> const& flux2,
                      amrex::Array4<amrex::Real const> const& qz,
#endif
                      const amrex::Real dt);


///
/// A general interface to the pure hydrodynamics and radiation
/// Riemann solvers which takes the left and right interface states,
/// qm and qp, and computes the flux through the interface and the
/// godunov state (interface_state) on the interface
///
/// @param bx          the box to operate over
/// @param qm          left state on the interface
/// @param qp          right state on the interface
/// @param flx         flux through the interface
/// @param scaled_radiative_flux       radiation fluxes through the interface
/// @param godunov_state       Godunov state on the interface [either NQ or NGDSTATE]
/// @param qthermo        auxiliary state
/// @param shk         shock flag
/// @param idir        coordinate direction of the solve (0 = x, 1 = y, 2 = z)
/// @param keep_full_state do we store all NQ or just the NGDSTATE subset in godunov_state
///
    void compute_godunov_flux(const amrex::Box& bx,
                             amrex::Array4<amrex::Real> const& qm,
                             amrex::Array4<amrex::Real> const& qp,
                             amrex::Array4<amrex::Real> const& flx,
#ifdef RADIATION
                             amrex::Array4<amrex::Real> const& radiative_flux_out,
#endif
                             amrex::Array4<amrex::Real> const& godunov_state,
                             amrex::Array4<amrex::Real const> const& qthermo,
                             amrex::Array4<amrex::Real const> const& shk,
                             const int idir, const bool keep_full_state);

    void
    flux_from_state(const amrex::Box& bx,
                        amrex::Array4<amrex::Real const> const& interface_state,
                        amrex::Array4<amrex::Real> const& F,
                        const int idir);

    static void
    save_godunov_state(const amrex::Box& bx,
                        amrex::Array4<amrex::Real const> const& interface_state,
#ifdef RADIATION
                        amrex::Array4<amrex::Real const> const& lambda,
#endif
                        amrex::Array4<amrex::Real> const& godunov_state);

    static void balance_species_flux(const amrex::Box& bx, amrex::Array4<amrex::Real> const& flux);

#ifndef MHD
    static void
    clip_fluxes_at_density_floor(const amrex::Box& bx,
                                     int idir,
                                     amrex::Array4<amrex::Real const> const& u,
                                     amrex::Array4<amrex::Real const> const& vol,
                                     amrex::Array4<amrex::Real> const& flux,
                                     amrex::Array4<amrex::Real const> const& area,
                                     amrex::Real dt,
                                     bool scale_by_area_dt = true);
#endif


///
/// For a reflecting boundary, we simply reflect the interface state
/// just inside the domain to overwrite the state on the same
/// interface but just outside the domain.  This ensures that the flux
/// through the reflecting boundary is zero.
///
   void set_reflecting_boundary_states(const amrex::Box& bx, const int idir,
                               amrex::Array4<amrex::Real> const& qm,
                               amrex::Array4<amrex::Real> const& qp);


    void scale_flux_by_area(const amrex::Box& bx,
                    amrex::Array4<amrex::Real> const& flux,
                    amrex::Array4<amrex::Real const> const& area,
                    const amrex::Real dt);

#ifdef RADIATION
    void scale_radiative_flux(const amrex::Box& bx,
                        amrex::Array4<amrex::Real> const& scaled_radiative_flux,
                        amrex::Array4<amrex::Real const> const& area,
                        const amrex::Real dt);
#endif

#ifndef MHD
    static void reset_edge_thermo(const amrex::Box& bx,
                                        amrex::Array4<amrex::Real> const& edge_state);

    static void edge_temp_to_pressure(const amrex::Box& bx,
                                 amrex::Array4<amrex::Real> const& qm,
                                 amrex::Array4<amrex::Real> const& qp);
#endif

    void clamp_state_density(const amrex::Box& bx,
                                    amrex::Array4<amrex::Real> const& state_arr,
                                    const int warning_level);

#ifdef HYBRID_MOMENTUM

///
/// Construct hybrid source terms at old time
///
/// @param source   MultiFab to save source to
/// @param state    Old state
/// @param time     current time
/// @param dt       timestep
///
    void assemble_old_hybrid_source(amrex::MultiFab& source, amrex::MultiFab& state, amrex::Real time, amrex::Real dt);


///
/// Construct hybrid source terms at new time
///
/// @param source       MultiFab to save source to
/// @param field_old    Old state
/// @param field_new    New state
/// @param time         current time
/// @param dt           timestep
///
    void assemble_new_hybrid_source(amrex::MultiFab& source, amrex::MultiFab& field_old, amrex::MultiFab& field_new, amrex::Real time, amrex::Real dt);


///
/// Fill ``source`` with hybrid source terms
///
/// @param state        Current state
/// @param source       MultiFab to save source to
/// @param term_weight scaling factor applied to the source terms
///
    void fill_hybrid_source_mf(amrex::MultiFab& source, const amrex::MultiFab& state, const amrex::Real term_weight);


///
/// Synchronize linear momentum with hybrid momentum
///
/// @param state    Current state
/// @param ng       Number of ghost cells
///
    void split_to_linear_momentum(amrex::MultiFab& state, int ng = 0);


///
/// Synchronize hybrid momentum with linear momentum
///
/// @param state    Current state
/// @param ng       Number of ghost cells
///
    void linear_to_split_momentum(amrex::MultiFab& state, int ng = 0);

#endif

#ifdef TRUE_SDC

    void
    fourth_face_states(const amrex::Box& bx,
                      const int idir, const int ncomp,
                      amrex::Array4<amrex::Real const> const& a,
                      amrex::Array4<amrex::Real> const& face_values);

    void
    states(const amrex::Box& bx,
           const int idir, const int ncomp,
           amrex::Array4<amrex::Real const> const& a,
           amrex::Array4<amrex::Real const> const& face_values,
           amrex::Array4<amrex::Real const> const& flatten_factor,
           amrex::Array4<amrex::Real> const& al,
           amrex::Array4<amrex::Real> const& ar);

    void
    fourth_viscosity(const amrex::Box& bx,
                 amrex::Array4<amrex::Real const> const& q,
                 amrex::Array4<amrex::Real const> const& qthermo,
                 amrex::Array4<amrex::Real> const& viscosity_field,
                 const int idir);

    void
    fourth_accumulate_diffusion(const amrex::Box& bx,
                              amrex::Array4<amrex::Real const> const& q,
                              const int temperature_index,
                              amrex::Array4<amrex::Real const> const& interface_state,
                              amrex::Array4<amrex::Real> const& F,
                              const int idir, const bool holds_cell_average);

    static
    void build_center_state(const Box& bx,
                          Array4<Real const> const& U,
                          Array4<Real> const& centered_state,
                          GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);

    static
    void build_center_state_in_place(const Box& bx,
                                   Array4<Real> const& U,
                                   Array4<Real> const& tmp,
                                   GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);

    static
    void laplacian_of_state(const Box& bx,
                          Array4<Real const> const& U,
                          Array4<Real> const& lap, const int ncomp,
                          GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);

    static
    void build_fourth_average(const Box& bx,
                             Array4<Real> const& q,
                             Array4<Real const> const& q_second,
                             GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);

    static
    void build_fourth_in_place(const Box& bx,
                              Array4<Real> const& q,
                              Array4<Real> const& tmp,
                              GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);

    static
    void build_fourth_in_place_n(const Box& bx,
                                Array4<Real> const& q, const int ncomp,
                                Array4<Real> const& tmp,
                                GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);

#endif
