
#ifdef TRUE_SDC
///
/// Calculate primitive variables from conserved variables (uses StateData)
///
/// @param time     current time
///
    void cons_to_prim(const amrex::Real time);
#endif

///
/// convert the conservative state cell averages to primitive cell
/// averages with 4th order accuracy
///
/// @param time     current time
///
    void cons_to_prim_fourth(const amrex::Real time);

///
/// Check to see if the CFL condition has been violated
///
/// @param State MultiFab of conserved variables
/// @param dt timestep
///
    void check_for_cfl_violation(const amrex::MultiFab& State, const amrex::Real dt);

///
/// this constructs the hydrodynamic source (essentially the flux
/// divergence) using the CTU framework for unsplit hydrodynamics
///
/// @param time     current time
/// @param dt       timestep
///
    advance_status construct_ctu_hydro_source(amrex::Real time, amrex::Real dt);

///
/// this constructs the hydrodynamic source (essentially the flux
/// divergence) using method of lines integration.  The output, is the
/// divergence of the fluxes, A = -div{F(U)}
///
/// @param time      current time
/// @param dt        timestep
/// @param A_update  divergence of the fluxes
///
    void construct_mol_hydro_source(amrex::Real time, amrex::Real dt, amrex::MultiFab& A_update);


///
/// convert the conserved form of the source terms, S(U), into source terms
/// for the primitive variable equations, S(q).
///
/// @param bx      the box to operate over
/// @param dt      timestep
/// @param q_arr   primitive variable state array
/// @param old_src conserved source array
#ifndef TRUE_SDC
/// @param src_corr source correction array
#endif
/// @param srcQ    primitive variable source array
///
    void src_to_prim(const amrex::Box& bx, const Real dt,
                     amrex::Array4<amrex::Real const> const& q_arr,
                     amrex::Array4<amrex::Real const> const& old_src,
#ifndef TRUE_SDC
                     amrex::Array4<amrex::Real const> const& src_corr,
#endif
                     amrex::Array4<amrex::Real> const& srcQ);

///
/// A multidimensional shock detection algorithm
///
/// @param bx         the box to operate over
/// @param q_arr      the primitive variable state
/// @param U_scr_arr  the conservative state sources
/// @param shk        the shock flag (1 = shock, 0 = no shock)
///
    void shock(const amrex::Box& bx,
               amrex::Array4<amrex::Real const> const& q_arr,
               amrex::Array4<amrex::Real const> const& U_src_arr,
               amrex::Array4<amrex::Real> const& shk);

///
/// Compute the node-centered velocity divergence (DU)_{i-1/2.j-1/2,k-1/2}
///
/// @param bx     the box to operate over
/// @param q_arr  input primitive variable state
/// @param div    output velocity divergence
///
    void divu(const amrex::Box& bx,
              amrex::Array4<amrex::Real const> const& q_arr,
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
    void apply_art_visc(const amrex::Box& bx,
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
/// @param Erin     the radiation energy
/// @param radflux  the radiation energy flux in direction idir
///
    void apply_rad_art_visc(const amrex::Box& bx,
                      const int idir,
                      amrex::Array4<amrex::Real const> const& div,
                      amrex::Array4<amrex::Real const> const& Erin,
                      amrex::Array4<amrex::Real> const& radflux);

///
/// The radiation part of the CTU conservative update.  Returns the
/// largest number of frequency-space advection steps a zone took, or
/// -1 if the comoving-frame terms were not evaluated.
///
    int
    ctu_rad_cons_update(const amrex::Box& bx,
                   amrex::Array4<amrex::Real> const& update,
                   amrex::Array4<amrex::Real const> const& Erin,
                   amrex::Array4<amrex::Real> const& Erout,
                   amrex::Array4<amrex::Real const> const& radflux1,
                   amrex::Array4<amrex::Real const> const& qx,
                   amrex::Array4<amrex::Real const> const& area1,
#if AMREX_SPACEDIM >= 2
                   amrex::Array4<amrex::Real const> const& radflux2,
                   amrex::Array4<amrex::Real const> const& qy,
                   amrex::Array4<amrex::Real const> const& area2,
#endif
#if AMREX_SPACEDIM == 3
                   amrex::Array4<amrex::Real const> const& radflux3,
                   amrex::Array4<amrex::Real const> const& qz,
                   amrex::Array4<amrex::Real const> const& area3,
#endif
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
/// @param U_arr    the conserved variable state
/// @param rho_inv_arr the inverse density array
/// @param q_arr    the primitive variable state
/// @param qaux_arr the auxiliary state
/// @param srcQ     the source terms for the primitive variable equations
/// @param qm       left interface state in each direction, e.g. q_{i-1/2,j,k,L} in x
/// @param qp       right interface state in each direction, e.g. q_{i-1/2,j,k,R} in x
/// @param dloga    the geometric factor d(log area) in each direction (x = r,
///                 y = theta); only read in 1-d and 2-d
/// @param dt       timestep
///
    void ppm_interface_states(const amrex::Box& bx, const amrex::Box& vbx,
                        amrex::Array4<amrex::Real const> const& U_arr,
                        amrex::Array4<amrex::Real const> const& rho_inv_arr,
                        amrex::Array4<amrex::Real const> const& q_arr,
                        amrex::Array4<amrex::Real const> const& qaux_arr,
                        amrex::Array4<amrex::Real const> const& srcQ,
                        amrex::Array<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> const& qm,
                        amrex::Array<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> const& qp,
                        amrex::Array<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& dloga,
                        const amrex::Real dt);

///
/// Compute the normal left and right primitive variable interface
/// states at each interface using piecewise linear reconstruction with
/// characteristic tracing.  This implementation is for pure
/// hydrodynamics
///
/// @param bx       the box to operate over
/// @param vbx      the valid region box (excludes ghost cells)
/// @param U_arr    the conserved variable state
/// @param rho_inv_arr the inverse density array
/// @param q_arr    the primitive variable state
/// @param qaux_arr the auxiliary state
/// @param srcQ     the source terms for the primitive variable equations
/// @param qm       left interface state in each direction, e.g. q_{i-1/2,j,k,L} in x
/// @param qp       right interface state in each direction, e.g. q_{i-1/2,j,k,R} in x
/// @param dloga    the geometric factor d(log area) in each direction (x = r,
///                 y = theta); only read in 1-d and 2-d
/// @param dt       timestep
///
    void plm_interface_states(const amrex::Box& bx, const amrex::Box& vbx,
                        amrex::Array4<amrex::Real const> const& U_arr,
                        amrex::Array4<amrex::Real const> const& rho_inv_arr,
                        amrex::Array4<amrex::Real const> const& q_arr,
                        amrex::Array4<amrex::Real const> const& qaux_arr,
                        amrex::Array4<amrex::Real const> const& srcQ,
                        amrex::Array<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> const& qm,
                        amrex::Array<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> const& qp,
                        amrex::Array<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& dloga,
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
/// @param U_arr    the conserved variable state
/// @param rho_inv_arr the inverse density array
/// @param q_arr    the primitive variable state
/// @param qaux_arr the auxiliary state
/// @param srcQ     the source terms for the primitive variable equations
/// @param qm       left interface state in each direction, e.g. q_{i-1/2,j,k,L} in x
/// @param qp       right interface state in each direction, e.g. q_{i-1/2,j,k,R} in x
/// @param dloga    the geometric factor d(log area) in each direction (x = r,
///                 y = theta); only read in 1-d and 2-d
/// @param dt       timestep
///
    void ppm_rad_interface_states(const amrex::Box& bx, const amrex::Box& vbx,
                            amrex::Array4<amrex::Real const> const& U_arr,
                            amrex::Array4<amrex::Real const> const& rho_inv_arr,
                            amrex::Array4<amrex::Real const> const& q_arr,
                            amrex::Array4<amrex::Real const> const& qaux_arr,
                            amrex::Array4<amrex::Real const> const& srcQ,
                            amrex::Array<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> const& qm,
                            amrex::Array<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> const& qp,
                            amrex::Array<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& dloga,
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
/// @param q_arr       the primitive variable state
/// @param flatcoef_arr   flattening coefficient
/// @param src_q_arr   source terms for the primitive variable equations
/// @param dq          slope of the primitive variables
/// @param qm          left interface state, e.g., q_{i-1/2,j,k,L}
/// @param qp          right interface state, e.g., q_{i-1/2,j,k,R}
///
    void plm_face_states_mol(const amrex::Box& bx,
                             const int idir,
                             amrex::Array4<amrex::Real const> const& q_arr,
                             amrex::Array4<amrex::Real const> const& flatcoef_arr,
                             amrex::Array4<amrex::Real const> const& src_q_arr,
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
/// @param q_arr       the primitive variable state
/// @param flatcoef_arr   flattening coefficient
/// @param qm          left interface state, e.g., q_{i-1/2,j,k,L}
/// @param qp          right interface state, e.g., q_{i-1/2,j,k,R}
///
    void ppm_face_states_mol(const amrex::Box& bx,
                             const int idir,
                             amrex::Array4<amrex::Real const> const& q_arr,
                             amrex::Array4<amrex::Real const> const& flatcoef_arr,
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
/// @param srcU    source terms for the conservative equations
/// @param update  the final conserved state update, -div{F} + S(U)
/// @param flux    flux in each direction
/// @param face_area  area of the faces in each direction
/// @param qe      Godunov state on the faces in each direction (only
///                read in 1-d and 2-d)
/// @param vol     cell volume
///
    void mol_cons_update(const amrex::Box& bx,
#ifdef SHOCK_VAR
                    amrex::Array4<amrex::Real const> const& shk,
#endif
                    amrex::Array4<amrex::Real const> const& srcU,
                    amrex::Array4<amrex::Real> const& update,
                    const amrex::Real dt,
                    amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& flux,
                    amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& face_area,
                    amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& qe,
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
     add_mol_conduction_flux(const amrex::Box& bx,
                        const int idir,
                        amrex::Array4<amrex::Real const> const& U,
                        amrex::Array4<amrex::Real const> const& cond,
                        amrex::Array4<amrex::Real> const& flux);

///
/// The data on the faces of a transverse direction that a transverse
/// correction differences
///
/// flux       flux through the faces
/// qface      Godunov state on the faces
/// area       face area (2-d only)
/// vol        cell volume (2-d only)
/// rad_flux   radiation flux through the faces
///
    struct TransverseFaces
    {
        amrex::Array4<amrex::Real const> flux;
        amrex::Array4<amrex::Real const> qface;
#if AMREX_SPACEDIM == 2
        amrex::Array4<amrex::Real const> area;
        amrex::Array4<amrex::Real const> vol;
#endif
#ifdef RADIATION
        amrex::Array4<amrex::Real const> rad_flux;
#endif
    };

///
/// Add the transverse flux difference in idir_t to the interface states in
/// direction idir_n as part of the CTU hydrodynamics update.  This is the main driver.
///
/// @param bx        the box to operate over
/// @param idir_t    direction for the transverse flux difference (0 = x, 1 = y, 2 = z)
/// @param idir_n    direction of the interface states normal (0 = x, 1 = y, 2 = z)
/// @param qm        input left interface state
/// @param qmo       updated left interface state
/// @param qp        input right interface state
/// @param qpo       updated right interface state
/// @param qaux      auxiliary state
/// @param faces_t   fluxes, Godunov state and geometry on the idir_t faces
/// @param hdt       1/2 * timestep
/// @param coef_dtdx     weighted timestep over cell width from the CTU algorithm
///
     void transverse_correct_single(const amrex::Box& bx,
                       int idir_t, int idir_n,
                       amrex::Array4<amrex::Real const> const& qm,
                       amrex::Array4<amrex::Real> const& qmo,
                       amrex::Array4<amrex::Real const> const& qp,
                       amrex::Array4<amrex::Real> const& qpo,
                       amrex::Array4<amrex::Real const> const& qaux,
                       TransverseFaces const& faces_t,
                       amrex::Real hdt, amrex::Real coef_dtdx);

///
/// Add the transverse flux difference in idir_t to the interface states in
/// direction idir_n as part of the CTU hydrodynamics update.
///
/// @param bx        the box to operate over
/// @param idir_t    direction for the transverse flux difference (0 = x, 1 = y, 2 = z)
/// @param idir_n    direction of the interface states normal (0 = x, 1 = y, 2 = z)
/// @param d         a flag set by the calling driver that determines which interface
///                  we are updating
/// @param q_arr     input interface state
/// @param qo_arr    updated interface state
/// @param qaux      auxiliary state
/// @param faces_t   fluxes, Godunov state and geometry on the idir_t faces
/// @param hdt       1/2 * timestep
/// @param coef_dtdx     weighted timestep over cell width from the CTU algorithm
///
     void transverse_correct_single_impl(const amrex::Box& bx,
                              int idir_t, int idir_n, int d,
                              amrex::Array4<amrex::Real const> const& q_arr,
                              amrex::Array4<amrex::Real> const& qo_arr,
                              amrex::Array4<amrex::Real const> const& qaux,
                              TransverseFaces const& faces_t,
                              amrex::Real hdt, amrex::Real coef_dtdx);

///
/// The final transverse update where the flux difference in both
/// perpendicular directions are added to the normal flux.  This is
/// the main driver.
///
/// @param bx        the box to operate over
/// @param idir_n    direction of the interface states normal (0 = x, 1 = y, 2 = z)
/// @param idir_t1   direction for the first transverse flux difference (0 = x, 1 = y, 2 = z)
/// @param idir_t2   direction for the second transverse flux difference (0 = x, 1 = y, 2 = z)
/// @param qm        input left interface state
/// @param qmo       updated left interface state
/// @param qp        input right interface state
/// @param qpo       updated right interface state
/// @param qaux      auxiliary state
/// @param faces_t1  fluxes and Godunov state on the idir_t1 faces
/// @param faces_t2  fluxes and Godunov state on the idir_t2 faces
/// @param coef_dtdx_t1  weight * timestep in idir_t1 direction
/// @param coef_dtdx_t2  weight * timestep in idir_t2 direction
///
     static void transverse_correct_final(const amrex::Box& bx,
                      int idir_n, int idir_t1, int idir_t2,
                      amrex::Array4<amrex::Real const> const& qm,
                      amrex::Array4<amrex::Real> const& qmo,
                      amrex::Array4<amrex::Real const> const& qp,
                      amrex::Array4<amrex::Real> const& qpo,
                      amrex::Array4<amrex::Real const> const& qaux,
                      TransverseFaces const& faces_t1,
                      TransverseFaces const& faces_t2,
                      amrex::Real coef_dtdx_t1, amrex::Real coef_dtdx_t2);

///
/// The final transverse update where the flux difference in both
/// perpendicular directions are added to the normal flux.
///
/// @param bx        the box to operate over
/// @param idir_n    direction of the interface states normal (0 = x, 1 = y, 2 = z)
/// @param idir_t1   direction for the first transverse flux difference (0 = x, 1 = y, 2 = z)
/// @param idir_t2   direction for the second transverse flux difference (0 = x, 1 = y, 2 = z)
/// @param d         a flag set by the driver to indicate which state we are operating on
/// @param q_arr     input interface state
/// @param qo_arr    updated interface state
/// @param qaux      auxiliary state
/// @param faces_t1  fluxes and Godunov state on the idir_t1 faces
/// @param faces_t2  fluxes and Godunov state on the idir_t2 faces
/// @param coef_dtdx_t1  weight * timestep in idir_t1 direction
/// @param coef_dtdx_t2  weight * timestep in idir_t2 direction
///
     static void transverse_correct_final_impl(const amrex::Box& bx,
                             int idir_n, int idir_t1, int idir_t2, int d,
                             amrex::Array4<amrex::Real const> const& q_arr,
                             amrex::Array4<amrex::Real> const& qo_arr,
                             amrex::Array4<amrex::Real const> const& qaux,
                             TransverseFaces const& faces_t1,
                             TransverseFaces const& faces_t2,
                             amrex::Real coef_dtdx_t1, amrex::Real coef_dtdx_t2);

///
/// The zone data the characteristic tracing reads in one direction
///
/// U          the conserved variable state
/// rho_inv    the inverse density
/// q          the primitive variable state
/// qaux       the auxiliary state
/// srcQ       the source terms for the primitive variable equations
/// dloga      geometric factor dlog(area) in the traced direction (1-d and 2-d only)
///
    struct TraceFields
    {
        amrex::Array4<amrex::Real const> U;
        amrex::Array4<amrex::Real const> rho_inv;
        amrex::Array4<amrex::Real const> q;
        amrex::Array4<amrex::Real const> qaux;
        amrex::Array4<amrex::Real const> srcQ;
        amrex::Array4<amrex::Real const> dloga;
    };

///
/// Reconstruct the primitive state as parabola, integrate under them,
/// and perform the characteristic tracing to get the interface states.
/// This is for the CTU hydrodynamics scheme.
///
/// @param bx        the box to operate over
/// @param idir      coordinate direction of the interface (0 = x, 1 = y, 2 = z)
/// @param fields    the zone data to trace
/// @param qm        left interface state (e.g., q_{i-1/2,j,k,L})
/// @param qp        right interface state (e.g., q_{i-1/2,j,k,R})
/// @param vbx       the valid region box (excluding ghost cells)
/// @param dt        timestep
///
    void trace_ppm(const amrex::Box& bx,
                   const int idir,
                   TraceFields const& fields,
                   amrex::Array4<amrex::Real> const& qm,
                   amrex::Array4<amrex::Real> const& qp,
                   const amrex::Box& vbx,
                   const amrex::Real dt);

///
/// Reconstruct the primitive state as piecewise linear, integrate under them,
/// and perform the characteristic tracing to get the interface states.
/// This is for the CTU hydrodynamics scheme.
///
/// @param bx           the box to operate over
/// @param idir         coordinate direction of the interface (0 = x, 1 = y, 2 = z)
/// @param fields       the zone data to trace
/// @param qm           left interface state (e.g., q_{i-1/2,j,k,L})
/// @param qp           right interface state (e.g., q_{i-1/2,j,k,R})
/// @param vbx          the valid region box (excluding ghost cells)
/// @param dt           timestep
///
    void trace_plm(const amrex::Box& bx,
                   const int idir,
                   TraceFields const& fields,
                   amrex::Array4<amrex::Real> const& qm,
                   amrex::Array4<amrex::Real> const& qp,
                   const amrex::Box& vbx,
                   const amrex::Real dt);

///
/// Reconstruct the primitive state as parabola, integrate under them,
/// and perform the characteristic tracing to get the interface states.
/// This is for the CTU radiation hydrodynamics scheme.
///
/// @param bx        the box to operate over
/// @param idir      coordinate direction of the interface (0 = x, 1 = y, 2 = z)
/// @param fields    the zone data to trace
/// @param qm        left interface state (e.g., q_{i-1/2,j,k,L})
/// @param qp        right interface state (e.g., q_{i-1/2,j,k,R})
/// @param vbx       the valid region box (excluding ghost cells)
/// @param dt        timestep
///
    void trace_ppm_rad(const amrex::Box& bx,
                       const int idir,
                       TraceFields const& fields,
                       amrex::Array4<amrex::Real> const& qm,
                       amrex::Array4<amrex::Real> const& qp,
                       const amrex::Box& vbx,
                       const amrex::Real dt);

///
/// Compute the conservative update of the hydrodynamics state and store it
/// in update so it can be applied later to advance the state
///
/// @param bx      the box to operate over
/// @param U_new   the new hydrodynamics conserved state
/// @param flux    flux in each direction
/// @param qe      Godunov state on the faces in each direction
/// @param dt      timestep
///
    void cons_update_hydro(const amrex::Box& bx,
                      amrex::Array4<amrex::Real> const& U_new,
                      amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM> const& flux,
                      amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& qe,
                      const amrex::Real dt);


///
/// The arrays a Riemann solve writes on the interfaces
///
/// flux       flux through the interface
/// qface      Godunov state on the interface [either NQ or NIFC]
/// rad_flux   radiation fluxes through the interface
///
    struct InterfaceOutput
    {
        amrex::Array4<amrex::Real> flux;
        amrex::Array4<amrex::Real> qface;
#ifdef RADIATION
        amrex::Array4<amrex::Real> rad_flux;
#endif
    };

///
/// A general interface to the pure hydrodynamics and radiation
/// Riemann solvers which takes the left and right interface states,
/// qm and qp, and computes the flux through the interface and the
/// godunov state (q_riemann) on the interface
///
/// @param bx          the box to operate over
/// @param qm          left state on the interface
/// @param qp          right state on the interface
/// @param out         flux, Godunov state and radiation fluxes on the interface
/// @param qaux        auxiliary state
/// @param shk         shock flag
/// @param idir        coordinate direction of the solve (0 = x, 1 = y, 2 = z)
/// @param qface_comps do we store all NQ or just the NIFC subset in qface
///
    void riemann_flux_and_state(const amrex::Box& bx,
                             amrex::Array4<amrex::Real> const& qm,
                             amrex::Array4<amrex::Real> const& qp,
                             InterfaceOutput const& out,
                             amrex::Array4<amrex::Real const> const& qaux,
                             amrex::Array4<amrex::Real const> const& shk,
                             const int idir, const FaceComponents qface_comps);

    void
    flux_from_face_state(const amrex::Box& bx,
                        amrex::Array4<amrex::Real const> const& q_riemann,
                        amrex::Array4<amrex::Real> const& F,
                        const int idir);


///
/// For a reflecting boundary, we simply reflect the interface state
/// just inside the domain to overwrite the state on the same
/// interface but just outside the domain.  This ensures that the flux
/// through the reflecting boundary is zero.
///
   void mirror_states_at_walls(const amrex::Box& bx, const int idir,
                               amrex::Array4<amrex::Real> const& qm,
                               amrex::Array4<amrex::Real> const& qp);


    void scale_flux(const amrex::Box& bx,
                    amrex::Array4<amrex::Real> const& flux,
                    amrex::Array4<amrex::Real const> const& area,
                    const amrex::Real dt);

#ifdef RADIATION
    void scale_rad_flux(const amrex::Box& bx,
                        amrex::Array4<amrex::Real> const& rflux,
                        amrex::Array4<amrex::Real const> const& area,
                        const amrex::Real dt);
#endif

    void apply_density_floor_box(const amrex::Box& bx,
                                    amrex::Array4<amrex::Real> const& state_arr,
                                    const DensityResetReport report);

#ifdef HYBRID_MOMENTUM

///
/// Construct hybrid source terms at old time
///
/// @param state    Old state
/// @param time     current time
/// @param dt       timestep
/// @param source   MultiFab to save source to
///
    void assemble_old_hybrid_source(amrex::MultiFab& state, amrex::Real time, amrex::Real dt, amrex::MultiFab& source);


///
/// Construct hybrid source terms at new time
///
/// @param state_old    Old state
/// @param state_new    New state
/// @param time         current time
/// @param dt           timestep
/// @param source       MultiFab to save source to
///
    void assemble_new_hybrid_source(amrex::MultiFab& state_old, amrex::MultiFab& state_new, amrex::Real time, amrex::Real dt, amrex::MultiFab& source);


///
/// Fill ``source`` with hybrid source terms
///
/// @param state        Current state
/// @param source       MultiFab to save source to
/// @param mult_factor scaling factor applied to the source terms
///
    void fill_hybrid_hydro_source(amrex::MultiFab& source, const amrex::MultiFab& state, const amrex::Real mult_factor);


///
/// Synchronize linear momentum with hybrid momentum
///
/// @param state    Current state
/// @param ng       Number of ghost cells
///
    void hybrid_to_linear_momentum(amrex::MultiFab& state, int ng = 0);


///
/// Synchronize hybrid momentum with linear momentum
///
/// @param state    Current state
/// @param ng       Number of ghost cells
///
    void linear_to_hybrid_momentum(amrex::MultiFab& state, int ng = 0);

#endif

#ifdef TRUE_SDC

    void
    fourth_order_face_values(const amrex::Box& bx,
                      const int idir, const int ncomp,
                      amrex::Array4<amrex::Real const> const& a,
                      amrex::Array4<amrex::Real> const& a_int);

    void
    states(const amrex::Box& bx,
           const int idir, const int ncomp,
           amrex::Array4<amrex::Real const> const& a,
           amrex::Array4<amrex::Real const> const& a_int,
           amrex::Array4<amrex::Real const> const& flatcoef,
           amrex::Array4<amrex::Real> const& al,
           amrex::Array4<amrex::Real> const& ar);

    void
    fourth_order_art_visc(const amrex::Box& bx,
                 amrex::Array4<amrex::Real const> const& q,
                 amrex::Array4<amrex::Real const> const& qaux,
                 amrex::Array4<amrex::Real> const& avis,
                 const int idir);

    void
    fourth_order_conduction_flux(const amrex::Box& bx,
                              amrex::Array4<amrex::Real const> const& q,
                              const int temp_comp,
                              amrex::Array4<amrex::Real const> const& q_riemann,
                              amrex::Array4<amrex::Real> const& F,
                              const int idir, const bool is_avg);

    static
    void to_cell_center(const Box& bx,
                          Array4<Real const> const& U,
                          Array4<Real> const& U_cc,
                          GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);

    static
    void to_cell_center_in_place(const Box& bx,
                                   Array4<Real> const& U,
                                   Array4<Real> const& tmp,
                                   GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);

    static
    void laplacian_term(const Box& bx,
                          Array4<Real const> const& U,
                          Array4<Real> const& lap, const int ncomp,
                          GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);

    static
    void to_fourth_order_average(const Box& bx,
                             Array4<Real> const& q,
                             Array4<Real const> const& q_bar,
                             GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);

    static
    void to_fourth_order_in_place(const Box& bx,
                              Array4<Real> const& q,
                              Array4<Real> const& tmp,
                              GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);

    static
    void to_fourth_order_in_place_n(const Box& bx,
                                Array4<Real> const& q, const int ncomp,
                                Array4<Real> const& tmp,
                                GpuArray<int, 3> const& domlo, GpuArray<int, 3> const& domhi);

#endif
