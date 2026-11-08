

    void make_primitives_fourth(const amrex::Real time);

    void verify_courant_limit(const amrex::MultiFab& State, const amrex::Real dt);

    step_result assemble_ctu_hydro_source(amrex::Real time, amrex::Real dt);

    void assemble_mol_hydro_source(amrex::Real time, amrex::Real dt, amrex::MultiFab& A_divergence);


    static void inject_deferred_source(const Box& bx, const int idir, const Real dt,
                                      Array4<Real> const& prim_left,
                                      Array4<Real> const& prim_right,
                                      Array4<Real const> const& deferred_source);

    void source_to_primitive(const amrex::Box& bx, const Real dt,
                     amrex::Array4<amrex::Real const> const& prim_arr,
                     amrex::Array4<amrex::Real const> const& previous_source,
                     amrex::Array4<amrex::Real const> const& source_correction,
                     amrex::Array4<amrex::Real> const& prim_source);

    static void ctoprim(const amrex::Box& bx,
                 const amrex::Real time,
                 amrex::Array4<amrex::Real const> const& uin,
                 amrex::Array4<amrex::Real> const& prim_arr,
                 amrex::Array4<amrex::Real> const& qthermo_arr);


    void shock(const amrex::Box& bx,
               amrex::Array4<amrex::Real const> const& prim_arr,
               amrex::Array4<amrex::Real const> const& U_source_arr,
               amrex::Array4<amrex::Real> const& shk);

    void divu(const amrex::Box& bx,
              amrex::Array4<amrex::Real const> const& prim_arr,
              amrex::Array4<amrex::Real> const& div);

    void apply_viscosity(const amrex::Box& bx,
                  const int idir,
                  amrex::Array4<amrex::Real const> const& div,
                  amrex::Array4<amrex::Real const> const& uin,
                  amrex::Array4<amrex::Real> const& flux);


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


    void lines_plm_reconstruct(const amrex::Box& bx,
                             const int idir,
                             amrex::Array4<amrex::Real const> const& prim_arr,
                             amrex::Array4<amrex::Real const> const& flatten_arr,
                             amrex::Array4<amrex::Real const> const& prim_source_arr,
                             amrex::Array4<amrex::Real> const& dq,
                             amrex::Array4<amrex::Real> const& qm,
                             amrex::Array4<amrex::Real> const& qp);
    void lines_ppm_reconstruct(const amrex::Box& bx,
                             const int idir,
                             amrex::Array4<amrex::Real const> const& prim_arr,
                             amrex::Array4<amrex::Real const> const& flatten_arr,
                             amrex::Array4<amrex::Real> const& qm,
                             amrex::Array4<amrex::Real> const& qp);

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

     void
     lines_diffusion_flux(const amrex::Box& bx,
                        const int idir,
                        amrex::Array4<amrex::Real const> const& U,
                        amrex::Array4<amrex::Real const> const& cond,
                        amrex::Array4<amrex::Real> const& flux);

     void tangential_pass(const amrex::Box& bx,
                       int idir_tan, int idir_normal,
                       amrex::Array4<amrex::Real const> const& qm,
                       amrex::Array4<amrex::Real> const& qmo,
                       amrex::Array4<amrex::Real const> const& qp,
                       amrex::Array4<amrex::Real> const& qpo,
                       amrex::Array4<amrex::Real const> const& qthermo,
                       amrex::Array4<amrex::Real const> const& flux_tan,
                       amrex::Array4<amrex::Real const> const& q_t,
#if AMREX_SPACEDIM == 2
                       amrex::Array4<amrex::Real const> const& area_t,
                       amrex::Array4<amrex::Real const> const& vol,
#endif
                       amrex::Real hdt, amrex::Real third_dt_dx);

     enum class TraceSide : int { Minus = -1, Plus = 0 };

     void apply_tangential_pass(const amrex::Box& bx,
                              int idir_tan, int idir_normal, TraceSide side,
                              amrex::Array4<amrex::Real const> const& prim_arr,
                              amrex::Array4<amrex::Real> const& q_out_arr,
                              amrex::Array4<amrex::Real const> const& qthermo,
                              amrex::Array4<amrex::Real const> const& flux_tan,
                              amrex::Array4<amrex::Real const> const& q_t,
#if AMREX_SPACEDIM == 2
                              amrex::Array4<amrex::Real const> const& area_t,
                              amrex::Array4<amrex::Real const> const& vol,
#endif
                              amrex::Real hdt, amrex::Real third_dt_dx);

     static void tangential_correction(const amrex::Box& bx,
                      int idir_normal, int idir_tan1, int idir_tan2,
                      amrex::Array4<amrex::Real const> const& qm,
                      amrex::Array4<amrex::Real> const& qmo,
                      amrex::Array4<amrex::Real const> const& qp,
                      amrex::Array4<amrex::Real> const& qpo,
                      amrex::Array4<amrex::Real const> const& qthermo,
                      amrex::Array4<amrex::Real const> const& flux_tan1,
                      amrex::Array4<amrex::Real const> const& flux_tan2,
                      amrex::Array4<amrex::Real const> const& q_tan1,
                      amrex::Array4<amrex::Real const> const& q_tan2,
                      amrex::Real third_dt_dx_tan1, amrex::Real third_dt_dx_tan2);

     static void apply_tangential_correction(const amrex::Box& bx,
                             int idir_normal, int idir_tan1, int idir_tan2, TraceSide side,
                             amrex::Array4<amrex::Real const> const& prim_arr,
                             amrex::Array4<amrex::Real> const& q_out_arr,
                             amrex::Array4<amrex::Real const> const& qthermo,
                             amrex::Array4<amrex::Real const> const& flux_tan1,
                             amrex::Array4<amrex::Real const> const& flux_tan2,
                             amrex::Array4<amrex::Real const> const& q_tan1,
                             amrex::Array4<amrex::Real const> const& q_tan2,
                             amrex::Real third_dt_dx_tan1, amrex::Real third_dt_dx_tan2);

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


    enum class GodunovStore { Faces, FullState };

    void compute_godunov_flux(const amrex::Box& bx,
                             amrex::Array4<amrex::Real> const& qm,
                             amrex::Array4<amrex::Real> const& qp,
                             amrex::Array4<amrex::Real> const& flx,
                             amrex::Array4<amrex::Real> const& godunov_state,
                             amrex::Array4<amrex::Real const> const& qthermo,
                             amrex::Array4<amrex::Real const> const& shk,
                             const int idir, const GodunovStore store);

    void
    flux_from_state(const amrex::Box& bx,
                        amrex::Array4<amrex::Real const> const& interface_state,
                        amrex::Array4<amrex::Real> const& F,
                        const int idir);

    static void
    save_godunov_state(const amrex::Box& bx,
                        amrex::Array4<amrex::Real const> const& interface_state,
                        amrex::Array4<amrex::Real> const& godunov_state);

    static void balance_species_flux(const amrex::Box& bx, amrex::Array4<amrex::Real> const& flux);

    static void
    clip_fluxes_at_density_floor(const amrex::Box& bx,
                                     int idir,
                                     amrex::Array4<amrex::Real const> const& u,
                                     amrex::Array4<amrex::Real const> const& vol,
                                     amrex::Array4<amrex::Real> const& flux,
                                     amrex::Array4<amrex::Real const> const& area,
                                     amrex::Real dt,
                                     bool scale_by_area_dt = true);


   void set_reflecting_boundary_states(const amrex::Box& bx, const int idir,
                               amrex::Array4<amrex::Real> const& qm,
                               amrex::Array4<amrex::Real> const& qp);


    void scale_flux_by_area(const amrex::Box& bx,
                    amrex::Array4<amrex::Real> const& flux,
                    amrex::Array4<amrex::Real const> const& area,
                    const amrex::Real dt);


    static void reset_edge_thermo(const amrex::Box& bx,
                                        amrex::Array4<amrex::Real> const& edge_state);

    static void edge_temp_to_pressure(const amrex::Box& bx,
                                 amrex::Array4<amrex::Real> const& qm,
                                 amrex::Array4<amrex::Real> const& qp);

    void clamp_state_density(const amrex::Box& bx,
                                    amrex::Array4<amrex::Real> const& state_arr,
                                    const int warning_level);


