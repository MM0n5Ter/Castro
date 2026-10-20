#ifndef HYDROSIM_SOURCES_H
#define HYDROSIM_SOURCES_H

    bool source_enabled(int src);


    bool apply_source_terms();

    static int get_write_on_finish();


    static void apply_source_term_to_state (amrex::MultiFab& state, const amrex::MultiFab& source, amrex::Real dt, int ng);


    void run_old_sources(
#ifdef MHD
                        amrex::MultiFab& Bx,
                        amrex::MultiFab& By,
                        amrex::MultiFab& Bz,
#endif
                        amrex::MultiFab& source,
                        amrex::MultiFab& field_old, amrex::MultiFab& field_new,
                        amrex::Real time, amrex::Real dt, bool modify_state_flag = true);


    step_result run_old_sources (amrex::Real time, amrex::Real dt, bool modify_state_flag = true);


    void assemble_old_source(int src, amrex::MultiFab& source,
                              amrex::MultiFab& state,
                              amrex::Real time, amrex::Real dt);

    void run_new_sources(
#ifdef MHD
                        amrex::MultiFab& Bx,
                        amrex::MultiFab& By,
                        amrex::MultiFab& Bz,
#endif
                        amrex::MultiFab& source,
                        amrex::MultiFab& field_old, amrex::MultiFab& field_new,
                        amrex::Real time, amrex::Real dt, bool modify_state_flag = true);


    step_result run_new_sources (amrex::Real time, amrex::Real dt);


    void assemble_new_source(int src, amrex::MultiFab& source,
                              amrex::MultiFab& field_old, amrex::MultiFab& field_new,
                              amrex::Real time, amrex::Real dt);


    amrex::Vector<amrex::Real> measure_source_change(const amrex::MultiFab& source, amrex::Real dt,
                                                      bool local = false);


    static void log_source_change(const amrex::Vector<amrex::Real>& update);


    void log_all_source_changes(amrex::Real dt, bool use_new_state);


    void measure_and_log_source_change(const amrex::MultiFab& source, amrex::Real dt,
                                          const std::string& source_name);



#ifdef SPONGE

    static void assemble_old_damping_source(amrex::MultiFab& source, amrex::MultiFab& state,
                                     amrex::Real time, amrex::Real dt);


    void assemble_new_damping_source(amrex::MultiFab& source, const amrex::MultiFab& field_old,
                                     amrex::MultiFab& field_new, amrex::Real time, amrex::Real dt);


    void add_damping_term(const amrex::Box& bx,
                      amrex::Array4<amrex::Real const> const& state,
                      amrex::Array4<amrex::Real> const& source,
                      amrex::Real dt);

#endif


    void assemble_old_external_source(amrex::MultiFab& source, const amrex::MultiFab& state,
                                  amrex::Real time, amrex::Real dt);


    void assemble_new_external_source(amrex::MultiFab& source,
                                  const amrex::MultiFab& field_old, const amrex::MultiFab& field_new,
                                  amrex::Real time, amrex::Real dt);


    void fill_external_source(const amrex::Real time, const amrex::Real dt,
                         const amrex::MultiFab& field_old, const amrex::MultiFab& field_new,
                         amrex::MultiFab& external_source);


    void assemble_old_thermodynamic_source(amrex::MultiFab& source, amrex::MultiFab& state,
                                     amrex::Real time, amrex::Real dt);


    void assemble_new_thermodynamic_source(amrex::MultiFab& source, const amrex::MultiFab& field_old,
                                     amrex::MultiFab& field_new, amrex::Real time, amrex::Real dt);


    void fill_thermodynamic_source(const amrex::MultiFab& state_in, amrex::MultiFab& thermodynamic_src);


    void assemble_old_geometry_source(amrex::MultiFab& source, const amrex::MultiFab& state,
                                   amrex::Real time, amrex::Real dt);


    void assemble_new_geometry_source(amrex::MultiFab& source, const amrex::MultiFab& field_old,
                                   const amrex::MultiFab& field_new, amrex::Real time, amrex::Real dt);


    void fill_axisymmetric_geometry_source(amrex::Real time, amrex::Real dt,
                             const amrex::MultiFab& conserved_state, amrex::MultiFab& geometry_source);


    void fill_spherical_geometry_source(amrex::Real time, amrex::Real dt,
                                 const amrex::MultiFab& conserved_state, amrex::MultiFab& geometry_source);



    step_result before_advance_hooks (amrex::Real time, amrex::Real dt);


    step_result before_hydro_hooks (amrex::Real time, amrex::Real dt);



    step_result after_hydro_hooks (amrex::Real time, amrex::Real dt);



    step_result after_advance_hooks (amrex::Real time, amrex::Real dt);

#endif
