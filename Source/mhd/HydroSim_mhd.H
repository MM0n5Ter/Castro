#ifndef HYDROSIM_MHD_H
#define HYDROSIM_MHD_H

    step_result assemble_ctu_mhd_source(amrex::Real time, amrex::Real dt);

    void
    plm(const amrex::Box& bx,
        const int idir,
        amrex::Array4<amrex::Real const> const& s,
        amrex::Array4<amrex::Real const> const& qthermo,
        amrex::Array4<amrex::Real const> const& flatten_factor,
        amrex::Array4<amrex::Real const> const& Bx,
        amrex::Array4<amrex::Real const> const& By,
        amrex::Array4<amrex::Real const> const& Bz,
        amrex::Array4<amrex::Real> const& prim_left,
        amrex::Array4<amrex::Real> const& prim_right,
        amrex::Array4<amrex::Real const> const& prim_source,
        const amrex::Real dt);

    void
    parabolic_reconstruct_mhd(const amrex::Box& bx,
            const int idir,
            amrex::Array4<amrex::Real const> const& prim_arr,
            amrex::Array4<amrex::Real const> const& qthermo,
            amrex::Array4<amrex::Real const> const& flatten_factor,
            amrex::Array4<amrex::Real const> const& Bx,
            amrex::Array4<amrex::Real const> const& By,
            amrex::Array4<amrex::Real const> const& Bz,
            amrex::Array4<amrex::Real> const& prim_left,
            amrex::Array4<amrex::Real> const& prim_right,
            amrex::Array4<amrex::Real const> const& prim_source,
            const amrex::Real dt);

    void
    cons_update_mhd(const amrex::Box& bx, const Real dt,
               amrex::Array4<amrex::Real> const& cons_new,
               amrex::Array4<amrex::Real const> const& xface_flux,
               amrex::Array4<amrex::Real const> const& flux1,
               amrex::Array4<amrex::Real const> const& flux2);

    static void
    PrimitiveToConserved(const amrex::Box& bx,
               amrex::Array4<amrex::Real const> const& prim_arr,
               amrex::Array4<amrex::Real> const& state_view);

    void
    primitive_half_update(const amrex::Box& bx,
              amrex::Array4<amrex::Real> const& q2D,
              amrex::Array4<amrex::Real const> const& prim_arr,
              amrex::Array4<amrex::Real const> const& flxx,
              amrex::Array4<amrex::Real const> const& flxy,
              amrex::Array4<amrex::Real const> const& flxz,
              const amrex::Real dt);


    static void
    edge_emf_x(const amrex::Box& bx,
                    amrex::Array4<amrex::Real const> const& prim_arr,
                    amrex::Array4<amrex::Real> const& E,
                    amrex::Array4<amrex::Real const> const& flxy,
                    amrex::Array4<amrex::Real const> const& flxz);

    static void
    edge_emf_y(const amrex::Box& bx,
                    amrex::Array4<amrex::Real const> const& prim_arr,
                    amrex::Array4<amrex::Real> const& E,
                    amrex::Array4<amrex::Real const> const& flxx,
                    amrex::Array4<amrex::Real const> const& flxz);

    static void
    edge_emf_z(const amrex::Box& bx,
                    amrex::Array4<amrex::Real const> const& prim_arr,
                    amrex::Array4<amrex::Real> const& E,
                    amrex::Array4<amrex::Real const> const& flxx,
                    amrex::Array4<amrex::Real const> const& flxy);

    void
    corner_coupling(const amrex::Box& bx,
                  amrex::Array4<amrex::Real const> const& ur,
                  amrex::Array4<amrex::Real const> const& ul,
                  amrex::Array4<amrex::Real const> const& flux_dir2,
                  amrex::Array4<amrex::Real const> const& Ed1,
                  amrex::Array4<amrex::Real const> const& Ed3,
                  const int d1, const int d2, const int d3,
                  const amrex::Real dt,
                  amrex::Array4<amrex::Real> const& right_state_out,
                  amrex::Array4<amrex::Real> const& left_state_out);

    void
    predict_half_step(const amrex::Box& bx,
              amrex::Array4<amrex::Real const> const& ur,
              amrex::Array4<amrex::Real const> const& ul,
              amrex::Array4<amrex::Real const> const& flux_dir1,
              amrex::Array4<amrex::Real const> const& flux_dir2,
              amrex::Array4<amrex::Real const> const& Ed,
              amrex::Array4<amrex::Real const> const& Ed1,
              amrex::Array4<amrex::Real const> const& Ed2,
              const int d, const int d1, const int d2, const amrex::Real dt,
              amrex::Array4<amrex::Real> const& right_state_out,
              amrex::Array4<amrex::Real> const& left_state_out);

    static void
    mhd_riemann_flux(const amrex::Box& bx,
         amrex::Array4<amrex::Real const> const& prim_left,
         amrex::Array4<amrex::Real const> const& prim_right,
         amrex::Array4<amrex::Real> const& flx,
         const int dir);


#endif
