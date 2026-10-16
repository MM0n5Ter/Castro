#ifndef HYDROSIM_MHD_H
#define HYDROSIM_MHD_H

    advance_status assemble_ctu_mhd_source(amrex::Real time, amrex::Real dt);

    void
    plm(const amrex::Box& bx,
        amrex::Array4<amrex::Real const> const& s,
        amrex::Array4<amrex::Real const> const& qaux,
        amrex::Array4<amrex::Real const> const& flatcoef,
        amrex::Array4<amrex::Real const> const& Bx,
        amrex::Array4<amrex::Real const> const& By,
        amrex::Array4<amrex::Real const> const& Bz,
        amrex::Array4<amrex::Real const> const& srcQ,
        const int idir,
        const amrex::Real dt,
        amrex::Array4<amrex::Real> const& qleft,
        amrex::Array4<amrex::Real> const& qright);

    void
    ppm_mhd(const amrex::Box& bx,
            amrex::Array4<amrex::Real const> const& q_arr,
            amrex::Array4<amrex::Real const> const& qaux,
            amrex::Array4<amrex::Real const> const& flatcoef,
            amrex::Array4<amrex::Real const> const& Bx,
            amrex::Array4<amrex::Real const> const& By,
            amrex::Array4<amrex::Real const> const& Bz,
            amrex::Array4<amrex::Real const> const& srcQ,
            const int idir,
            const amrex::Real dt,
            amrex::Array4<amrex::Real> const& qleft,
            amrex::Array4<amrex::Real> const& qright);

    void
    cons_update_mhd(const amrex::Box& bx, const Real dt,
               amrex::Array4<amrex::Real> const& U_new,
               amrex::Array4<amrex::Real const> const& flux0,
               amrex::Array4<amrex::Real const> const& flux1,
               amrex::Array4<amrex::Real const> const& flux2);

    void
    prim_half(const amrex::Box& bx,
              amrex::Array4<amrex::Real> const& q2D,
              amrex::Array4<amrex::Real const> const& q_arr,
              amrex::Array4<amrex::Real const> const& flxx,
              amrex::Array4<amrex::Real const> const& flxy,
              amrex::Array4<amrex::Real const> const& flxz,
              const amrex::Real dt);


    // the conserved right/left interface states on the faces of one
    // direction, and the primitive states that the transverse updates
    // of corner_couple and half_step write from them
    struct InterfaceStates
    {
        amrex::Array4<amrex::Real const> u_right;
        amrex::Array4<amrex::Real const> u_left;
        amrex::Array4<amrex::Real> q_right;
        amrex::Array4<amrex::Real> q_left;
    };

    void
    corner_couple(const amrex::Box& bx,
                  InterfaceStates const& states,
                  amrex::Array4<amrex::Real const> const& flxd2,
                  amrex::Array4<amrex::Real const> const& Ed1,
                  amrex::Array4<amrex::Real const> const& Ed3,
                  const int d1, const int d2, const int d3,
                  const amrex::Real dt);

    void
    half_step(const amrex::Box& bx,
              InterfaceStates const& states,
              amrex::Array4<amrex::Real const> const& flxd1,
              amrex::Array4<amrex::Real const> const& flxd2,
              amrex::Array4<amrex::Real const> const& Ed,
              amrex::Array4<amrex::Real const> const& Ed1,
              amrex::Array4<amrex::Real const> const& Ed2,
              const int d, const int d1, const int d2, const amrex::Real dt);


#endif
