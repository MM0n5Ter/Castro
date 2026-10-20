#ifndef HYDROSIM_CONDUCTION_COEFFS_H
#define HYDROSIM_CONDUCTION_COEFFS_H

void
fill_cell_conductivity(const amrex::Box& bx,
               amrex::Array4<amrex::Real const> const& field_arr,
               amrex::Array4<amrex::Real> const& cell_conductivity_arr);

void
fill_conduction_coeff(const amrex::Box& bx,
                     amrex::Array4<amrex::Real const> const& field_arr,
                     amrex::Array4<amrex::Real> const& cell_conductivity_arr);

#endif
