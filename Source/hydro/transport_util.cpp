#include <HydroSim.H>

#ifdef ROTATION
#include <Rotation.H>
#endif

#include <HydroSim_helpers.H>
#include <transport_util.H>

#ifdef HYBRID_MOMENTUM
#include <hybrid.H>
#endif

#ifdef RADIATION
#include <Radiation.H>
#include <flux_limiter.H>
#include <radiation_helpers.H>
#endif

#include <eos.H>

using namespace amrex;


void
HydroSim::ctoprim(const Box& bx,
                const Real time,
                Array4<Real const> const& uin,
#ifdef MHD
                Array4<Real const> const& Bx,
                Array4<Real const> const& By,
                Array4<Real const> const& Bz,
#endif
#ifdef RADIATION
                Array4<Real const> const& RadEnergyIn,
                Array4<Real const> const& lam,
#endif
                Array4<Real> const& prim_arr,
                Array4<Real> const& qthermo_arr) {

  amrex::ignore_unused(time);

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
      auto q = [&] (int n) -> Real& { return prim_arr(i,j,k,n); };
      auto qthermo = [&] (int n) -> Real& { return qthermo_arr(i,j,k,n); };

      hydro::cell_to_primitive(i, j, k, uin,
#ifdef MHD
                                       Bx, By, Bz,
#endif
#ifdef RADIATION
                                       RadEnergyIn, lam,
#endif
                                       q, qthermo, prim_arr.nComp() == NQ);
  });
}


void
HydroSim::shock(const Box& bx,
              Array4<Real const> const& prim_arr,
              Array4<Real const> const& U_source_arr,
              Array4<Real> const& shk) {

  // This is a basic multi-dimensional shock detection algorithm.
  // we look for |grad P . dx| / P > 2/3 and div u < 0

  const auto dx = geom.CellSizeArray();
  const auto problo = geom.ProbLoArray();
  const int coord_type = geom.Coord();

  Real dxinv = 1.0_rt / dx[0];
#if AMREX_SPACEDIM >= 2
  Real dyinv = 1.0_rt / dx[1];
#endif
#if AMREX_SPACEDIM == 3
  Real dzinv = 1.0_rt / dx[2];
#endif

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
    Real div_vel = 0.0_rt;

    // construct div{U}
    if (coord_type == 0) {

      // Cartesian
      div_vel += 0.5_rt * (prim_arr(i+1,j,k,QU) - prim_arr(i-1,j,k,QU)) * dxinv;
#if (AMREX_SPACEDIM >= 2)
      div_vel += 0.5_rt * (prim_arr(i,j+1,k,QV) - prim_arr(i,j-1,k,QV)) * dyinv;
#endif
#if (AMREX_SPACEDIM == 3)
      div_vel += 0.5_rt * (prim_arr(i,j,k+1,QW) - prim_arr(i,j,k-1,QW)) * dzinv;
#endif

#if AMREX_SPACEDIM <= 2
   } else if (coord_type == 1) {

     // r-z
     Real rc = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
     Real rm = problo[0] + (static_cast<Real>(i) - 0.5_rt) * dx[0];
     Real rp = problo[0] + (static_cast<Real>(i) + 1.5_rt) * dx[0];

#if (AMREX_SPACEDIM == 1)
     div_vel += 0.5_rt * (rp * prim_arr(i+1,j,k,QU) - rm * prim_arr(i-1,j,k,QU)) / (rc * dx[0]);
#endif
#if (AMREX_SPACEDIM == 2)
     div_vel += 0.5_rt * (rp * prim_arr(i+1,j,k,QU) - rm * prim_arr(i-1,j,k,QU)) / (rc * dx[0]) +
              0.5_rt * (prim_arr(i,j+1,k,QV) - prim_arr(i,j-1,k,QV)) * dyinv;
#endif
#endif

    } else if (coord_type == 2) {

      // 1-d spherical
      Real rc = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
      Real rm = problo[0] + (static_cast<Real>(i) - 0.5_rt) * dx[0];
      Real rp = problo[0] + (static_cast<Real>(i) + 1.5_rt) * dx[0];

      div_vel += 0.5_rt * (rp * rp * prim_arr(i+1,j,k,QU) - rm * rm * prim_arr(i-1,j,k,QU)) / (rc * rc * dx[0]);
#if AMREX_SPACEDIM == 2

      Real polar_center = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1];
      Real theta_minus = problo[1] + (static_cast<Real>(j) - 0.5_rt) * dx[1];
      Real theta_plus = problo[1] + (static_cast<Real>(j) + 1.5_rt) * dx[1];

      div_vel += 0.5_rt * (std::sin(theta_plus) * prim_arr(i,j+1,k,QV) -
                         std::sin(theta_minus) * prim_arr(i,j-1,k,QV)) /
          (rc * std::sin(polar_center) * dx[1]);
#endif

#ifndef AMREX_USE_GPU

    } else {
      amrex::Error("ERROR: invalid coord_type in shock");
#endif
    }


    // now compute (grad P - rho g) . dx
    // We subtract off the hydrostatic force, since the pressure that
    // balances that is not available to make a shock.  We compute this
    // as:
    //
    // P'_{i+1} = P_{i+1} - [ P_i + \int_{x_i}^{x_{i+1}} rho g dx ]
    //
    // where the term in the [ ] is the hydrostatic pressure in i+1
    // computed by integrating from x_i to x_{i+1}
    //
    // We'll use a centered diff for the pressure gradient.
    Real pressure_diff_x = 0.5_rt * (prim_arr(i+1,j,k,QPRES) - prim_arr(i-1,j,k,QPRES));
    if (shock_sensor_subtract_src == 1) {
        pressure_diff_x += -0.25_rt * dx[0] * (U_source_arr(i+1,j,k,UMX) + 2.0_rt * U_source_arr(i,j,k,UMX) + U_source_arr(i-1,j,k,UMX));
    }
    Real pressure_diff_y = 0.0_rt;
    Real pressure_diff_z = 0.0_rt;
#if AMREX_SPACEDIM >= 2
    pressure_diff_y = 0.5_rt * (prim_arr(i,j+1,k,QPRES) - prim_arr(i,j-1,k,QPRES));
    if (shock_sensor_subtract_src == 1) {
        Real dy{dx[1]};
        if (coord_type == 2) {
            // dx[1] is just theta_cell_width
            Real rc = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
            dy *= rc;
        }
        pressure_diff_y += -0.25_rt * dy * (U_source_arr(i,j+1,k,UMY) + 2.0_rt * U_source_arr(i,j,k,UMY) + U_source_arr(i,j-1,k,UMY));
    }
#endif
#if AMREX_SPACEDIM == 3
    pressure_diff_z = 0.5_rt * (prim_arr(i,j,k+1,QPRES) - prim_arr(i,j,k-1,QPRES));
    if (shock_sensor_subtract_src == 1) {
        pressure_diff_z += -0.25_rt * dx[2] * (U_source_arr(i,j,k+1,UMZ) + 2.0_rt * U_source_arr(i,j,k,UMZ) + U_source_arr(i,j,k-1,UMZ));
    }
#endif

    //Real relative_pressure_grad = std::sqrt(pressure_diff_x * pressure_diff_x + pressure_diff_y * pressure_diff_y + pressure_diff_z * pressure_diff_z) / prim_arr(i,j,k,QPRES);
    Real vel = std::sqrt(prim_arr(i,j,k,QU) * prim_arr(i,j,k,QU) +
                         prim_arr(i,j,k,QV) * prim_arr(i,j,k,QV) +
                         prim_arr(i,j,k,QW) * prim_arr(i,j,k,QW));

    Real relative_pressure_grad{0.0_rt};
    if (vel != 0.0) {
        relative_pressure_grad = std::abs(pressure_diff_x * prim_arr(i,j,k,QU) +
                                  pressure_diff_y * prim_arr(i,j,k,QV) +
                                  pressure_diff_z * prim_arr(i,j,k,QW)) / vel;
    }
    relative_pressure_grad /= prim_arr(i,j,k,QPRES);

    if (relative_pressure_grad > hydrosim::shock_sensor_threshold && div_vel < 0.0_rt) {
      shk(i,j,k) = 1.0_rt;
    } else {
      shk(i,j,k) = 0.0_rt;
    }
  });

}


void
HydroSim::divu(const Box& bx,
             Array4<Real const> const& prim_arr,
             Array4<Real> const& div) {
  // this computes the *node-centered* divergence

  const auto dx = geom.CellSizeArray();

#if AMREX_SPACEDIM <= 2
  const int coord_type = geom.Coord();
  const auto problo = geom.ProbLoArray();
#endif

  Real dxinv = 1.0_rt / dx[0];
#if AMREX_SPACEDIM >= 2
  Real dyinv = 1.0_rt / dx[1];
#endif
#if AMREX_SPACEDIM == 3
  Real dzinv = 1.0_rt / dx[2];
#endif

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

#if AMREX_SPACEDIM == 1
    if (coord_type == 0) {
        div(i,j,k) = (prim_arr(i,j,k,QU) - prim_arr(i-1,j,k,QU)) * dxinv;

    } else if (coord_type == 1) {
        // axisymmetric
        if (i == 0) {
            div(i,j,k) = 0.0_rt;
        } else {
            Real rl = (i - 0.5_rt) * dx[0] + problo[0];
            Real rr = (i + 0.5_rt) * dx[0] + problo[0];
            Real rc = (i) * dx[0] + problo[0];

            div(i,j,k) = (rr * prim_arr(i,j,k,QU) - rl * prim_arr(i-1,j,k,QU)) * dxinv / rc;
        }
    } else {
        // spherical
        if (i == 0) {
            div(i,j,k) = 0.0_rt;
        } else {
            Real rl = (i - 0.5_rt) * dx[0] + problo[0];
            Real rr = (i + 0.5_rt) * dx[0] + problo[0];
            Real rc = (i) * dx[0] + problo[0];

            div(i,j,k) = (rr * rr * prim_arr(i,j,k,QU) - rl * rl * prim_arr(i-1,j,k,QU)) * dxinv / (rc * rc);
        }
    }
#endif

#if AMREX_SPACEDIM == 2
    Real ux = 0.0_rt;
    Real vy = 0.0_rt;

    if (coord_type == 0) {

        // Cartesian

        ux = 0.5_rt * (prim_arr(i,j,k,QU) - prim_arr(i-1,j,k,QU) + prim_arr(i,j-1,k,QU) - prim_arr(i-1,j-1,k,QU)) * dxinv;
        vy = 0.5_rt * (prim_arr(i,j,k,QV) - prim_arr(i,j-1,k,QV) + prim_arr(i-1,j,k,QV) - prim_arr(i-1,j-1,k,QV)) * dyinv;

    } else if (coord_type == 1) {

        // Cylindrical R-Z

        if (i == 0) {
            ux = 0.0_rt;
        } else {
            Real rl = (i - 0.5_rt) * dx[0] + problo[0];
            Real rr = (i + 0.5_rt) * dx[0] + problo[0];
            Real rc = (i) * dx[0] + problo[0];

            // These are transverse averages in the y-direction
            Real ul = 0.5_rt * (prim_arr(i-1,j,k,QU) + prim_arr(i-1,j-1,k,QU));
            Real ur = 0.5_rt * (prim_arr(i,j,k,QU) + prim_arr(i,j-1,k,QU));

            // Take 1/r d/dr(r*u)
            ux = (rr * ur - rl * ul) * dxinv / rc;
        }

        // These are transverse averages in the x-direction
        Real vb = 0.5_rt * (prim_arr(i,j-1,k,QV) + prim_arr(i-1,j-1,k,QV));
        Real vt = 0.5_rt * (prim_arr(i,j,k,QV) + prim_arr(i-1,j,k,QV));

        vy = (vt - vb) * dyinv;

    } else {

        // Spherical R-Theta

        Real rl = (i - 0.5_rt) * dx[0] + problo[0];
        Real rr = (i + 0.5_rt) * dx[0] + problo[0];
        Real rc = (i) * dx[0] + problo[0];

        // cell-centered sin(theta) of top, bot cell and face-centered
        Real sint = std::sin((j + 0.5_rt) * dx[1] + problo[1]);
        Real sinb = std::sin((j - 0.5_rt) * dx[1] + problo[1]);
        Real sinc = std::sin(j  * dx[1] + problo[1]);

        // These are transverse averages in the y-direction
        Real ul = 0.5_rt * (prim_arr(i-1,j,k,QU) + prim_arr(i-1,j-1,k,QU));
        Real ur = 0.5_rt * (prim_arr(i,j,k,QU) + prim_arr(i,j-1,k,QU));

        // Finite difference to get divergence. ux = 1/r^2 d/dr(r^2 * u)
        ux = (ur * rr * rr - ul * rl * rl) * dxinv / (rc * rc);

        // If sinc == 0, then vy goes inf.
        // But due to Phi-symmetry, vt*sint = vb*sinb, so set to 0.
        if (sinc == 0.0_rt) {
            vy = 0.0_rt;
        } else {
            // These are transverse averages in the x-direction
            Real vb = 0.5_rt * (prim_arr(i,j-1,k,QV) + prim_arr(i-1,j-1,k,QV));
            Real vt = 0.5_rt * (prim_arr(i,j,k,QV) + prim_arr(i-1,j,k,QV));

            // Finite difference to get divergence. vy = 1/(r sin) * d/theta_cell_width(v sin)
            vy = (vt * sint - vb * sinb) * dyinv / (rc * sinc);
        }
    }

    div(i,j,k) = ux + vy;
#endif

#if AMREX_SPACEDIM == 3
    Real ux = 0.25_rt * (prim_arr(i,j,k,QU) - prim_arr(i-1,j,k,QU) +
                         prim_arr(i,j,k-1,QU) - prim_arr(i-1,j,k-1,QU) +
                         prim_arr(i,j-1,k,QU) - prim_arr(i-1,j-1,k,QU) +
                         prim_arr(i,j-1,k-1,QU) - prim_arr(i-1,j-1,k-1,QU)) * dxinv;

    Real vy = 0.25_rt * (prim_arr(i,j,k,QV) - prim_arr(i,j-1,k,QV) +
                         prim_arr(i,j,k-1,QV) - prim_arr(i,j-1,k-1,QV) +
                         prim_arr(i-1,j,k,QV) - prim_arr(i-1,j-1,k,QV) +
                         prim_arr(i-1,j,k-1,QV) - prim_arr(i-1,j-1,k-1,QV)) * dyinv;

    Real wz = 0.25_rt * (prim_arr(i,j,k,QW) - prim_arr(i,j,k-1,QW) +
                         prim_arr(i,j-1,k,QW) - prim_arr(i,j-1,k-1,QW) +
                         prim_arr(i-1,j,k,QW) - prim_arr(i-1,j,k-1,QW) +
                         prim_arr(i-1,j-1,k,QW) - prim_arr(i-1,j-1,k-1,QW)) * dzinv;

    div(i,j,k) = ux + vy + wz;
#endif

  });

}


void
HydroSim::apply_viscosity(const Box& bx,
                 const int idir,
                 Array4<Real const> const& div,
                 Array4<Real const> const& uin,
                 Array4<Real> const& flux) {

  const auto dx = geom.CellSizeArray();
  const auto coord = geom.Coord();
  const auto problo = geom.ProbLoArray();

  Real diff_coeff = viscosity_coeff;

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {
      Real dL = dx[idir];
      if (coord == 2 && idir == 1) {
          Real r = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
          dL *= r;
      }

      Real face_divergence;
      if (idir == 0) {
          face_divergence = 0.25_rt * (div(i,j,k) + div(i,j+dg1,k) +
                            div(i,j,k+dg2) + div(i,j+dg1,k+dg2));
      } else if (idir == 1) {
          face_divergence = 0.25_rt * (div(i,j,k) + div(i+1,j,k) +
                            div(i,j,k+dg2) + div(i+1,j,k+dg2));
      } else {
          face_divergence = 0.25_rt * (div(i,j,k) + div(i+1,j,k) +
                            div(i,j+dg1,k) + div(i+1,j+dg1,k));
      }

      face_divergence = diff_coeff * std::min(0.0_rt, face_divergence);

      for (int n = 0; n < NUM_STATE; ++n) {

          if (n == UTEMP) {
              continue;
          }
#ifdef SHOCK_VAR
          if (n == UFLAG_SHOCK) {
              continue;
          }
#endif
#ifdef NSE_NET
          if (n == UPROTON_MU || n == UNEUTRON_MU) {
              continue;
          }
#endif

          Real face_jump{};

          if (idir == 0) {
              face_jump = face_divergence * (uin(i,j,k,n) - uin(i-1,j,k,n));
          } else if (idir == 1) {
              face_jump = face_divergence * (uin(i,j,k,n) - uin(i,j-dg1,k,n));
          } else {
              face_jump = face_divergence * (uin(i,j,k,n) - uin(i,j,k-dg2,n));
          }

          flux(i,j,k,n) += dL * face_jump;
      }
  });
}


#ifdef RADIATION
void
HydroSim::apply_viscosity_radiative(const Box& bx,
                     const int idir,
                     Array4<Real const> const& div,
                     Array4<Real const> const& RadEnergyIn,
                     Array4<Real> const& radiative_flux) {

  const auto dx = geom.CellSizeArray();

  amrex::Real diff_coeff = viscosity_coeff;

  int ngroups = Radiation::nGroups;

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

      Real face_divergence;
      if (idir == 0) {
          face_divergence = 0.25_rt * (div(i,j,k) + div(i,j+dg1,k) +
                            div(i,j,k+dg2) + div(i,j+dg1,k+dg2));
      } else if (idir == 1) {
          face_divergence = 0.25_rt * (div(i,j,k) + div(i+1,j,k) +
                            div(i,j,k+dg2) + div(i+1,j,k+dg2));
      } else {
          face_divergence = 0.25_rt * (div(i,j,k) + div(i+1,j,k) +
                            div(i,j+dg1,k) + div(i+1,j+dg1,k));
      }

      face_divergence = diff_coeff * std::min(0.0_rt, face_divergence);

      for (int n = 0; n < ngroups; ++n) {

          Real face_jump{};

          if (idir == 0) {
              face_jump = face_divergence * (RadEnergyIn(i,j,k,n) - RadEnergyIn(i-1,j,k,n));
          } else if (idir == 1) {
              face_jump = face_divergence * (RadEnergyIn(i,j,k,n) - RadEnergyIn(i,j-dg1,k,n));
          } else {
              face_jump = face_divergence * (RadEnergyIn(i,j,k,n) - RadEnergyIn(i,j,k-dg2,n));
          }

          radiative_flux(i,j,k,n) += dx[idir] * face_jump;
      }
  });
}
#endif


void
HydroSim::balance_species_flux(const Box& bx,
                                 Array4<Real> const& flux) {

  // Normalize the fluxes of the mass fractions so that
  // they sum to 0.  This is essentially the CMA procedure.

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real sum = 0.0_rt;

    for (int n = UFS; n < UFS+NumSpec; n++) {
      sum += flux(i,j,k,n);
    }

    Real fac = 1.0_rt;

    // We skip the normalization if the sum is zero or within epsilon.
    // There can be numerical problems here if the density flux is
    // approximately zero at the interface but not exactly, resulting in
    // division by a small number and/or resulting in one of the species
    // fluxes being negative because of roundoff error. There are also other
    // terms like artificial viscosity which can cause these problems.
    // So checking that sum is sufficiently large helps avoid this.

    if (std::abs(sum) > std::numeric_limits<Real>::epsilon() * std::abs(flux(i,j,k,URHO))) {
      fac = flux(i,j,k,URHO) / sum;
    }

    for (int n = UFS; n < UFS+NumSpec; n++) {
      flux(i,j,k,n) = flux(i,j,k,n) * fac;
    }
  });
}


void  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
HydroSim::scale_flux_by_area(const Box& bx,
                   Array4<Real> const& flux,
                   Array4<Real const> const& face_area_arr,
                   const Real dt) {

  amrex::ParallelFor(bx, NUM_STATE,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
  {
    flux(i,j,k,n) = dt * flux(i,j,k,n) * face_area_arr(i,j,k);
  });
}


#ifdef RADIATION
void
HydroSim::scale_radiative_flux(const Box& bx,
                       Array4<Real> const& scaled_radiative_flux,
                       Array4<Real const> const& face_area_arr,
                       const Real dt) {

  amrex::ParallelFor(bx, Radiation::nGroups,
  [=] AMREX_GPU_DEVICE (int i, int j, int k, int g) noexcept
  {
    scaled_radiative_flux(i,j,k,g) = dt * scaled_radiative_flux(i,j,k,g) * face_area_arr(i,j,k);
  });
}
#endif


#ifndef MHD
void
HydroSim::clip_fluxes_at_density_floor(const Box& bx,
                                         int idir,
                                         Array4<Real const> const& u,
                                         Array4<Real const> const& vol,
                                         Array4<Real> const& flux,
                                         Array4<Real const> const& face_area_arr,
                                         Real dt,
                                         bool scale_by_area_dt)
{
    // a positivity-preserving advection scheme. That algorithm blends the actual
    // (second-order) hydro flux with the first-order Lax-Friedrichs flux to ensure
    // positivity. However, we demand a stronger requirement, that rho > small_dens.
    // Additionally, the blending approach can cause problems with multiple advecting
    // species, since adding the Lax-Friedrichs flux does not guarantee physical consistency
    // between the species advection and the density advection. So instead of trying the
    // blending approach, we simply apply a linear scaling to each flux such that it does
    // not violate the density floor.

    const Real floor_margin = 1.1_rt;

    // The density floor is the small density, modified by a small factor.
    // In practice numerical error can cause the density that is created
    // by this flux limiter to be slightly lower than the target density,
    // so we set the target to be slightly larger than the real density floor
    // to avoid density resets.

    Real floor_density = small_dens * floor_margin;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // Grab the states on either side of the interface we are working with,
        // depending on which dimension we're currently calling this with.

        Real right_density = u(i,j,k,URHO);
        Real right_volume = vol(i,j,k);

        Real left_density, left_volume;

        if (idir == 0) {
            left_density = u(i-1,j,k,URHO);
            left_volume = vol(i-1,j,k);
        }
        else if (idir == 1) {
            left_density = u(i,j-1,k,URHO);
            left_volume = vol(i,j-1,k);
        }
        else {
            left_density = u(i,j,k-1,URHO);
            left_volume = vol(i,j,k-1);
        }

        // Coefficients of fluxes on either side of the interface.

        Real right_flux_scale = 1.0_rt / right_volume;
        Real left_flux_scale = 1.0_rt / left_volume;

        if (scale_by_area_dt) {
            right_flux_scale *= dt * face_area_arr(i,j,k);
            left_flux_scale *= dt * face_area_arr(i,j,k);
        }

        // Updates to the zones on either side of the interface.

        Real right_density_change = right_flux_scale * flux(i,j,k,URHO);
        Real left_density_change = left_flux_scale * flux(i,j,k,URHO);

        // Limit all fluxes such that the zone does not go negative in density.
        // We apply this flux limiter on a per-edge basis. So we can guarantee
        // that any individual flux cannot cause a small density in one step,
        // but with the above floor we cannot guarantee that the sum of the
        // fluxes will enforce this constraint. A simple way to guarantee that
        // is to note that in the worst case, all six faces adjacent to a zone
        // could be evacuating that zone (more generally, 2 * AMREX_SPACEDIM).
        // So instead of checking whether rho + drho is less than the floor,
        // we can check whether rho + 6 * drho is less than the floor, and
        // strengthen the limiter by that same factor of 6, so that even in the
        // worst case, the zone will remain positive. This is simple to implement
        // at the risk of being overly aggressive on flux limiting. A more
        // sophisticated version of this might check the ratio of the fluxes
        // (or, similarly, the ratio of the edge velocities coming from the
        // Riemann solver) and scale the larger fluxes more strongly than the
        // weaker fluxes, and perhaps only consider fluxes that subtract from
        // the zone, but this would also be more complicated to implement.

        if (right_density >= floor_density && std::abs(right_density_change) > 0.0_rt && right_density + 2 * AMREX_SPACEDIM * right_density_change < floor_density) {
            Real floor_scale = std::abs((floor_density - right_density) / (2 * AMREX_SPACEDIM * right_density_change));
            for (int n = 0; n < NUM_STATE; ++n) {
                flux(i,j,k,n) = flux(i,j,k,n) * floor_scale;
            }
        }
        else if (left_density >= floor_density && std::abs(left_density_change) > 0.0_rt && left_density - 2 * AMREX_SPACEDIM * left_density_change < floor_density) {
            Real floor_scale = std::abs((floor_density - left_density) / (2 * AMREX_SPACEDIM * left_density_change));
            for (int n = 0; n < NUM_STATE; ++n) {
                flux(i,j,k,n) = flux(i,j,k,n) * floor_scale;
            }
        }

    });
}
#endif


void  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
HydroSim::clamp_state_density(const Box& bx,
                                   Array4<Real> const& state_arr,
                                   const int warning_level) {

#ifdef HYBRID_MOMENTUM
  GeometryData geomdata = geom.data();
#endif

  amrex::ignore_unused(warning_level);

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    if (state_arr(i,j,k,URHO) < small_dens) {

#ifndef AMREX_USE_GPU
      if (warning_level > 1 ||
          (warning_level > 0 && state_arr(i,j,k,URHO) > hydrosim::retry_density_cutoff)) {
        std::cout << " " << std::endl;
        if (state_arr(i,j,k,URHO) < 0.0_rt) {
          std::cout << ">>> RESETTING NEG.  DENSITY AT " << i << ", " << j << ", " << k << std::endl;
        }
        else if (state_arr(i,j,k,URHO) == 0.0_rt) {
          // If the density is *exactly* zero, that almost certainly means something has gone wrong,
          // like we failed to properly fill the state data on grid creation.
          amrex::Error("Density exactly zero at " + std::to_string(i) + ", " +
                                                    std::to_string(j) + ", " +
                                                    std::to_string(k));
        }
        else {
          std::cout << ">>> RESETTING SMALL DENSITY AT " << i << ", " << j << ", " << k << std::endl;
        }
        std::cout << ">>> FROM " << state_arr(i,j,k,URHO) << " TO " << small_dens << std::endl;
        std::cout << ">>> IN GRID " << bx << std::endl;
        std::cout << " " << std::endl;
      }
#endif

      for (int passive_index = 0; passive_index < npassive; passive_index++) {
          const int n = passive_cons_index(passive_index);
          state_arr(i,j,k,n) *= (small_dens / state_arr(i,j,k,URHO));
      }

      eos_re_t eos_state;
      eos_state.rho = small_dens;
      eos_state.T = small_temp;
      for (int n = 0; n < NumSpec; n++) {
        eos_state.xn[n] = state_arr(i,j,k,UFS+n) / small_dens;
      }
#if NAUX_NET > 0
      for (int n = 0; n < NumAux; n++) {
        eos_state.aux[n] = state_arr(i,j,k,UFX+n) / small_dens;
      }
#endif

      eos(eos_input_rt, eos_state);

      state_arr(i,j,k,URHO ) = eos_state.rho;
      state_arr(i,j,k,UTEMP) = eos_state.T;

      state_arr(i,j,k,UMX) = 0.0_rt;
      state_arr(i,j,k,UMY) = 0.0_rt;
      state_arr(i,j,k,UMZ) = 0.0_rt;

      state_arr(i,j,k,UEINT) = eos_state.rho * eos_state.e;
      state_arr(i,j,k,UEDEN) = state_arr(i,j,k,UEINT);

#ifdef HYBRID_MOMENTUM
      GpuArray<Real, 3> loc;

      position(i, j, k, geomdata, loc);

      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        loc[dir] -= problem::center[dir];
      }

      GpuArray<Real, 3> cartesian_vector;

      for (int dir = 0; dir < 3; ++dir) {
        cartesian_vector[dir] = state_arr(i,j,k,UMX+dir);
      }

      GpuArray<Real, 3> hybrid_vector;

      cartesian_to_hybrid(loc, cartesian_vector, hybrid_vector);

      for (int dir = 0; dir < 3; ++dir) {
        state_arr(i,j,k,UMR+dir) = hybrid_vector[dir];
      }
#endif
    }
  });
}


void
HydroSim::set_reflecting_boundary_states(const Box& bx, const int idir,
                               Array4<Real> const& qm,
                               Array4<Real> const& qp) {

    // special care for reflecting BCs
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();

    const auto domlo = geom.Domain().loVect3d();
    const auto domhi = geom.Domain().hiVect3d();

    bool at_lo_symmetry = lo_bc[idir] == amrex::PhysBCType::symmetry;
    bool at_hi_symmetry = hi_bc[idir] == amrex::PhysBCType::symmetry;

    // normal velocity
    const int QUN = QU + idir;

    // this is a loop over interfaces

    if (at_lo_symmetry) {

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {

            // reset the left state at domlo if needed -- it is outside the domain
            if ((idir == 0 && i == domlo[0]) ||
                (idir == 1 && j == domlo[1]) ||
                (idir == 2 && k == domlo[2])) {
                for (int n = 0; n < NQ; n++) {
                    if (n == QUN) {
                        qm(i,j,k,QUN) = -qp(i,j,k,QUN);
                    } else {
                        qm(i,j,k,n) = qp(i,j,k,n);
                    }
                }
            }
        });
    }

    if (at_hi_symmetry) {

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {

            // reset the right state at domhi+1 if needed -- it is outside the domain
            if ((idir == 0 && i == domhi[0]+1) ||
                (idir == 1 && j == domhi[1]+1) ||
                (idir == 2 && k == domhi[2]+1)) {
                for (int n = 0; n < NQ; n++) {
                    if (n == QUN) {
                        qp(i,j,k,QUN) = -qm(i,j,k,QUN);
                    } else {
                        qp(i,j,k,n) = qm(i,j,k,n);
                    }
                }
            }
        });

    }
}
