#include <HydroSim.H>
#include <HydroSim_helpers.H>
#include <Rotation.H>
#ifdef HYBRID_MOMENTUM
#include <hybrid.H>
#endif

void
HydroSim::rsrc(const Box& bx,
             Array4<Real const> const& uold,
             Array4<Real> const& source,
             const Real dt) {

  GeometryData geomdata = geom.data();
  const auto coord = geomdata.Coord();

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real Sr[3] = {};
    Real src[NUM_SOURCE] = {};


    Real snew[NUM_STATE] = {};

    GpuArray<Real, 3> loc;
    position(i, j, k, geomdata, loc);

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      loc[dir] -= problem::center[dir];
    }

    auto omega = get_angular_velocity(geomdata, j);

    Real rho = uold(i,j,k,URHO);
    Real rhoInv = 1.0_rt / rho;

    for (int n = 0; n < NUM_STATE; n++) {
      snew[n] = uold(i,j,k,n);
    }

    Real kinetic_energy_old = 0.5_rt * (snew[UMX] * snew[UMX] + snew[UMY] * snew[UMY] + snew[UMZ] * snew[UMZ]) * rhoInv;

    GpuArray<Real, 3> v;

    v[0] = uold(i,j,k,UMX) * rhoInv;
    v[1] = uold(i,j,k,UMY) * rhoInv;
    v[2] = uold(i,j,k,UMZ) * rhoInv;

    bool with_coriolis = true;
    fictitious_acceleration(loc, v, omega, coord, with_coriolis, Sr);

    for (auto& e : Sr) {
        e *= rho;
    }

    src[UMX] = Sr[0];
    src[UMY] = Sr[1];
    src[UMZ] = Sr[2];

    snew[UMX] += dt * src[UMX];
    snew[UMY] += dt * src[UMY];
    snew[UMZ] += dt * src[UMZ];

#ifdef HYBRID_MOMENTUM
    GpuArray<Real, 3> momentum_source_vec;
    momentum_source_vec[0] = src[UMX];
    momentum_source_vec[1] = src[UMY];
    momentum_source_vec[2] = src[UMZ];

    GpuArray<Real, 3> hybrid_source_vector;
    compute_hybrid_momentum_source(loc, momentum_source_vec, hybrid_source_vector);

    snew[UMR] += dt * hybrid_source_vector[0];
    snew[UML] += dt * hybrid_source_vector[1];
    snew[UMP] += dt * hybrid_source_vector[2];

    src[UMR] = hybrid_source_vector[0];
    src[UML] = hybrid_source_vector[1];
    src[UMP] = hybrid_source_vector[2];
#endif


    Real SrE;

    if (rotation_source_mode == 1 || rotation_source_mode == 2) {

      SrE = uold(i,j,k,UMX) * rhoInv * Sr[0] +
            uold(i,j,k,UMY) * rhoInv * Sr[1] +
            uold(i,j,k,UMZ) * rhoInv * Sr[2];

    } else if (rotation_source_mode == 3) {

      Real kinetic_energy_updated = 0.5_rt * (snew[UMX] * snew[UMX] + snew[UMY] * snew[UMY] + snew[UMZ] * snew[UMZ]) * rhoInv;
      SrE = kinetic_energy_updated - kinetic_energy_old;

    } else if (rotation_source_mode == 4) {


      SrE = uold(i,j,k,UMX) * rhoInv * Sr[0] +
            uold(i,j,k,UMY) * rhoInv * Sr[1] +
            uold(i,j,k,UMZ) * rhoInv * Sr[2];

    } else {
#ifndef AMREX_USE_GPU
      amrex::Error("Error:: rotation_sources_nd.F90 :: invalid rotation_source_mode");
#endif
    }

    src[UEDEN] += SrE;

    snew[UEDEN] += dt * src[UEDEN];


    for (int n = 0; n < NUM_SOURCE; n++) {
      source(i,j,k,n) += src[n];
    }

  });

}


void
HydroSim::apply_rotation_correction(const Box& bx,
                 Array4<Real const> const& uold,
                 Array4<Real const> const& unew,
                 Array4<Real> const& source,
                 Array4<Real const> const& xface_flux,
                 Array4<Real const> const& flux1,
                 Array4<Real const> const& flux2,
                 const Real dt,
                 Array4<Real const> const& vol) {








  GeometryData geomdata = geom.data();
  const auto coord = geomdata.Coord();

  Real halfInvDt = 0.5_rt / dt;

  GpuArray<Real, 3> dx;
  for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      dx[i] = geom.CellSizeArray()[i];
  }
  for (int i = AMREX_SPACEDIM; i < 3; ++i) {
      dx[i] = 0.0_rt;
  }

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
  {

    Real mom_source_old[3] = {};
    Real mom_source_new[3] = {};
    Real mom_source_corr[3] = {};
    Real src[NUM_SOURCE] = {};


    Real snew[NUM_STATE] = {};

    GpuArray<Real, 3> loc;
    position(i, j, k, geomdata, loc);

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      loc[dir] -= problem::center[dir];
    }

    auto omega = get_angular_velocity(geomdata, j);

    Real rho_at_old = uold(i,j,k,URHO);
    Real rho_old_inv = 1.0_rt / uold(i,j,k,URHO);

    Real rho_at_new = unew(i,j,k,URHO);
    Real rho_new_inv = 1.0_rt / unew(i,j,k,URHO);

    for (int n = 0; n < NUM_STATE; n++) {
      snew[n] = unew(i,j,k,n);
    }

    Real kinetic_energy_old = 0.5_rt * (snew[UMX] * snew[UMX] + snew[UMY] * snew[UMY] + snew[UMZ] * snew[UMZ]) * rho_new_inv;



    GpuArray<Real, 3> vold;

    vold[0] = uold(i,j,k,UMX) * rho_old_inv;
    vold[1] = uold(i,j,k,UMY) * rho_old_inv;
    vold[2] = uold(i,j,k,UMZ) * rho_old_inv;

    bool with_coriolis = true;
    fictitious_acceleration(loc, vold, omega, coord, with_coriolis, mom_source_old);

    for (auto& e : mom_source_old) {
        e *= rho_at_old;
    }

    Real energy_source_old = vold[0] * mom_source_old[0] + vold[1] * mom_source_old[1] + vold[2] * mom_source_old[2];



    GpuArray<Real, 3> velocity_new;

    velocity_new[0] = unew(i,j,k,UMX) * rho_new_inv;
    velocity_new[1] = unew(i,j,k,UMY) * rho_new_inv;
    velocity_new[2] = unew(i,j,k,UMZ) * rho_new_inv;

    fictitious_acceleration(loc, velocity_new, omega, coord, with_coriolis, mom_source_new);

    for (auto& e : mom_source_new) {
        e *= rho_at_new;
    }

    Real energy_source_new = velocity_new[0] * mom_source_new[0] + velocity_new[1] * mom_source_new[1] + velocity_new[2] * mom_source_new[2];



    for (int n = 0; n < 3; n++) {
      mom_source_corr[n] = 0.5_rt * (mom_source_new[n] - mom_source_old[n]);
    }

    if (rotation_implicit_step == 1) {



      Real acc[3];
      with_coriolis = false;
      fictitious_acceleration(loc, velocity_new, omega, coord, with_coriolis, acc);

      Real predicted_momentum[3];
      for (int n = 0; n < 3; n++) {
        predicted_momentum[n] = unew(i,j,k,UMX+n) - 0.5_rt * mom_source_old[n] * dt + 0.5_rt * rho_at_new * acc[n] * dt;
      }



      Array2D<Real, 0, 2, 0, 2> omega_times_dt_matrix = {};
      fill_omega_times_dt_matrix(dt, omega, omega_times_dt_matrix);

      Real updated_momentum[3] = {};


      for (int l = 0; l < 3; l++) {
        for (int m = 0; m < 3; m++) {
          updated_momentum[l] += omega_times_dt_matrix(l,m) * predicted_momentum[m];
        }
      }



      for (int n = 0; n < 3; n++) {
        mom_source_corr[n] = (updated_momentum[n] - unew(i,j,k,UMX+n)) / dt;
      }

    }


    src[UMX] = mom_source_corr[0];
    src[UMY] = mom_source_corr[1];
    src[UMZ] = mom_source_corr[2];

    snew[UMX] += dt * src[UMX];
    snew[UMY] += dt * src[UMY];
    snew[UMZ] += dt * src[UMZ];

#ifdef HYBRID_MOMENTUM
    GpuArray<Real, 3> hybrid_source_vector;

    GpuArray<Real, 3> momentum_source_vec;
    momentum_source_vec[0] = src[UMX];
    momentum_source_vec[1] = src[UMY];
    momentum_source_vec[2] = src[UMZ];

    compute_hybrid_momentum_source(loc, momentum_source_vec, hybrid_source_vector);

    snew[UMR] += dt * hybrid_source_vector[0];
    snew[UML] += dt * hybrid_source_vector[1];
    snew[UMP] += dt * hybrid_source_vector[2];

    src[UMR] = hybrid_source_vector[0];
    src[UML] = hybrid_source_vector[1];
    src[UMP] = hybrid_source_vector[2];
#endif


    Real energy_source_corr;

    if (rotation_source_mode == 1) {


      energy_source_corr = 0.5_rt * (energy_source_new - energy_source_old);

    } else if (rotation_source_mode == 2) {


      velocity_new[0] = snew[UMX] * rho_new_inv;
      velocity_new[1] = snew[UMY] * rho_new_inv;
      velocity_new[2] = snew[UMZ] * rho_new_inv;

      Real acc[3];
      with_coriolis = true;
      fictitious_acceleration(loc, velocity_new, omega, coord, with_coriolis, acc);

      mom_source_new[0] = rho_at_new * acc[0];
      mom_source_new[1] = rho_at_new * acc[1];
      mom_source_new[2] = rho_at_new * acc[2];

      energy_source_new = velocity_new[0] * mom_source_new[0] + velocity_new[1] * mom_source_new[1] + velocity_new[2] * mom_source_new[2];

      energy_source_corr = 0.5_rt * (energy_source_new - energy_source_old);

    } else if (rotation_source_mode == 3) {


      Real kinetic_energy_updated = 0.5_rt * (snew[UMX] * snew[UMX] + snew[UMY] * snew[UMY] + snew[UMZ] * snew[UMZ]) * rho_new_inv;
      energy_source_corr = kinetic_energy_updated - kinetic_energy_old;

    } else if (rotation_source_mode == 4) {



      energy_source_corr = - energy_source_old;


      Real edge_spin_source[3][2];


      for (int dir = 0; dir < 3; ++dir) {
          for (int edge = 0; edge <= 1; ++edge) {

              bool ccx = dir == 0 ? true : false;
              bool ccy = dir == 1 ? true : false;
              bool ccz = dir == 2 ? true : false;

              int ie = dir == 0 ? i + edge : i;
              int je = dir == 1 ? j + edge : j;
              int ke = dir == 2 ? k + edge : k;

              position(ie, je, ke, geomdata, loc, ccx, ccy, ccz);

              for (int n = 0; n < AMREX_SPACEDIM; ++n) {
                  loc[n] -= problem::center[n];
              }

              GpuArray<Real, 3> temp_velocity{};
              Real temp_spin_source[3];

              with_coriolis = false;
              fictitious_acceleration(loc, temp_velocity, omega, coord, with_coriolis, temp_spin_source);

              edge_spin_source[dir][edge] = temp_spin_source[dir];

          }
      }

      energy_source_corr += halfInvDt * (xface_flux(i      ,j,k) * edge_spin_source[0][0] * dx[0] +
                           xface_flux(i+1*dg0,j,k) * edge_spin_source[0][1] * dx[0] +
                           flux1(i,j      ,k) * edge_spin_source[1][0] * dx[1] +
                           flux1(i,j+1*dg1,k) * edge_spin_source[1][1] * dx[1] +
                           flux2(i,j,k      ) * edge_spin_source[2][0] * dx[2] +
                           flux2(i,j,k+1*dg2) * edge_spin_source[2][1] * dx[2]) / vol(i,j,k);

    } else {
#ifndef AMREX_USE_GPU
      amrex::Error("Error:: rotation_sources_nd.F90 :: invalid rotation_source_mode");
#endif
    }

    src[UEDEN] = energy_source_corr;


    for (int n = 0; n < NUM_SOURCE; n++) {
      source(i,j,k,n) += src[n];
    }

  });

}
