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

    // Temporary array for seeing what the new state would be if the update were applied here.

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

    // Kinetic energy source: this is v . the momentum source.
    // We don't apply in the case of the conservative energy
    // formulation.

    Real SrE;

    if (rotation_source_mode == 1 || rotation_source_mode == 2) {  // NOLINT(bugprone-branch-clone)

      SrE = uold(i,j,k,UMX) * rhoInv * Sr[0] +
            uold(i,j,k,UMY) * rhoInv * Sr[1] +
            uold(i,j,k,UMZ) * rhoInv * Sr[2];

    } else if (rotation_source_mode == 3) {

      Real kinetic_energy_updated = 0.5_rt * (snew[UMX] * snew[UMX] + snew[UMY] * snew[UMY] + snew[UMZ] * snew[UMZ]) * rhoInv;
      SrE = kinetic_energy_updated - kinetic_energy_old;

    } else if (rotation_source_mode == 4) {

      // The conservative energy formulation does not strictly require
      // any energy source-term here, because it depends only on the
      // fluid motions from the hydrodynamical fluxes which we will only
      // have when we get to the 'corrector' step. Nevertheless we add a
      // predictor energy source term in the way that the other methods
      // do, for consistency. We will fully subtract this predictor value
      // during the corrector step, so that the final result is correct.
      // Here we use the same approach as rotation_source_mode == 2.

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

    // Add to the outgoing source array.

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

  // Corrector step for the rotation source terms. This is applied
  // after the hydrodynamics update to fix the time-level n
  // prediction and add the time-level n+1 data.

  // uold and unew are the old and new time state data

  // source is the source term to send back

  // xface_flux, flux1, and flux2 are the hydrodynamical mass fluxes


  // Rotation source options for how to add the work to (rho E):
  // rotation_source_mode =
  // 1: Standard version ("does work")
  // 2: Modification of type 1 that updates the momentum before constructing the energy corrector
  // 3: Puts all rotational work into KE, not (rho e)
  // 4: Conservative energy formulation

  // Note that the time passed to this function
  // is the new time at time-level n+1.

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

    // Temporary array for seeing what the new state would be if the update were applied here.

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


    // Define old source terms

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


    // Define new source terms

    GpuArray<Real, 3> velocity_new;

    velocity_new[0] = unew(i,j,k,UMX) * rho_new_inv;
    velocity_new[1] = unew(i,j,k,UMY) * rho_new_inv;
    velocity_new[2] = unew(i,j,k,UMZ) * rho_new_inv;

    fictitious_acceleration(loc, velocity_new, omega, coord, with_coriolis, mom_source_new);

    for (auto& e : mom_source_new) {
        e *= rho_at_new;
    }

    Real energy_source_new = velocity_new[0] * mom_source_new[0] + velocity_new[1] * mom_source_new[1] + velocity_new[2] * mom_source_new[2];


    // Define correction terms

    for (int n = 0; n < 3; n++) {
      mom_source_corr[n] = 0.5_rt * (mom_source_new[n] - mom_source_old[n]);
    }

    if (rotation_implicit_step == 1) {

      // Coupled/implicit momentum update

      // Do the full corrector step with the old contribution (subtract 1/2 times the old term) and do
      // the non-Coriolis parts of the new contribution (add 1/2 of the new term).

      Real acc[3];
      with_coriolis = false;
      fictitious_acceleration(loc, velocity_new, omega, coord, with_coriolis, acc);

      Real predicted_momentum[3];
      for (int n = 0; n < 3; n++) {
        predicted_momentum[n] = unew(i,j,k,UMX+n) - 0.5_rt * mom_source_old[n] * dt + 0.5_rt * rho_at_new * acc[n] * dt;
      }


      // The following is the general solution to the 3D coupled system,
      // assuming that the rotation vector has components along all three
      // axes, obtained using Cramer's rule (the coefficient matrix is
      // defined above). In practice the user will probably only be using
      // one axis for rotation. Note that this will
      // have the correct form regardless of whether the state variables are
      // measured in the rotating frame or not; we handled that in the construction
      // of the omega_times_dt_matrix. It also has the correct form if we have disabled
      // the Coriolis force entirely; at that point it reduces to the identity matrix.

      Array2D<Real, 0, 2, 0, 2> omega_times_dt_matrix = {};
      fill_omega_times_dt_matrix(dt, omega, omega_times_dt_matrix);

      Real updated_momentum[3] = {};

      // updated_momentum = matmul(omega_times_dt_matrix, updated_momentum)

      for (int l = 0; l < 3; l++) {
        for (int m = 0; m < 3; m++) {
          updated_momentum[l] += omega_times_dt_matrix(l,m) * predicted_momentum[m];
        }
      }


      // Obtain the effective source term; remember that we're ultimately going
      // to multiply the source term by dt to get the update to the state.

      for (int n = 0; n < 3; n++) {
        mom_source_corr[n] = (updated_momentum[n] - unew(i,j,k,UMX+n)) / dt;
      }

    }

    // Correct momenta

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

    // Correct energy

    Real energy_source_corr;

    if (rotation_source_mode == 1) {

      // If rotation_source_mode == 1, then we calculated energy_source_corr before updating the velocities.

      energy_source_corr = 0.5_rt * (energy_source_new - energy_source_old);

    } else if (rotation_source_mode == 2) {

      // For this source type, we first update the momenta
      // before we calculate the energy source term.

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

      // Instead of calculating the energy source term explicitly,
      // we simply update the kinetic energy.

      Real kinetic_energy_updated = 0.5_rt * (snew[UMX] * snew[UMX] + snew[UMY] * snew[UMY] + snew[UMZ] * snew[UMZ]) * rho_new_inv;
      energy_source_corr = kinetic_energy_updated - kinetic_energy_old;

    } else if (rotation_source_mode == 4) {

      // Conservative energy update

      // First, subtract the predictor step we applied earlier.

      energy_source_corr = - energy_source_old;

      // See corrgsrc for an explanation of this algorithm. We can implement this identically
      // to the conservative gravity, swapping out the gravitational acceleration for the
      // rotational acceleration. We only need to calculate the rotational acceleration on
      // edges, which is straightforward to do. The Coriolis term does not play a role in
      // the energy update (it does no work), so we turn it off and pass in a temporary
      // velocity array which will be ignored.

      Real edge_spin_source[3][2];

      // Loop over directions and edges

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

    // Add to the outgoing source array.

    for (int n = 0; n < NUM_SOURCE; n++) {
      source(i,j,k,n) += src[n];
    }

  });

}
