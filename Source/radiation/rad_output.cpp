#include <Radiation.H>
#include <flux_limiter.H>
#include <AMReX_REAL.H>

#include <AMReX.H>

#include <RadFieldDerive.H>

#include <HydroSim.H>
#include <algorithm>
#include <iomanip>

#include <cstdio>

#include <iostream>

#include <numbers>

using namespace amrex;
using std::cout;
using std::cerr;
using std::endl;

void Radiation::store_flux_plotvar(int level, const MultiFab& Snew,
                                     const FaceMultiFabs& lambda,
                                     const MultiFab& Er, const MultiFab& Fr, int flux_comp_base,
                                     const Real frame_sign)
{
    int icomp_flux_index = -1;
    if (radiation::plot_moving_flux) {
        icomp_flux_index = icomp_comoving_flux;
    }
    else if (radiation::plot_lab_rad_flux) {
        icomp_flux_index = icomp_lab_flux;
    }

    int num_limiter_comps = lambda[0].nComp();

    GpuArray<Real, NGROUPS> log_group_width = {0.0};

    if (NGROUPS > 1) {
        std::copy_n(log_group_widths.begin(), NGROUPS, log_group_width.begin());
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(*plot_fields[level],true); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto newtime_state_arr = Snew[mfi].array();
        auto rad_eden_arr = Er[mfi].array();
        auto rflux_arr = Fr[mfi].array();
        auto plot_flux_arr = (*plot_fields[level])[mfi].array();

        auto limiter_x = lambda[0][mfi].array();
#if AMREX_SPACEDIM >= 2
        auto limiter_y = lambda[1][mfi].array();
#endif
#if AMREX_SPACEDIM == 3
        auto limiter_z = lambda[2][mfi].array();
#endif

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Array1D<Real, 0, NGROUPS-1> RadClosure = {0.0};

            for (int g = 0; g < NGROUPS; ++g) {
                int limiter_index = amrex::min(g, num_limiter_comps - 1);

#if AMREX_SPACEDIM == 1
                Real limiter_center = 0.5_rt * (limiter_x(i,j,k,limiter_index) + limiter_x(i+1,j,k,limiter_index));
#elif AMREX_SPACEDIM >= 2
                Real limiter_center = 0.25_rt * (limiter_x(i,j,k,limiter_index) + limiter_x(i+1,j,k,limiter_index) +
                                        limiter_y(i,j,k,limiter_index) + limiter_y(i,j+1,k,limiter_index));
#else
                Real limiter_center = (1.0_rt / 6.0_rt) * (limiter_x(i,j,k,limiter_index) + limiter_x(i+1,j,k,limiter_index) +
                                                  limiter_y(i,j,k,limiter_index) + limiter_y(i,j+1,k,limiter_index) +
                                                  limiter_z(i,j,k,limiter_index) + limiter_z(i,j,k+1,limiter_index));
#endif

                RadClosure(g) = Eddington_factor(limiter_center);
            }

            int in_flux_x_comp = flux_comp_base;
            int out_flux_x_comp = icomp_flux_index;

#if AMREX_SPACEDIM >= 2
            int in_flux_y_comp = flux_comp_base + NGROUPS;
            int out_flux_y_comp = icomp_flux_index + NGROUPS;
#endif

#if AMREX_SPACEDIM == 3
            int in_flux_z_comp = flux_comp_base + 2 * NGROUPS;
            int out_flux_z_comp = icomp_flux_index + 2 * NGROUPS;
#endif

            Real rhoInv = 1.0_rt / newtime_state_arr(i,j,k,URHO);

            Real vx = newtime_state_arr(i,j,k,UMX) * rhoInv * frame_sign;
#if AMREX_SPACEDIM >= 2
            Real vy = newtime_state_arr(i,j,k,UMY) * rhoInv * frame_sign;
#endif
#if AMREX_SPACEDIM == 3
            Real vz = newtime_state_arr(i,j,k,UMZ) * rhoInv * frame_sign;
#endif

            Array1D<Real, 0, NGROUPS-1> vel_dot_pressure_x;
#if AMREX_SPACEDIM >= 2
            Array1D<Real, 0, NGROUPS-1> vel_dot_pressure_y;
#endif
#if AMREX_SPACEDIM == 3
            Array1D<Real, 0, NGROUPS-1> vel_dot_pressure_z;
#endif

            for (int g = 0; g < NGROUPS; ++g) {
                Real f1 = (1.0_rt - RadClosure(g));
                Real f2 = (3.0_rt * RadClosure(g) - 1.0_rt);
                Real foo = 1.0_rt / std::sqrt(rflux_arr(i,j,k,in_flux_x_comp+g) * rflux_arr(i,j,k,in_flux_x_comp+g) +
#if AMREX_SPACEDIM >= 2
                                              rflux_arr(i,j,k,in_flux_y_comp+g) * rflux_arr(i,j,k,in_flux_y_comp+g) +
#endif
#if AMREX_SPACEDIM == 3
                                              rflux_arr(i,j,k,in_flux_z_comp+g) * rflux_arr(i,j,k,in_flux_z_comp+g) +
#endif
                                              1.e-50_rt);

                Real nx = rflux_arr(i,j,k,in_flux_x_comp+g) * foo;
#if AMREX_SPACEDIM >= 2
                Real ny = rflux_arr(i,j,k,in_flux_y_comp+g) * foo;
#endif
#if AMREX_SPACEDIM == 3
                Real nz = rflux_arr(i,j,k,in_flux_z_comp+g) * foo;
#endif

                Real vel_dot_normal = vx * nx;
#if AMREX_SPACEDIM >= 2
                vel_dot_normal += vy * ny;
#endif
#if AMREX_SPACEDIM == 3
                vel_dot_normal += vz * vz;
#endif

                vel_dot_pressure_x(g) = 0.5_rt * rad_eden_arr(i,j,k,g) * (f1 * vx + f2 * vel_dot_normal * nx);
#if AMREX_SPACEDIM >= 2
                vel_dot_pressure_y(g) = 0.5_rt * rad_eden_arr(i,j,k,g) * (f1 * vy + f2 * vel_dot_normal * ny);
#endif
#if AMREX_SPACEDIM == 3
                vel_dot_pressure_z(g) = 0.5_rt * rad_eden_arr(i,j,k,g) * (f1 * vz + f2 * vel_dot_normal * nz);
#endif

                plot_flux_arr(i,j,k,out_flux_x_comp+g) = rflux_arr(i,j,k,in_flux_x_comp+g) + vx * rad_eden_arr(i,j,k,g) + vel_dot_pressure_x(g);
#if AMREX_SPACEDIM >= 2
                plot_flux_arr(i,j,k,out_flux_y_comp+g) = rflux_arr(i,j,k,in_flux_y_comp+g) + vy * rad_eden_arr(i,j,k,g) + vel_dot_pressure_y(g);
#endif
#if AMREX_SPACEDIM == 3
                plot_flux_arr(i,j,k,out_flux_z_comp+g) = rflux_arr(i,j,k,in_flux_z_comp+g) + vz * rad_eden_arr(i,j,k,g) + vel_dot_pressure_z(g);
#endif
            }

            if (NGROUPS > 1) {
                Array1D<Real, -1, NGROUPS> vdotp_per_logbin_x;
#if AMREX_SPACEDIM >= 2
                Array1D<Real, -1, NGROUPS> vdotp_per_logbin_y;
#endif
#if AMREX_SPACEDIM == 3
                Array1D<Real, -1, NGROUPS> vdotp_per_logbin_z;
#endif

                for (int g = 0; g < NGROUPS; ++g) {
                    vdotp_per_logbin_x(g) = vel_dot_pressure_x(g) / log_group_width[g];
#if AMREX_SPACEDIM >= 2
                    vdotp_per_logbin_y(g) = vel_dot_pressure_y(g) / log_group_width[g];
#endif
#if AMREX_SPACEDIM == 3
                    vdotp_per_logbin_z(g) = vel_dot_pressure_z(g) / log_group_width[g];
#endif
                }

                vdotp_per_logbin_x(-1) = -vdotp_per_logbin_x(0);
                vdotp_per_logbin_x(NGROUPS) = -vdotp_per_logbin_x(NGROUPS-1);

#if AMREX_SPACEDIM >= 2
                vdotp_per_logbin_y(-1) = -vdotp_per_logbin_y(0);
                vdotp_per_logbin_y(NGROUPS) = -vdotp_per_logbin_y(NGROUPS-1);
#endif

#if AMREX_SPACEDIM == 3
                vdotp_per_logbin_z(-1) = -vdotp_per_logbin_z(0);
                vdotp_per_logbin_z(NGROUPS) = -vdotp_per_logbin_z(NGROUPS-1);
#endif

                for (int g = 0; g < NGROUPS; ++g) {
                    plot_flux_arr(i,j,k,out_flux_x_comp+g) -= 0.5_rt * (vdotp_per_logbin_x(g+1)-vdotp_per_logbin_x(g-1));
#if AMREX_SPACEDIM >= 2
                    plot_flux_arr(i,j,k,out_flux_y_comp+g) -= 0.5_rt * (vdotp_per_logbin_y(g+1)-vdotp_per_logbin_y(g-1));
#endif
#if AMREX_SPACEDIM == 3
                    plot_flux_arr(i,j,k,out_flux_z_comp+g) -= 0.5_rt * (vdotp_per_logbin_z(g+1)-vdotp_per_logbin_z(g-1));
#endif
                }
            }
        });
    }
}

void Radiation::store_lab_energy_plotvar(int level, const MultiFab& Snew,
                                       const MultiFab& comoving_energy, const MultiFab& F, int flux_comp_base)
{
    int num_flux_comps = F.nComp();

    int ifx = flux_comp_base;

#if AMREX_SPACEDIM >= 2
    int ify = flux_comp_base + NGROUPS;
#else
    int ify = 0;
#endif

#if AMREX_SPACEDIM == 3
    int ifz = flux_comp_base + 2 * NGROUPS;
#else
    int ifz = 0;
#endif

    int ier = comp_lab_energy;

    Real c2 = 1.e0_rt / (Radiation::clight * Radiation::clight);

    GpuArray<Real, NGROUPS> log_group_width = {0.0};

    if (NGROUPS > 1) {
        for (int i = 0; i < NGROUPS; ++i) {
            log_group_width[i] = log_group_widths[i];
        }
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(*plot_fields[level],true); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto newtime_state_arr = Snew[mfi].array();
        auto comoving_energy_arr = comoving_energy[mfi].array();
        auto radflux_arr = F[mfi].array();
        auto lab_energy_arr = (*plot_fields[level])[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Real rhoInv = 1.0_rt / newtime_state_arr(i,j,k,URHO);

            Real vel_x_scaled = newtime_state_arr(i,j,k,UMX) * rhoInv * c2;

#if AMREX_SPACEDIM >= 2
            Real vel_y_scaled = newtime_state_arr(i,j,k,UMY) * rhoInv * c2;
#else
            Real vel_y_scaled = 0.0_rt;
#endif

#if AMREX_SPACEDIM == 3
            Real vel_z_scaled = newtime_state_arr(i,j,k,UMZ) * rhoInv * c2;
#else
            Real vel_z_scaled = 0.0_rt;
#endif

            for (int g = 0; g < NGROUPS; ++g) {
                lab_energy_arr(i,j,k,g+ier) = comoving_energy_arr(i,j,k,g) + 2.0_rt * (vel_x_scaled * radflux_arr(i,j,k,ifx+g) +
                                                                      vel_y_scaled * radflux_arr(i,j,k,ify+g) +
                                                                      vel_z_scaled * radflux_arr(i,j,k,ifz+g));
            }

            if (NGROUPS > 1) {
                Array1D<Real, -1, NGROUPS> flux_per_logbin_x, flux_per_logbin_y, flux_per_logbin_z;

                for (int g = 0; g < NGROUPS; ++g) {
                    flux_per_logbin_x(g) = radflux_arr(i,j,k,ifx+g) / log_group_width[g];
                    flux_per_logbin_y(g) = radflux_arr(i,j,k,ify+g) / log_group_width[g];
                    flux_per_logbin_z(g) = radflux_arr(i,j,k,ifz+g) / log_group_width[g];
                }

                flux_per_logbin_x(-1) = -flux_per_logbin_x(0);
                flux_per_logbin_y(-1) = -flux_per_logbin_y(0);
                flux_per_logbin_z(-1) = -flux_per_logbin_z(0);
                flux_per_logbin_x(NGROUPS) = -flux_per_logbin_x(NGROUPS-1);
                flux_per_logbin_y(NGROUPS) = -flux_per_logbin_y(NGROUPS-1);
                flux_per_logbin_z(NGROUPS) = -flux_per_logbin_z(NGROUPS-1);

                for (int g = 0; g < NGROUPS; ++g) {
                    lab_energy_arr(i,j,k,g+ier) -= vel_x_scaled * 0.5_rt * (flux_per_logbin_x(g+1) - flux_per_logbin_x(g-1)) +
                                             vel_y_scaled * 0.5_rt * (flux_per_logbin_y(g+1) - flux_per_logbin_y(g-1)) +
                                             vel_z_scaled * 0.5_rt * (flux_per_logbin_z(g+1) - flux_per_logbin_z(g-1));
                }
            }
        });
    }
}

void Radiation::store_limiter_plotvar(int level, const FaceMultiFabs& lambda)
{
    int num_limiter_comps = lambda[0].nComp();

    if (!plot_fields[level]) {
        amrex::Abort("store_limiter_plotvar: plot fields not allocated");
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(*plot_fields[level],true); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto limiter_x = lambda[0][mfi].array();
#if AMREX_SPACEDIM >= 2
        auto limiter_y = lambda[1][mfi].array();
#endif
#if AMREX_SPACEDIM == 3
        auto limiter_z = lambda[2][mfi].array();
#endif

        auto lamc = (*plot_fields[level])[mfi].array();

        int dcomp = comp_limiter;

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            for (int n = 0; n < num_limiter_comps; ++n) {
#if AMREX_SPACEDIM == 1
                lamc(i,j,k,dcomp+n) = (limiter_x(i,j,k,n) + limiter_x(i+1,j,k,n)) * 0.5_rt;
#elif AMREX_SPACEDIM == 2
                lamc(i,j,k,dcomp+n) = (limiter_x(i,j,k,n) + limiter_x(i+1,j,k,n) +
                                       limiter_y(i,j,k,n) + limiter_y(i,j+1,k,n)) * 0.25_rt;
#else
                lamc(i,j,k,dcomp+n) = (limiter_x(i,j,k,n) + limiter_x(i+1,j,k,n) +
                                       limiter_y(i,j,k,n) + limiter_y(i,j+1,k,n) +
                                       limiter_z(i,j,k,n) + limiter_z(i,j,k+1,n)) * (1.0_rt / 6.0_rt);
#endif
            }
        });
    }
}

void derive_total_radiative_energy(const Box& bx, FArrayBox& derfab, int  , int  ,
                 const FArrayBox& datfab, const Geometry&  ,
                 Real  , const int*  , int  )
{

    auto const dat = datfab.array();
    auto const der = derfab.array();

    Real radiativeToEnergy = Radiation::radiativeToEnergy;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {

        der(i,j,k,0) = 0.0_rt;

        for (int g = 0; g < NGROUPS; g++) {
            der(i,j,k,0) += dat(i,j,k,g) * radiativeToEnergy;
        }
    });
}

void
HydroSim::print_energy_budget()
{
  int finest_level = parent->finestLevel();

  int v = verbose;
  v = (!((do_radiative_transfer && radiation->verbose > v)) ? (v) : (radiation->verbose));

  if (do_radiative_transfer && (v > 2 || (level == 0 && v > 0))) {
    Real prev_time = state[State_Type].prevTime();
    Real dt = parent->dtLevel(level);

    Real m = 0.0, s = 0.0, r = 0.0, rr = 0.0, rry = 0.0;

    for (int lev = 0; lev <= finest_level; lev++) {
      m += getLevel(lev).volWeightedTotal("density", prev_time + dt);
      s += getLevel(lev).volWeightedTotal("rho_E",   prev_time + dt);
      if (!(!Radiation::groups_enabled)) {
          std::string rad_label;
          for (int rad_group_idx = 0; rad_group_idx < Radiation::nGroups; rad_group_idx++) {
              rad_label = "rad" + std::to_string(rad_group_idx);
              r += getLevel(lev).volWeightedTotal(rad_label, prev_time + dt);
          }
      } else {
          r += getLevel(lev).volWeightedTotal("rad", prev_time + dt);
      }
      if (lev < finest_level) {
          FluxRegister* fine_correction_register = radiation->conservativeRegister(lev + 1);
          if (fine_correction_register) {
              for (int k = 0; k < Radiation::nGroups; k++) {
                  Real tmp = -fine_correction_register->SumReg(k) * parent->dtLevel(lev);
                  rr  += tmp;
              }
          }
      }
    }


#if (AMREX_SPACEDIM == 1)
    if (Geom().IsSPHERICAL()) {
      rr  *= (4.0 * std::numbers::pi);
      rry *= (4.0 * std::numbers::pi);
    }
#elif (AMREX_SPACEDIM == 2)
    if (Geom().IsRZ()) {
      rr  *= (2.0 * std::numbers::pi);
      rry *= (2.0 * std::numbers::pi);
    }
#endif

    if (ParallelDescriptor::IOProcessor()) {
      int oldprec = cout.precision(20);
      cout << "Integrated  Fluid   Mass  is " << m << '\n';
      cout << "Integrated  Fluid  Energy is " << s << '\n';
      cout << "Integrated Radiant Energy is " << r << '\n';
      cout << "     Flux Register Energy is " << rr << '\n';
      cout << "Integrated  Total  Energy is " << s + r + rr << endl;
      cout.precision(oldprec);
    }
  }
  else if (v > 2 || (level == 0 && v > 0)) {
    Real prev_time = state[State_Type].prevTime();
    Real dt = parent->dtLevel(level);

    Real m = 0.0, s = 0.0;
    for (int lev = 0; lev <= finest_level; lev++) {
      m += getLevel(lev).volWeightedTotal("density", prev_time + dt);
      s += getLevel(lev).volWeightedTotal("rho_E",   prev_time + dt);
    }
    if (ParallelDescriptor::IOProcessor()) {
      int oldprec = cout.precision(20);
      cout << "Integrated  Fluid   Mass  is " << m << '\n';
      cout << "Integrated  Fluid  Energy is " << s << '\n';
      cout.precision(oldprec);
    }
  }
}
