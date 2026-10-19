#include <AMReX_REAL.H>

#include <Derive.H>
#include <HydroSim.H>
#include <Radiation.H>
#include <prob_parameters.H>
#include <extern_parameters.H>
#include <global.H>
#include <gravity_params.H>
#include <problem_util.H>

using namespace amrex;


void deranalytic(const Box& bx, FArrayBox& derfab, int dcomp, int /*ncomp*/,
                 const FArrayBox& datfab, const Geometry& geomdata,
                 Real time, const int* /*bcrec*/, int /*level*/)
{

  const auto dx = geomdata.CellSizeArray();
  const auto problo = geomdata.ProbLoArray();

  auto const dat = datfab.array();
  auto const der = derfab.array();

  GpuArray<Real, NGROUPS> freq_center = {0.0};
  GpuArray<Real, NGROUPS> freq_width = {0.0};

  for (int i = 0; i < NGROUPS; ++i) {
      freq_center[i] = global::radiation_instance->freq_center[i];
      freq_width[i] = global::radiation_instance->freq_width[i];
  }

  amrex::ParallelFor(bx,
  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
  {

    Real loc[3] = {0.0};

    loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];
#if AMREX_SPACEDIM >= 2
    loc[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
#endif
#if AMREX_SPACEDIM == 3
    loc[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
#endif

    Real r = std::sqrt(loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2]);

    // compute the analytic radiating sphere solution for each point
    for (int g = 0; g < NGROUPS; g++) {
      Real F = F_radsphere(r, time, freq_center[g]);
      Real E = planck(freq_center[g], problem::T_0) +
        (R_sphere / r) * (planck(freq_center[g], T_sphere) - planck(freq_center[g], problem::T_0)) * F;
      der(i,j,k,g) = E * freq_width[g];
    }

  });

}
