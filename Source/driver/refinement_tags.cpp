#ifndef WIN32
#include <unistd.h>
#endif

#include <iomanip>

#include <algorithm>
#include <cstdio>
#include <vector>
#include <iostream>
#include <string>
#include <ctime>
#include <memory>
#include <numbers>
#include <format>

#include <AMReX_Utility.H>
#include <AMReX_CONSTANTS.H>
#include <HydroSim.H>
#include <global.H>
#include <runtime_parameters.H>
#include <AMReX_VisMF.H>
#include <AMReX_TagBox.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_ParmParse.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef AMREX_PARTICLES
#include <AMReX_Particles.H>
#endif

#ifdef GRAVITY
#include <Gravity.H>
#endif

#ifdef DIFFUSION
#include <Diffusion.H>
#endif

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <extern_parameters.H>
#include <prob_parameters.H>

#include <user_tagging.H>

#include <ambient.H>
#include <hydrosim_limits.H>

#include <solver_constants.H>

using namespace amrex;

void
HydroSim::errorEst (TagBoxArray& tags,
                  int           ,
                  int           ,
                  Real         time,
                  int           ,
                  int           )
{
    BL_PROFILE("HydroSim::errorEst()");

    Real tag_time = time;

    if (post_step_regrid) {
      tag_time = get_state_data(State_Type).curTime();
    }

    for (const auto & etag : tag_criteria) {
        std::unique_ptr<MultiFab> mf;
        if (! etag.Field().empty()) {
            mf = derive(etag.Field(), tag_time, etag.NGrow());
        }
        etag(tags, mf.get(), TagBox::CLEAR, TagBox::SET, tag_time, level, geom);
    }

    apply_user_tags(tags, tag_time);

    apply_tag_limits(tags, tag_time);

}

void
HydroSim::apply_tag_limits(TagBoxArray& tags, [[maybe_unused]] Real time)
{
    BL_PROFILE("HydroSim::apply_tag_limits()");

#ifdef GRAVITY
    if (gravity::gravity_model_name == "PoissonGrav") {

        int lev = level;

        int n_error_buf[3] = {0};
        int ref_ratio[3] = {0};
        int domlo[3] = {0};
        int domhi[3] = {0};
        int physbc_lo[3] = {-1};
        int physbc_hi[3] = {-1};
        int blocking_factor[3] = {0};
        for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
            n_error_buf[dim] = parent->nErrorBuf(lev, dim);
            ref_ratio[dim] = parent->refRatio(lev)[dim];
            domlo[dim] = geom.Domain().loVect()[dim];
            domhi[dim] = geom.Domain().hiVect()[dim];
            physbc_lo[dim] = phys_bc.lo()[dim];
            physbc_hi[dim] = phys_bc.hi()[dim];
            blocking_factor[dim] = parent->blockingFactor(lev)[dim];
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();

            auto tag = tags[mfi].array();

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                bool domain_edge_flags[3] = {false};

                const int idx[3] = {i, j, k};

                for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {

                    int boundary_width = n_error_buf[dim] + blocking_factor[dim] / ref_ratio[dim];

                    if ((physbc_lo[dim] != amrex::PhysBCType::symmetry &&
                         physbc_lo[dim] != amrex::PhysBCType::interior) &&
                        (idx[dim] <= domlo[dim] + boundary_width)) {
                        domain_edge_flags[dim] = true;
                    }

                    if ((physbc_hi[dim] != amrex::PhysBCType::symmetry &&
                         physbc_hi[dim] != amrex::PhysBCType::interior) &&
                        (idx[dim] >= domhi[dim] - boundary_width)) {
                        domain_edge_flags[dim] = true;
                    }
                }

                if (domain_edge_flags[0] || domain_edge_flags[1] || domain_edge_flags[2]) {

                    tag(i,j,k) = TagBox::CLEAR;

                }
            });
        }

    }
#endif

    auto geomdata = geom.data();

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto tag = tags[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real* problo = geomdata.ProbLo();
            const Real* probhi = geomdata.ProbHi();
            const Real* dx = geomdata.CellSize();

            GpuArray<Real, 3> loc = {0.0};

            loc[0] = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0] - problem::center[0];
#if AMREX_SPACEDIM >= 2
            loc[1] = problo[1] + (static_cast<Real>(j) + 0.5_rt) * dx[1] - problem::center[1];
#endif
#if AMREX_SPACEDIM == 3
            loc[2] = problo[2] + (static_cast<Real>(k) + 0.5_rt) * dx[2] - problem::center[2];
#endif

            Real r = distance(geomdata, loc);

            Real extent_lo = 0.0;
            Real extent_hi = 0.0;

            for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
                extent_lo = amrex::max(extent_lo, std::abs(problo[dim] - problem::center[dim]));
                extent_hi = amrex::max(extent_hi, std::abs(probhi[dim] - problem::center[dim]));
            }

            if (r > hydrosim::tagging_radius_cap * amrex::max(extent_lo, extent_hi)) {
                tag(i,j,k) = TagBox::CLEAR;
            }
        });
    }
}

void
HydroSim::apply_user_tags (TagBoxArray& tags, Real time)
{

    amrex::ignore_unused(time);

    BL_PROFILE("HydroSim::apply_user_tags()");

    MultiFab& S_new = get_new_data(State_Type);

    int lev = level;

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
        for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            TagBox& tagfab = tags[mfi];

            auto tag_view = tagfab.array();
            const auto state_arr = S_new[mfi].array();
            const GeometryData& geomdata = geom.data();

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                user_tagging(i, j, k, tag_view, state_arr, lev, geomdata);
            });
        }
    }

}
