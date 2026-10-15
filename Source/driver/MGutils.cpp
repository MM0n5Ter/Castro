#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Amr.H>

#include <MGutils.H>
#include <hydrosim_limits.H>

using namespace amrex;

void
apply_metric(const Box& bx,
             Array4<Real> const& rhs, const Box& rbx,
             const GpuArray<Array4<Real>, AMREX_SPACEDIM>& edge_coefs,
             const GpuArray<Box, AMREX_SPACEDIM>& edge_boxes,
             const GpuArray<Real, AMREX_SPACEDIM>& problo,
             const GpuArray<Real, AMREX_SPACEDIM>& dx,
             const int coord_type)
{
    // r-z
    if (coord_type == 1) {

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {

            IntVect idx(AMREX_D_DECL(i, j, k));

            // at centers
            if (rbx.contains(idx)) {
                Real r = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
                rhs(i,j,k) *= r;
            }

            // On x-edges
            if (edge_boxes[0].contains(idx)) {
                Real r = problo[0] + static_cast<Real>(i) * dx[0];
                edge_coefs[0](i,j,k) *= r;
            }

#if AMREX_SPACEDIM >= 2
            // On y-edges
            if (edge_boxes[1].contains(idx)) {
                Real r = problo[0] + (static_cast<Real>(i) + 0.5_rt) * dx[0];
                edge_coefs[1](i,j,k) *= r;
            }
#endif

        });

    } else {
        amrex::Print() << "Bogus coord_type in apply_metric " << coord_type << std::endl;
        amrex::Error("Error:: MGutils.cpp :: hs_apply_metric");
    }
}

LevelSolverBase::LevelSolverBase(Amr* Parent, BCRec* _phys_bc)
  :
    parent(Parent),
    LevelData(MAX_LEV),
    volume(MAX_LEV),
    area(MAX_LEV),
    phys_bc(_phys_bc)
{
    AMREX_ALWAYS_ASSERT(parent->maxLevel() < MAX_LEV);
}

void
LevelSolverBase::store_level (int level, AmrLevel* level_data, MultiFab& _volume, MultiFab* _area)
{
    LevelData[level] = level_data;

    volume[level] = &_volume;

    area[level] = _area;
}
