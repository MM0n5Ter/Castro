#include <AMReX_REAL.H>
#include <AMReX.H>

#include <RadFieldDerive.H>
#include <HydroSim.H>

#include <Radiation.H>

using namespace amrex;

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

