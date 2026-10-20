#ifndef STATE_BC_TABLES_H
#define STATE_BC_TABLES_H

#include <array>
#include <AMReX_StateDescriptor.H>
#include <AMReX_BCRec.H>

using BndryFunc = StateDescriptor::BndryFunc;

namespace {
    constexpr std::array<int, 6> scalar_bc{
        amrex::BCType::int_dir,
        amrex::BCType::ext_dir,
        amrex::BCType::foextrap,
        amrex::BCType::reflect_even,
        amrex::BCType::reflect_even,
        amrex::BCType::reflect_even
    };
    constexpr std::array<int, 6> norm_vel_bc{
        amrex::BCType::int_dir,
        amrex::BCType::ext_dir,
        amrex::BCType::foextrap,
        amrex::BCType::reflect_odd,
        amrex::BCType::reflect_odd,
        amrex::BCType::reflect_odd
    };
    constexpr std::array<int, 6> tang_vel_bc{
        amrex::BCType::int_dir,
        amrex::BCType::ext_dir,
        amrex::BCType::foextrap,
        amrex::BCType::reflect_even,
        amrex::BCType::reflect_even,
        amrex::BCType::reflect_even
    };

#ifdef MHD
    constexpr std::array<int, 6> bfield_bc{
        amrex::BCType::int_dir,
        amrex::BCType::ext_dir,
        amrex::BCType::foextrap,
        amrex::BCType::reflect_even,
        amrex::BCType::foextrap,
        amrex::BCType::hoextrap
    };
#endif

    Box the_same_box (const Box& b) { return b; }

    Box grow_box_by_one (const Box& b) { return amrex::grow(b,1); }

    void
    set_x_vel_bc(BCRec& bc, const BCRec& phys_bc)
    {
        const int* lo_bc = phys_bc.lo();
        const int* hi_bc = phys_bc.hi();
        bc.setLo(0,norm_vel_bc[lo_bc[0]]);
        bc.setHi(0,norm_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
        bc.setLo(1,tang_vel_bc[lo_bc[1]]);
        bc.setHi(1,tang_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
        bc.setLo(2,tang_vel_bc[lo_bc[2]]);
        bc.setHi(2,tang_vel_bc[hi_bc[2]]);
#endif
    }

    void
    set_y_vel_bc(BCRec& bc, const BCRec& phys_bc)
    {
        const int* lo_bc = phys_bc.lo();
        const int* hi_bc = phys_bc.hi();
        bc.setLo(0,tang_vel_bc[lo_bc[0]]);
        bc.setHi(0,tang_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
        bc.setLo(1,norm_vel_bc[lo_bc[1]]);
        bc.setHi(1,norm_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
        bc.setLo(2,tang_vel_bc[lo_bc[2]]);
        bc.setHi(2,tang_vel_bc[hi_bc[2]]);
#endif
    }

    void
    set_z_vel_bc(BCRec& bc, const BCRec& phys_bc)
    {
        const int* lo_bc = phys_bc.lo();
        const int* hi_bc = phys_bc.hi();
        bc.setLo(0,tang_vel_bc[lo_bc[0]]);
        bc.setHi(0,tang_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
        bc.setLo(1,tang_vel_bc[lo_bc[1]]);
        bc.setHi(1,tang_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
        bc.setLo(2,norm_vel_bc[lo_bc[2]]);
        bc.setHi(2,norm_vel_bc[hi_bc[2]]);
#endif
    }

    void
    set_scalar_bc (BCRec& bc, const BCRec& phys_bc)
    {
        const int* lo_bc = phys_bc.lo();
        const int* hi_bc = phys_bc.hi();
        for (int i = 0; i < AMREX_SPACEDIM; i++)
        {
            bc.setLo(i,scalar_bc[lo_bc[i]]);
            bc.setHi(i,scalar_bc[hi_bc[i]]);
        }
    }

#ifdef MHD
    void
    set_bfield_bc(BCRec& bc, const BCRec& phys_bc)
    {
        const int* lo_bc = phys_bc.lo();
        const int* hi_bc = phys_bc.hi();
        for (int i = 0; i < AMREX_SPACEDIM; i++)
        {
            bc.setLo(i, bfield_bc[lo_bc[i]]);
            bc.setHi(i, bfield_bc[hi_bc[i]]);
        }
    }
#endif

    void
    swap_inflow_bc (BCRec& bc)
    {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (bc.lo(dir) == amrex::BCType::ext_dir) {
                bc.setLo(dir, amrex::BCType::foextrap);
            }
            if (bc.hi(dir) == amrex::BCType::ext_dir) {
                bc.setHi(dir, amrex::BCType::foextrap);
            }
        }
    }
}

#endif
