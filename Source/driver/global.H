#ifndef GLOBAL_PTRS_H
#define GLOBAL_PTRS_H

#include <AMReX_Amr.H>
#ifdef RADIATION
#include <Radiation.H>
#endif

namespace global
{
    AMREX_INLINE Amr* amr_instance;
#ifdef RADIATION
    AMREX_INLINE Radiation* radiation_instance;
#endif
}

#endif
