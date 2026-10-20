#ifndef PARAM_LOAD_H
#define PARAM_LOAD_H

#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <hydrosim_limits.H>

#ifdef DIFFUSION
#include <diffusion_params.H>
#endif
#ifdef GRAVITY
#include <gravity_params.H>
#endif
#ifdef RADIATION
#include <radiation_solve_params.H>
#include <radiation_params.H>
#endif

AMREX_INLINE
void
load_compiled_params()
{

    {
        amrex::ParmParse pp("hydrosim");
#include <hydrosim_queries.H>
    }

#ifdef AMREX_PARTICLES
    {
        amrex::ParmParse pp("particles");
#include <particles_queries.H>
    }
#endif

#ifdef DIFFUSION
    {
        amrex::ParmParse pp("diffusion");
#include <diffusion_queries.H>
    }
#endif

#ifdef GRAVITY
    {
        amrex::ParmParse pp("gravity");
#include <gravity_queries.H>
    }
#endif

#ifdef RADIATION
    {
        amrex::ParmParse pp("radiation");
#include <radiation_queries.H>
    }
#endif

}


AMREX_INLINE
void
check_runparams()
{

    amrex::Vector<std::string> valid_namespaces = {"hydrosim", "problem"};
#ifdef AMREX_PARTICLES
    valid_namespaces.push_back("particles");
#endif
#ifdef DIFFUSION
    valid_namespaces.push_back("diffusion");
#endif
#ifdef GRAVITY
    valid_namespaces.push_back("gravity");
#endif
#ifdef RADIATION
    valid_namespaces.push_back("radiation_solve");
#endif


    valid_namespaces.push_back("eos");
#ifdef REACTIONS
    valid_namespaces.push_back("network");
    valid_namespaces.push_back("integrator");
#endif
#ifdef RADIATION
    valid_namespaces.push_back("opacity");
#endif
#ifdef DIFFUSION
    valid_namespaces.push_back("conductivity");
#endif

    for (const auto& nm: valid_namespaces)
    {
        if (amrex::ParmParse::hasUnusedInputs(nm)) {
            amrex::Print() << Font::Bold << FGColor::Yellow
                           << "[WARNING] the following " + nm + ".* parameters are ignored\n";
            auto unused = amrex::ParmParse::getUnusedInputs(nm);
            for (const auto& p: unused) {
                amrex::Print() << p << "\n";
            }
            amrex::Print() << ResetDisplay << std::endl;
            if (hydrosim::stop_on_bad_params) {
                amrex::Error("Error: invalid parameters");
            }
        }

    }

}

#endif
