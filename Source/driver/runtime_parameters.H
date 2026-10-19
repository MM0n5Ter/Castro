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

///
/// Initialize all of the runtime parameters defined in _cpp_parameters
/// regardless of the namespace
///
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


///
/// Check to ensure that any runtime parameters set in inputs or on
/// the command line are valid -- everything should have been accessed
/// by now.
///
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
    //valid_namespaces.push_back("radiation");  // this doesn't yet work because RadiativeBoundary isn't setup yet
    valid_namespaces.push_back("radiation_solve");
#endif

    // now Microphysics

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
        // "hydrosim"
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
