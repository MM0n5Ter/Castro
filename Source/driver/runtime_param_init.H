#ifndef PARAM_LOAD_H
#define PARAM_LOAD_H

#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <hydrosim_limits.H>

#ifdef GRAVITY
#include <gravity_params.H>
#endif

AMREX_INLINE
void
load_compiled_params()
{

    {
        amrex::ParmParse pp("hydrosim");
#include <hydrosim_queries.H>
    }



#ifdef GRAVITY
    {
        amrex::ParmParse pp("gravity");
#include <gravity_queries.H>
    }
#endif


}


AMREX_INLINE
void
check_runparams()
{

    amrex::Vector<std::string> valid_namespaces = {"hydrosim", "problem"};
#ifdef GRAVITY
    valid_namespaces.push_back("gravity");
#endif


    valid_namespaces.push_back("eos");

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
