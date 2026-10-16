#ifndef user_problem_setup_H
#define user_problem_setup_H

AMREX_INLINE
void user_problem_setup()
{
    const Geometry& dgeom = DefaultGeometry();

    const Real* problo = dgeom.ProbLo();
    const Real* probhi = dgeom.ProbHi();

    for (int n=0; n < AMREX_SPACEDIM; ++n)
    {
        problem::center[n] = 0.5_rt * (problo[n] + probhi[n]);
    }



}

#endif
