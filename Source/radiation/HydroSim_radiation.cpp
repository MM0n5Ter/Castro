
#include <HydroSim.H>

using std::string;

#ifdef RADIATION
#include <Radiation.H>

using namespace amrex;

void
HydroSim::finish_radiation_step (MultiFab& S_new, int iteration, int ncycle)
{
    if (do_radiation) {

        if (Radiation::pure_hydro) {
          HydroSim::computeTemp(S_new, state[State_Type].curTime(), S_new.nGrow());
          return;
        }

        HydroSim::computeTemp(S_new, state[State_Type].curTime(), S_new.nGrow());

        if (Radiation::filter_prim_int > 0 && Radiation::filter_prim_T>0
            && iteration==ncycle) {
            int nstep = parent->levelSteps(0);
            if (nstep % Radiation::filter_prim_int == 0) {
                radiation->filter_prim(level, S_new);
            }
        }

        if (Radiation::SolverType == Radiation::MGFLDSolver) {
            if (! Radiation::rad_hydro_combined) {
                MultiFab& Er_old = get_old_data(Rad_Type);
                MultiFab& Er_new = get_new_data(Rad_Type);
                MultiFab::Copy(Er_new, Er_old, 0, 0, Er_old.nComp(), 0);
            }
            radiation->MGFLD_implicit_update(level, iteration, ncycle);
        }
        else if (!(Radiation::SolverType == Radiation::SGFLDSolver)) {
            MultiFab& Er_old = get_old_data(Rad_Type);
            MultiFab& Er_new = get_new_data(Rad_Type);
            MultiFab::Copy(Er_new, Er_old, 0, 0, Er_old.nComp(), 0);
            radiation->single_group_update(level, iteration, ncycle);
        } else {
            if (! Radiation::rad_hydro_combined) {
                MultiFab& Er_old = get_old_data(Rad_Type);
                MultiFab& Er_new = get_new_data(Rad_Type);
                MultiFab::Copy(Er_new, Er_old, 0, 0, Er_old.nComp(), 0);
            }
            radiation->single_group_update(level, iteration, ncycle);
        }

    }
}
#endif

