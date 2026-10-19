
#include <HydroSim.H>

using std::string;

#ifdef RADIATION
#include <Radiation.H>

using namespace amrex;

void
HydroSim::close_radiation_step (MultiFab& S_new, int iteration, int ncycle)
{
    if (do_radiative_transfer) {

        if (Radiation::hydro_only) {
          HydroSim::computeTemp(S_new, state[State_Type].curTime(), S_new.nGrow());
          return;
        }

        HydroSim::computeTemp(S_new, state[State_Type].curTime(), S_new.nGrow());

        if (Radiation::filter_primitive_interval > 0 && Radiation::filter_prim_temperature>0
            && iteration==ncycle) {
            int nstep = parent->levelSteps(0);
            if (nstep % Radiation::filter_primitive_interval == 0) {
                radiation->filter_primitives(level, S_new);
            }
        }

        if (Radiation::SolverKind == Radiation::MultiGroupFLDSolver) {
            if (! Radiation::rad_hydro_coupled) {
                MultiFab& rad_energy_old = get_old_data(Radiation_Type);
                MultiFab& rad_energy_new = get_new_data(Radiation_Type);
                MultiFab::Copy(rad_energy_new, rad_energy_old, 0, 0, rad_energy_old.nComp(), 0);
            }
            radiation->MultigroupDiffusion_implicit_update(level, iteration, ncycle);
        }
        else if (Radiation::SolverKind == Radiation::SingleGroupFLDSolver) {
            if (! Radiation::rad_hydro_coupled) {
                MultiFab& rad_energy_old = get_old_data(Radiation_Type);
                MultiFab& rad_energy_new = get_new_data(Radiation_Type);
                MultiFab::Copy(rad_energy_new, rad_energy_old, 0, 0, rad_energy_old.nComp(), 0);
            }
            radiation->advance_single_group(level, iteration, ncycle);
        }
        else {
            MultiFab& rad_energy_old = get_old_data(Radiation_Type);
            MultiFab& rad_energy_new = get_new_data(Radiation_Type);
            MultiFab::Copy(rad_energy_new, rad_energy_old, 0, 0, rad_energy_old.nComp(), 0);
            radiation->advance_single_group(level, iteration, ncycle);
        }

    }
}
#endif

