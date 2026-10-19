#ifndef user_diagnostics_H
#define user_diagnostics_H

AMREX_INLINE
void
HydroSim::user_diagnostics ()
{
    int finest_level = parent->finestLevel();
    Real time        = state[State_Type].curTime();

    Real T_max = -1.0e200;
    Real T_min =  1.0e200;
    Real grad_T_max = -1.0e200;
    Real rho_fuel_dot = 0.0;
    Real rho_fuel_initial = 1.0;

    Real flame_width = 0.0;
    Real flame_speed = 0.0;

    for (int lev = 0; lev <= finest_level; lev++)
    {
        HydroSim& hs_level = getLevel(lev);

        hs_level.flame_width_properties(time, T_max, T_min, grad_T_max);


        if (lev == 0)
            hs_level.flame_speed_properties(time, rho_fuel_dot);
    }

    ParallelDescriptor::ReduceRealMax(T_max);
    ParallelDescriptor::ReduceRealMin(T_min);
    ParallelDescriptor::ReduceRealMax(grad_T_max);

    flame_width = (T_max - T_min) / grad_T_max;

    ParallelDescriptor::ReduceRealSum(rho_fuel_dot);


    flame_speed = -rho_fuel_dot / rho_fuel_initial;

    if (ParallelDescriptor::IOProcessor()) {

        if (verbose > 0) {
            std::cout << '\n';
            std::cout << "TIME= " << time << " FLAME WIDTH = "   << flame_width << '\n';
            std::cout << "TIME= " << time << " FLAME SPEED = "   << flame_speed << '\n';
        }

        std::ostream& datalog = *HydroSim::user_run_logs[0];

        if (time == 0.0) {
            datalog << std::setw(14) <<  "      time    ";
            datalog << std::setw(14) <<  "  flame width ";
            datalog << std::setw(14) <<  "  flame speed " << std::endl;
        }

        datalog << std::setw(14) <<  time;
        datalog << std::setw(14) <<  std::setprecision(6) << flame_width;
        datalog << std::setw(14) <<  std::setprecision(6) << flame_speed << std::endl;

    }
}

#endif
