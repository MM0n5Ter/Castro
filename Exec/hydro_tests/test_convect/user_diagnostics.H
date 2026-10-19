#ifndef user_diagnostics_H
#define user_diagnostics_H

AMREX_INLINE
void
HydroSim::user_diagnostics ()
{
    int finest_level = parent->finestLevel();
    Real time        = state[State_Type].curTime();
    Real mass        = 0.0;
    Real rho_E       = 0.0;
    Real Tmax        = -std::numeric_limits<Real>::max();
    Real MachMax     = -std::numeric_limits<Real>::max();
    Real Tmax_level;
    Real MachMax_level;

    for (int lev = 0; lev <= finest_level; lev++)
    {
        HydroSim& hs_level = getLevel(lev);

        mass     += hs_level.volWeightedTotal("density", time);

        rho_E    += hs_level.volWeightedTotal("rho_E", time);

        auto temp_mf = hs_level.derive("Temp", time, 0);
        Tmax_level = temp_mf->max(0, 0);
        if (Tmax_level > Tmax)
        {
            Tmax = Tmax_level;
        }

        auto mach_mf = hs_level.derive("MachNumber", time, 0);
        MachMax_level = mach_mf->max(0, 0);
        if (MachMax_level > MachMax)
        {
            MachMax = MachMax_level;
        }

    }


    if (verbose > 0 && ParallelDescriptor::IOProcessor())
    {
        std::cout << '\n';
        std::cout << "TIME= " << time << " MASS  = "   << mass  << '\n';
        std::cout << "TIME= " << time << " RHO*E = "   << rho_E     << '\n';

        std::ostream& primary_data_log = *HydroSim::user_run_logs[0];

        if (time == 0.0) {
           primary_data_log << std::setw(14) <<  "#     time    ";
           primary_data_log << std::setw(14) <<  "       max(T) ";
           primary_data_log << std::setw(14) <<  "       max(M) ";
           primary_data_log << std::setw(14) <<  "        rho_E ";
           primary_data_log << std::endl;
        }

        primary_data_log << std::setw(14) <<  time;
        primary_data_log << std::setw(14) <<  std::setprecision(6) << Tmax;
        primary_data_log << std::setw(14) <<  std::setprecision(6) << MachMax;
        primary_data_log << std::setw(14) <<  std::setprecision(6) << rho_E;
        primary_data_log << std::endl;

        std::cout<<'\n';
    }
}

#endif
