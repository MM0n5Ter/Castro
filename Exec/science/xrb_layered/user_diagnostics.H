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
        HydroSim& sim_lev = getLevel(lev);

        mass     += sim_lev.volumeIntegralSum("density", time);

        rho_E    += sim_lev.volumeIntegralSum("rho_E", time);

        auto temp_mf = sim_lev.derive("Temp", time, 0);
        Tmax_level = temp_mf->max(0, 0);
        if (Tmax_level > Tmax)
        {
            Tmax = Tmax_level;
        }

        auto mach_mf = sim_lev.derive("LocalMach", time, 0);
        MachMax_level = mach_mf->max(0, 0);
        if (MachMax_level > MachMax)
        {
            MachMax = MachMax_level;
        }

    }


    if (ParallelDescriptor::IOProcessor())
    {
        if (verbose > 0) {
            std::cout << '\n';
            std::cout << "TIME= " << time << " MASS  = "   << mass  << '\n';
            std::cout << "TIME= " << time << " RHO*E = "   << rho_E     << '\n';
        }

        std::ostream& datalog = *HydroSim::user_data_logs[0];

        if (time == 0.0) {
            datalog << std::setw(14) <<  "#     time    ";
            datalog << std::setw(14) <<  "       max(T) ";
            datalog << std::setw(14) <<  "       max(M) ";
            datalog << std::setw(14) <<  "        rho_E ";
            datalog << std::endl;
        }

        // Write the quantities at this time
        datalog << std::setw(14) <<  time;
        datalog << std::setw(14) <<  std::setprecision(6) << Tmax;
        datalog << std::setw(14) <<  std::setprecision(6) << MachMax;
        datalog << std::setw(14) <<  std::setprecision(6) << rho_E;
        datalog << std::endl;

        if (verbose > 0) {
            std::cout<<'\n';
        }
    }
}

#endif
