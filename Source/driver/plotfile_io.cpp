#ifndef WIN32
#include <unistd.h>
#endif

#include <iomanip>
#include <iostream>
#include <format>
#include <string>
#include <ctime>

#include <AMReX_Utility.H>
#include <HydroSim.H>
#include <run_output.H>
#include <AMReX_ParmParse.H>

#ifdef RADIATION
#include <Radiation.H>
#endif

#ifdef GRAVITY
#include <Gravity.H>
#endif

#ifdef DIFFUSION
#include <diffusion_operator.H>
#endif

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include <user_initialize_state_data.H>
#include <user_checkpoint.H>
#include <user_restart.H>
#include <AMReX_buildInfo.H>
#include <algorithm>

#include <cstdio>

#include <vector>

#include <memory>

#include <numbers>

#include <AMReX_CONSTANTS.H>

#include <global.H>

#include <runtime_param_init.H>

#include <AMReX_VisMF.H>

#include <AMReX_TagBox.H>

#include <AMReX_FillPatchUtil.H>

#ifdef AMREX_PARTICLES
#include <AMReX_Particles.H>
#endif

#include <extern_parameters.H>

#include <prob_parameters.H>

#include <user_initialize.H>

#include <ambient.H>

#include <hydrosim_limits.H>

#include <solver_constants.H>

using std::string;
using namespace amrex;

std::unique_ptr<MultiFab>
HydroSim::derive (const std::string& name,
                Real           time,
                int            ngrow)
{

    BL_PROFILE("HydroSim::derive()");

#ifdef AMREX_PARTICLES
  return ParticleQuantity(name,time,ngrow);
#else
   return AmrLevel::derive(name,time,ngrow);
#endif
}

void
HydroSim::derive (const std::string& name,
                Real           time,
                MultiFab&      mf,
                int            dcomp)
{

    BL_PROFILE("HydroSim::derive()");

    AmrLevel::derive(name,time,mf,dcomp);
}

void
HydroSim::writePlotFile(const std::string& dir,
                      ostream& os,
                      VisMF::How how)
{
  writeSnapshot(dir, os, how, 0);
}

void
HydroSim::writeSmallPlotFile (const std::string& dir,
                            ostream&       os,
                            VisMF::How     how)
{
  writeSnapshot(dir, os, how, 1);
}

void
HydroSim::writeSnapshot(const std::string& dir,
                       ostream& os,
                       VisMF::How how,
                       const int is_small)
{
#ifdef AMREX_PARTICLES
  ParticleOutputFile(dir);
#endif

    auto wants_state_var = [&] (const std::string& name) {
        return (parent->isStatePlotVar(name) && is_small == 0) ||
               (parent->isStateSmallPlotVar(name) && is_small == 1);
    };

    auto wants_derived_var = [&] (const std::string& name) {
        return (parent->isDerivePlotVar(name) && is_small == 0) ||
               (parent->isDeriveSmallPlotVar(name) && is_small == 1);
    };

    std::vector<std::pair<int,int> > plot_var_map;
    for (int typ = 0; typ < desc_lst.size(); typ++) {
      for (int comp = 0; comp < desc_lst[typ].nComp(); comp++) {
        if (wants_state_var(desc_lst[typ].name(comp)) &&
            desc_lst[typ].getType() == IndexType::TheCellType()) {
            plot_var_map.emplace_back(typ, comp);
        }
      }
    }

    int num_derive = 0;
    std::list<std::string> derive_names;
    const std::list<DeriveRec>& dlist = derive_lst.dlist();

    for (const auto & dd : dlist) {

        if (wants_derived_var(dd.name()))
        {
#ifdef AMREX_PARTICLES
            if (dd.name() == "particle_count" ||
                dd.name() == "total_particle_count")
            {
                if (HydroSim::theTracerPC())
                {
                    derive_names.push_back(dd.name());
                    num_derive = num_derive + dd.numDerive();
                }
            } else
#endif
            {
               derive_names.push_back(dd.name());
               num_derive = num_derive + dd.numDerive();
            }
        }
    }

    int n_data_items = static_cast<int>(plot_var_map.size()) + num_derive;

#ifdef RADIATION
    if (Radiation::num_plot_vars > 0) n_data_items += Radiation::num_plot_vars;
#endif

#ifdef REACTIONS
#ifndef TRUE_SDC
    if (keep_burn_weights) {
        n_data_items += static_cast<int>(HydroSim::burn_split_names.size());
    }
#endif
#endif

    Real cur_time = state[State_Type].curTime();

    if (level == 0 && ParallelDescriptor::IOProcessor())
    {
        os << thePlotFileType() << '\n';

        if (n_data_items == 0) {
          amrex::Error("Must specify at least one valid data item to plot");
        }

        os << n_data_items << '\n';

        for (const auto& [typ, comp] : plot_var_map)
        {
            os << desc_lst[typ].name(comp) << '\n';
        }

        for (const auto &name : derive_names)
        {
            const DeriveRec* rec = derive_lst.get(name);
            if (rec->numDerive() > 1) {
                for (int i = 0; i < rec->numDerive(); ++i) {
                    os << std::format("{}_{}\n", rec->variableName(0), i);
                }
            }
            else {
                os << rec->variableName(0) << '\n';
            }
        }

#ifdef RADIATION
        for (int i=0; i<Radiation::num_plot_vars; ++i) {
          os << Radiation::plot_var_names[i] << '\n';
        }
#endif

#ifdef REACTIONS
#ifndef TRUE_SDC
        if (keep_burn_weights) {
            for (const auto& name: HydroSim::burn_split_names) {
                os << name << '\n';
            }
        }
#endif
#endif

        os << AMREX_SPACEDIM << '\n';
        os << parent->cumTime() << '\n';
        int f_lev = parent->finestLevel();
        os << f_lev << '\n';
        for (int i = 0; i < AMREX_SPACEDIM; i++) {
            os << geom.ProbLo(i) << ' ';
        }
        os << '\n';
        for (int i = 0; i < AMREX_SPACEDIM; i++) {
            os << geom.ProbHi(i) << ' ';
        }
        os << '\n';
        for (int i = 0; i < f_lev; i++) {
          os << parent->refRatio(i)[0] << ' ';
        }
        os << '\n';
        for (int i = 0; i <= f_lev; i++) {
          os << parent->Geom(i).Domain() << ' ';
        }
        os << '\n';
        for (int i = 0; i <= f_lev; i++) {
          os << parent->levelSteps(i) << ' ';
        }
        os << '\n';
        for (int i = 0; i <= f_lev; i++)
        {
            for (int k = 0; k < AMREX_SPACEDIM; k++) {
              os << parent->Geom(i).CellSize()[k] << ' ';
            }
            os << '\n';
        }
        os << (int) geom.Coord() << '\n';
        os << "0\n";

#ifdef RADIATION
        if (do_radiative_transfer && Radiation::groups_enabled) {
          std::ofstream energy_group_file;
          std::string FullPathEnergyGroupFile = dir;
          FullPathEnergyGroupFile += "/RadiationGroups";
          energy_group_file.open(FullPathEnergyGroupFile.c_str(), std::ios::out);

          radiation->dump_group_table(energy_group_file);

          energy_group_file.close();
        }
#endif
    }
    static const std::string BaseName = "/Cell";
    std::string Level = std::format("Level_{}", level);
    std::string FullPath = dir;
    if (!FullPath.ends_with('/')) {
      FullPath += '/';
    }
    FullPath += Level;
    if (ParallelDescriptor::IOProcessor()) {
        if (!amrex::UtilCreateDirectory(FullPath, 0755)) {
            amrex::CreateDirectoryFailed(FullPath);
        }
    }
    ParallelDescriptor::Barrier();

    if (ParallelDescriptor::IOProcessor())
    {
        os << level << ' ' << grids.size() << ' ' << cur_time << '\n';
        os << parent->levelSteps(level) << '\n';

        for (int i = 0; i < grids.size(); ++i)
        {
            RealBox gridloc = RealBox(grids[i],geom.CellSize(),geom.ProbLo());
            for (int n = 0; n < AMREX_SPACEDIM; n++) {
              os << gridloc.lo(n) << ' ' << gridloc.hi(n) << '\n';
            }
        }
        if (n_data_items > 0)
        {
            std::string PathNameInHeader = Level;
            PathNameInHeader += BaseName;
            os << PathNameInHeader << '\n';
        }
    }
    int       cnt   = 0;
    const int nGrow = 0;
    MultiFab  plotMF(grids,dmap,n_data_items,nGrow);
    MultiFab* this_dat = nullptr;
    for (const auto& [typ, comp] : plot_var_map) {
        this_dat = &state[typ].newData();
        MultiFab::Copy(plotMF, *this_dat, comp, cnt, 1, nGrow);
        cnt++;
    }
    if (!dlist.empty())
    {
        for (const auto & dd : dlist) {

            if (wants_derived_var(dd.name())) {

                auto derived_mf = derive(dd.variableName(0), cur_time, nGrow);
                MultiFab::Copy(plotMF, *derived_mf, 0, cnt, dd.numDerive(), nGrow);
                cnt = cnt + dd.numDerive();
            }
        }
    }

#ifdef RADIATION
    if (Radiation::num_plot_vars > 0) {
        MultiFab::Copy(plotMF,*(radiation->plot_fields[level]),0,cnt,Radiation::num_plot_vars,0);
        cnt += Radiation::num_plot_vars;
    }
#endif

#ifdef REACTIONS
#ifndef TRUE_SDC
    if (keep_burn_weights) {
        MultiFab::Copy(plotMF, getLevel(level).burn_split_data, 0, cnt, static_cast<int>(HydroSim::burn_split_names.size()), 0);
        cnt += static_cast<int>(HydroSim::burn_split_names.size());
    }
#endif
#endif

    std::string TheFullPath = FullPath;
    TheFullPath += BaseName;

    const Real io_begin_clock = ParallelDescriptor::second();

    if (amrex::AsyncOut::UseAsyncOut()) {
        VisMF::AsyncWrite(std::move(plotMF),TheFullPath);
    } else {
        VisMF::Write(plotMF,TheFullPath,how,true);
    }

    const Real write_seconds = ParallelDescriptor::second() - io_begin_clock;

    if (level == 0 && ParallelDescriptor::IOProcessor()) {
        writeRunRecord(dir, write_seconds);
    }

#ifdef GRAVITY
    if (use_central_mass && level == 0) {

        std::ofstream PointMassFile;
        std::string FullPathPointMassFile = dir;
        FullPathPointMassFile += "/point_mass";
        PointMassFile.open(FullPathPointMassFile.c_str(), std::ios::out);

        PointMassFile << std::setprecision(17) << point_mass << std::endl;

        PointMassFile.close();

    }
#endif

}

void
HydroSim::setPlotVariables ()
{
  AmrLevel::setPlotVariables();


  for (int i = 0; i < desc_lst[Forcing_Type].nComp(); i++) {
    parent->deleteStatePlotVar(desc_lst[Forcing_Type].name(i));
  }

#ifdef REACTIONS
  if (time_scheme == SimplifiedIterCorrection) {
      for (int i = 0; i < desc_lst[Burn_Corr_Type].nComp(); i++) {
          parent->deleteStatePlotVar(desc_lst[Burn_Corr_Type].name(i));
      }
  }
#endif

}

std::string
HydroSim::thePlotFileType () const
{
    static const std::string the_plot_file_type("HyperCLaw-V1.1");

    return the_plot_file_type;
}
