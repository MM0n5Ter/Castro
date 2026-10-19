#include <iostream>
#include <AMReX_DataServices.H>
#include <Radiation_F.H>
#include <Radiation_utils.H>

using namespace amrex;

std::string inputs_name = "";

int main(int argc, char* argv[])
{

    amrex::Initialize(argc, argv, false);

    BL_PROFILE_VAR("main()", pmain);

    string pltfile;
    string slcfile;
    int idir = 1;

    GetInputArgs (argc, argv, pltfile, slcfile, idir);

    Print() << "idir = " << idir << std::endl;

    if (idir > AMREX_SPACEDIM)
        Abort("ERROR: idir must be <= DIM");

    DataServices::SetBatchMode();

    Amrvis::FileType fileType(Amrvis::NEWPLT);
    DataServices dataServices (pltfile, fileType);

    if (!dataServices.AmrDataOk())
        DataServices::Dispatch(DataServices::ExitRequest, NULL);

    AmrData& data = dataServices.AmrDataRef();

    int finestLevel = data.FinestLevel();

    const Vector<string>& varNames = data.PlotVarNames();

    Box domain = data.ProbDomain()[finestLevel];
    Vector<Real> dx = data.CellSize(finestLevel);
    Vector<Real> problo = data.ProbLo();
    Vector<Real> probhi = data.ProbHi();

    Vector<int> rr = data.RefRatio();

    int nbins = domain.length()[idir-1];

    Vector<Real> r(nbins);

    for (auto i = 0; i < nbins; i++)
        r[i] = (i + 0.5) * dx[idir-1] + problo[idir-1];

#if (AMREX_SPACEDIM == 1)
    Vector<std::string> compVarNames = {"density", "eint_E", "Temp",
                                    "pressure", "rad", "x_velocity"};
#elif (AMREX_SPACEDIM == 2)
    Vector<std::string> compVarNames = {"density",
                                    "eint_E", "Temp", "pressure", "rad",
                                    "x_velocity", "y_velocity"};
#else
    Vector<std::string> compVarNames = {"density", "eint_E", "Temp", "pressure",
                                    "rad", "x_velocity", "y_velocity","z_velocity"};
#endif
    auto varComps = GetComponents(data, compVarNames);

    auto r1 = 1.0;

    const int nvars = data.NComp();

    Vector<int> fill_comps(nvars);
    for (auto i = 0; i < nvars; i++)
        fill_comps[i] = i;

    Vector<Real> vars_bin(nbins * (nvars + 1), 0.);

    int mask_size = domain.length().max();
    Vector<int> imask(std::pow(mask_size, AMREX_SPACEDIM), 1);

    auto cnt = 0;

    for (int l = finestLevel; l >= 0; l--) {

        int refratio = 1;
        for (auto lev = 0; lev < l; lev++) refratio *= rr[lev];

        Vector<Real> level_dx = data.DxLevel()[l];

        const BoxArray& ba = data.boxArray(l);
        const DistributionMapping& dm = data.DistributionMap(l);

        MultiFab lev_data_mf(ba, dm, nvars, data.NGrow());
        data.FillVar(lev_data_mf, l, varNames, fill_comps);

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(lev_data_mf, true); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();

            fradshock(AMREX_ARLIM_3D(bx.loVect()), AMREX_ARLIM_3D(bx.hiVect()),
                      problo.dataPtr(), probhi.dataPtr(),
                      BL_TO_FORTRAN_FAB(lev_data_mf[mfi]),
                      nbins, vars_bin.dataPtr(),
                      imask.dataPtr(), mask_size, r1, refratio,
                      dx.dataPtr(), idir, &cnt);

        }

        if (l != 0) r1 *= rr[l-1];
    }

    Vector<Real> coords(cnt);
    for (auto i = 0; i < cnt; i++)
        coords[i] = vars_bin[i];

    auto isv = sort_indexes(coords);

#if (AMREX_SPACEDIM == 1)
    compVarNames = {"density", "x_velocity", "pressure",
                "eint_E", "Temp", "rad", "rad"};
#elif (AMREX_SPACEDIM == 2)
    compVarNames = {"density", "x_velocity", "y_velocity", "pressure",
                "eint_E", "Temp", "rad", "rad"};
#else
    compVarNames = {"density", "x_velocity", "y_velocity", "z_velocity",
                "pressure","eint_E", "Temp", "rad", "rad"};
#endif
    Vector<int> comps = GetComponents(data, compVarNames);

#if (AMREX_SPACEDIM == 1)
    Vector<std::string> slcvarNames = {"density", "x-velocity",
                                   "pressure", "int. energy", "temperature",
                                   "rad energy", "rad temp"};
#elif (AMREX_SPACEDIM == 2)
    Vector<std::string> slcvarNames = {"density", "x-velocity", "y-velocity",
                                   "pressure", "int. energy", "temperature",
                                   "rad energy", "rad temp"};
#else
    Vector<std::string> slcvarNames = {"density", "x-velocity", "y-velocity",
                                   "z-velocity", "pressure", "int. energy", "temperature", "rad energy", "rad temp"};
#endif

    std::ofstream slicefile;
    slicefile.open(slcfile);
    slicefile.setf(std::ios::scientific);
    slicefile.precision(12);
    const auto w = 24;

    if (idir == 1)
        slicefile << std::setw(w) << "x";
    else if (idir == 2)
        slicefile << std::setw(w) << "y";
    else
        slicefile << std::setw(w) << "z";

    for (auto it=slcvarNames.begin(); it!=slcvarNames.end(); ++it)
        slicefile << std::setw(w) << *it;

    slicefile << std::endl;

    const Real arad = 1.0;

    for (auto i = 0; i < cnt; i++) {

        slicefile << std::setw(w) << vars_bin[isv[i]];

        for (auto it=comps.begin(); it!=(comps.end()-1); ++it)
            slicefile << std::setw(w) << vars_bin[isv[i]+((*it)+1)*nbins];

        auto it = comps.end()-1;
        slicefile << std::setw(w) << std::pow(vars_bin[isv[i]+((*it)+1)*nbins] / arad, 0.25);

        slicefile << std::endl;
    }

    slicefile.close();

    BL_PROFILE_VAR_STOP(pmain);

    amrex::Finalize();
}
