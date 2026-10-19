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

    string pltfile, slcfile;
    int dir;

    GetInputArgs(argc, argv, pltfile, slcfile, dir);

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
    const Vector<Real>& problo = data.ProbLo();
    const Vector<Real>& probhi = data.ProbHi();

    Vector<int> rr = data.RefRatio();

    double maxdist = std::abs(probhi[0] - problo[0]);
    double dx_fine = *(std::min_element(dx.begin(), dx.end()));
    int nbins = int(maxdist / dx_fine);

    Vector<Real> r(nbins);

    for (auto i = 0; i < nbins; i++)
        r[i] = (i + 0.5) * dx_fine;

    Vector<std::string> compVarNames = {"density", "xmom", "pressure", "rad"};

    auto varComps = GetComponents(data, compVarNames);
    auto dens_comp = varComps[0];
    auto xmom_comp = varComps[1];
    auto pressure_index = varComps[2];
    auto rad_comp = varComps[3];

    Vector<Real> dens_bin(nbins, 0.);
    Vector<Real> vel_bin(nbins, 0.);
    Vector<Real> pres_bin(nbins, 0.);
    Vector<Real> rad_bin(nbins, 0.);
    Vector<int> ncount(nbins, 0);

    auto r1 = 1.0;

    Vector<int> fill_comps(data.NComp());
    for (auto i = 0; i < data.NComp(); i++)
        fill_comps[i] = i;

    int mask_size = domain.length().max();
    Vector<int> imask(std::pow(mask_size, AMREX_SPACEDIM), 1);

    for (int l = finestLevel; l >= 0; l--) {

        Vector<Real> level_dx = data.DxLevel()[l];

        const BoxArray& ba = data.boxArray(l);
        const DistributionMapping& dm = data.DistributionMap(l);

        MultiFab lev_data_mf(ba, dm, data.NComp(), data.NGrow());
        data.FillVar(lev_data_mf, l, varNames, fill_comps);

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(lev_data_mf, true); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();

            flgt_frnt1d(AMREX_ARLIM_3D(bx.loVect()), AMREX_ARLIM_3D(bx.hiVect()),
                        BL_TO_FORTRAN_FAB(lev_data_mf[mfi]),
                        nbins, dens_bin.dataPtr(), vel_bin.dataPtr(),
                        pres_bin.dataPtr(), rad_bin.dataPtr(),
                        imask.dataPtr(), mask_size, r1,
                        dens_comp, xmom_comp, pressure_index, rad_comp,
                        AMREX_ZFILL(dx), dx_fine);
        }

        if (l != 0) r1 *= rr[l-1];
    }

    for (int i = 0; i < nbins; i++)
        if (ncount[i] != 0)
            rad_bin[i] /= ncount[i];

    Vector<Vector<Real> > vars = {dens_bin, vel_bin, pres_bin, rad_bin};
    Vector<std::string> slcvarNames = {"density", "velocity", "pressure", "rad"};

    WriteSlicefile(nbins, r, slcvarNames, vars, slcfile);

    BL_PROFILE_VAR_STOP(pmain);

    amrex::Finalize();
}
