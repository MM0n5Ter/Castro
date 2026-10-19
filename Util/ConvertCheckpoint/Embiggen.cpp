
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <string>

#ifndef WIN32
#include <unistd.h>
#endif

#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_DataServices.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>
#include <AMReX_Geometry.H>
#include <AMReX_StateDescriptor.H>
#include <AMReX_StateData.H>
#include <AMReX_BCRec.H>
#include <AMReX_LevelBld.H>
#include <AMReX_AmrLevel.H>

using namespace amrex;

LevelBld *getLevelBld() {
  return 0;
}

#ifdef BL_USE_SETBUF
#define pubsetbuf setbuf
#endif

using std::cout;
using std::cerr;
using std::endl;

using namespace amrex;





std::string mode("full_hierarchy");
std::string CheckFileIn;
std::string CheckFileOut;
int nFiles(64);
bool verbose(true);
int num_new_levels(1);
int      ref_ratio(1);
int   restart_grow_factor(1);
int body_at_center(-1);
Real theta_new_hi;
int   max_grid_size(4096);
int blocking_factor(16);
int   coord(-1);

const std::string CheckPointVersion = "CheckPointVersion_1.0";

Vector<int> nsets_save(1);

VisMF::How how = VisMF::OneFilePerCPU;


struct FakeStateData {
    struct TimeInterval {
        Real start, stop;
    };
    const StateDescriptor *desc;
    Box domain;
    BoxArray grids;
    TimeInterval new_time;
    TimeInterval old_time;
    MultiFab *new_data;
    MultiFab *old_data;
    Vector< Vector<BCRec> > bc;
};


struct FakeAmrLevel {
    int level;
    Geometry geom;
    BoxArray grids;
    IntVect crse_ratio;
    IntVect fine_ratio;
    Vector<FakeStateData> state;
    Vector<FakeStateData> new_state;
};


struct FakeAmr {
  int                  finest_level;
  Real                 cumtime;
  Vector<Real>          dt_level;
  Vector<int>           level_steps;
  Vector<int>           level_count;
  Vector<int>           n_cycle;
  Vector<Real>          dt_min;
  Vector<IntVect>       ref_ratio;
  Vector<Geometry>      geom;
  Vector< Vector<Real> > dx;
  Vector<FakeAmrLevel> fakeAmrLevels;
};

FakeAmr fakeAmr;

static void ScanArguments() {
    ParmParse pp;

    if (pp.contains("mode")) {
        pp.get("mode", mode);
    }
    if(pp.contains("checkin")) {
      pp.get("checkin", CheckFileIn);
    }
    if(pp.contains("checkout")) {
      pp.get("checkout", CheckFileOut);
    }
    if(pp.contains("nfiles")) {
      pp.get("nfiles", nFiles);
    }
    if(pp.contains("verbose")) {
      pp.get("verbose", verbose);
    }

    if (mode == "full_hierarchy") {

        if(pp.contains("ref_ratio")) {
            pp.get("ref_ratio", ref_ratio);
        }

        if(pp.contains("restart_grow_factor")) {
            pp.get("restart_grow_factor", restart_grow_factor);
        }

        pp.get("body_at_center", body_at_center);

        if (body_at_center != 0 && body_at_center != 1) {
            amrex::Abort("body_at_center must be 0 or 1");
        }

        if (ref_ratio != 2 && ref_ratio != 4) {
            amrex::Abort("ref_ratio must be 2 or 4");
        }

        if (restart_grow_factor <= 1) {
            amrex::Abort("must have restart_grow_factor > 1");
        }

        if (body_at_center == 1 && restart_grow_factor != 2 && restart_grow_factor != 3) {
                amrex::Abort("must have restart_grow_factor = 2 or 3 for star at center");
        }

    } else if (mode == "theta_extend") {
        pp.get("theta_new_hi", theta_new_hi);
        if (pp.contains("blocking_factor"))
            pp.get("blocking_factor", blocking_factor);
    } else {
        amrex::Abort("Unknown mode");
    }
}


static void PrintUsage(char* progName)
{
    cout << "\n";

    cout << "Full hierarchy grow:\n";
    cout << progName
         << " mode=full_hierarchy "
         << "checkin=filename "
         << "checkout=outfilename "
         << "ref_ratio=2|4 "
         << "restart_grow_factor=integer "
         << "body_at_center=0|1 "
         << "[nfiles=nfilesout] "
         << "[verbose=true|false]" << endl;

    cout << "Theta extension:\n";
    cout << progName
         << " mode=theta_extend "
         << "checkin=filename "
         << "checkout=outfilename "
         << "theta_new_hi=value_in_radian "
         << "[nfiles=nfilesout] "
         << "[verbose=true|false]" << endl;

    exit(1);
}


static void ReadCheckpointFile(const std::string& fileName) {
    int i;
    std::string File = fileName;

    File += '/';
    File += "Header";

    VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

    std::ifstream is;
    is.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
    is.open(File.c_str(), std::ios::in);
    if( ! is.good()) {
        amrex::FileOpenFailed(File);
    }

    int         spdim;
    bool        new_checkpoint_format = false;
    std::string first_line;

    std::getline(is,first_line);

    if (first_line == CheckPointVersion) {
        new_checkpoint_format = true;
        is >> spdim;
    } else {
        spdim = atoi(first_line.c_str());
    }

    if(spdim != AMREX_SPACEDIM) {
        cerr << "Amr::restart(): bad spacedim = " << spdim << '\n';
        amrex::Abort();
    }

    if (mode == "theta_extend" && AMREX_SPACEDIM != 2) {
        amrex::Abort("Currently only works for Spherical 2D geometry");
    }

    is >> fakeAmr.cumtime;
    int mx_lev;
    is >> mx_lev;
    is >> fakeAmr.finest_level;

    if(ParallelDescriptor::IOProcessor())
       std::cout << "previous finest_lev is " << fakeAmr.finest_level <<  std::endl;

    int n = 0;
    if (mode == "full_hierarchy") {

        n = num_new_levels;
        mx_lev = mx_lev + n;
        fakeAmr.finest_level = fakeAmr.finest_level + n;

        if(ParallelDescriptor::IOProcessor()) {
            std::cout << "     new finest_lev is " << fakeAmr.finest_level << std::endl;
            std::cout << "previous     mx_lev is " << mx_lev-n << std::endl;
            std::cout << "     new     mx_lev is " << mx_lev << std::endl;
        }
    }

    fakeAmr.geom.resize(mx_lev + 1);
    fakeAmr.ref_ratio.resize(mx_lev);
    fakeAmr.dt_level.resize(mx_lev + 1);
    fakeAmr.dt_min.resize(mx_lev + 1);
    fakeAmr.n_cycle.resize(mx_lev + 1);
    fakeAmr.level_steps.resize(mx_lev + 1);
    fakeAmr.level_count.resize(mx_lev + 1);
    fakeAmr.fakeAmrLevels.resize(mx_lev + 1);

    for (i = n; i <= mx_lev; i++) {
      is >> fakeAmr.geom[i];
    }

    if(ParallelDescriptor::IOProcessor()) {
       std::cout << " " << std::endl;
       for (i = n; i <= mx_lev; i++) {
          std::cout << "Old checkpoint level    " << i-n << std::endl;
          std::cout << " ... domain is       " << fakeAmr.geom[i].Domain() << std::endl;
          std::cout << " ...     dx is       " << fakeAmr.geom[i].CellSize()[0] << std::endl;
#if AMREX_SPACEDIM == 2
          std::cout << " ...     dy is       " << fakeAmr.geom[i].CellSize()[1] << std::endl;
#elif AMREX_SPACEDIM >= 2
          std::cout << " ...     dz is       " << fakeAmr.geom[i].CellSize()[2] << std::endl;
#endif
          std::cout << "  " << std::endl;
       }
    }

    if (mode == "full_hierarchy") {
        Box dom0(fakeAmr.geom[0].Domain());
        for (int d = 0; d < AMREX_SPACEDIM; d++)
            {
                int dlen = dom0.size()[d];
                int scaled = dlen / (2*ref_ratio);
                if ( (scaled * 2 * ref_ratio) != dlen )
                    amrex::Abort("must have domain divisible by 2*ref_ratio");
            }

        if (restart_grow_factor <= 1)
            amrex::Abort("must have restart_grow_factor > 1");
    }

    for (i = n; i <  mx_lev; i++) {
      is >> fakeAmr.ref_ratio[i];
    }
    for (i = n; i <= mx_lev; i++) {
      is >> fakeAmr.dt_level[i];
    }

    Box          domain(fakeAmr.geom[n].Domain());
    RealBox prob_domain(fakeAmr.geom[n].ProbDomain());
    coord = fakeAmr.geom[n].Coord();
    if (mode == "theta_extend" && coord != 2) {
        amrex::Abort("theta_extend mode currently only for Spherical 2D geometry");
    }

    if (mode == "full_hierarchy") {
        domain.coarsen(ref_ratio);
        fakeAmr.geom[0].define(domain,&prob_domain,coord);

        fakeAmr.ref_ratio[0] = ref_ratio * IntVect::TheUnitVector();

        fakeAmr.dt_level[0] = fakeAmr.dt_level[1] * ref_ratio;
    }

    if (new_checkpoint_format) {
        for (i = n; i <= mx_lev; i++) {
            is >> fakeAmr.dt_min[i];
        }
        if (mode == "full_hierarchy") {
            fakeAmr.dt_min[0] = fakeAmr.dt_min[1] * ref_ratio;
        }
    } else {
      for (i = 0; i <= mx_lev; i++) fakeAmr.dt_min[i] = fakeAmr.dt_level[i];
    }

    for (i = n; i <= mx_lev; i++) {
      is >> fakeAmr.n_cycle[i];
    }
    for (i = n; i <= mx_lev; i++) {
      is >> fakeAmr.level_steps[i];
    }
    for (i = n; i <= mx_lev; i++) {
      is >> fakeAmr.level_count[i];
    }

    if (mode == "full_hierarchy") {

        fakeAmr.n_cycle[0] = 1;

        fakeAmr.n_cycle[1] = ref_ratio;

        fakeAmr.level_steps[0] = fakeAmr.level_steps[1] / ref_ratio;
        if ( (fakeAmr.level_steps[0]*ref_ratio) != fakeAmr.level_steps[1] )
            amrex::Abort("Number of steps in original checkpoint must be divisible by ref_ratio");

        if (fakeAmr.level_count[1] == fakeAmr.level_steps[1])
            {
                fakeAmr.level_count[0] = fakeAmr.level_steps[0];

            } else {
            fakeAmr.level_count[0] = std::min(fakeAmr.level_count[1],fakeAmr.level_steps[0]);;
        }
    }
    int ndesc_save;

    for(int lev = n; lev <= fakeAmr.finest_level; ++lev) {

      FakeAmrLevel &falRef = fakeAmr.fakeAmrLevels[lev];

      is >> falRef.level;
      falRef.level = falRef.level + n;

      is >> falRef.geom;

      falRef.fine_ratio = IntVect::TheUnitVector();
      falRef.fine_ratio.scale(-1);
      falRef.crse_ratio = IntVect::TheUnitVector();
      falRef.crse_ratio.scale(-1);

      if(falRef.level > 0) {
        falRef.crse_ratio = fakeAmr.ref_ratio[falRef.level-1];
      }
      if(falRef.level < mx_lev) {
        falRef.fine_ratio = fakeAmr.ref_ratio[falRef.level];
      }

      falRef.grids.readFrom(is);

      int nstate;
      is >> nstate;
      int ndesc = nstate;

      if (lev == n) {
          ndesc_save = ndesc;

          nsets_save.resize(ndesc_save);
      }

      falRef.state.resize(ndesc);
      falRef.new_state.resize(ndesc);

      for(int i = 0; i < ndesc; i++) {

        is >> falRef.state[i].domain;

        falRef.state[i].grids.readFrom(is);

        is >> falRef.state[i].old_time.start;
        is >> falRef.state[i].old_time.stop;
        is >> falRef.state[i].new_time.start;
        is >> falRef.state[i].new_time.stop;

        int nsets;
        is >> nsets;

        nsets_save[i] = nsets;

        falRef.state[i].old_data = 0;
        falRef.state[i].new_data = 0;

        std::string mf_name;
        std::string FullPathName;

        if (nsets >= 1) {
           falRef.state[i].new_data = new MultiFab;
           is >> mf_name;
           FullPathName = fileName;
           if( ! fileName.empty() && fileName[fileName.length()-1] != '/') {
             FullPathName += '/';
           }
           FullPathName += mf_name;
           VisMF::Read(*(falRef.state[i].new_data), FullPathName);
        }

        if (nsets == 2) {
          falRef.state[i].old_data = new MultiFab;
          is >> mf_name;
          FullPathName = fileName;
          if( ! fileName.empty() && fileName[fileName.length()-1] != '/') {
            FullPathName += '/';
          }
          FullPathName += mf_name;
          VisMF::Read(*(falRef.state[i].old_data), FullPathName);
        }

      }
    }

    int max_len = 0;
    BoxArray g(fakeAmr.fakeAmrLevels[n].grids);
    for (int b = 0; b < g.size(); b++)
       for (int d = 0; d < AMREX_SPACEDIM; d++)
         max_len = std::max(max_len, g[b].length(d));
    max_grid_size = max_len;

    if (mode == "full_hierarchy") {
        FakeAmrLevel &falRef_orig = fakeAmr.fakeAmrLevels[n];
        for(int lev(n-1); lev >= 0; lev--) {
            FakeAmrLevel &falRef = fakeAmr.fakeAmrLevels[lev];
            falRef.level = lev;

            BoxArray new_grids(domain);
            new_grids.maxSize(max_grid_size);

            falRef.grids = new_grids;

            falRef.geom.define(domain,&prob_domain,coord);

            if(falRef.level > 0)
                falRef.crse_ratio = ref_ratio * IntVect::TheUnitVector();
            falRef.fine_ratio = ref_ratio * IntVect::TheUnitVector();

            falRef.state.resize(ndesc_save);
            falRef.new_state.resize(ndesc_save);

            for(int i = 0; i < ndesc_save; i++) {

                falRef.state[i].domain = domain;
                falRef.state[i].grids = falRef.grids;
                falRef.state[i].new_time.start = falRef_orig.state[i].new_time.start;
                falRef.state[i].new_time.stop  = falRef_orig.state[i].new_time.stop;
                falRef.state[i].old_time.start = falRef.state[i].new_time.start - fakeAmr.dt_level[lev];
                falRef.state[i].old_time.stop  = falRef.state[i].new_time.stop  - fakeAmr.dt_level[lev];

                falRef.state[i].old_data = 0;
                falRef.state[i].new_data = 0;

                if (nsets_save[i] >= 1) {

                    int ncomp = falRef_orig.state[i].new_data->nComp();
                    int ngrow = falRef_orig.state[i].new_data->nGrow();

                    DistributionMapping dmap {falRef.grids};
                    falRef.state[i].new_data = new MultiFab(falRef.grids, dmap, ncomp, ngrow);
                    falRef.state[i].new_data->setVal(0.);

                    if (nsets_save[i] == 2) {
                        falRef.state[i].old_data = new MultiFab(falRef.grids, dmap, ncomp, ngrow);
                        falRef.state[i].old_data->setVal(0.);
                    }

                }
            }
        }
    }
}

static void WriteCheckpointFile(const std::string& inFileName, const std::string &outFileName) {
    VisMF::SetNOutFiles(nFiles);
    FABio::Format thePrevFormat = FArrayBox::getFormat();
    FArrayBox::setFormat(FABio::FAB_NATIVE);

    const std::string ckfile = outFileName;

    if(ParallelDescriptor::IOProcessor()) {
      if( ! amrex::UtilCreateDirectory(ckfile, 0755)) {
        amrex::CreateDirectoryFailed(ckfile);
      }
    }
    ParallelDescriptor::Barrier();


    if (ParallelDescriptor::IOProcessor()) {

      std::ifstream oldHydroSimHeaderFile;

      std::string oldHydroSimHeaderName = inFileName + "/CastroHeader";
      oldHydroSimHeaderFile.open(oldHydroSimHeaderName.c_str(), std::ios::binary);

      if (oldHydroSimHeaderFile.good()) {

        std::ofstream newHydroSimHeaderFile;

        std::string newHydroSimHeaderName = outFileName + "/CastroHeader";
        newHydroSimHeaderFile.open(newHydroSimHeaderName.c_str(), std::ios::binary);

        if (newHydroSimHeaderFile.good()) {
          newHydroSimHeaderFile << oldHydroSimHeaderFile.rdbuf();
          newHydroSimHeaderFile.close();
        }

        oldHydroSimHeaderFile.close();

      }

    }


    std::string HeaderFileName = ckfile + "/Header";

    VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);

    std::ofstream HeaderFile;

    HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());

    int old_prec(0), i;

    if(ParallelDescriptor::IOProcessor()) {
        HeaderFile.open(HeaderFileName.c_str(),
                        std::ios::out|std::ios::trunc|std::ios::binary);

        if( ! HeaderFile.good()) {
          amrex::FileOpenFailed(HeaderFileName);
        }

        old_prec = HeaderFile.precision(15);

        int max_level(fakeAmr.finest_level);
        HeaderFile << CheckPointVersion << '\n'
                   << AMREX_SPACEDIM       << '\n'
                   << fakeAmr.cumtime           << '\n'
                   << max_level                 << '\n'
                   << fakeAmr.finest_level      << '\n';
        for (i = 0; i <= max_level; i++) HeaderFile << fakeAmr.geom[i]        << ' ';
        HeaderFile << '\n';
        for (i = 0; i < max_level; i++)  HeaderFile << fakeAmr.ref_ratio[i]   << ' ';
        HeaderFile << '\n';
        for (i = 0; i <= max_level; i++) HeaderFile << fakeAmr.dt_level[i]    << ' ';
        HeaderFile << '\n';
        for (i = 0; i <= max_level; i++) HeaderFile << fakeAmr.dt_min[i]      << ' ';
        HeaderFile << '\n';
        for (i = 0; i <= max_level; i++) HeaderFile << fakeAmr.n_cycle[i]     << ' ';
        HeaderFile << '\n';
        for (i = 0; i <= max_level; i++) HeaderFile << fakeAmr.level_steps[i] << ' ';
        HeaderFile << '\n';
        for (i = 0; i <= max_level; i++) HeaderFile << fakeAmr.level_count[i] << ' ';
        HeaderFile << '\n';
    }

    for(int lev(0); lev <= fakeAmr.finest_level; ++lev) {

      std::ostream &os = HeaderFile;
      FakeAmrLevel &falRef = fakeAmr.fakeAmrLevels[lev];
      int ndesc = falRef.state.size();

      char buf[64];
      sprintf(buf, "Level_%d", lev);
      std::string Level = buf;

      std::string FullPath = ckfile;
      if( ! FullPath.empty() && FullPath[FullPath.length()-1] != '/') {
        FullPath += '/';
      }
      FullPath += Level;

      if(ParallelDescriptor::IOProcessor()) { if( ! amrex::UtilCreateDirectory(FullPath, 0755)) {
          amrex::CreateDirectoryFailed(FullPath);
        }
      }
      ParallelDescriptor::Barrier();

      if(ParallelDescriptor::IOProcessor()) {
        os << lev << '\n' << falRef.geom  << '\n';
        falRef.grids.writeOn(os);
        os << ndesc << '\n';
      }
      for(int i(0); i < ndesc; ++i) {
        std::string PathNameInHeader = Level;
        sprintf(buf, "/SD_%d", i);
        PathNameInHeader += buf;
        std::string FullPathName = FullPath;
        FullPathName += buf;
          static const std::string NewSuffix("_New_MF");
          static const std::string OldSuffix("_Old_MF");
          const std::string name(PathNameInHeader);
          const std::string fullpathname(FullPathName);

          bool dump_old(true);
          if(dump_old == true && falRef.state[i].old_data == 0) {
            dump_old = false;
          }

          if(ParallelDescriptor::IOProcessor()) {
            std::string mf_name_old = name;
            mf_name_old += OldSuffix;
            std::string mf_name_new = name;
            mf_name_new += NewSuffix;

            os << falRef.state[i].domain << '\n';

            falRef.state[i].grids.writeOn(os);

            os << falRef.state[i].old_time.start << '\n'
               << falRef.state[i].old_time.stop  << '\n'
               << falRef.state[i].new_time.start << '\n'
               << falRef.state[i].new_time.stop  << '\n';

            if (nsets_save[i] > 0) {
               if(dump_old) {
                 os << 2 << '\n' << mf_name_new << '\n' << mf_name_old << '\n';
               } else {
                 os << 1 << '\n' << mf_name_new << '\n';
               }
            } else {
              os << 0 << '\n';
            }

          }

          if (nsets_save[i] > 0) {
             BL_ASSERT(falRef.state[i].new_data);
             std::string mf_fullpath_new = fullpathname;
             mf_fullpath_new += NewSuffix;
             VisMF::Write(*(falRef.state[i].new_data),mf_fullpath_new,how);
          }

          if (nsets_save[i] > 1) {
            BL_ASSERT(dump_old);
            BL_ASSERT(falRef.state[i].old_data);
            std::string mf_fullpath_old = fullpathname;
            mf_fullpath_old += OldSuffix;
            VisMF::Write(*(falRef.state[i].old_data),mf_fullpath_old,how);
          }
      }

       if (ParallelDescriptor::IOProcessor()) {
          if (lev == 0) {
             std::cout << " " << std::endl;
             std::cout << " **************************************** " << std::endl;
             std::cout << " " << std::endl;
          }
          std::cout << "New checkpoint level    " << lev << std::endl;
          std::cout << " ... domain is       " << fakeAmr.geom[lev].Domain() << std::endl;
          std::cout << " ...     dx is       " << fakeAmr.geom[lev].CellSize()[0] << std::endl;
#if AMREX_SPACEDIM == 2
          std::cout << " ...     dy is       " << fakeAmr.geom[lev].CellSize()[1] << std::endl;
#elif AMREX_SPACEDIM >= 2
          std::cout << " ...     dz is       " << fakeAmr.geom[lev].CellSize()[2] << std::endl;
#endif
          std::cout << "  " << std::endl;
       }

    }

    if(ParallelDescriptor::IOProcessor()) {
        HeaderFile.precision(old_prec);

        if( ! HeaderFile.good()) {
          amrex::Error("Amr::checkpoint() failed");
        }
    }

    FArrayBox::setFormat(thePrevFormat);
}


static void ConvertData() {

   int max_level(fakeAmr.finest_level);


   FakeAmrLevel &falRef0 = fakeAmr.fakeAmrLevels[0];
   Box domain(fakeAmr.geom[0].Domain());
   BoxList(newgrid_list);

   int dlenx = domain.size()[0];
#if (AMREX_SPACEDIM >= 2)
   int dleny = domain.size()[1];
#if (AMREX_SPACEDIM == 3)
   int dlenz = domain.size()[1];
#endif
#endif

#if (AMREX_SPACEDIM == 2)
   if (coord == 1 && body_at_center == 1)
   {
     for (int jy = 0; jy < restart_grow_factor; jy++)
      for (int jx = 0; jx < restart_grow_factor; jx++)
        for (int n = 0; n < falRef0.grids.size(); n++)
        {
          int shiftx(jx*dlenx);
          int shifty(jy*dleny);
          IntVect shift_box(AMREX_D_DECL(shiftx,shifty,shiftz));

          Box bx(falRef0.grids[n]);
          bx.shift(shift_box);
          newgrid_list.push_back(bx);
        }

      Box new_domain(fakeAmr.geom[0].Domain());
      new_domain.refine(restart_grow_factor);
      newgrid_list.intersect(new_domain);

      newgrid_list.maxSize(dlenx);

   } else {
#endif

   if (body_at_center == 1 && restart_grow_factor == 2)
   {

#if (AMREX_SPACEDIM == 3)
   for (int jz = 0; jz < 3; jz++)
#endif
#if (AMREX_SPACEDIM >= 2)
   for (int jy = 0; jy < 3; jy++)
#endif
    for (int jx = 0; jx < 3; jx++)
      for (int n = 0; n < falRef0.grids.size(); n++) {
        int shiftx(jx*dlenx - dlenx/2);
#if (AMREX_SPACEDIM >= 2)
        int shifty(jy*dleny - dleny/2);
#if (AMREX_SPACEDIM == 3)
        int shiftz(jz*dlenz - dlenz/2);
#endif
#endif
        IntVect shift_box(AMREX_D_DECL(shiftx,shifty,shiftz));

        Box bx(falRef0.grids[n]);
        bx.shift(shift_box);
        newgrid_list.push_back(bx);
      }

      Box new_domain(fakeAmr.geom[0].Domain());
      new_domain.refine(restart_grow_factor);
      newgrid_list.intersect(new_domain);

   } else {

#if (AMREX_SPACEDIM == 3)
   for (int jz = 0; jz < restart_grow_factor; jz++)
#endif
#if (AMREX_SPACEDIM >= 2)
   for (int jy = 0; jy < restart_grow_factor; jy++)
#endif
    for (int jx = 0; jx < restart_grow_factor; jx++)
      for (int n = 0; n < falRef0.grids.size(); n++)
      {
        int shiftx(jx*dlenx);
#if (AMREX_SPACEDIM >= 2)
        int shifty(jy*dleny);
#if (AMREX_SPACEDIM == 3)
        int shiftz(jz*dlenz);
#endif
#endif
        IntVect shift_box(AMREX_D_DECL(shiftx,shifty,shiftz));

        Box bx(falRef0.grids[n]);
        bx.shift(shift_box);
        newgrid_list.push_back(bx);
      }

   }

#if (AMREX_SPACEDIM == 2)
   }
#endif

   BoxArray newgrids(newgrid_list);
   falRef0.grids = newgrids;

   int nstatetypes = falRef0.state.size();

   for (int n = 0; n < nstatetypes; n++)
      falRef0.state[n].grids = newgrids;

   RealBox rb(fakeAmr.geom[0].ProbDomain());

   if (body_at_center == 0)
   {
      for (int dm = 0; dm < AMREX_SPACEDIM; dm++)
         rb.setHi(dm,restart_grow_factor*rb.hi(dm));
   }

#if (AMREX_SPACEDIM == 2)
   else if (coord == 1)
   {
     int dm = 0;
     rb.setHi(dm,restart_grow_factor*rb.hi(dm));

     dm = 1;
     Real dist   = 0.5 * (rb.hi(dm)-rb.lo(dm));
     Real center = 0.5 * (rb.hi(dm)+rb.lo(dm));
     Real newlo = center - restart_grow_factor * dist;
     Real newhi = center + restart_grow_factor * dist;
     rb.setLo(dm,newlo);
     rb.setHi(dm,newhi);
   }
#endif

   else
   {
      for (int dm = 0; dm < AMREX_SPACEDIM; dm++)
      {
         Real dist   = 0.5 * (rb.hi(dm)-rb.lo(dm));
         Real center = 0.5 * (rb.hi(dm)+rb.lo(dm));
         Real newlo = center - restart_grow_factor * dist;
         Real newhi = center + restart_grow_factor * dist;
         rb.setLo(dm,newlo);
         rb.setHi(dm,newhi);
      }
   }

   Geometry::ResetDefaultProbDomain(rb);
   for (auto& g : fakeAmr.geom) {
       g.ProbDomain(rb);
   }
   for (auto& l : fakeAmr.fakeAmrLevels) {
       l.geom.ProbDomain(rb);
   }

   const Real* xlo = fakeAmr.geom[0].ProbLo();

   fakeAmr.geom[0].SetOffset(xlo);

   IntVect shift_iv[max_level+1];

   if (body_at_center == 1)
   {
      if (coord == 1)
      {
         for (int i = 0; i <= max_level; i++)
         {
            Box domain(fakeAmr.geom[i].Domain());
            shift_iv[i][0] = 0;
            shift_iv[i][1] = domain.size()[1] / 2;
         }
      } else if (coord == 0) {
         for (int i = 0; i <= max_level; i++)
         {
            Box domain(fakeAmr.geom[i].Domain());
            if (restart_grow_factor == 3) {
               shift_iv[i] = domain.size();
            } else if (restart_grow_factor == 2) {
               shift_iv[i] = domain.size() / 2;
            }
         }
      }
   }

   for (int i = 0; i <= max_level; i++)
   {
      Box domain(fakeAmr.geom[i].Domain());
      domain.refine(restart_grow_factor);
      fakeAmr.geom[i].Domain(domain);

      FakeAmrLevel &falRef = fakeAmr.fakeAmrLevels[i];
      falRef.geom.Domain(domain);
   }

   for (int i = 0; i <= max_level; i++)
   {
      FakeAmrLevel &falRef = fakeAmr.fakeAmrLevels[i];
      for (int n = 0; n < nstatetypes; n++)
         falRef.state[n].domain.refine(restart_grow_factor);
   }

   DistributionMapping newdm {newgrids};

   for (int n = 0; n < nstatetypes; n++)
   {
      if (falRef0.state[n].new_data != 0) {
         int ncomps = (falRef0.state[n].new_data)->nComp();
         MultiFab * newNewData = new MultiFab(newgrids,newdm,ncomps,1);

         newNewData->setVal(0.);

         if (body_at_center == 1)
            (falRef0.state[n].new_data)->shift(shift_iv[0]);

         falRef0.state[n].new_data = newNewData;

      }
   }

   for (int n = 0; n < nstatetypes; n++)
   {
      if (falRef0.state[n].old_data != 0) {
         int ncomps = (falRef0.state[n].old_data)->nComp();
         MultiFab * newOldData = new MultiFab(newgrids,newdm,ncomps,1);
         newOldData->setVal(0.);

         if (body_at_center == 1)
            (falRef0.state[n].old_data)->shift(shift_iv[0]);

         falRef0.state[n].old_data = newOldData;

      }
   }

   if (body_at_center == 1) {
      for (int i = 1; i <= max_level; i++)
      {
         FakeAmrLevel &falRef = fakeAmr.fakeAmrLevels[i];

         falRef.grids.shift(shift_iv[i]);

         for (int n = 0; n < nstatetypes; n++)
         {
            falRef.state[n].grids.shift(shift_iv[i]);

            if (falRef0.state[n].new_data != 0)
               (falRef.state[n].new_data)->shift(shift_iv[i]);
            if (falRef0.state[n].old_data != 0)
               (falRef.state[n].old_data)->shift(shift_iv[i]);
         }
      }
   }
}


static void ExtendThetaDomain() {

    int max_level = fakeAmr.finest_level;

    Box dom0 = fakeAmr.geom[0].Domain();
    RealBox rb0 = fakeAmr.geom[0].ProbDomain();

    int old_theta_hi = dom0.bigEnd()[1];
    Real theta_lo = rb0.lo()[1];
    Real theta_hi = rb0.hi()[1];
    int  ncells_old = dom0.size()[1];
    Real theta_cell_width = (theta_hi - theta_lo) / ncells_old;

    int theta_cells_requested = (int)std::round((theta_new_hi - theta_lo) / theta_cell_width);
    int theta_cells_new = blocking_factor *
        (int)std::ceil((Real)theta_cells_requested / blocking_factor);
    Real theta_snapped = theta_lo + theta_cells_new * theta_cell_width;

    if (ParallelDescriptor::IOProcessor()) {
        cout << "Requested theta_new_hi:    " << theta_new_hi   << " rad ("
             << theta_cells_requested << " cells)" << endl;
        cout << "Snapped  theta_new_hi:     " << theta_snapped  << " rad ("
             << theta_cells_new   << " cells, divisible by "
             << blocking_factor  << ")" << endl;
    }

    theta_new_hi = theta_snapped;

    for (int lev = 0; lev <= max_level; lev++) {

        FakeAmrLevel &falRef = fakeAmr.fakeAmrLevels[lev];
        Box old_domain = fakeAmr.geom[lev].Domain();
        RealBox old_rb = fakeAmr.geom[lev].ProbDomain();

        int refine_theta = 1;
        for (int l = 0; l < lev; l++) {
            refine_theta *= fakeAmr.ref_ratio[l][1];
        }

        int theta_cells_new_lev = theta_cells_new * refine_theta;
        int ncells_old_lev = old_domain.size()[1];
        int theta_cells_add_lev = theta_cells_new_lev - ncells_old_lev;

        Box new_domain = old_domain;
        new_domain.growHi(1, theta_cells_add_lev);

        RealBox new_rb = old_rb;
        new_rb.setHi(1, theta_snapped);

        fakeAmr.geom[lev].define(new_domain, &new_rb, coord);
        falRef.geom.define(new_domain, &new_rb, coord);

        if (lev == 0) {

            BoxArray new_ba(new_domain);

            new_ba.maxSize(IntVect(max_grid_size, max_grid_size));
            DistributionMapping new_dm(new_ba);
            falRef.grids = new_ba;

            int nstatetypes = falRef.state.size();
            for (int n = 0; n < nstatetypes; n++) {

                falRef.state[n].domain = new_domain;
                falRef.state[n].grids  = new_ba;

                if (falRef.state[n].new_data != 0) {

                    MultiFab* old_mf = falRef.state[n].new_data;
                    int ncomp = old_mf->nComp();
                    int ngrow = old_mf->nGrow();

                    MultiFab* new_mf = new MultiFab(new_ba, new_dm, ncomp, ngrow);

                    new_mf->setVal(0., 0, ncomp, ngrow);


                    new_mf->ParallelCopy(*old_mf, 0, 0, ncomp);

                    delete old_mf;
                    falRef.state[n].new_data = new_mf;
                }

                if (falRef.state[n].old_data != 0) {
                    MultiFab* old_mf = falRef.state[n].old_data;
                    int ncomp = old_mf->nComp();
                    int ngrow = old_mf->nGrow();

                    MultiFab* new_mf = new MultiFab(new_ba, new_dm, ncomp, ngrow);
                    new_mf->setVal(0., 0, ncomp, ngrow);
                    new_mf->ParallelCopy(*old_mf, 0, 0, ncomp);

                    delete old_mf;
                    falRef.state[n].old_data = new_mf;
                }
            }

            if (ParallelDescriptor::IOProcessor()) {
                cout << "Level 0 old amr.n_cell = " << old_domain.length(0)
                     << " " << old_domain.length(1) << endl;
                cout << "Level 0 old geometry.prob_hi = " << old_rb.hi()[0]
                     << " " << old_rb.hi()[1] << endl;

                cout << "Level 0 new amr.n_cell = " << new_domain.length(0)
                     << " " << new_domain.length(1) << endl;
                cout << "Level 0 new geometry.prob_hi = " << new_rb.hi()[0]
                     << " " << new_rb.hi()[1] << endl;
            }

        } else {

            int nstatetypes = falRef.state.size();
            for (int n = 0; n < nstatetypes; n++) {
                falRef.state[n].domain = new_domain;
            }

            if (ParallelDescriptor::IOProcessor()) {
                cout << "Level " << lev
                     << ": grids/data left untouched (old extent), "
                     << "domain metadata extended for consistency" << endl;
            }
        }

    }
}

int main(int argc, char *argv[]) {
    amrex::Initialize(argc,argv);

    if(argc < 3) {
      PrintUsage(argv[0]);
    }

    ScanArguments();

    if(verbose && ParallelDescriptor::IOProcessor()) {
      cout << " " << std::endl;
      cout << "Reading from old checkpoint file: " <<  CheckFileIn << endl;
      cout << " " << std::endl;
    }

    ReadCheckpointFile(CheckFileIn);

    if (mode == "full_hierarchy") {
        if (verbose && ParallelDescriptor::IOProcessor()) {
            if (body_at_center == 0) cout << "Star at corner " << endl;
            if (body_at_center == 1) cout << "Star at center " << endl;
            cout << " " << std::endl;
        }
        ConvertData();
    } else {
        ExtendThetaDomain();
        if (ParallelDescriptor::IOProcessor()) {
            cout << "Remember to update `amr.n_cell` and `geometry.prob_hi` to the new values "
                 << " also use `amr.regrid_on_restart=1` and set `hydrosim.prev_theta_cells` to the number of cells "
                 << " along the theta-dir in the original domain when restarting from new chkfile!!" << endl;
        }
    }

    WriteCheckpointFile(CheckFileIn, CheckFileOut);

    if(verbose && ParallelDescriptor::IOProcessor()) {
      cout << " " << std::endl;
      cout << "Finished writing to new checkpoint file: " <<  CheckFileOut << endl;
      cout << " " << std::endl;
    }

    amrex::Finalize();
}
