#ifndef _Radiation_utils_H_
#define _Radiation_utils_H_
#include <AMReX_BLFort.H>
#include <AMReX_DataServices.H>
#include <regex>

using namespace amrex;

template <typename T>
Vector<size_t> sort_indexes(const Vector<T> &v);

void GetInputArgs ( const int argc, char** argv,
                    string& pltfile, string& slcfile, int& dir);

Vector<int> GetComponents(AmrData& data, const Vector<std::string> varNames);

void PrintHelp ();

string GetVarFromJobInfo (const string pltfile, const string varname);

Vector<Real> GetCenter (const string pltfile);


template <typename T>
Vector<size_t> sort_indexes(const Vector<T> &v) {

    Vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {
        return v[i1] < v[i2];
    });

    return idx;
}

void GetInputArgs ( const int argc, char** argv,
                    string& pltfile, string& slcfile, int& dir)
{

    int i = 1;

    while ( i < argc)
    {

        if ( !strcmp(argv[i], "-p") || !strcmp(argv[i],"--pltfile") )
        {
            pltfile = argv[++i];
        }
        else if ( !strcmp(argv[i], "-s") || !strcmp(argv[i],"--slicefile") )
        {
            slcfile = argv[++i];
        }
        else if ( !strcmp(argv[i], "-d") || !strcmp(argv[i],"--direction") )
        {
            dir = std::atoi(argv[++i]);
        }
        else
        {
            std::cout << "\n\nOption " << argv[i] << " not recognized" << std::endl;
            PrintHelp ();
            exit ( EXIT_FAILURE );
        }

        ++i;
    }

    if (pltfile.empty() && slcfile.empty())
    {
        PrintHelp();
        Abort("Missing input file");
    }

    Print() << "\nplotfile  = \"" << pltfile << "\"" << std::endl;
    Print() << "slicefile = \"" << slcfile << "\"" << std::endl;
    Print() << std::endl;
}

Vector<int> GetComponents(AmrData& data, const Vector<std::string> varNames) {

    Vector<int> varComps;

    for (auto it=varNames.begin(); it!=varNames.end(); ++it)
        varComps.push_back(data.StateNumber(*it));

    return varComps;

}

void WriteSlicefile(const int nbins, const Vector<Real> r,
                    const Vector<std::string> varNames,
                    Vector<Vector<Real> > vars,
                    const std::string slcfile) {

    std::ofstream slicefile;
    slicefile.open(slcfile);
    slicefile.setf(std::ios::scientific);
    slicefile.precision(12);
    const auto w = 24;

    slicefile << std::setw(w) << "x";
    for (auto it=varNames.begin(); it!=varNames.end(); ++it)
        slicefile << std::setw(w) << *it;

    slicefile << std::endl;

    const auto SMALL = 1.e-20;
    for (auto i = 0; i < nbins; i++) {

        slicefile << std::setw(w) << r[i];

        for (auto it=vars.begin(); it!=vars.end(); ++it) {
            if(std::abs((*it)[i]) < SMALL) (*it)[i] = 0.0;
            slicefile << std::setw(w) << (*it)[i];
        }

        slicefile << std::endl;

    }

    slicefile.close();
}


void PrintHelp ()
{
    Print() << "\nusage: executable_name args"
            << "\nargs [-p|--pltfile]     plotfile : plot file directory (required)"
            << "\n     [-s|--slicefile] slice file : slice file          (required)"
            << "\n     [--xctr]               xctr : central x coord     (non-cartesian only)"
            << "\n     [--yctr]               yctr : central y coord     (non-cartesian only)"
            << "\n     [--zctr]               zctr : central z coord     (non-cartesian only)"
            << "\n\n" << std::endl;

}

string GetVarFromJobInfo (const string pltfile, const string varname) {
    string filename = pltfile + "/job_info";
    std::regex re("(?:[ \\t]*)" + varname + "\\s*:\\s*(.*)\\s*\\n");

    std::smatch m;

    std::ifstream jobfile(filename);
    if (jobfile.is_open()) {
        std::stringstream buf;
        buf << jobfile.rdbuf();
        string file_contents = buf.str();

        if (std::regex_search(file_contents, m, re)) {
            return m[1];
        } else {
            Print() << "Unable to find " << varname << " in job_info file!" << std::endl;
        }
    } else {
        Print() << "Could not open job_info file!" << std::endl;
    }

    return "";
}

Vector<Real> GetCenter (const string pltfile) {
    auto center_str = GetVarFromJobInfo(pltfile, "center");

    std::istringstream iss {center_str};
    Vector<Real> center;

    std::string s;
    while (std::getline(iss, s, ','))
        center.push_back(stod(s));

    return center;
}

#endif
