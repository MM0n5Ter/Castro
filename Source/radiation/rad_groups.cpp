#ifndef RADIATION
#define RADIATION
#endif

#include <Radiation.H>

#include <AMReX_ParmParse.H>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace amrex;

void Radiation::dump_group_table(ostream& os)
{
  os << "# total number of groups = " << nGroups << std::endl;

  if (nGroups > 1) {
    os << "# group center, group weight" << group_unit_label << std::endl;
    int oldprec = os.precision(10);
    for (int i = 0; i < nGroups; i++) {
      os.width(3);
      os << i << ": ";
      os.width(15);
      os <<  freq_center[i] * group_output_scale << ", ";
      os.width(15);
      os << freq_width[i] * group_output_scale << std::endl;
    }
    os.precision(oldprec);
  }

  if (xnu.size() > 0) {
    os << "# group lower boundaries" << std::endl;
    for (int i = 0; i < xnu.size(); i++) {
      os << "group(" << i << ") = "
         << xnu[i] * group_output_scale << std::endl;
    }
  }
}

void Radiation::build_groups(int verbose)
{
  group_output_scale   = 1.0;
  group_unit_label          = " (units are Hz)";

  xnu.resize(nGroups+1, 0.0);
  freq_center.resize(nGroups, 1.0);
  freq_width.resize(nGroups, 0.0);
  log_group_center.resize(nGroups, 0.0);
  log_group_widths.resize(nGroups, 0.0);

  ParmParse pp("radiation");

  if (nGroups > 1) {
    Real lowest;
    pp.get("lowestGroupHz", lowest);

    Real groupGrowthRatio;
    if (pp.query("groupGrowthRatio", groupGrowthRatio)) {
      xnu[0] = lowest;

      Real initialGroupWidthHz;
      pp.get("initialGroupWidthHz", initialGroupWidthHz);
      freq_width[0] = initialGroupWidthHz;

      xnu[1] = xnu[0] + freq_width[0];

      if (lowest == 0.0) {
        freq_center[0] = 0.5*freq_width[0];
        log_group_widths[0] = 2.0 * (std::log(xnu[1]) - std::log(freq_center[0]));
      }
      else {
        freq_center[0] = std::sqrt(xnu[0]*xnu[1]);
        log_group_widths[0] = std::log(xnu[1]) - std::log(xnu[0]);
      }

      for (int i=1; i<nGroups; i++) {
        freq_width[i] = freq_width[i-1] * groupGrowthRatio;
        xnu[i+1] = xnu[i] + freq_width[i];
        freq_center[i] = std::sqrt(xnu[i]*xnu[i+1]);
        log_group_widths[i] = std::log(xnu[i+1]) - std::log(xnu[i]);
      }
    }
    else {
      Real highest;
      pp.get("highestGroupHz", highest);

      Real loglowbound = std::log10(lowest);
      Real loghighbound = std::log10(highest);
      Real log_group_width = (loghighbound - loglowbound) / Real(nGroups);

      for (int i=0; i<nGroups; i++) {
        xnu[i] = std::pow(10.0, loglowbound+i*log_group_width);
        freq_center[i] = std::pow(10.0, loglowbound+(i+0.5)*log_group_width);
      }
      xnu[nGroups] = highest;

      for (int i=0; i<nGroups; i++) {
        freq_width[i] = xnu[i+1] - xnu[i];
        log_group_widths[i] = std::log(xnu[i+1]) - std::log(xnu[i]);
      }
    }

    for (int i = 0; i < nGroups; ++i) {
        log_group_center[i] = std::log(freq_center[i]);
    }
  }

  if (ParallelDescriptor::IOProcessor()) {

    {
        std::ofstream energy_group_file("group_structure.dat");

        energy_group_file << "# total number of groups = " << nGroups << std::endl;
        energy_group_file << "# group center, group weight" << group_unit_label << std::endl;

        energy_group_file.precision(10);
        for (int i = 0; i < nGroups; i++) {
          energy_group_file.width(15);
          energy_group_file <<  freq_center[i] * group_output_scale << "\t";
          energy_group_file.width(15);
          energy_group_file << freq_width[i] * group_output_scale << std::endl;
        }
        energy_group_file << std::endl;

        if (xnu.size() > 0) {
          energy_group_file << "# group lower boundaries" << std::endl;
          for (int i = 0; i < xnu.size(); i++) {
            energy_group_file << "group(" << i << ") = "
                      << xnu[i] * group_output_scale << std::endl;
          }
        }

        if (log_group_widths.size() > 0) {
          energy_group_file << std::endl;
          energy_group_file << "# group width in log space" << std::endl;
          for (int i = 0; i < log_group_widths.size(); i++) {
            energy_group_file << "group(" << i << ") = "
                      << log_group_widths[i] << std::endl;
          }
        }
    }

    if (verbose >= 1) {
      dump_group_table(std::cout);
    }
  }
}
