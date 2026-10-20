#include <AMReX_LO_BCTYPES.H>
#include <AMReX_ParmParse.H>
#include <Radiation.H>
#include <RadTransportSolver.H>
#include <radiation_helpers.H>

#include <opacity.H>

#include <format>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace amrex;

Radiation::Solver_Kind Radiation::SolverKind = Radiation::UnsetSolver;
Real Radiation::radiativeToEnergy = 0.;
Real Radiation::EnergyToRadFactor = 0.;
Real Radiation::fluxFromRad = 0.;
int Radiation::groups_enabled = 0;
int Radiation::nGroups = NGROUPS;
int Radiation::accelerate = 1;
int Radiation::rad_hydro_coupled = 0;
int Radiation::lorentz_energy_term = 1;
int Radiation::comp_limiter  = -1;
int Radiation::icomp_kappa_planck      = -1;
int Radiation::icomp_kappa_rosseland      = -1;
int Radiation::comp_lab_energy  = -1;
int Radiation::icomp_lab_flux  = -1;
int Radiation::icomp_comoving_flux  = -1;
int Radiation::num_plot_vars      = 0;
Vector<std::string>  Radiation::plot_var_names;
int Radiation::filter_limiter_T = 0;
int Radiation::limiter_filter_strength = 0;
int Radiation::filter_primitive_interval = 0;
int Radiation::filter_prim_temperature = 4;
int Radiation::filter_primitive_S = 0;
Real Radiation::mev_to_erg = 0.0;
Real Radiation::clight          = 0.0;
Real Radiation::planckConst         = 0.0;
Real Radiation::boltzmannConst          = 0.0;
Real Radiation::Avogadro        = 0.0;
Real Radiation::c               = 0.0;
Real Radiation::sigma           = 0.0;
Real Radiation::radConstant            = 0.0;
int Radiation::active_group_id = -1;
std::string Radiation::group_label = "Radiation";
int Radiation::hydro_only = 0;

Radiation::Radiation(Amr* Parent, HydroSim* hydrosim, int restart)
  : parent(Parent)
{

  ParmParse pp("radiation");

  apply_sync = 1; pp.query("apply_sync", apply_sync);

  {

    clight = C::c_light;
    planckConst = C::hplanck;
    boltzmannConst = C::k_B;
    Avogadro = C::n_A;
    mev_to_erg = 1.e6_rt * C::ev2erg;

    radConstant = 4.*C::sigma_SB / C::c_light;

    c        = clight;
    sigma    = C::sigma_SB;

    if (!groups_enabled) {
        pp.query("c", c);
        pp.query("sigma", sigma);
    }

    AlphaBetaSolver::fluxScale() = c;
    MultiLevelEllipticSolver::fluxScale() = c;

  }

  radiativeToEnergy = 1.0;
  EnergyToRadFactor = 1.0;
  fluxFromRad = 1.0;

  reltol   = 1.e-6;          pp.query("reltol", reltol);
  if (SolverKind == SingleGroupFLDSolver || SolverKind == MultiGroupFLDSolver) {
    abstol = 0.0;
  }
  else {
    abstol   = 1.e-6;
  }
  pp.query("abstol", abstol);
  maxiter  = 50;             pp.query("maxiter", maxiter);
  min_outer_iter  =  1;             pp.query("min_outer_iter", min_outer_iter);
  convergence_check_mode = 0;
  pp.query("convergence_check_mode", convergence_check_mode);
  if (SolverKind == SingleGroupFLDSolver && radiation::limiter == 1) {
    amrex::Abort("SingleGroupFLDSolver does not support limiter = 1");
  }
  if (SolverKind == MultiGroupFLDSolver && radiation::limiter == 1) {
    amrex::Abort("MultiGroupFLDSolver does not support limiter = 1");
  }

  inner_limiter_update_iters = 0;
  pp.query("inner_limiter_update_iters", inner_limiter_update_iters);

  opacity_lag_iter    = 1000;

  if (SolverKind == SingleGroupFLDSolver || SolverKind == MultiGroupFLDSolver) {
    planck_lag_iter     = 1000;
    rosseland_lag_iter  = 1000;
    limiter_lag_iter    = 1000;
  }
  else {
    planck_lag_iter     = 50;
    rosseland_lag_iter  = 50;
    limiter_lag_iter    = 4;
  }
  pp.query("planck_lag_iter", planck_lag_iter);
  pp.query("rosseland_lag_iter", rosseland_lag_iter);
  pp.query("opacity_lag_iter", opacity_lag_iter);
  pp.query("limiter_lag_iter", limiter_lag_iter);

  dT  = 1.0;                 pp.query("delta_temp", dT);

  relInnerTol = 1.e-4;          pp.query("relInnerTol", relInnerTol);
  if (SolverKind == SingleGroupFLDSolver || SolverKind == MultiGroupFLDSolver) {
    absInnerTolerance = 0.0;
  }
  else {
    absInnerTolerance = 1.e-4;
  }
  pp.query("absInnerTolerance", absInnerTolerance);
  maxInnerIter = 30;            pp.query("maxInnerIter", maxInnerIter);
  minInnerIter =  1;            pp.query("minInnerIter", minInnerIter);

  skipAccelIfNoGain = 0;
  pp.query("skipAccelIfNoGain", skipAccelIfNoGain);

  matter_update_mode = 0;
  pp.query("matter_update_mode", matter_update_mode);

  bisect_after = 1000;
  pp.query("bisect_after", bisect_after);
  dedT_safety = 1.0;
  pp.query("dedT_safety", dedT_safety);

  inner_convergence_mode = 2;
  pp.query("inner_convergence_mode", inner_convergence_mode);

  delta_e_rat_dt_tol = 100.0;
  pp.query("delta_e_rat_dt_tol", delta_e_rat_dt_tol);
  delta_T_rat_dt_tol = 100.0;
  pp.query("delta_T_rat_dt_tol", delta_T_rat_dt_tol);

  underrelax_factor = 1.0;    pp.query("underrelax_factor", underrelax_factor);

  use_WienForm = 0;
  pp.query("use_WienForm", use_WienForm);
  Wien_Temp_Limit = -1.0;
  pp.query("Wien_Temp_Limit", Wien_Temp_Limit);

  verbose  = 0;      pp.query("v", verbose);  pp.query("verbose", verbose);

  do_stimulated_emission = 0;
  pp.query("do_stimulated_emission", do_stimulated_emission);

  use_kappa_deriv = 0;
  pp.query("use_kappa_deriv", use_kappa_deriv);

  if (verbose > 2) {
    Vector<int> temp;
    if (pp.queryarr("spot",temp,0,AMREX_SPACEDIM)) {
      IntVect tempi(temp);
      spot = tempi;
    }
    if (ParallelDescriptor::IOProcessor()) std::cout << "Spot: " << spot << std::endl;
  }

  if (verbose > 0 && ParallelDescriptor::IOProcessor()) {
    std::cout << "Creating Radiation object" << std::endl;
  }
  if (verbose >= 1 && ParallelDescriptor::IOProcessor()) {
    std::cout << "processors = " << ParallelDescriptor::NProcs() << std::endl;
    std::cout << "apply_sync            = " << apply_sync << std::endl;

    std::cout << "c        = " << c << std::endl;
    std::cout << "sigma    = " << sigma << std::endl;
    std::cout << "reltol   = " << reltol << std::endl;
    std::cout << "abstol   = " << abstol << std::endl;
    std::cout << "maxiter  = " << maxiter << std::endl;
    std::cout << "relInnerTol = " << relInnerTol << std::endl;
    std::cout << "absInnerTolerance = " << absInnerTolerance << std::endl;
    std::cout << "maxInnerIter = " << maxInnerIter << std::endl;
    std::cout << "delta_e_rat_dt_tol = " << delta_e_rat_dt_tol << std::endl;
    std::cout << "delta_T_rat_dt_tol = " << delta_T_rat_dt_tol << std::endl;
    std::cout << "limiter  = " << radiation::limiter << std::endl;
    std::cout << "closure  = " << radiation::closure << std::endl;
    std::cout << "limiter_lag_iter   = " << limiter_lag_iter << std::endl;
    std::cout << "planck_lag_iter    = " << planck_lag_iter << std::endl;
    std::cout << "rosseland_lag_iter = " << rosseland_lag_iter << std::endl;
    std::cout << "delta_temp = " << dT << std::endl;
    std::cout << "underrelax_factor = " << underrelax_factor << std::endl;
    std::cout << "groups_enabled = " << groups_enabled << std::endl;
    std::cout << "accelerate = " << accelerate << std::endl;
    std::cout << "verbose  = " << verbose << std::endl;
    if (SolverKind == OneGroupSolver) {
      std::cout << "SolverKind = 0: OneGroupSolver " << std::endl;
    }
    else if (SolverKind == SingleGroupFLDSolver) {
      std::cout << "SolverKind = 5: SingleGroupFLDSolver " << std::endl;
    }
    else if (SolverKind == MultiGroupFLDSolver) {
      std::cout << "SolverKind = 6: MultiGroupFLDSolver " << std::endl;
    }
    if (SolverKind == MultiGroupFLDSolver || SolverKind == SingleGroupFLDSolver) {
      std::cout << "rad_hydro_coupled = " << rad_hydro_coupled << std::endl;
      std::cout << "moving_frame = " << radiation::moving_frame << std::endl;
    }
    if (SolverKind == MultiGroupFLDSolver) {
      std::cout << "frequency_advection_mode = " << radiation::frequency_advection_mode << std::endl;
    }
    if (SolverKind == SingleGroupFLDSolver && radiation::moving_frame == 0) {
      std::cout << "lorentz_energy_term = " << lorentz_energy_term << std::endl;
    }
  }

  if (groups_enabled) {
    build_groups(verbose);
  }
  else {
    xnu.resize(2, 1.0);
    freq_center.resize(1, 1.0);
  }

  Vector<int> lo_bc(AMREX_SPACEDIM), hi_bc(AMREX_SPACEDIM);
  pp.getarr("lo_bc",lo_bc,0,AMREX_SPACEDIM);
  pp.getarr("hi_bc",hi_bc,0,AMREX_SPACEDIM);
  for (int i = 0; i < AMREX_SPACEDIM; i++) {
    rad_boundary.setLo(i,lo_bc[i]);
    rad_boundary.setHi(i,hi_bc[i]);
    if (verbose > 1 && ParallelDescriptor::IOProcessor()) {
      std::cout << "dimension " << i << " rad boundary conditions = "
            << lo_bc[i] << ", " << hi_bc[i] << std::endl;
    }
  }

  int levels = parent->maxLevel() + 1;

  flux_accum_reg.resize(levels);
  conservative_register_old.resize(levels);
  flux_candidate_regs.resize(levels);

  flux_div.resize(levels);

  plot_fields.resize(levels);

  dt_previous.resize(levels, 0.0);

  delta_e_rat_level.resize(levels, 0.0);
  delta_T_rat_level.resize(levels, 0.0);

  pp.query("hydro_only", hydro_only);

}

void Radiation::load_static_settings()
{
  ParmParse pp("radiation");

  {
    int solver_type = Radiation::SolverKind;
    pp.get("SolverKind", solver_type);
    SolverKind = static_cast<Solver_Kind>(solver_type);
  }

  if (Radiation::SolverKind == Radiation::MultiGroupFLDSolver) {
    groups_enabled = 1;
  }
  else {
    groups_enabled = 0;
  }

  if (!groups_enabled && Radiation::nGroups > 1) {
      amrex::Error("Radiation::nGroups > 1 but using single group solver");
  }

  if (Radiation::SolverKind == Radiation::MultiGroupFLDSolver) {
    accelerate = 2;
  }
  pp.query("accelerate", accelerate);

  if (Radiation::SolverKind == Radiation::SingleGroupFLDSolver ||
      Radiation::SolverKind == Radiation::MultiGroupFLDSolver ) {
    Radiation::rad_hydro_coupled = 1;
    pp.query("rad_hydro_coupled", rad_hydro_coupled);
  }

  if (SolverKind == SingleGroupFLDSolver) {
    radiation::frequency_advection_mode = 1;
    if (radiation::moving_frame) {
      lorentz_energy_term = 0;
    }
    else {
      lorentz_energy_term = 1;
      pp.query("lorentz_energy_term", lorentz_energy_term);
    }
  }

  pp.query("filter_limiter_T", filter_limiter_T);
  limiter_filter_strength = filter_limiter_T - 1;
  pp.query("limiter_filter_strength", limiter_filter_strength);
  if (filter_limiter_T > 4) {
    amrex::Error("filter_limiter_T > 4");
  }
  if (filter_limiter_T < 0) {
    amrex::Error("filter_limiter_T < 0");
  }
  if (filter_limiter_T > 0) {
    if (limiter_filter_strength >= filter_limiter_T) {
      amrex::Error("Invalid limiter_filter_strength; S must be less than T when T > 0.");
    }
  }

  pp.query("filter_primitive_interval", filter_primitive_interval);
  pp.query("filter_prim_temperature", filter_prim_temperature);
  filter_primitive_S = filter_prim_temperature - 1;
  pp.query("filter_primitive_S", filter_primitive_S);
  if (filter_prim_temperature > 4) {
    amrex::Error("filter_prim_temperature > 4");
  }
  if (filter_prim_temperature < 0) {
    amrex::Error("filter_prim_temperature < 0");
  }
  if (filter_prim_temperature > 0) {
    if (filter_primitive_S >= filter_prim_temperature) {
      amrex::Error("Invalid filter_primitive_S; S must be less than T when T > 0.");
    }
  }

  if (Radiation::SolverKind == Radiation::MultiGroupFLDSolver) {

    Radiation::nGroups = NGROUPS;

    int configured_group_count = -1;
    pp.query("nGroups", configured_group_count);

    if (configured_group_count > 0 && configured_group_count != Radiation::nGroups) {
      amrex::Error("you set the number of groups at runtime, but this does not match the compiled value");
    }

  }
  else if (Radiation::SolverKind != Radiation::OneGroupSolver &&
           Radiation::SolverKind != Radiation::SingleGroupFLDSolver) {
      amrex::Error("Unknown Radiation::SolverKind");
  }

  {
      if (radiation::plot_flux_limiter) {
          comp_limiter = plot_var_names.size();

          if (!groups_enabled || radiation::limiter == 0) {
              plot_var_names.push_back("lambda");
          } else {
              for (int g=0; g<nGroups; ++g) {
                  std::ostringstream ss;
                  ss << "lambda" << g;
                  plot_var_names.push_back(ss.str());
              }
          }
      }
      if (radiation::plot_planck_opacity) {
          icomp_kappa_planck = plot_var_names.size();
          if (!groups_enabled) {
              plot_var_names.push_back("kappa_P");
          } else {
              for (int g=0; g<nGroups; ++g) {
                  std::ostringstream ss;
                  ss << "kappa_P" << g;
                  plot_var_names.push_back(ss.str());
              }
          }
      }
      if (radiation::plot_rosseland_opacity) {
          icomp_kappa_rosseland = plot_var_names.size();
          if (!groups_enabled) {
              plot_var_names.push_back("kappa_R");
          } else {
              for (int g=0; g<nGroups; ++g) {
                  std::ostringstream ss;
                  ss << "kappa_R" << g;
                  plot_var_names.push_back(ss.str());
              }
          }
      }
      if (radiation::plot_lab_rad_energy) {
          comp_lab_energy = plot_var_names.size();
          if (!groups_enabled) {
              plot_var_names.push_back("Erlab");
          } else {
              for (int g=0; g<nGroups; ++g) {
                  std::ostringstream ss;
                  ss << "Erlab" << g;
                  plot_var_names.push_back(ss.str());
              }
          }
      }
      if (radiation::plot_lab_rad_flux) {
          icomp_lab_flux = plot_var_names.size();
          std::string frame = "lab";
          Vector<std::string> axis_labels;
          axis_labels.push_back("x");
          axis_labels.push_back("y");
          axis_labels.push_back("z");
          if (!groups_enabled) {
              for (int idim=0; idim<AMREX_SPACEDIM; ++idim) {
                  std::ostringstream ss;
                  ss << "Fr" << frame << axis_labels[idim];
                  plot_var_names.push_back(ss.str());
              }
          } else {
              for (int idim=0; idim<AMREX_SPACEDIM; ++idim) {
                  for (int g=0; g<nGroups; ++g) {
                      std::ostringstream ss;
                      ss << "Fr" << frame << g << axis_labels[idim];
                      plot_var_names.push_back(ss.str());
                  }
              }
          }
      }
      if (radiation::plot_moving_flux) {
          icomp_comoving_flux = plot_var_names.size();
          std::string frame = "com";
          Vector<std::string> axis_labels;
          axis_labels.push_back("x");
          axis_labels.push_back("y");
          axis_labels.push_back("z");
          if (!groups_enabled) {
              for (int idim=0; idim<AMREX_SPACEDIM; ++idim) {
                  std::ostringstream ss;
                  ss << "Fr" << frame << axis_labels[idim];
                  plot_var_names.push_back(ss.str());
              }
          } else {
              for (int idim=0; idim<AMREX_SPACEDIM; ++idim) {
                  for (int g=0; g<nGroups; ++g) {
                      std::ostringstream ss;
                      ss << "Fr" << frame << g << axis_labels[idim];
                      plot_var_names.push_back(ss.str());
                  }
              }
          }
      }
      num_plot_vars = plot_var_names.size();
  }

  amrex::Print() << "radiation initialized, nGroups = " << Radiation::nGroups << std::endl;

}

void Radiation::post_init(int level)
{
    return;
}

void Radiation::regrid(int level, const BoxArray& grids, const DistributionMapping& dmap)
{
  BL_PROFILE("Radiation::Regrid");
  if (verbose > 0 && ParallelDescriptor::IOProcessor()) {
    std::cout << "Regridding radiation object at level " << level
         << "..." << std::endl;
  }
  if (level > 0) {
    IntVect crse_ratio = parent->refRatio(level-1);

    flux_accum_reg[level].reset(new FluxRegister(grids, dmap, crse_ratio, level, nGroups));
    flux_accum_reg[level]->setVal(0.0);

    flux_candidate_regs[level].reset(new FluxRegister(grids, dmap, crse_ratio, level, nGroups));
    flux_candidate_regs[level]->setVal(0.0);

  }

  flux_div[level].reset(new MultiFab(grids, dmap, 1, 0));

  if (num_plot_vars > 0) {
      plot_fields[level].reset(new MultiFab(grids, dmap, num_plot_vars, 0));
      plot_fields[level]->setVal(0.0);
  }

  if (verbose > 1 && ParallelDescriptor::IOProcessor()) {
    std::cout << "                                         done" << std::endl;
  }
}

void Radiation::begin_timestep(int level)
{
  BL_PROFILE("Radiation::begin_timestep");
  int fine_level = parent->finestLevel();
  int ncycle     = parent->nCycle(level);

  static int done = 0;
  if (level < fine_level) {

      int fine_lvl = level + 1;
      if (!flux_accum_reg[fine_lvl]) {
          const BoxArray& grids = parent->getLevel(fine_lvl).boxArray();
          const DistributionMapping& dmap = parent->getLevel(fine_lvl).DistributionMap();
          const IntVect& crse_ratio = parent->refRatio(level);
          flux_accum_reg[fine_lvl].reset(new FluxRegister(grids, dmap, crse_ratio, fine_lvl, nGroups));
          flux_accum_reg[fine_lvl]->setVal(0.0);
      }
  }

  int iteration = parent->levelSteps(level);
  iteration = (level > 0) ? iteration % ncycle : done;
  if (level < fine_level || iteration == 0) {
      reset_flux_records(level, ncycle);
      done = 1;
  }
}

void Radiation::select_active_group(int rad_group_idx)
{
  active_group_id = rad_group_idx;
}

void Radiation::restart(int level, const BoxArray& grids,
                        const DistributionMapping& dmap,
                        const std::string& dir, std::istream& is)
{

  std::string Path, stream_token;

  do {
    is >> stream_token;
    if (stream_token.find("delta_e_rat") == 0) {
      is >> delta_e_rat_level[level];
    }
    else if (stream_token.find("delta_T_rat") == 0) {
      is >> delta_T_rat_level[level];
    }
    else {
      Path = stream_token;
    }
  } while (Path.empty());

  std::string Flag;
  is >> Flag;

  if (Flag == "Present") {
    BL_ASSERT(level > 0);
    is >> dt_previous[level-1];
    std::string FullPathName = dir;
    if (!dir.empty() && dir[dir.length()-1] != '/')
        FullPathName += '/';
    FullPathName += Path;
    const IntVect& crse_ratio = parent->refRatio(level-1);
    conservative_register_old[level].reset(new FluxRegister(grids, dmap, crse_ratio, level, nGroups));
    conservative_register_old[level]->read(FullPathName, is);
  }
}

void Radiation::checkPoint(int level,
                           const std::string& dir,
                           std::ostream&  os,
                           VisMF::How     how)
{

  char buf[64];

  if (ParallelDescriptor::IOProcessor()) {
    int oldprec = os.precision(20);
    auto line_label = std::format("delta_e_rat_level[{}]= ", level);
    os << line_label << delta_e_rat_level[level] << '\n';
    line_label = std::format("delta_T_rat_level[{}]= ", level);
    os << line_label << delta_T_rat_level[level] << '\n';
    os.precision(oldprec);
  }

  std::string Level = std::format("Level_{}", level);

  std::string PathNameInHeader = Level;
  PathNameInHeader += "/RadFlux";
  if (ParallelDescriptor::IOProcessor()) {
    os << PathNameInHeader;
  }

  if (conservative_register_old[level]) {
    if (ParallelDescriptor::IOProcessor()) {
      BL_ASSERT(level > 0);
      int oldprec = os.precision(20);
      os << " Present  " << dt_previous[level-1] << '\n';
      os.precision(oldprec);
    }
    std::string FullPathName = dir;
    if (!FullPathName.empty() &&
        FullPathName[FullPathName.length()-1] != '/') {
      FullPathName += '/';
    }
    FullPathName += Level;
    FullPathName += "/RadFlux";
    conservative_register_old[level]->write(FullPathName, os   );
  }
  else {
    if (ParallelDescriptor::IOProcessor()) {
      os << " Absent\n";
    }
  }
}

void Radiation::close(int level)
{
  if (level > parent->finestLevel()) {
    if (verbose > 0 && ParallelDescriptor::IOProcessor()) {
      std::cout << "Clearing radiation object at level " << level
            << "..." << std::endl;
    }
    flux_accum_reg[level].reset();
    flux_candidate_regs[level].reset();

    flux_div[level].reset();

    plot_fields[level].reset();

    if (verbose > 1 && ParallelDescriptor::IOProcessor()) {
      std::cout << "                                       done" << std::endl;
    }

    BL_ASSERT(level > 0);
    int clev = level - 1;

    if (clev == parent->finestLevel()) {
      int ncycle    = parent->nCycle(clev);
      int iteration = parent->levelSteps(clev);
      iteration = (clev > 0) ? iteration % ncycle : iteration;
      if (iteration > 0) {
        reset_flux_records(clev, ncycle);
      }
    }
  }
}
