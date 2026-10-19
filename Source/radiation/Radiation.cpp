#include <AMReX_LO_BCTYPES.H>
#include <AMReX_ParmParse.H>
#include <Radiation.H>
#include <RadTransportSolver.H>
#include <radiation_helpers.H>
#include <primitive_filter_cell.H>

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

void Radiation::post_init(int level)
{
    return;
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

void Radiation::reset_flux_records(int level, int ncycle)
{
  BL_PROFILE("Radiation::reset_flux_records");
  if (verbose > 0 && ParallelDescriptor::IOProcessor()) {
    std::cout << "Radiation flux initialization at level " << level
         << "..." << std::endl;
  }

  int fine_level = parent->finestLevel();

  flux_div[level]->setVal(0.0);

  if (level < fine_level) {
      flux_accum_reg[level+1]->setVal(0.0);
  }

  if (verbose > 1 && ParallelDescriptor::IOProcessor()) {
    std::cout << "                                           done" << std::endl;
  }
}



void Radiation::evaluate_energy_transfer(MultiFab& exch,
                                 MultiFab& Er,
                                 MultiFab& fkp, int rad_group_idx)
{
    BL_PROFILE("Radiation::evaluate_energy_transfer");

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(exch, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto energy_exchange_arr = exch[mfi].array();
        auto rad_eden_arr = Er[mfi].array();
        auto scaled_planck_arr = fkp[mfi].array();

        Real local_sb_const = sigma;
        Real lc = c;

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            energy_exchange_arr(i,j,k) = scaled_planck_arr(i,j,k) * (4.e0_rt * local_sb_const * std::pow(energy_exchange_arr(i,j,k), 4) - lc * rad_eden_arr(i,j,k));
        });
    }
}

void Radiation::compute_coupling_factors(MultiFab& eta, MultiFab& coupling_inverse,
                            MultiFab& state, MultiFab& temp,
                            MultiFab& fkp, MultiFab& Er,
                            Real delta_t, Real c,
                            Real underrelax_scale, int freeze_planck, int rad_group_idx)
{
    BL_PROFILE("Radiation::compute_coupling_factors");

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        FArrayBox c_v;

        for (MFIter mfi(eta, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

            const Box& bx = mfi.tilebox();

            if (freeze_planck) {

                Array4<Real> const coupling_arr = eta.array(mfi);
                Array4<Real> const scaled_planck_arr = fkp.array(mfi);
                AMREX_PARALLEL_FOR_3D(bx, i, j, k,
                                      { coupling_arr(i,j,k) = scaled_planck_arr(i,j,k); });

            }
            else {


                auto coupling_arr = eta[mfi].array();
                auto state_arr = state[mfi].array();

                const Real nu = freq_center[rad_group_idx];
                const Real local_temp_step = dT;

                amrex::ParallelFor(bx,
                [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
                {
                    Real rho = state_arr(i,j,k,URHO);
                    Real temp = state_arr(i,j,k,UTEMP) + local_temp_step;
                    Real Ye;
                    if (NumAux > 0)  {
                        Ye = state_arr(i,j,k,UFX);
                    } else {
                        Ye = 0.e0_rt;
                    }

                    Real kp, kr;
                    bool need_planck_opac = true;
                    bool need_ross_opac = false;
                    opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

                    coupling_arr(i,j,k,rad_group_idx) = kp;
                });

            }

            c_v.resize(bx);
            Elixir heat_capacity_hold = c_v.elixir();

            get_heat_capacity(c_v, temp[mfi], state[mfi], bx);

            auto coupling_arr = eta[mfi].array();
            auto coupling_inverse_arr = coupling_inverse[mfi].array();
            auto fluid_density_arr = state[mfi].array(URHO);
            auto temp_field_arr = temp[mfi].array();
            auto heat_capacity_arr = c_v.array();
            auto scaled_planck_arr = fkp[mfi].array();
            auto rad_eden_arr = Er[mfi].array(rad_group_idx);

            Real local_temp_step = dT;

            const Real fac1 = 16.e0_rt * sigma * delta_t;
            const Real fac0 = 0.25e0_rt * fac1 / dT;
            const Real fac2 = delta_t * c / dT;

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                Real d;

                if (freeze_planck != 0)
                {
                    d = fac1 * scaled_planck_arr(i,j,k) * std::pow(temp_field_arr(i,j,k), 3);
                }
                else
                {
                    d = fac0 * (coupling_arr(i,j,k) * std::pow(temp_field_arr(i,j,k) + local_temp_step, 4) -
                                scaled_planck_arr(i,j,k) * std::pow(temp_field_arr(i,j,k), 4)) -
                        fac2 * (coupling_arr(i,j,k) - scaled_planck_arr(i,j,k)) * rad_eden_arr(i,j,k);
                }

                Real frc = fluid_density_arr(i,j,k) * heat_capacity_arr(i,j,k) + 1.0e-50_rt;
                coupling_arr(i,j,k) = d / (d + frc);
                coupling_inverse_arr(i,j,k) = underrelax_scale * frc / (d + frc);
                coupling_arr(i,j,k) = 1.e0_rt - coupling_inverse_arr(i,j,k);
            });
        }
    }
}

void Radiation::apply_implicit_energy(Real& relative, Real& absolute,
                                       MultiFab& state_energy_density,
                                       MultiFab& matter_energy_density,
                                       MultiFab& eta,
                                       MultiFab& coupling_inverse,
                                       MultiFab& flux_div_old,
                                       MultiFab& flux_div_new,
                                       MultiFab& exch,
                                       Real delta_t)
{
  BL_PROFILE("Radiation::apply_implicit_energy");

  ReduceOps<ReduceOpMax, ReduceOpMax> reduce_op;
  ReduceData<Real, Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

  Real theta = 1.0;

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(eta, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      const auto coupling_arr = eta[mfi].array();
      const auto coupling_inverse_arr = coupling_inverse[mfi].array();
      const auto density_matter_energy_arr = matter_energy_density[mfi].array();
      const auto energy_exchange_arr = exch[mfi].array();
      const auto dfo = flux_div_old[mfi].array();
      const auto dfn = flux_div_new[mfi].array();
      auto density_state_energy_arr = state_energy_density[mfi].array();

      reduce_op.eval(bx, reduce_data,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
      {
          Real chg = 0.e0_rt;
          Real tot = 0.e0_rt;

          Real tmp = coupling_arr(i,j,k) * density_state_energy_arr(i,j,k) +
                     coupling_inverse_arr(i,j,k) *
                     (density_matter_energy_arr(i,j,k) -
                      delta_t * ((1.e0_rt - theta) *
                                 (dfo(i,j,k) - dfn(i,j,k)) +
                                 energy_exchange_arr(i,j,k)));

          chg = std::abs(tmp - density_state_energy_arr(i,j,k));
          tot = std::abs(density_state_energy_arr(i,j,k));

          density_state_energy_arr(i,j,k) = tmp;

          Real abs_residual = chg;
          Real relative_residual = chg / (tot + 1.e-50_rt);

          return {relative_residual, abs_residual};
      });
  }

  ReduceTuple hv = reduce_data.value();

  relative = amrex::get<0>(hv);
  absolute = amrex::get<1>(hv);

  ParallelDescriptor::ReduceRealMax(relative);
  ParallelDescriptor::ReduceRealMax(absolute);
}

void Radiation::apply_implicit_energy(Real& relative, Real& absolute,
                                       MultiFab& state_energy_density,
                                       MultiFab& matter_energy_density,
                                       MultiFab& eta,
                                       MultiFab& coupling_inverse,
                                       MultiFab& flux_div_old,
                                       MultiFab& flux_div_new,
                                       MultiFab& exch,
                                       MultiFab& diffusive_face_term,
                                       Real delta_t)
{
  BL_PROFILE("Radiation::internal_energy_update_d");

  relative = 0.0;
  absolute = 0.0;

  ReduceOps<ReduceOpMax, ReduceOpMax> reduce_op;
  ReduceData<Real, Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

  const Real theta = 1.0;
  const Real tiny = 1.e-50_rt;

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(eta,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box &reg = mfi.tilebox();

      const auto coupling_arr = eta[mfi].array();
      const auto coupling_inverse_arr = coupling_inverse[mfi].array();
      const auto density_matter_energy_arr = matter_energy_density[mfi].array();
      const auto energy_exchange_arr = exch[mfi].array();
      const auto dfo = flux_div_old[mfi].array();
      const auto dfn = flux_div_new[mfi].array();
      const auto diffusion_term_arr = diffusive_face_term[mfi].array();
      auto density_state_energy_arr = state_energy_density[mfi].array();

      reduce_op.eval(reg, reduce_data,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
      {

          Real chg = 0.e0_rt;
          Real tot = 0.e0_rt;

          Real tmp = coupling_arr(i,j,k) * density_state_energy_arr(i,j,k) +
              coupling_inverse_arr(i,j,k) * (density_matter_energy_arr(i,j,k) -
                                   delta_t * ((1.e0_rt - theta) *
                                              (dfo(i,j,k) - dfn(i,j,k)) +
                                              energy_exchange_arr(i,j,k))) +
              delta_t * diffusion_term_arr(i,j,k);

          chg = std::abs(tmp - density_state_energy_arr(i,j,k));
          tot = std::abs(density_state_energy_arr(i,j,k));

          density_state_energy_arr(i,j,k) = tmp;

          Real abs_residual = chg;
          Real relative_residual = chg / (tot + tiny);

          return {relative_residual, abs_residual};

      });

  }

  ReduceTuple hv = reduce_data.value();

  relative = amrex::get<0>(hv);
  absolute = amrex::get<1>(hv);

  ParallelDescriptor::ReduceRealMax(relative);
  ParallelDescriptor::ReduceRealMax(absolute);

}

void Radiation::apply_nonconservative_correction(Real& relative, Real& absolute,
                                              MultiFab& state_energy_density,
                                              MultiFab& matter_energy_density,
                                              MultiFab& eta,
                                              MultiFab& coupling_inverse,
                                              MultiFab& rad_energy_new,
                                              MultiFab& flux_div_old,
                                              MultiFab& flux_div_new,
                                              MultiFab& temp,
                                              MultiFab& fkp,
                                              MultiFab& state,
                                              Real delta_t)
{
  BL_PROFILE("Radiation::apply_nonconservative_correction");

  ReduceOps<ReduceOpMax, ReduceOpMax> reduce_op;
  ReduceData<Real, Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

  const Real theta = 1.0;
  const Real tiny = 1.e-50_rt;

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
      FArrayBox c_v;

      for (MFIter mfi(eta,true); mfi.isValid(); ++mfi) {
          const Box &reg = mfi.tilebox();

          c_v.resize(reg);
          Elixir hold_heat_capacity = c_v.elixir();

          get_heat_capacity(c_v, temp[mfi], state[mfi], reg);

          const auto state_arr = state[mfi].array();
          const auto temp_field_arr = temp[mfi].array();
          const auto scaled_planck_arr = fkp[mfi].array();
          const auto rad_energy_arr = rad_energy_new[mfi].array();
          const auto coupling_arr = eta[mfi].array();
          const auto coupling_inverse_arr = coupling_inverse[mfi].array();
          const auto density_matter_energy_arr = matter_energy_density[mfi].array();
          const auto density_state_energy_arr = state_energy_density[mfi].array();
          const auto flux_div_old_view = flux_div_old[mfi].array();
          const auto flux_div_new_arr = flux_div_new[mfi].array();
          const auto heat_capacity_arr = c_v.array();

          reduce_op.eval(reg, reduce_data,
          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
          {

              Real chg = 0.e0_rt;
              Real tot = 0.e0_rt;

              Real density_heat_capacity = state_arr(i,j,k,URHO) * heat_capacity_arr(i,j,k);

              Real planck_temp_slope = 16.0_rt * C::sigma_SB * std::pow(temp_field_arr(i,j,k), 3);
              Real b = 4.0_rt * C::sigma_SB * std::pow(temp_field_arr(i,j,k), 4);
              Real exch = scaled_planck_arr(i,j,k) * (b - C::c_light * rad_energy_arr(i,j,k));
              Real tmp = coupling_arr(i,j,k) * density_state_energy_arr(i,j,k) +
                  coupling_inverse_arr(i,j,k) * (density_matter_energy_arr(i,j,k) -
                                       delta_t * ((1.0_rt - theta) * (flux_div_old_view(i,j,k) - flux_div_new_arr(i,j,k)) + exch));

              if (density_heat_capacity > tiny && tmp > density_state_energy_arr(i,j,k)) {
                  Real db = (tmp - density_state_energy_arr(i,j,k)) * planck_temp_slope / density_heat_capacity;
                  tmp = std::pow((b + db) / (4.0_rt * C::sigma_SB), 0.25_rt);
                  tmp = density_state_energy_arr(i,j,k) + density_heat_capacity * (tmp - temp_field_arr(i,j,k));
              }

              chg = std::abs(tmp - density_state_energy_arr(i,j,k));
              tot = std::abs(density_state_energy_arr(i,j,k));
              density_state_energy_arr(i,j,k) = tmp;

              Real abs_residual = chg;
              Real relative_residual = chg / (tot + tiny);

              return {abs_residual, relative_residual};
          });

      }

  }

  ReduceTuple hv = reduce_data.value();

  absolute = amrex::get<0>(hv);
  relative = amrex::get<1>(hv);

  ParallelDescriptor::ReduceRealMax(relative);
  ParallelDescriptor::ReduceRealMax(absolute);
}

void Radiation::sync_state_from_energy(MultiFab& state, MultiFab& state_energy_density)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto state_arr = state[mfi].array();
        auto density_state_energy_arr = state_energy_density[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Real kin = state_arr(i,j,k,UEDEN) - state_arr(i,j,k,UEINT);
            state_arr(i,j,k,UEINT) = density_state_energy_arr(i,j,k);
            state_arr(i,j,k,UEDEN) = density_state_energy_arr(i,j,k) + kin;


            if (state_arr(i,j,k,UEINT) <= 0.e0_rt)
            {
                density_state_energy_arr(i,j,k) = small_temp;
            }
            else
            {
                Real rhoInv = 1.e0_rt / state_arr(i,j,k,URHO);

                eos_re_t eos_state;
                eos_state.rho = state_arr(i,j,k,URHO);
                eos_state.T   = state_arr(i,j,k,UTEMP);
                eos_state.e   = state_arr(i,j,k,UEINT) * rhoInv;
                for (int n = 0; n < NumSpec; ++n) {
                    eos_state.xn[n] = state_arr(i,j,k,UFS+n) * rhoInv;
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; ++n) {
                    eos_state.aux[n] = state_arr(i,j,k,UFX+n) * rhoInv;
                }
#endif

                eos(eos_input_re, eos_state);

                density_state_energy_arr(i,j,k) = eos_state.T;

                state_arr(i,j,k,UTEMP) = density_state_energy_arr(i,j,k);
            }
        });
    }
}


void Radiation::extendBorderValues(MultiFab& f, int indx)
{
    BL_PROFILE("Radiation::extendBorderValues");

    BL_ASSERT(f.nGrow() >= 1);

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(f); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const Box& grown_valid_box = amrex::grow(bx, 1);

        Array4<Real> const face_flux_arr = f[mfi].array(indx);

        amrex::LoopOnCpu(grown_valid_box, [=] (int i, int j, int k) noexcept
        {

            if (i == bx.smallEnd(0)) {
                face_flux_arr(i-1,j,k) = 2.0_rt * face_flux_arr(i,j,k) - face_flux_arr(i+1,j,k);
            }

            if (i == bx.bigEnd(0)) {
                face_flux_arr(i+1,j,k) = 2.0_rt * face_flux_arr(i,j,k) - face_flux_arr(i-1,j,k);
            }

#if AMREX_SPACEDIM >= 2
            if (j == bx.smallEnd(1)) {
                face_flux_arr(i,j-1,k) = 2.0_rt * face_flux_arr(i,j,k) - face_flux_arr(i,j+1,k);
            }

            if (j == bx.bigEnd(1)) {
                face_flux_arr(i,j+1,k) = 2.0_rt * face_flux_arr(i,j,k) - face_flux_arr(i,j-1,k);
            }
#endif

#if AMREX_SPACEDIM == 3
            if (k == bx.smallEnd(2)) {
                face_flux_arr(i,j,k-1) = 2.0_rt * face_flux_arr(i,j,k) - face_flux_arr(i,j,k+1);
            }

            if (k == bx.bigEnd(2)) {
                face_flux_arr(i,j,k+1) = 2.0_rt * face_flux_arr(i,j,k) - face_flux_arr(i,j,k-1);
            }
#endif
        });
    }
}


void Radiation::fetchBndryData(RadiativeBoundary& bd, MultiFab& Er,
                             Real time, int level)
{
  BL_PROFILE("Radiation::fetchBndryData");
  HydroSim      *hydrosim = (HydroSim*)&parent->getLevel(level);
  const BoxArray& grids = hydrosim->boxArray();
  const DistributionMapping& dmap = hydrosim->DistributionMap();

  if (level == 0) {
    bd.setBndryValues(Er, Rad, 0, 1, rad_boundary);
  }
  else {
    BoxArray coarse_boxes(grids);
    IntVect crse_ratio = parent->refRatio(level-1);
    coarse_boxes.coarsen(crse_ratio);
    BndryRegister crse_br(coarse_boxes, dmap, 0, 1, 1, 1);
    crse_br.setVal(1.0e30);
    fillBndryRegister(crse_br, level-1, time);

    bd.setBndryValues(crse_br, 0, Er, Rad, 0, 1, crse_ratio, rad_boundary);
  }


  bd.setTime(time);

  bd.setBoundaryFluxConditions(rad_boundary);
}

void Radiation::fetchMultigridBndry(MultiGroupBoundary& multigroup_boundary, MultiFab& Er,
                               Real time, int level)
{
  BL_PROFILE("Radiation::fetchMultigridBndry");
  HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
  const BoxArray& grids = hydrosim->boxArray();
  const DistributionMapping& dmap = hydrosim->DistributionMap();

  if(level == 0) {
    multigroup_boundary.setBndryValues(Er, 0, 0, Radiation::nGroups, rad_boundary);
  }
  else {
    BoxArray coarse_boxes(grids);
    IntVect crse_ratio = parent->refRatio(level-1);
    coarse_boxes.coarsen(crse_ratio);
    BndryRegister crse_br(coarse_boxes, dmap, 0, 1, 1, Radiation::nGroups);
    crse_br.setVal(1.0e30);
    fillBndryRegister(crse_br, level-1, time);

    multigroup_boundary.setBndryValues(crse_br, 0, Er, 0,
                        0, Radiation::nGroups, crse_ratio, rad_boundary);
  }

  multigroup_boundary.setTime(time);
  multigroup_boundary.setBoundaryFluxConditions(rad_boundary);
}

void Radiation::fetchMultigridBndryZeroRad(MultiGroupBoundary& multigroup_boundary, MultiFab& Er, int level)
{
  BL_PROFILE("Radiation::fetchMultigridBndryZeroRad");
  HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
  const BoxArray& grids = hydrosim->boxArray();
  const DistributionMapping& dmap = hydrosim->DistributionMap();

  if(level == 0) {
    multigroup_boundary.setBndryValues(Er, 0, 0, 1, rad_boundary);
  }
  else {
    BoxArray coarse_boxes(grids);
    IntVect crse_ratio = parent->refRatio(level-1);
    coarse_boxes.coarsen(crse_ratio);
    BndryRegister crse_br(coarse_boxes, dmap, 0, 1, 1, 1);
    crse_br.setVal(0.0);

    multigroup_boundary.setBndryValues(crse_br, 0, Er, 0,
                        0, 1, crse_ratio, rad_boundary);
  }

  multigroup_boundary.setBoundaryFluxConditions(rad_boundary);
}

void Radiation::fillBndryRegister(BndryRegister& bdry, int level, Real time)
{
  BL_PROFILE("Radiation::fillBndryRegister");

  HydroSim      *hydrosim = (HydroSim*)&parent->getLevel(level);
  const BoxArray& grids = hydrosim->boxArray();
  const DistributionMapping& dmap = hydrosim->DistributionMap();
  const Geometry& geom  = parent->Geom(level);

  Real old_time = hydrosim->get_state_data(Radiation_Type).prevTime();
  Real new_time = hydrosim->get_state_data(Radiation_Type).curTime();
  Real eps = (new_time > old_time) ? 0.001*(new_time - old_time) : 1.0;

  BL_ASSERT( (time > old_time-eps) && (time < new_time + eps));

  MultiFab& S_new = hydrosim->get_new_data(Radiation_Type);
  MultiFab& S_old = hydrosim->get_old_data(Radiation_Type);

  if (!geom.isAnyPeriodic()) {
    int n_ghost = 0;

    if (time > new_time - eps) {
      bdry.copyFrom(S_new, n_ghost, 0, 0, Radiation::nGroups);
    }
    else if (time < old_time + eps) {
      bdry.copyFrom(S_old, n_ghost, 0, 0, Radiation::nGroups);
    }
    else {
      Real a = (new_time - time) / (new_time - old_time);
      Real b = (time - old_time) / (new_time - old_time);
      bdry.linComb(a, S_old, 0, b, S_new, 0, 0, Radiation::nGroups, n_ghost);
    }
  }
  else {



    int need_old_state = (time <= new_time - eps);
    int need_new_state = (time >= old_time + eps);

    int n_grow = 1;

    MultiFab old_state_buffer;
    MultiFab new_state_buffer;

    if (need_old_state) {
      old_state_buffer.define(grids, dmap, 1, n_grow);
      old_state_buffer.setVal(0.0);
      MultiFab::Copy(old_state_buffer, S_old, Rad, 0, 1, 0);
      old_state_buffer.FillBoundary(geom.periodicity());
    }

    if (need_new_state) {
      new_state_buffer.define(grids, dmap, 1, n_grow);
      new_state_buffer.setVal(0.0);
      MultiFab::Copy(new_state_buffer, S_new, Rad, 0, 1, 0);
      new_state_buffer.FillBoundary(geom.periodicity());
    }

    int n_ghost = 1;

    if (time > new_time - eps) {
      bdry.copyFrom(new_state_buffer, n_ghost, 0, 0, 1);
    }
    else if (time < old_time + eps) {
      bdry.copyFrom(old_state_buffer, n_ghost, 0, 0, 1);
    }
    else {
      Real a = (new_time - time) / (new_time - old_time);
      Real b = (time - old_time) / (new_time - old_time);
      bdry.linComb(a, old_state_buffer, 0, b, new_state_buffer, 0, 0, 1, n_ghost);
    }
  }
}

void Radiation::get_heat_capacity(FArrayBox& c_v, FArrayBox& temp, FArrayBox& state,
                        const Box& reg)
{
    BL_PROFILE("Radiation::get_heat_capacity");

    auto heat_capacity_arr = c_v.array();
    auto temp_field_arr = temp.array();
    auto state_arr = state.array();

    amrex::ParallelFor(reg,
    [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
    {
        Real rhoInv = 1.e0_rt / state_arr(i,j,k,URHO);

        eos_re_t eos_state;
        eos_state.rho = state_arr(i,j,k,URHO);
        eos_state.T   = temp_field_arr(i,j,k);
        for (int n = 0; n < NumSpec; ++n) {
            eos_state.xn[n] = state_arr(i,j,k,UFS+n) * rhoInv;
        }
#if NAUX_NET > 0
        for (int n = 0; n < NumAux; ++n) {
            eos_state.aux[n] = state_arr(i,j,k,UFX+n) * rhoInv;
        }
#endif

        eos(eos_input_rt, eos_state);

        heat_capacity_arr(i,j,k) = eos_state.cv;
    });
    Gpu::synchronize();
}

void Radiation::get_density_energy(MultiFab& fluid_rho_eint,
                          MultiFab& state)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter si(state,TilingIfNotGPU()); si.isValid(); ++si) {
        const Box& reg = si.tilebox();

        auto density_energy_arr = fluid_rho_eint[si].array();
        auto state_arr = state[si].array();

        amrex::ParallelFor(reg,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            density_energy_arr(i,j,k) = state_arr(i,j,k,UEINT);
        });
    }
}


void Radiation::derive_planck_and_temp(MultiFab& fkp,
                                    MultiFab& temp,
                                    MultiFab& state,
                                    int rad_group_idx, Real delta_t)
{
    BL_PROFILE("Radiation::derive_planck_and_temp");

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        auto scaled_planck_arr = fkp[mfi].array(rad_group_idx);
        auto temp_field_arr = temp[mfi].array();
        auto state_arr = state[mfi].array();


        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            if (temp_field_arr(i,j,k) <= 0.e0_rt)
            {
                temp_field_arr(i,j,k) = small_temp;
            }
            else
            {
                Real rhoInv = 1.e0_rt / state_arr(i,j,k,URHO);

                eos_re_t eos_state;
                eos_state.rho = state_arr(i,j,k,URHO);
                eos_state.T   = state_arr(i,j,k,UTEMP);
                eos_state.e   = temp_field_arr(i,j,k) * rhoInv;
                for (int n = 0; n < NumSpec; ++n) {
                    eos_state.xn[n] = state_arr(i,j,k,UFS+n) * rhoInv;
                }
#if NAUX_NET > 0
                for (int n = 0; n < NumAux; ++n) {
                    eos_state.aux[n] = state_arr(i,j,k,UFX+n) * rhoInv;
                }
#endif

                eos(eos_input_re, eos_state);

                temp_field_arr(i,j,k) = eos_state.T;
            }
        });

        const Real nu = freq_center[rad_group_idx];

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            Real rho = state_arr(i,j,k,URHO);
            Real temp = state_arr(i,j,k,UTEMP);
            Real Ye;
            if (NumAux > 0)  {
                Ye = state_arr(i,j,k,UFX);
            } else {
                Ye = 0.e0_rt;
            }

            Real kp, kr;
            bool need_planck_opac = true;
            bool need_ross_opac = false;
            opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

            scaled_planck_arr(i,j,k) = kp;
        });

        int ncomp = temp[mfi].nComp();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            const Real min_temperature = 1.e-10_rt;

            for (int n = 0; n < ncomp; ++n) {
                if (temp_field_arr(i,j,k,n) < min_temperature) {
                    temp_field_arr(i,j,k,n) = min_temperature;
                }
            }
        });
    }
}


void Radiation::derive_rosseland_opacity(MultiFab& kappa_r,
                              AmrLevel* hydrosim,
                              int rad_group_idx)
{
  BL_PROFILE("Radiation::derive_rosseland_opacity");

  BL_ASSERT(kappa_r.nGrow() == 1);

  if(Radiation::nGroups == 0) {
      kappa_r.setVal(0.);
      return;
  }

  Real time = hydrosim->get_state_data(State_Type).curTime();
  MultiFab& S_new = hydrosim->get_new_data(State_Type);
  int nstate = S_new.nComp();

  FillPatchIterator fpi(*hydrosim, S_new, 1, time, State_Type, 0, nstate);
  MultiFab& state = fpi.get_mf();

  const Real nu = freq_center[rad_group_idx];

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
      FArrayBox fluid_rho_eint;
      for(MFIter mfi(kappa_r, TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
          const Box& bx = mfi.growntilebox();
          fluid_rho_eint.resize(bx, 1);
          Elixir density_energy_hold = fluid_rho_eint.elixir();

          auto density_energy_arr = fluid_rho_eint.array();
          auto state_arr = state[mfi].array();
          auto kpr = kappa_r[mfi].array();

          amrex::ParallelFor(bx,
          [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
          {

              if (state_arr(i,j,k,UEINT) <= 0.e0_rt)
              {
                  density_energy_arr(i,j,k) = small_temp;
              }
              else
              {
                  Real rhoInv = 1.e0_rt / state_arr(i,j,k,URHO);

                  eos_re_t eos_state;
                  eos_state.rho = state_arr(i,j,k,URHO);
                  eos_state.T   = state_arr(i,j,k,UTEMP);
                  eos_state.e   = state_arr(i,j,k,UEINT) * rhoInv;
                  for (int n = 0; n < NumSpec; ++n) {
                      eos_state.xn[n] = state_arr(i,j,k,UFS+n) * rhoInv;
                  }
#if NAUX_NET > 0
                  for (int n = 0; n < NumAux; ++n) {
                      eos_state.aux[n] = state_arr(i,j,k,UFX+n) * rhoInv;
                  }
#endif

                  eos(eos_input_re, eos_state);

                  density_energy_arr(i,j,k) = eos_state.T;

                  state_arr(i,j,k,UTEMP) = density_energy_arr(i,j,k);
              }

              Real rho = state_arr(i,j,k,URHO);
              Real temp = state_arr(i,j,k,UTEMP);
              Real Ye;
              if (NumAux > 0) {
                  Ye = state_arr(i,j,k,UFX);
              } else {
                  Ye = 0.e0_rt;
              }

              Real kp, kr;
              bool need_planck_opac = false;
              bool need_ross_opac = true;

              opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

              kpr(i,j,k,rad_group_idx) = kr;
          });
      }
  }
}


void Radiation::refresh_rosseland_from_temp(MultiFab& kappa_r,
                                           MultiFab& temp,
                                           MultiFab& state,
                                           const Geometry& geom,
                                           int rad_group_idx)
{
  BL_PROFILE("refresh_rosseland_from_temp");

  BL_ASSERT(kappa_r.nGrow() == 1);
  BL_ASSERT(temp.nGrow()    == 0);
  BL_ASSERT(kappa_r.nComp() == Radiation::nGroups);

  const Real nu = freq_center[rad_group_idx];

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();

      auto state_arr = state[mfi].array();
      auto temp_field_arr = temp[mfi].array();
      auto kpr = kappa_r[mfi].array();

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          state_arr(i,j,k,UTEMP) = temp_field_arr(i,j,k);

          Real rho = state_arr(i,j,k,URHO);
          Real temp = state_arr(i,j,k,UTEMP);
          Real Ye;
          if (NumAux > 0) {
              Ye = state_arr(i,j,k,UFX);
          } else {
              Ye = 0.e0_rt;
          }

          Real kp, kr;
          bool need_planck_opac = false;
          bool need_ross_opac = true;

          opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

          kpr(i,j,k,rad_group_idx) = kr;
      });
  }

  kappa_r.FillBoundary(geom.periodicity());
}

void Radiation::compute_rosseland_opacity(MultiFab& kappa_r, const MultiFab& state)
{
  BL_PROFILE("Radiation::compute_rosseland_opacity (MultiFab)");

  const int rad_group_idx = 0;
  const Real nu = freq_center[rad_group_idx];

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(kappa_r, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& kbox = mfi.growntilebox();

      auto state_arr = state[mfi].array();
      auto kpr = kappa_r[mfi].array();

      amrex::ParallelFor(kbox,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          Real rho = state_arr(i,j,k,URHO);
          Real temp = state_arr(i,j,k,UTEMP);
          Real Ye;
          if (NumAux > 0) {
              Ye = state_arr(i,j,k,UFX);
          } else {
              Ye = 0.e0_rt;
          }

          Real kp, kr;
          bool need_planck_opac = false;
          bool need_ross_opac = true;

          opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

          kpr(i,j,k,rad_group_idx) = kr;
      });
  }
}

void Radiation::compute_rosseland_opacity(FArrayBox& kappa_r, const FArrayBox& state)
{
  BL_PROFILE("Radiation::compute_rosseland_opacity (FArrayBox)");

  const Box& kbox = kappa_r.box();

  const int rad_group_idx = 0;
  const Real nu = freq_center[rad_group_idx];

  auto state_arr = state.array();
  auto kpr = kappa_r.array();

  amrex::ParallelFor(kbox,
  [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
  {
      Real rho = state_arr(i,j,k,URHO);
      Real temp = state_arr(i,j,k,UTEMP);
      Real Ye;
      if (NumAux > 0) {
          Ye = state_arr(i,j,k,UFX);
      } else {
          Ye = 0.e0_rt;
      }

      Real kp, kr;
      bool need_planck_opac = false;
      bool need_ross_opac = true;

      opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

      kpr(i,j,k,rad_group_idx) = kr;
  });
  Gpu::synchronize();
}

void Radiation::prepare_delayed_reflux(int crse_level)
{
  BL_PROFILE("Radiation::prepare_delayed_reflux");
  if (verbose && ParallelDescriptor::IOProcessor()) {
    std::cout << "Radiation deferred sync setup for coarse level " << crse_level << "..." << std::endl;
  }


  int level = crse_level + 1;

  const Orientation ori(0, Orientation::low);

  int do_swap = (conservative_register_old[level] &&
                 (conservative_register_old[level]->coarsenedBoxes() ==
                  flux_accum_reg[level]->coarsenedBoxes()) &&
                 ((*conservative_register_old[level])[ori].DistributionMap() ==
                  (*flux_accum_reg[level])[ori].DistributionMap()));

  if (do_swap) {
      std::swap(conservative_register_old[level], flux_accum_reg[level]);
  }
  else {
      conservative_register_old[level] = std::move(flux_accum_reg[level]);
  }

  dt_previous[crse_level] = parent->dtLevel(crse_level);

  if (verbose && ParallelDescriptor::IOProcessor()) {
    if (apply_sync) {
      std::cout << "                                                   done"
           << std::endl;
    }
    else {
      std::cout << "                                                   zeroed out"
           << std::endl;
    }
  }
}

void Radiation::apply_delayed_reflux(int level, MultiFab& rhs, int indx)
{
  int fine_level = parent->finestLevel();
  HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
  const BoxArray& grids = hydrosim->boxArray();
  const DistributionMapping& dmap = hydrosim->DistributionMap();
  Real delta_t = parent->dtLevel(level);

  if (level < parent->maxLevel() &&
      conservative_register_old[level+1]) {

    FluxRegister& fine_correction_register = *conservative_register_old[level+1];

    if (indx == 0) {
        fine_correction_register.ClearInternalBorders(hydrosim->Geom());
    }

    Real scale = dt_previous[level] / delta_t;
    if (!apply_sync) scale = 0.0;

    fine_correction_register.Reflux(rhs, scale, indx, 0, 1, hydrosim->Geom());


    if (level+1 < parent->maxLevel() &&
        conservative_register_old[level+2]) {

      BoxArray fgrids;

      if (level+1 <= fine_level) {


        fgrids = parent->getLevel(level+1).boxArray();
      }

      const IntVect& rat = fine_correction_register.refRatio();
      BoxArray prior_cover_boxes = fine_correction_register.coarsenedBoxes();
      prior_cover_boxes.refine(rat);


      if (fgrids != prior_cover_boxes) {


        IntVect level_refine_ratio = IntVect::TheUnitVector();

        for (int flev = level+1; flev < parent->maxLevel(); flev++) {


          level_refine_ratio *= parent->refRatio(flev-1);

          if (conservative_register_old[flev+1]) {

            if (flev > level+1) {
              fgrids.refine(parent->refRatio(flev-1));
            }

            FluxRegister& fine_fine_register = *conservative_register_old[flev+1];
            BoxArray fine_fine_boxes = fine_fine_register.coarsenedBoxes();
            fine_fine_boxes.grow(1);

            const Box& domain = parent->Geom(flev).Domain();

            for (int i = 0; i < fine_fine_boxes.size(); i++) {
              fine_fine_boxes.set(i, (fine_fine_boxes[i] & domain));
            }


            if (fgrids.contains(fine_fine_boxes)) {

              break;
            }
            else {




              if (indx == 0) {
                  fine_fine_register.ClearInternalBorders(parent->Geom(flev));
              }

              BoxArray coarse_cover_boxes(fine_fine_boxes);


              coarse_cover_boxes.coarsen(level_refine_ratio);


              BoxArray restored_fine_boxes(coarse_cover_boxes);
              restored_fine_boxes.refine(level_refine_ratio);


              const DistributionMapping& fine_fine_dmap = fine_fine_register.DistributionMap();

              MultiFab fine_level_data(restored_fine_boxes, fine_fine_dmap, 1, 0);
              fine_level_data.setVal(0.0);




              Real scale = dt_previous[flev] / delta_t;
              if (!apply_sync) scale = 0.0;

              fine_fine_register.Reflux(fine_level_data, scale, indx, 0, 1,
                             parent->Geom(flev));


              MultiFab rhs_tmp(grids, dmap, 1, 0);
              rhs_tmp.setVal(0.0);

              amrex::average_down(fine_level_data, rhs_tmp, 0, 1, level_refine_ratio);

              rhs.plus(rhs_tmp, 0, 1, 0);
            }
          }
        }
      }

    }

  }


  IntVect level_refine_ratio = IntVect::TheUnitVector();

  for (int flev = level; flev > 0; flev--) {


    if (conservative_register_old[flev] &&
        dt_previous[flev-1] > 0.0) {

      FluxRegister& coarse_correction_register = *conservative_register_old[flev];

      const IntVect& rat = coarse_correction_register.refRatio();
      BoxArray prior_cover_boxes = coarse_correction_register.coarsenedBoxes();
      prior_cover_boxes.refine(rat);

      const BoxArray& fgrids = parent->getLevel(flev).boxArray();
      if (fgrids == prior_cover_boxes) {


        break;
      }
      else {


        prior_cover_boxes.refine(level_refine_ratio);



        if (prior_cover_boxes.contains(grids)) {
          break;
        }


        const DistributionMapping& coarse_correction_dmap = coarse_correction_register.DistributionMap();

        FluxRegister refined_correction_register(prior_cover_boxes, coarse_correction_dmap,
                                   IntVect::TheUnitVector(),
                                   level+1,
                                   1);
        level_refine_ratio *= rat;



        for (int dir = 0; dir < AMREX_SPACEDIM; dir++) {
          const Orientation lo_face(dir,Orientation::low);
          const Orientation hi_face(dir,Orientation::high);

          for (FabSetIter fsi(refined_correction_register[lo_face]);
               fsi.isValid(); ++fsi) {

            coarse_to_fine_face(refined_correction_register[lo_face][fsi].array(),
                   coarse_correction_register[lo_face][fsi].array(indx),
                   dir, level_refine_ratio);
          }

          for (FabSetIter fsi(refined_correction_register[hi_face]);
               fsi.isValid(); ++fsi) {

            coarse_to_fine_face(refined_correction_register[hi_face][fsi].array(),
                   coarse_correction_register[hi_face][fsi].array(indx),
                   dir, level_refine_ratio);
          }
        }




        Real scale = dt_previous[flev-1] / delta_t;
        if (!apply_sync) scale = 0.0;

        refined_correction_register.Reflux(rhs, scale, 0, 0, 1, hydrosim->Geom());

      }

    }

  }

}

void Radiation::reflux(int level)
{
  return;
}


void Radiation::evaluateGradientRatio(int level,
                               Array<MultiFab, AMREX_SPACEDIM>& R,
                               MultiFab& kappa_r, int kcomp,
                               MultiFab& Er, int rad_group_idx,
                               int nGrow_rad_energy, int GradComp)
{
  BL_PROFILE("Radiation::evaluateGradientRatio");
  BL_ASSERT(kappa_r.nGrow() == 1);

  MultiFab er_periodic_buffer;
  if (nGrow_rad_energy == 0) {
    if (radiation::limiter > 0) {
      const BoxArray& grids = parent->boxArray(level);
      const DistributionMapping& dmap = parent->DistributionMap(level);
      er_periodic_buffer.define(grids,dmap,1,1);
      er_periodic_buffer.setVal(-1.0);
      MultiFab::Copy(er_periodic_buffer, Er, rad_group_idx, 0, 1, 0);


      er_periodic_buffer.FillBoundary(parent->Geom(level).periodicity());
    }
  }

  MultiFab& RadEnergyGrown = (nGrow_rad_energy==0) ? er_periodic_buffer : Er;

  auto dx = parent->Geom(level).CellSizeArray();

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

      for (MFIter mfi(R[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

          const Box& nbx = mfi.tilebox();

          auto scaled_grad_arr = R[idim][mfi].array(GradComp);

          if (radiation::limiter == 0) {

              R[idim][mfi].setVal<RunOn::Device>(0.0, GradComp);

          }
          else {

              int cross_term_flag = 0;

              if (radiation::limiter == 1) {
                  cross_term_flag = 0;
              } else if (radiation::limiter == 2) {
                  cross_term_flag = 1;
              } else {
                  amrex::Abort("Unknown limiter");
              }

              auto opacity_arr = kappa_r[mfi].array(kcomp);
              auto rad_eden_arr = RadEnergyGrown[mfi].array();

              amrex::ParallelFor(nbx,
              [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
              {
                  Real dxInv[3] = {0.0};

                  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                      dxInv[d] = 1.e0_rt / dx[d];
                  }

                  Real dal, dar, dbl, dbr;

                  const Real tiny = 1.e-50_rt;

                  if (idim == 0)
                  {
                      if (cross_term_flag == 1)
                      {
#if (AMREX_SPACEDIM >= 2)
                          dal = rad_eden_arr(i-1,j+1,k) - rad_eden_arr(i-1,j-1,k);
                          dar = rad_eden_arr(i  ,j+1,k) - rad_eden_arr(i  ,j-1,k);

                          if      (rad_eden_arr(i-1,j-1,k) == -1.e0_rt)
                          {
                              dal = 2.e0_rt * (rad_eden_arr(i-1,j+1,k) - rad_eden_arr(i-1,j  ,k));
                          }
                          else if (rad_eden_arr(i-1,j+1,k) == -1.e0_rt)
                          {
                              dal = 2.e0_rt * (rad_eden_arr(i-1,j  ,k) - rad_eden_arr(i-1,j-1,k));
                          }

                          if      (rad_eden_arr(i  ,j-1,k) == -1.e0_rt)
                          {
                              dar = 2.e0_rt * (rad_eden_arr(i  ,j+1,k) - rad_eden_arr(i  ,j  ,k));
                          }
                          else if (rad_eden_arr(i  ,j+1,k) == -1.e0_rt)
                          {
                              dar = 2.e0_rt * (rad_eden_arr(i  ,j  ,k) - rad_eden_arr(i  ,j-1,k));
                          }
#else
                          dal = 0.e0_rt;
                          dar = 0.e0_rt;
#endif

#if (AMREX_SPACEDIM == 3)
                          dbl = rad_eden_arr(i-1,j,k+1) - rad_eden_arr(i-1,j,k-1);
                          dbr = rad_eden_arr(i  ,j,k+1) - rad_eden_arr(i  ,j,k-1);

                          if      (rad_eden_arr(i-1,j,k-1) == -1.e0_rt)
                          {
                              dbl = 2.e0_rt * (rad_eden_arr(i-1,j,k+1) - rad_eden_arr(i-1,j,k  ));
                          }
                          else if (rad_eden_arr(i-1,j,k+1) == -1.e0_rt)
                          {
                              dbl = 2.e0_rt * (rad_eden_arr(i-1,j,k  ) - rad_eden_arr(i-1,j,k-1));
                          }

                          if      (rad_eden_arr(i  ,j,k-1) == -1.e0_rt)
                          {
                              dbr = 2.e0_rt * (rad_eden_arr(i  ,j,k+1) - rad_eden_arr(i  ,j,k  ));
                          }
                          else if (rad_eden_arr(i  ,j,k+1) == -1.e0_rt)
                          {
                              dbr = 2.e0_rt * (rad_eden_arr(i  ,j,k  ) - rad_eden_arr(i  ,j,k-1));
                          }
#else
                          dbl = 0.e0_rt;
                          dbr = 0.e0_rt;
#endif

                      }
                      else
                      {
                          dal = 0.e0_rt;
                          dar = 0.e0_rt;
                          dbl = 0.e0_rt;
                          dbr = 0.e0_rt;
                      }

                      Real rg;

                      if (rad_eden_arr(i-1,j,k) == -1.e0_rt)
                      {
                          rg = std::pow((rad_eden_arr(i+1,j,k) - rad_eden_arr(i,j,k)) * dxInv[0], 2) +
                               std::pow(0.5_rt * dar * dxInv[1], 2) +
                               std::pow(0.5_rt * dbr * dxInv[2], 2);
                      }
                      else if (rad_eden_arr(i,j,k) == -1.e0_rt)
                      {
                          rg = std::pow((rad_eden_arr(i-1,j,k) - rad_eden_arr(i-2,j,k)) * dxInv[0], 2) +
                               std::pow(0.5_rt * dal * dxInv[1], 2) +
                               std::pow(0.5_rt * dbl * dxInv[2], 2);
                      }
                      else
                      {
                          rg = std::pow((rad_eden_arr(i,j,k) - rad_eden_arr(i-1,j,k)) * dxInv[0], 2) +
                               std::pow((1.0_rt / 4.0_rt) * (dal + dar) * dxInv[1], 2) +
                               std::pow((1.0_rt / 4.0_rt) * (dbl + dbr) * dxInv[2], 2);
                      }

                      Real kap = face_opacity_average(opacity_arr(i-1,j,k), opacity_arr(i,j,k), dx[0], -1);
                      scaled_grad_arr(i,j,k) = std::sqrt(rg) / (kap * amrex::max(rad_eden_arr(i-1,j,k), rad_eden_arr(i,j,k), tiny));

                  }
                  else if (idim == 1)
                  {
                      if (cross_term_flag == 1)
                      {
                          dal = rad_eden_arr(i+1,j-1,k  ) - rad_eden_arr(i-1,j-1,k  );
                          dar = rad_eden_arr(i+1,j  ,k  ) - rad_eden_arr(i-1,j  ,k  );

                          if      (rad_eden_arr(i-1,j-1,k  ) == -1.e0_rt)
                          {
                              dal = 2.e0_rt * (rad_eden_arr(i+1,j-1,k  ) - rad_eden_arr(i  ,j-1,k  ));
                          }
                          else if (rad_eden_arr(i+1,j-1,k  ) == -1.e0_rt)
                          {
                              dal = 2.e0_rt * (rad_eden_arr(i  ,j-1,k  ) - rad_eden_arr(i-1,j-1,k  ));
                          }

                          if      (rad_eden_arr(i-1,j  ,k  ) == -1.e0_rt)
                          {
                              dar = 2.e0_rt * (rad_eden_arr(i+1,j  ,k  ) - rad_eden_arr(i  ,j  ,k  ));
                          }
                          else if (rad_eden_arr(i+1,j  ,k  ) == -1.e0_rt)
                          {
                              dar = 2.e0_rt * (rad_eden_arr(i  ,j  ,k  ) - rad_eden_arr(i-1,j  ,k  ));
                          }

#if (AMREX_SPACEDIM == 3)
                          dbl = rad_eden_arr(i  ,j-1,k+1) - rad_eden_arr(i  ,j-1,k-1);
                          dbr = rad_eden_arr(i  ,j  ,k+1) - rad_eden_arr(i  ,j  ,k-1);

                          if      (rad_eden_arr(i  ,j-1,k-1) == -1.e0_rt)
                          {
                              dbl = 2.e0_rt * (rad_eden_arr(i  ,j-1,k+1) - rad_eden_arr(i  ,j-1,k  ));
                          }
                          else if (rad_eden_arr(i  ,j-1,k+1) == -1.e0_rt)
                          {
                              dbl = 2.e0_rt * (rad_eden_arr(i  ,j-1,k  ) - rad_eden_arr(i  ,j-1,k-1));
                          }

                          if      (rad_eden_arr(i  ,j  ,k-1) == -1.e0_rt)
                          {
                              dbr = 2.e0_rt * (rad_eden_arr(i  ,j  ,k+1) - rad_eden_arr(i  ,j,  k  ));
                          }
                          else if (rad_eden_arr(i  ,j  ,k+1) == -1.e0_rt)
                          {
                              dbr = 2.e0_rt * (rad_eden_arr(i  ,j  ,k  ) - rad_eden_arr(i  ,j,  k-1));
                          }
#else
                          dbl = 0.e0_rt;
                          dbr = 0.e0_rt;
#endif
                      }
                      else
                      {
                          dal = 0.e0_rt;
                          dar = 0.e0_rt;
                          dbl = 0.e0_rt;
                          dbr = 0.e0_rt;
                      }

                      Real rg;

                      if (rad_eden_arr(i,j-1,k) == -1.e0_rt)
                      {
                          rg = std::pow((rad_eden_arr(i,j+1,k) - rad_eden_arr(i,j,k)) * dxInv[1], 2) +
                               std::pow(0.5_rt * dar * dxInv[0], 2) +
                               std::pow(0.5_rt * dbr * dxInv[2], 2);
                      }
                      else if (rad_eden_arr(i,j,k) == -1.e0_rt)
                      {
                          rg = std::pow((rad_eden_arr(i,j-1,k) - rad_eden_arr(i,j-2,k)) * dxInv[1], 2) +
                               std::pow(0.5_rt * dal * dxInv[0], 2) +
                               std::pow(0.5_rt * dbl * dxInv[2], 2);
                      }
                      else
                      {
                          rg = std::pow((rad_eden_arr(i,j,k) - rad_eden_arr(i,j-1,k)) * dxInv[1], 2) +
                               std::pow((1.0_rt / 4.0_rt) * (dal + dar) * dxInv[0], 2) +
                               std::pow((1.0_rt / 4.0_rt) * (dbl + dbr) * dxInv[2], 2);
                      }

                      Real kap = face_opacity_average(opacity_arr(i,j-1,k), opacity_arr(i,j,k), dx[1], -1);
                      scaled_grad_arr(i,j,k) = std::sqrt(rg) / (kap * amrex::max(rad_eden_arr(i,j-1,k), rad_eden_arr(i,j,k), tiny));
                  }
                  else
                  {
                      if (cross_term_flag == 1)
                      {
                          dal = rad_eden_arr(i+1,j  ,k-1) - rad_eden_arr(i-1,j  ,k-1);
                          dar = rad_eden_arr(i+1,j  ,k  ) - rad_eden_arr(i-1,j  ,k  );

                          if      (rad_eden_arr(i-1,j  ,k-1) == -1.e0_rt)
                          {
                              dal = 2.e0_rt * (rad_eden_arr(i+1,j  ,k-1) - rad_eden_arr(i  ,j  ,k-1));
                          }
                          else if (rad_eden_arr(i+1,j  ,k-1) == -1.e0_rt)
                          {
                              dal = 2.e0_rt * (rad_eden_arr(i  ,j  ,k-1) - rad_eden_arr(i-1,j  ,k-1));
                          }

                          if      (rad_eden_arr(i-1,j  ,k  ) == -1.e0_rt)
                          {
                              dar = 2.e0_rt * (rad_eden_arr(i+1,j  ,k  ) - rad_eden_arr(i  ,j  ,k  ));
                          }
                          else if (rad_eden_arr(i+1,j  ,k  ) == -1.e0_rt)
                          {
                              dar = 2.e0_rt * (rad_eden_arr(i  ,j  ,k  ) - rad_eden_arr(i-1,j  ,k  ));
                          }

                          dbl = rad_eden_arr(i  ,j+1,k-1) - rad_eden_arr(i  ,j-1,k-1);
                          dbr = rad_eden_arr(i  ,j+1,k  ) - rad_eden_arr(i  ,j-1,k  );

                          if      (rad_eden_arr(i  ,j-1,k-1) == -1.e0_rt)
                          {
                              dbl = 2.e0_rt * (rad_eden_arr(i  ,j+1,k-1) - rad_eden_arr(i  ,j  ,k-1));
                          }
                          else if (rad_eden_arr(i  ,j+1,k-1) == -1.e0_rt)
                          {
                              dbl = 2.e0_rt * (rad_eden_arr(i  ,j  ,k-1) - rad_eden_arr(i  ,j-1,k-1));
                          }

                          if      (rad_eden_arr(i  ,j-1,k  ) == -1.e0_rt)
                          {
                              dbr = 2.e0_rt * (rad_eden_arr(i  ,j+1,k  ) - rad_eden_arr(i  ,j,  k  ));
                          }
                          else if (rad_eden_arr(i  ,j+1,k  ) == -1.e0_rt)
                          {
                              dbr = 2.e0_rt * (rad_eden_arr(i  ,j  ,k  ) - rad_eden_arr(i  ,j-1,k  ));
                          }
                      }
                      else
                      {
                          dal = 0.e0_rt;
                          dar = 0.e0_rt;
                          dbl = 0.e0_rt;
                          dbr = 0.e0_rt;
                      }

                      Real rg;

                      if (rad_eden_arr(i,j,k-1) == -1.e0_rt)
                      {
                          rg = std::pow((rad_eden_arr(i,j,k+1) - rad_eden_arr(i,j,k)) * dxInv[2], 2) +
                               std::pow(0.5_rt * dar * dxInv[0], 2) +
                               std::pow(0.5_rt * dbr * dxInv[1], 2);
                      }
                      else if (rad_eden_arr(i,j,k) == -1.e0_rt)
                      {
                          rg = std::pow((rad_eden_arr(i,j,k-1) - rad_eden_arr(i,j,k-2)) * dxInv[2], 2) +
                               std::pow(0.5_rt * dal * dxInv[0], 2) +
                               std::pow(0.5_rt * dbl * dxInv[1], 2);
                      }
                      else
                      {
                          rg = std::pow((rad_eden_arr(i,j,k) - rad_eden_arr(i,j,k-1)) * dxInv[2], 2) +
                               std::pow((1.0_rt / 4.0_rt) * (dal + dar) * dxInv[0], 2) +
                               std::pow((1.0_rt / 4.0_rt) * (dbl + dbr) * dxInv[1], 2);
                      }

                      Real kap = face_opacity_average(opacity_arr(i,j,k-1), opacity_arr(i,j,k), dx[2], -1);
                      scaled_grad_arr(i,j,k) = std::sqrt(rg) / (kap * amrex::max(rad_eden_arr(i,j,k-1), rad_eden_arr(i,j,k), tiny));
                  }
              });
          }
      }
  }
}


void Radiation::computeFluxLimiter(int level,
                            Array<MultiFab, AMREX_SPACEDIM>& lambda,
                            int limiter_comp)
{
    BL_PROFILE("Radiation:computeFluxLimiter");

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        for (MFIter mfi(lambda[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();

            auto limiter_arr = lambda[idim][mfi].array(limiter_comp);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                limiter_arr(i,j,k) = LimitedDiffusionLambda(limiter_arr(i,j,k));
            });
        }
    }
}

void Radiation::derive_rosseland_velocity_coef(MultiFab& kappa_r, MultiFab& v, MultiFab& dcf,
                                    Real delta_t, Real c,
                                    AmrLevel* hydrosim, int rad_group_idx)
{
    BL_ASSERT(kappa_r.nGrow() == 1);
    BL_ASSERT(      v.nGrow() == 1);
    BL_ASSERT(    dcf.nGrow() == 1);

    int nstate = hydrosim->get_new_data(State_Type).nComp();
    Real time = hydrosim->get_state_data(State_Type).curTime();

    FillPatchIterator rad_fill_iter(*hydrosim, kappa_r, 1, time, Radiation_Type, 0, 1);
    MultiFab& Er = rad_fill_iter.get_mf();

    FillPatchIterator state_fill_iter(*hydrosim, kappa_r, 1, time, State_Type, 0, nstate);
    MultiFab& S = state_fill_iter.get_mf();

    Real local_stefan_const = sigma;
    Real c_loc = c;
    Real local_temp_step = dT;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        FArrayBox temp, c_v, kp, kp2;
        for (MFIter mfi(kappa_r,true); mfi.isValid(); ++mfi)
        {
            const Box& reg = mfi.growntilebox();

            temp.resize(reg);
            Elixir temperature_hold = temp.elixir();

            kp.resize(reg);
            Elixir planck_opacity_hold = kp.elixir();

            kp2.resize(reg);
            Elixir planck_opacity_shift_hold = kp2.elixir();

            c_v.resize(reg);
            Elixir heat_capacity_hold = c_v.elixir();

            auto temp_field_arr = temp.array();
            auto cons_state_arr = S[mfi].array();
            auto gas_vel_arr = v[mfi].array();
            auto rad_energy_arr = Er[mfi].array();
            auto rosseland_opacity_arr = kappa_r[mfi].array();
            auto planck_opacity_arr = kp.array();
            auto planck_opacity_shift_arr = kp2.array();
            auto heat_capacity_arr = c_v.array();
            auto diff_factor_arr = dcf[mfi].array();

            amrex::ParallelFor(reg,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {

                if (cons_state_arr(i,j,k,UEINT) <= 0.e0_rt)
                {
                    temp_field_arr(i,j,k) = small_temp;
                }
                else
                {
                    Real rhoInv = 1.e0_rt / cons_state_arr(i,j,k,URHO);

                    eos_re_t eos_state;
                    eos_state.rho = cons_state_arr(i,j,k,URHO);
                    eos_state.T   = cons_state_arr(i,j,k,UTEMP);
                    eos_state.e   = cons_state_arr(i,j,k,UEINT) * rhoInv;
                    for (int n = 0; n < NumSpec; ++n) {
                        eos_state.xn[n] = cons_state_arr(i,j,k,UFS+n) * rhoInv;
                    }
#if NAUX_NET > 0
                    for (int n = 0; n < NumAux; ++n) {
                        eos_state.aux[n] = cons_state_arr(i,j,k,UFX+n) * rhoInv;
                    }
#endif

                    eos(eos_input_re, eos_state);

                    temp_field_arr(i,j,k) = eos_state.T;
                }
            });
            Gpu::synchronize();

            get_heat_capacity(c_v, temp, S[mfi], reg);

            S[mfi].copy<RunOn::Device>(temp,reg,0,reg,UTEMP,1);

            const Real nu = freq_center[rad_group_idx];

            amrex::ParallelFor(reg,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                Real rho = cons_state_arr(i,j,k,URHO);
                Real temp = cons_state_arr(i,j,k,UTEMP);
                Real Ye;
                if (NumAux > 0)  {
                    Ye = cons_state_arr(i,j,k,UFX);
                } else {
                    Ye = 0.e0_rt;
                }

                Real kp, kr;
                bool need_planck_opac = true;
                bool need_ross_opac = true;
                opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

                planck_opacity_arr(i,j,k) = kp;
                rosseland_opacity_arr(i,j,k) = kr;

                temp += local_temp_step;

                opacity(kp, kr, rho, temp, Ye, nu, need_planck_opac, need_ross_opac);

                planck_opacity_shift_arr(i,j,k) = kp;
            });

            amrex::ParallelFor(reg,
            [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
            {
                const Real fac0 = 4.e0_rt * local_stefan_const * delta_t / local_temp_step;
                const Real fac2 = c_loc * delta_t / local_temp_step;

                gas_vel_arr(i,j,k,0) = cons_state_arr(i,j,k,UMX) / cons_state_arr(i,j,k,URHO);
#if AMREX_SPACEDIM >= 2
                gas_vel_arr(i,j,k,1) = cons_state_arr(i,j,k,UMY) / cons_state_arr(i,j,k,URHO);
#endif
#if AMREX_SPACEDIM == 3
                gas_vel_arr(i,j,k,2) = cons_state_arr(i,j,k,UMZ) / cons_state_arr(i,j,k,URHO);
#endif

                Real alpha = fac0 * (planck_opacity_shift_arr(i,j,k) * std::pow(temp_field_arr(i,j,k) + local_temp_step, 4) -
                                     planck_opacity_arr(i,j,k) * std::pow(temp_field_arr(i,j,k), 4)) -
                             fac2 * (planck_opacity_shift_arr(i,j,k) - planck_opacity_arr(i,j,k)) * rad_energy_arr(i,j,k);

                Real frc = cons_state_arr(i,j,k,URHO) * heat_capacity_arr(i,j,k) + 1.0e-50_rt;
                Real coupling_inverse = frc / (alpha + frc);

                diff_factor_arr(i,j,k) = 2.e0_rt * coupling_inverse * (planck_opacity_arr(i,j,k) / rosseland_opacity_arr(i,j,k));
            });
        }
    }
}

void Radiation::refresh_diffusion_coef(MultiFab& dcf, MultiFab& coupling_inverse, MultiFab& kp, MultiFab& kr,
                           const Geometry& geom)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(dcf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto diff_factor_arr = dcf[mfi].array();
        auto coupling_inverse_arr = coupling_inverse[mfi].array();
        auto planck_opacity_arr = kp[mfi].array();
        auto rosseland_opacity_arr = kr[mfi].array();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
        {
            diff_factor_arr(i,j,k) = 2.e0_rt * coupling_inverse_arr(i,j,k) * (planck_opacity_arr(i,j,k) / rosseland_opacity_arr(i,j,k));
        });
    }

    dcf.FillBoundary(geom.periodicity());
}

void Radiation::select_active_group(int rad_group_idx)
{
  active_group_id = rad_group_idx;
}


void Radiation::filter_primitives(int level, MultiFab& State)
{
  HydroSim *hydrosim = dynamic_cast<HydroSim*>(&parent->getLevel(level));
  const Geometry& geom = parent->Geom(level);
  auto geomdata = geom.data();

  int ngrow = filter_prim_temperature;
  int ncomp = State.nComp();
  Real time = hydrosim->get_state_data(Radiation_Type).curTime();

  FillPatchIterator fpi(*hydrosim,State,ngrow,time,State_Type,0,ncomp);
  MultiFab& S_fp = fpi.get_mf();

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi(State,true); mfi.isValid(); ++mfi)
  {
      const Box& bx = mfi.tilebox();

      auto filtered_state_arr = S_fp[mfi].array();
      auto source_state_arr = State[mfi].array();

      int T = filter_prim_temperature;
      int S = filter_primitive_S;

      amrex::ParallelFor(bx,
      [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k)
      {
          primitive_filter_cell(i, j, k,
                    filtered_state_arr, source_state_arr,
                    T, S,
                    geomdata, time);
      });
  }
}
