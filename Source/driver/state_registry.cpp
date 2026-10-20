#include <cstdio>

#include <array>
#include <format>

#include <AMReX_LevelBld.H>
#include <AMReX_ParmParse.H>
#include <HydroSim.H>
#include <boundary_fill.H>
#include <Derive.H>
#include <runtime_param_init.H>
#include <state_bc_tables.H>
#ifdef RADIATION
#include <Radiation.H>
#include <RadFieldDerive.H>
#include <opacity.H>
#endif

#include <AMReX_buildInfo.H>
#include <eos.H>
#include <ambient.H>
#ifndef WIN32
#include <unistd.h>
#endif

#include <iomanip>

#include <algorithm>

#include <vector>

#include <iostream>

#include <string>

#include <ctime>

#include <memory>

#include <numbers>

#include <AMReX_Utility.H>

#include <AMReX_CONSTANTS.H>

#include <global.H>

#include <AMReX_VisMF.H>

#include <AMReX_TagBox.H>

#include <AMReX_FillPatchUtil.H>

#ifdef AMREX_PARTICLES
#include <AMReX_Particles.H>
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

#include <extern_parameters.H>

#include <prob_parameters.H>

#include <user_initialize.H>

#include <hydrosim_limits.H>

#include <solver_constants.H>

using std::string;
using namespace amrex;

void
HydroSim::variableCleanUp ()
{
#ifdef GRAVITY
  if (gravity != nullptr) {
    if (verbose > 1 && ParallelDescriptor::IOProcessor()) {
      std::cout << "Deleting gravity in variableCleanUp..." << '\n';
    }
    gravity.reset();
  }
#endif

#ifdef DIFFUSION
  if (diffusion != nullptr) {
    if (verbose > 1 && ParallelDescriptor::IOProcessor()) {
      std::cout << "Deleting diffusion in variableCleanUp..." << '\n';
    }
    diffusion.reset();
  }
#endif

#ifdef RADIATION
  if (radiation != nullptr) {
      int report = (verbose || radiation->verbose);
      if (report && ParallelDescriptor::IOProcessor()) {
          std::cout << "Deleting radiation in variableCleanUp..." << '\n';
      }
      radiation.reset();
      if (report && ParallelDescriptor::IOProcessor()) {
          std::cout << "                                        done" << std::endl;
      }
  }
#endif

#ifdef AMREX_PARTICLES
  TracerPC.reset();
#endif

    desc_lst.clear();

    eos_finalize();

}

void
HydroSim::variableSetUp ()
{

  startCpuSeconds = ParallelDescriptor::second();

  if (ParallelDescriptor::IOProcessor()) {

    const char* hydrosim_hash  = buildInfoGetGitHash(1);
    const char* framework_hash   = buildInfoGetGitHash(2);
    const char* physics_lib_hash = buildInfoGetGitHash(3);
    const char* build_rev_hash = buildInfoGetBuildGitHash();
    const char* build_rev_name = buildInfoGetBuildGitName();

    if (strlen(hydrosim_hash) > 0) {
      std::cout << "\n" << "HydroSim git describe: " << hydrosim_hash << "\n";
    }
    if (strlen(framework_hash) > 0) {
      std::cout << "AMReX git describe: " << framework_hash << "\n";
    }
    if (strlen(physics_lib_hash) > 0) {
      std::cout << "Microphysics git describe: " << physics_lib_hash << "\n";
    }
    if (strlen(build_rev_hash) > 0){
      std::cout << build_rev_name << " git describe: " << build_rev_hash << "\n";
    }

    std::cout << "\n";
  }

  BL_ASSERT(desc_lst.size() == 0);

  read_params();

  init_prob_parameters();

  microphys_init();

  if (small_dens < 0.0_rt) {
    small_dens = 1.e-100_rt;
  }

  if (small_temp < 0.0_rt) {
    small_temp = 1.e-100_rt;
  }

  if (pressure_cutoff < 0.0_rt) {
    pressure_cutoff = 1.e-100_rt;
  }

  if (eint_floor < 0.0_rt) {
    eint_floor = 1.e-100_rt;
  }

#ifdef REACTIONS
  network_init();
#endif

  eos_init(hydrosim::small_temp, hydrosim::small_dens);

#ifdef RADIATION
  opacity_init();
#endif

  Real temp_floor_next = std::max(small_temp, EOSData::mintemp);
  small_temp = temp_floor_next;
  EOSData::mintemp = temp_floor_next;

  Real dens_floor_next = std::max(small_dens, EOSData::mindens);
  small_dens = dens_floor_next;
  EOSData::mindens = dens_floor_next;

  eos_t eos_state;

  eos_state.rho = hydrosim::small_dens;
  eos_state.T = hydrosim::small_temp;
  for (double& X : eos_state.xn) {
      X = 1.0_rt / NumSpec;
  }
#ifdef AUX_THERMO
  set_aux_comp_from_X(eos_state);
#endif

  eos(eos_input_rt, eos_state);

  hydrosim::pressure_cutoff = amrex::max(hydrosim::pressure_cutoff, eos_state.p);
  hydrosim::eint_floor = amrex::max(hydrosim::eint_floor, eos_state.e);

#ifdef REACTIONS
#ifdef TRUE_SDC
  if (allow_step_retry) {
    amrex::Warning("allow_step_retry = 1 is not supported with true SDC.  Disabling");
    allow_step_retry = false;
  }
#endif
#endif

  for (Real & s : ambient::ambient_reference_state) {
      s = 0.0;
  }

  ambient::ambient_reference_state[URHO]  = amrex::max(hydrosim::background_density, hydrosim::small_dens);
  ambient::ambient_reference_state[UTEMP] = amrex::max(hydrosim::background_temp, hydrosim::small_temp);
  ambient::ambient_reference_state[UEINT] = ambient::ambient_reference_state[URHO] * amrex::max(hydrosim::background_energy,
                                                                            hydrosim::eint_floor);
  ambient::ambient_reference_state[UEDEN] = ambient::ambient_reference_state[UEINT];
  for (int n = 0; n < NumSpec; ++n) {
      ambient::ambient_reference_state[UFS+n] = ambient::ambient_reference_state[URHO] * (1.0_rt / NumSpec);
  }

  MFInterpolater* interp = nullptr;

  if (state_interp_degree == 0) {
    interp = &mf_pc_interp;
  }
  else {
    if (linear_interp_limiter == 2) {
      interp = &mf_linear_slope_minmax_interp;
    }
    else if (linear_interp_limiter == 1) {
      interp = &mf_lincc_interp;
    }
    else {
      interp = &mf_cell_cons_interp;
    }
  }

  auto state_data_extrap = false;
  bool store_in_checkpoint;

  int state_ghosts = 0;

  store_in_checkpoint = true;
  desc_lst.addDescriptor(State_Type,IndexType::TheCellType(),
                         StateDescriptor::Point,state_ghosts,NUM_STATE,
                         interp,state_data_extrap,store_in_checkpoint);

#ifdef MHD
  store_in_checkpoint = true;
  IndexType xface(IntVect{AMREX_D_DECL(1,0,0)});
  desc_lst.addDescriptor(Bfield_Type_x, xface,
                         StateDescriptor::Point, 0, 1,
                         interp, state_data_extrap,
                         store_in_checkpoint);
  IndexType yface(IntVect{AMREX_D_DECL(0,1,0)});
  desc_lst.addDescriptor(Bfield_Type_y, yface,
                         StateDescriptor::Point, 0, 1,
                         interp, state_data_extrap,
                         store_in_checkpoint);
  IndexType z_face_type(IntVect{AMREX_D_DECL(0,0,1)});
  desc_lst.addDescriptor(Bfield_Type_z, z_face_type,
                         StateDescriptor::Point, 0, 1,
                         interp, state_data_extrap,
                         store_in_checkpoint);
#endif

#ifdef GRAVITY
  store_in_checkpoint = true;
  desc_lst.addDescriptor(Potential_Type, IndexType::TheCellType(),
                         StateDescriptor::Point, 1, 1,
                         interp, state_data_extrap,
                         store_in_checkpoint);

  store_in_checkpoint = false;
  desc_lst.addDescriptor(Accel_Type,IndexType::TheCellType(),
                         StateDescriptor::Point,SRC_GHOST_WIDTH,3,
                         interp,state_data_extrap,store_in_checkpoint);
#endif

  store_in_checkpoint = true;
  int forcing_ng = 0;
  if (time_scheme == CornerTrace ||
      time_scheme == SimplifiedIterCorrection) {
      forcing_ng = SRC_GHOST_WIDTH;
  } else if (time_scheme == IterativeCorrection) {
#ifdef SHOCK_VAR
      forcing_ng = SRC_GHOST_WIDTH;
#else
      if (sdc_accuracy_order == 2 && use_pressure_slope) {
          forcing_ng = SRC_GHOST_WIDTH;
      } else {
          forcing_ng = 1;
      }
#endif
  } else {
      amrex::Error("Unknown time_scheme");
  }
  desc_lst.addDescriptor(Forcing_Type, IndexType::TheCellType(),
                         StateDescriptor::Point, forcing_ng, NUM_SOURCE,
                         interp, state_data_extrap, store_in_checkpoint);

#ifdef REACTIONS
  store_in_checkpoint = false;

  int num_react_comps = 1;

  if (keep_omegadot == 1) {
      num_react_comps += NumSpec + NumAux;
  }
#ifdef NSE
  num_react_comps += 1;
#endif
  desc_lst.addDescriptor(Burning_Type,IndexType::TheCellType(),
                         StateDescriptor::Point, 0, num_react_comps,
                         interp, state_data_extrap, store_in_checkpoint);
#endif

#ifdef REACTIONS

  if (time_scheme == SimplifiedIterCorrection) {

      store_in_checkpoint = true;
      desc_lst.addDescriptor(Burn_Corr_Type, IndexType::TheCellType(),
                             StateDescriptor::Point, 1, NQ,
                             interp, state_data_extrap, store_in_checkpoint);

  }
#endif

  Vector<BCRec>       bcs(NUM_STATE);
  Vector<std::string> name(NUM_STATE);

  BCRec bc;
  set_scalar_bc(bc, phys_bc);
  bcs[URHO] = bc;
  name[URHO] = "density";

  set_x_vel_bc(bc, phys_bc);
  bcs[UMX] = bc;
  name[UMX] = "xmom";

  set_y_vel_bc(bc, phys_bc);
  bcs[UMY] = bc;
  name[UMY] = "ymom";

  set_z_vel_bc(bc, phys_bc);
  bcs[UMZ] = bc;
  name[UMZ] = "zmom";

#ifdef HYBRID_MOMENTUM
  set_scalar_bc(bc, phys_bc);
  bcs[UMR] = bc;
  name[UMR] = "rmom";

  set_scalar_bc(bc, phys_bc);
  bcs[UML] = bc;
  name[UML] = "lmom";

  set_scalar_bc(bc, phys_bc);
  bcs[UMP] = bc;
  name[UMP] = "pmom";

#endif
  set_scalar_bc(bc, phys_bc);
  bcs[UEDEN] = bc;
  name[UEDEN] = "rho_E";

  set_scalar_bc(bc, phys_bc);
  bcs[UEINT] = bc;
  name[UEINT] = "rho_e";

  set_scalar_bc(bc, phys_bc);
  bcs[UTEMP] = bc;
  name[UTEMP] = "Temp";

  for (int i=0; i<NumAdvected; ++i)
    {
      set_scalar_bc(bc, phys_bc);
      bcs[UFA+i] = bc;
      name[UFA+i] = std::format("adv_{}", i);
    }

  if ( ParallelDescriptor::IOProcessor())
    {
      std::cout << NumSpec << " Species: " << std::endl;
      for (int i = 0; i < NumSpec; i++) {
        std::cout << short_spec_names_cxx[i] << ' ' << ' ';
      }
      std::cout << std::endl;
    }

  for (int i=0; i<NumSpec; ++i)
    {
      set_scalar_bc(bc, phys_bc);
      bcs[UFS+i] = bc;
      name[UFS+i] = "rho_" + short_spec_names_cxx[i];
    }

#if NAUX_NET > 0
  if ( ParallelDescriptor::IOProcessor())
    {
      std::cout << NumAux << " Auxiliary Variables: " << std::endl;
      for (int i = 0; i < NumAux; i++) {
        std::cout << short_aux_names_cxx[i] << ' ' << ' ';
      }
      std::cout << std::endl;
    }

  for (int i=0; i<NumAux; ++i)
    {
      set_scalar_bc(bc, phys_bc);
      bcs[UFX+i] = bc;
      name[UFX+i] = "rho_" + short_aux_names_cxx[i];
    }
#endif

#ifdef SHOCK_VAR
  set_scalar_bc(bc, phys_bc);
  bcs[UFLAG_SHOCK] = bc;
  name[UFLAG_SHOCK] = "Shock";
#endif

#ifdef NSE_NET
  set_scalar_bc(bc, phys_bc);
  bcs[UPROTON_MU] = bc;
  name[UPROTON_MU] = "mu_p";

  set_scalar_bc(bc, phys_bc);
  bcs[UNEUTRON_MU] = bc;
  name[UNEUTRON_MU] = "mu_n";
#endif

  BndryFunc stateBoundaryFunc(hs_statefill);
  stateBoundaryFunc.setRunOnGPU(true);

  desc_lst.setComponent(State_Type,
                        URHO,
                        name,
                        bcs,
                        stateBoundaryFunc);

  BndryFunc genericBoundaryFunc(hs_generic_fill);
  genericBoundaryFunc.setRunOnGPU(true);

#ifdef MHD
  set_bfield_bc(bc, phys_bc);
  desc_lst.setComponent(Bfield_Type_x, 0, "b_x", bc, genericBoundaryFunc);
  desc_lst.setComponent(Bfield_Type_y, 0, "b_y", bc, genericBoundaryFunc);
  desc_lst.setComponent(Bfield_Type_z, 0, "b_z", bc, genericBoundaryFunc);
#endif

#ifdef GRAVITY
  set_scalar_bc(bc,phys_bc);
  swap_inflow_bc(bc);
  desc_lst.setComponent(Potential_Type,0,"phiGrav",bc,genericBoundaryFunc);
  set_x_vel_bc(bc,phys_bc);
  swap_inflow_bc(bc);
  desc_lst.setComponent(Accel_Type,0,"grav_x",bc,genericBoundaryFunc);
  set_y_vel_bc(bc,phys_bc);
  swap_inflow_bc(bc);
  desc_lst.setComponent(Accel_Type,1,"grav_y",bc,genericBoundaryFunc);
  set_z_vel_bc(bc,phys_bc);
  swap_inflow_bc(bc);
  desc_lst.setComponent(Accel_Type,2,"grav_z",bc,genericBoundaryFunc);
#endif

  Vector<BCRec> forcing_bcs(NUM_SOURCE);
  Vector<std::string> state_type_forcing_names(NUM_SOURCE);

  for (int i = 0; i < NUM_SOURCE; ++i) {
    state_type_forcing_names[i] = name[i] + "_source";
    forcing_bcs[i] = bcs[i];

    swap_inflow_bc(forcing_bcs[i]);
  }

  desc_lst.setComponent(Forcing_Type, URHO, state_type_forcing_names, forcing_bcs, genericBoundaryFunc);

#ifdef REACTIONS
  desc_lst.setComponent(Burning_Type, 0, "rho_enuc", bc, genericBoundaryFunc);

  if (keep_omegadot == 1) {

      std::string react_label;
      for (int i = 0; i < NumSpec; ++i)
      {
          set_scalar_bc(bc,phys_bc);
          swap_inflow_bc(bc);
          react_label = "rho_omegadot_" + short_spec_names_cxx[i];
          desc_lst.setComponent(Burning_Type, 1+i, react_label, bc,genericBoundaryFunc);
      }
#if NAUX_NET > 0
      std::string aux_label;
      for (int i = 0; i < NumAux; ++i) {
          set_scalar_bc(bc,phys_bc);
          swap_inflow_bc(bc);
          aux_label = "rho_auxdot_" + short_aux_names_cxx[i];
          desc_lst.setComponent(Burning_Type, 1+NumSpec+i, aux_label, bc, genericBoundaryFunc);
      }
#endif
  }
#ifdef NSE
  set_scalar_bc(bc,phys_bc);
  swap_inflow_bc(bc);
  if (keep_omegadot == 1) {
    desc_lst.setComponent(Burning_Type, NumSpec+NumAux+1, "in_nse", bc, genericBoundaryFunc);
  }
  else {
    desc_lst.setComponent(Burning_Type, 1, "in_nse", bc, genericBoundaryFunc);
  }
#endif

  if (keep_burn_weights) {

      if (hydrosim::time_scheme == CornerTrace) {
          burn_split_names.emplace_back("burn_weights_firsthalf");
          burn_split_names.emplace_back("burn_weights_secondhalf");
      }
      else if (hydrosim::time_scheme == SimplifiedIterCorrection) {
          for (int n = 0; n < deferred_iters+1; n++) {
              burn_split_names.emplace_back(std::format("burn_weights_iter_{}", n + 1));
          }
      }
  }
#endif

#ifdef REACTIONS
  if (time_scheme == SimplifiedIterCorrection) {
      for (int i = 0; i < NQ; ++i) {
          set_scalar_bc(bc,phys_bc);
          swap_inflow_bc(bc);

          desc_lst.setComponent(Burn_Corr_Type, i,
                                std::format("sdc_react_source_{}", i),
                                bc, genericBoundaryFunc);
      }
  }
#endif

#ifdef RADIATION
  int ngrow = 1;
  int ncomp = Radiation::nGroups;
  desc_lst.addDescriptor(Radiation_Type, IndexType::TheCellType(),
                         StateDescriptor::Point, ngrow, ncomp,
                         interp);
  set_scalar_bc(bc,phys_bc);
  swap_inflow_bc(bc);

  if (ParallelDescriptor::IOProcessor()) {
    std::cout << "Radiation::nGroups = " << Radiation::nGroups << std::endl;
  }

  std::string rad_label;
  if (!Radiation::groups_enabled) {
    desc_lst
      .setComponent(Radiation_Type, Rad, "rad", bc,
                    genericBoundaryFunc);
  }
  else {
      for (int i = 0; i < Radiation::nGroups; i++) {
        rad_label = std::format("rad{}", i);
        desc_lst
          .setComponent(Radiation_Type, i, rad_label, bc,
                        genericBoundaryFunc);
      }
  }
#endif

#ifdef REACTIONS
  if (time_scheme == IterativeCorrection && sdc_accuracy_order == 4) {

    Forcing_Type_Index = desc_lst.size();

    store_in_checkpoint = false;
    desc_lst.addDescriptor(Forcing_Type_Index, IndexType::TheCellType(),
                           StateDescriptor::Point, 2, NUM_STATE,
                           interp, state_data_extrap, store_in_checkpoint);

    Vector<BCRec> sdc_forcing_bcs(NUM_STATE);
    Vector<std::string> sdc_forcing_names(NUM_STATE);

    for (int i = 0; i < NUM_STATE; ++i) {
      sdc_forcing_names[i] = name[i] + "_sdc_source";
      sdc_forcing_bcs[i] = bcs[i];

      swap_inflow_bc(sdc_forcing_bcs[i]);
    }

    desc_lst.setComponent(Forcing_Type_Index, URHO, sdc_forcing_names, sdc_forcing_bcs,
                          genericBoundaryFunc);
  }
#endif

  num_field_types = desc_lst.size();

  derive_lst.add("pressure",IndexType::TheCellType(),1,derive_pressure,the_same_box);
  derive_lst.addComponent("pressure",desc_lst,State_Type, URHO, NUM_STATE);

  derive_lst.add("kineng",IndexType::TheCellType(),1,derive_kinetic_energy,the_same_box);
  derive_lst.addComponent("kineng",desc_lst,State_Type, URHO, 1);
  derive_lst.addComponent("kineng",desc_lst,State_Type, UMX, 3);

  derive_lst.add("soundspeed",IndexType::TheCellType(),1,derive_sound_speed,the_same_box);
  derive_lst.addComponent("soundspeed",desc_lst,State_Type, URHO, NUM_STATE);

  derive_lst.add("Gamma_1",IndexType::TheCellType(),1,derive_adiabatic_index,the_same_box);
  derive_lst.addComponent("Gamma_1",desc_lst,State_Type, URHO, NUM_STATE);

  derive_lst.add("MachNumber",IndexType::TheCellType(),1,derive_mach_ratio,the_same_box);
  derive_lst.addComponent("MachNumber",desc_lst,State_Type, URHO, NUM_STATE);

#if (AMREX_SPACEDIM == 1)
  derive_lst.add("uplusc",IndexType::TheCellType(),1,derive_u_plus_c,the_same_box);
  derive_lst.addComponent("uplusc",desc_lst,State_Type, URHO, NUM_STATE);

  derive_lst.add("uminusc",IndexType::TheCellType(),1,derive_u_minus_c,the_same_box);
  derive_lst.addComponent("uminusc",desc_lst,State_Type, URHO, NUM_STATE);
#endif

  derive_lst.add("entropy",IndexType::TheCellType(),1,derive_entropy,the_same_box);
  derive_lst.addComponent("entropy",desc_lst,State_Type,URHO,NUM_STATE);

#ifdef DIFFUSION
  if (conduction_enabled) {
    derive_lst.add("thermal_cond",IndexType::TheCellType(),1,derive_conductivity,the_same_box);
    derive_lst.addComponent("thermal_cond",desc_lst,State_Type,URHO,NUM_STATE);

    derive_lst.add("diff_coeff",IndexType::TheCellType(),1,derive_diffusivity,the_same_box);
    derive_lst.addComponent("diff_coeff",desc_lst,State_Type,URHO,NUM_STATE);

    derive_lst.add("diffusion_term",IndexType::TheCellType(),1,derive_diffusion_term,grow_box_by_one);
    derive_lst.addComponent("diffusion_term",desc_lst,State_Type,URHO,NUM_STATE);

  }
#endif

  derive_lst.add("magvort",IndexType::TheCellType(),1,derive_vorticity,grow_box_by_one);
  if (UMX != URHO+1) {
    amrex::Error("We are assuming UMX = URHO + 1 in HydroSim_setup.cpp");
  }
  derive_lst.addComponent("magvort",desc_lst,State_Type,URHO,4);

  derive_lst.add("divu",IndexType::TheCellType(),1,derive_flow_divergence,grow_box_by_one);
  derive_lst.addComponent("divu",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("divu",desc_lst,State_Type,UMX,3);

  derive_lst.add("eint_E",IndexType::TheCellType(),1,derive_internal_from_total,the_same_box);
  derive_lst.addComponent("eint_E",desc_lst,State_Type,URHO,NUM_STATE);

  derive_lst.add("eint_e",IndexType::TheCellType(),1,derive_internal_from_stored,the_same_box);
  derive_lst.addComponent("eint_e",desc_lst,State_Type,URHO,NUM_STATE);

  derive_lst.add("logden",IndexType::TheCellType(),1,derive_log_density,the_same_box);
  derive_lst.addComponent("logden",desc_lst,State_Type,URHO,NUM_STATE);

  derive_lst.add("StateErr",IndexType::TheCellType(),3,derive_state_error,grow_box_by_one);
  derive_lst.addComponent("StateErr",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("StateErr",desc_lst,State_Type,UTEMP,1);
  derive_lst.addComponent("StateErr",desc_lst,State_Type,UFS,1);

  for (int i = 0; i < NumSpec; i++){
    std::string species_label = "X(" + short_spec_names_cxx[i] + ")";
    derive_lst.add(species_label,IndexType::TheCellType(),1,derive_mass_fraction,the_same_box);
    derive_lst.addComponent(species_label,desc_lst,State_Type,URHO,1);
    derive_lst.addComponent(species_label,desc_lst,State_Type,UFS+i,1);
  }

#ifndef AUX_THERMO
  derive_lst.add("abar",IndexType::TheCellType(),1,derive_atomic_mass,the_same_box);
  derive_lst.addComponent("abar",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("abar",desc_lst,State_Type,UFS,NumSpec);

#ifdef REACTIONS
  derive_lst.add("Ye",IndexType::TheCellType(),1,derive_elec_frac,the_same_box);
  derive_lst.addComponent("Ye",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("Ye",desc_lst,State_Type,UFS,NumSpec);
#endif
#endif

  derive_lst.add("x_velocity",IndexType::TheCellType(),1,derive_velocity_component,the_same_box);
  derive_lst.addComponent("x_velocity",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("x_velocity",desc_lst,State_Type,UMX,1);

  derive_lst.add("y_velocity",IndexType::TheCellType(),1,derive_velocity_component,the_same_box);
  derive_lst.addComponent("y_velocity",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("y_velocity",desc_lst,State_Type,UMY,1);

  derive_lst.add("z_velocity",IndexType::TheCellType(),1,derive_velocity_component,the_same_box);
  derive_lst.addComponent("z_velocity",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("z_velocity",desc_lst,State_Type,UMZ,1);

#ifdef REACTIONS
  derive_lst.add("t_sound_t_enuc", IndexType::TheCellType(), 1, derive_timescale_ratio, the_same_box);
  derive_lst.addComponent("t_sound_t_enuc", desc_lst, State_Type, URHO, NUM_STATE);
  derive_lst.addComponent("t_sound_t_enuc", desc_lst, Burning_Type, 0, 1);

  derive_lst.add("enuc", IndexType::TheCellType(), 1, derive_energy_generation, the_same_box);
  derive_lst.addComponent("enuc", desc_lst, State_Type, URHO, 1);
  derive_lst.addComponent("enuc", desc_lst, Burning_Type, 0, 1);
#endif

  derive_lst.add("magvel",IndexType::TheCellType(),1,derive_speed,the_same_box);
  derive_lst.addComponent("magvel",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("magvel",desc_lst,State_Type,UMX,3);

  derive_lst.add("radvel",IndexType::TheCellType(),1,derive_radial_velocity,the_same_box);
  derive_lst.addComponent("radvel",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("radvel",desc_lst,State_Type,UMX,3);

  derive_lst.add("circvel",IndexType::TheCellType(),1,derive_circular_speed,the_same_box);
  derive_lst.addComponent("circvel",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("circvel",desc_lst,State_Type,UMX,3);

  derive_lst.add("magmom",IndexType::TheCellType(),1,derive_momentum_magnitude,the_same_box);
  derive_lst.addComponent("magmom",desc_lst,State_Type,UMX,3);

  derive_lst.add("angular_momentum_x",IndexType::TheCellType(),1,derive_ang_momentum_x,the_same_box);
  derive_lst.addComponent("angular_momentum_x",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("angular_momentum_x",desc_lst,State_Type,UMX,3);

  derive_lst.add("angular_momentum_y",IndexType::TheCellType(),1,derive_ang_momentum_y,the_same_box);
  derive_lst.addComponent("angular_momentum_y",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("angular_momentum_y",desc_lst,State_Type,UMX,3);

  derive_lst.add("angular_momentum_z",IndexType::TheCellType(),1,derive_ang_momentum_z,the_same_box);
  derive_lst.addComponent("angular_momentum_z",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("angular_momentum_z",desc_lst,State_Type,UMX,3);

#ifdef GRAVITY
  derive_lst.add("maggrav",IndexType::TheCellType(),1,derive_gravity_magnitude,the_same_box);
  derive_lst.addComponent("maggrav",desc_lst,Accel_Type,0,3);
#endif

#ifdef AMREX_PARTICLES
  derive_lst.add("particle_count",IndexType::TheCellType(),1,derive_noop,the_same_box);
  derive_lst.addComponent("particle_count",desc_lst,State_Type,URHO,1);

  derive_lst.add("total_particle_count",IndexType::TheCellType(),1,derive_noop,the_same_box);
  derive_lst.addComponent("total_particle_count",desc_lst,State_Type,URHO,1);
#endif

#ifdef RADIATION
  if (Radiation::groups_enabled) {
    derive_lst.add("Ertot", IndexType::TheCellType(),1,derive_total_radiative_energy,the_same_box);
    derive_lst.addComponent("Ertot",desc_lst,Radiation_Type,0,Radiation::nGroups);
  }
#endif

#ifdef MHD
  derive_lst.add("B_x", IndexType::TheCellType(), 1, derive_cell_bfield_x, the_same_box);
  derive_lst.addComponent("B_x", desc_lst, Bfield_Type_x, 0, 1);
  derive_lst.add("B_y", IndexType::TheCellType(), 1, derive_cell_bfield_y, the_same_box);
  derive_lst.addComponent("B_y", desc_lst, Bfield_Type_y, 0, 1);
  derive_lst.add("B_z", IndexType::TheCellType(), 1, derive_cell_bfield_z, the_same_box);
  derive_lst.addComponent("B_z", desc_lst, Bfield_Type_z, 0, 1);

  derive_lst.add("Div_B", IndexType::TheCellType(), 1, derive_b_divergence, the_same_box);
  derive_lst.addComponent("Div_B", desc_lst, Bfield_Type_x, 0, 1);
  derive_lst.addComponent("Div_B", desc_lst, Bfield_Type_y, 0, 1);
  derive_lst.addComponent("Div_B", desc_lst, Bfield_Type_z, 0, 1);

#endif

#if NAUX_NET > 0
  for (int i = 0; i < NumAux; i++)  {
    derive_lst.add(short_aux_names_cxx[i], IndexType::TheCellType(), 1, derive_mass_fraction, the_same_box);
    derive_lst.addComponent(short_aux_names_cxx[i], desc_lst, State_Type, URHO, 1);
    derive_lst.addComponent(short_aux_names_cxx[i], desc_lst, State_Type, UFX+i, 1);
  }
#endif

#include <User_Derives.H>

  tag_field_names.emplace_back("density");
  tag_field_ghosts.push_back(1);

  tag_field_names.emplace_back("Temp");
  tag_field_ghosts.push_back(1);

  tag_field_names.emplace_back("pressure");
  tag_field_ghosts.push_back(1);

  tag_field_names.emplace_back("x_velocity");
  tag_field_ghosts.push_back(1);

#if (AMREX_SPACEDIM >= 2)
  tag_field_names.emplace_back("y_velocity");
  tag_field_ghosts.push_back(1);
#endif

#if (AMREX_SPACEDIM == 3)
  tag_field_names.emplace_back("z_velocity");
  tag_field_ghosts.push_back(1);
#endif

#ifdef REACTIONS
  tag_field_names.emplace_back("t_sound_t_enuc");
  tag_field_ghosts.push_back(0);

  tag_field_names.emplace_back("enuc");
  tag_field_ghosts.push_back(0);
#endif

#ifdef RADIATION
  if (do_radiative_transfer && !Radiation::groups_enabled) {
      tag_field_names.emplace_back("rad");
      tag_field_ghosts.push_back(1);
  }
#endif

  num_tag_fields_default = static_cast<int>(tag_field_names.size());

  forcing_names.resize(num_forcing);

  for (int n = 0; n < num_forcing; ++n) {
    forcing_names[n] = "";
  }

  forcing_names[external_source] = "user-defined external";
  forcing_names[thermodynamic_src] = "pdivU source";
#ifdef SPONGE
  forcing_names[damping_src] = "sponge";
#endif

#ifdef DIFFUSION
  forcing_names[conduction_src] = "diffusion";
#endif

#ifdef HYBRID_MOMENTUM
  forcing_names[hybrid_mom_source] = "hybrid";
#endif

#ifdef GRAVITY
  forcing_names[gravity_src] = "gravity";
#endif

#ifdef ROTATION
  forcing_names[rotation_src] = "rotation";
#endif

#ifdef TRUE_SDC
  if (sdc_quadrature == 0) {

    if (sdc_accuracy_order == 2) {
      COLLOC_NODES = 2;

      node_dt_frac.resize(COLLOC_NODES);
      node_dt_frac = {0.0, 1.0};

      node_quad_weights.resize(COLLOC_NODES);
      node_quad_weights = {0.5, 0.5};

    } else if (sdc_accuracy_order == 4) {
      COLLOC_NODES = 3;

      node_dt_frac.resize(COLLOC_NODES);
      node_dt_frac = {0.0, 0.5, 1.0};

      node_quad_weights.resize(COLLOC_NODES);
      node_quad_weights = {1.0/6.0, 4.0/6.0, 1.0/6.0};

    } else {
      amrex::Error("invalid value of sdc_accuracy_order");
    }

  } else if (sdc_quadrature == 1) {

    if (sdc_accuracy_order == 2) {
      COLLOC_NODES = 3;

      node_dt_frac.resize(COLLOC_NODES);
      node_dt_frac = {0.0, 1.0/3.0, 1.0};

      node_quad_weights.resize(COLLOC_NODES);
      node_quad_weights = {0.0, 3.0/4.0, 1.0/4.0};

    } else if (sdc_accuracy_order == 4) {
      COLLOC_NODES = 4;

      node_dt_frac.resize(COLLOC_NODES);
      node_dt_frac = {0.0, (4.0 - std::sqrt(6.0))/10.0, (4.0 + std::sqrt(6.0))/10.0, 1.0};

      node_quad_weights.resize(COLLOC_NODES);
      node_quad_weights = {0.0, (16.0 - std::sqrt(6.0))/36.0, (16.0 + std::sqrt(6.0))/36.0, 1.0/9.0};

    } else {
      amrex::Error("invalid value of sdc_accuracy_order");
    }

  } else {
    amrex::Error("invalid value of sdc_quadrature");
  }
#endif

}
