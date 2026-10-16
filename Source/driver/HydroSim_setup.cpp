#include <cstdio>

#include <algorithm>
#include <array>
#include <format>

#include <AMReX_LevelBld.H>
#include <AMReX_ParmParse.H>
#include <HydroSim.H>
#include <HydroSim_bc_fill.H>
#include <HydroSim_default_fill.H>
#include <Derive.H>
#include <runtime_parameters.H>
#ifdef RADIATION
#include <Radiation.H>
#include <RadDerive.H>
#include <opacity.H>
#endif

#include <AMReX_buildInfo.H>
#include <eos.H>
#include <ambient.H>

using std::string;
using namespace amrex;

using BndryFunc = StateDescriptor::BndryFunc;

namespace {

    Box the_same_box (const Box& b) { return b; }
    Box grow_box_by_one (const Box& b) { return amrex::grow(b,1); }

    //
    // Components are:
    //  Interior, Inflow, Outflow,  Symmetry,     SlipWall,     NoSlipWall
    //
    using BCTypeMap = std::array<int, 6>;

    constexpr BCTypeMap scalar_bc{
        amrex::BCType::int_dir,
        amrex::BCType::ext_dir,
        amrex::BCType::foextrap,
        amrex::BCType::reflect_even,
        amrex::BCType::reflect_even,
        amrex::BCType::reflect_even
    };

    constexpr BCTypeMap norm_vel_bc{
        amrex::BCType::int_dir,
        amrex::BCType::ext_dir,
        amrex::BCType::foextrap,
        amrex::BCType::reflect_odd,
        amrex::BCType::reflect_odd,
        amrex::BCType::reflect_odd
    };

    constexpr BCTypeMap tang_vel_bc{
        amrex::BCType::int_dir,
        amrex::BCType::ext_dir,
        amrex::BCType::foextrap,
        amrex::BCType::reflect_even,
        amrex::BCType::reflect_even,
        amrex::BCType::reflect_even
    };

#ifdef MHD
    constexpr BCTypeMap mag_field_bc{
        amrex::BCType::int_dir,
        amrex::BCType::ext_dir,
        amrex::BCType::foextrap,
        amrex::BCType::reflect_even,
        amrex::BCType::foextrap,
        amrex::BCType::hoextrap
    };
#endif

    void
    set_scalar_bc (BCRec& bc, const BCRec& phys_bc)
    {
        const int* lo_bc = phys_bc.lo();
        const int* hi_bc = phys_bc.hi();
        for (int i = 0; i < AMREX_SPACEDIM; i++)
        {
            bc.setLo(i,scalar_bc[lo_bc[i]]);
            bc.setHi(i,scalar_bc[hi_bc[i]]);
        }
    }

    // Velocity-like vector components: at symmetry and wall boundaries
    // the component normal to the boundary is reflected with odd parity,
    // the tangential ones with even parity.

    template <int normal_dir>
    void
    set_vel_bc (BCRec& bc, const BCRec& phys_bc)
    {
        const int* lo_bc = phys_bc.lo();
        const int* hi_bc = phys_bc.hi();
        for (int i = 0; i < AMREX_SPACEDIM; i++)
        {
            const auto& vel_bc = (i == normal_dir) ? norm_vel_bc : tang_vel_bc;
            bc.setLo(i, vel_bc[lo_bc[i]]);
            bc.setHi(i, vel_bc[hi_bc[i]]);
        }
    }

#ifdef MHD
    void
    set_mag_field_bc(BCRec& bc, const BCRec& phys_bc)
    {
        const int* lo_bc = phys_bc.lo();
        const int* hi_bc = phys_bc.hi();
        for (int i = 0; i < AMREX_SPACEDIM; i++)
        {
            bc.setLo(i, mag_field_bc[lo_bc[i]]);
            bc.setHi(i, mag_field_bc[hi_bc[i]]);
        }
    }
#endif

    // In some cases we want to replace inflow boundaries with
    // first-order extrapolation boundaries. This is intended to
    // be used for state data that the user is not going to
    // provide inflow boundary conditions for, like gravity
    // and reactions, and it works in conjunction with the
    // generic_fill boundary routine.

    void
    replace_inflow_bc (BCRec& bc)
    {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (bc.lo(dir) == amrex::BCType::ext_dir) {
                bc.setLo(dir, amrex::BCType::foextrap);
            }
            if (bc.hi(dir) == amrex::BCType::ext_dir) {
                bc.setHi(dir, amrex::BCType::foextrap);
            }
        }
    }
}

void
HydroSim::variableSetUp ()
{

  // HydroSim::variableSetUp is called in the constructor of Amr.cpp, so
  // it should get called every time we start or restart a job


  // initialize the start time for our CPU-time tracker
  runStartCPUTime = ParallelDescriptor::second();


  // Output the git commit hashes used to build the executable.

  if (ParallelDescriptor::IOProcessor()) {

    const char* hydrosim_hash  = buildInfoGetGitHash(1);
    const char* amrex_hash   = buildInfoGetGitHash(2);
    const char* microphysics_hash = buildInfoGetGitHash(3);
    const char* buildgithash = buildInfoGetBuildGitHash();
    const char* buildgitname = buildInfoGetBuildGitName();

    if (strlen(hydrosim_hash) > 0) {
      std::cout << "\n" << "HydroSim git describe: " << hydrosim_hash << "\n";
    }
    if (strlen(amrex_hash) > 0) {
      std::cout << "AMReX git describe: " << amrex_hash << "\n";
    }
    if (strlen(microphysics_hash) > 0) {
      std::cout << "Microphysics git describe: " << microphysics_hash << "\n";
    }
    if (strlen(buildgithash) > 0){
      std::cout << buildgitname << " git describe: " << buildgithash << "\n";
    }

    std::cout << "\n";
  }

  BL_ASSERT(desc_lst.size() == 0);

  // read the C++ parameters that are set in inputs and do other
  // initializations (e.g., set phys_bc)
  read_params();

  // initialize the C++ values of the problem-specific runtime parameters.

  init_prob_parameters();

  // Initialize the runtime parameters for any of the external
  // microphysics
  init_external_params();

  // set small positive values of the "small" quantities if they are
  // negative
  if (small_dens < 0.0_rt) {
    small_dens = 1.e-100_rt;
  }

  if (small_temp < 0.0_rt) {
    small_temp = 1.e-100_rt;
  }

  if (small_pres < 0.0_rt) {
    small_pres = 1.e-100_rt;
  }

  if (small_ener < 0.0_rt) {
    small_ener = 1.e-100_rt;
  }

  // now initialize the C++ Microphysics
#ifdef REACTIONS
  network_init();
#endif

  eos_init(hydrosim::small_temp, hydrosim::small_dens);

#ifdef RADIATION
  opacity_init();
#endif

  // Ensure that HydroSim's small variables are consistent
  // with the minimum permitted by the EOS, and vice versa.

  Real new_min_T = std::max(small_temp, EOSData::mintemp);
  small_temp = new_min_T;
  EOSData::mintemp = new_min_T;

  Real new_min_rho = std::max(small_dens, EOSData::mindens);
  small_dens = new_min_rho;
  EOSData::mindens = new_min_rho;

  // Given small_temp and small_dens, compute small_pres
  // and small_ener, assuming a more restrictive value is
  // not already provided by the user. We'll arbitrarily
  // set the mass fraction for this call, since we presumably
  // don't need to be too accurate, we just need to set a
  // reasonable floor.

  eos_t eos_state;

  eos_state.rho = hydrosim::small_dens;
  eos_state.T = hydrosim::small_temp;
  std::fill(std::begin(eos_state.xn), std::end(eos_state.xn), 1.0_rt / NumSpec);
#ifdef AUX_THERMO
  set_aux_comp_from_X(eos_state);
#endif

  eos(eos_input_rt, eos_state);

  hydrosim::small_pres = amrex::max(hydrosim::small_pres, eos_state.p);
  hydrosim::small_ener = amrex::max(hydrosim::small_ener, eos_state.e);

  // some consistency checks on the parameters
#ifdef REACTIONS
#ifdef TRUE_SDC
  // for TRUE_SDC, we don't support retry
  if (use_retry) {
    amrex::Warning("use_retry = 1 is not supported with true SDC.  Disabling");
    use_retry = false;
  }
#endif
#endif

  // Set some initial data in the ambient state for safety, though the
  // intent is that any problems using this may override these. We use
  // the user-specified parameters if they were set, but if they were
  // not (which is reflected by whether ambient_density is positive)
  // then we use the "small" quantities.

  std::fill(std::begin(ambient::ambient_state), std::end(ambient::ambient_state), 0.0);

  ambient::ambient_state[URHO]  = amrex::max(hydrosim::ambient_density, hydrosim::small_dens);
  ambient::ambient_state[UTEMP] = amrex::max(hydrosim::ambient_temp, hydrosim::small_temp);
  ambient::ambient_state[UEINT] = ambient::ambient_state[URHO] * amrex::max(hydrosim::ambient_energy,
                                                                            hydrosim::small_ener);
  ambient::ambient_state[UEDEN] = ambient::ambient_state[UEINT];
  for (int n = 0; n < NumSpec; ++n) {
      ambient::ambient_state[USPEC+n] = ambient::ambient_state[URHO] * (1.0_rt / NumSpec);
  }

  MFInterpolater* interp = nullptr;

  if (!(state_interp_order == 0)) {
    if (lin_limit_state_interp == 2) {
      interp = &mf_linear_slope_minmax_interp;
    }
    else if (!(lin_limit_state_interp == 1)) {
      interp = &mf_cell_cons_interp;
    } else {
      interp = &mf_lincc_interp;
    }
  } else {
    interp = &mf_pc_interp;
  }

  // Note that the default is state_data_extrap = false,
  // store_in_checkpoint = true.  We only need to put these in
  // explicitly if we want to do something different,
  // like not store the state data in a checkpoint directory
  bool state_data_extrap = false;
  bool store_in_checkpoint;

  int ngrow_state = 0;

  store_in_checkpoint = true;
  desc_lst.addDescriptor(State_Type,IndexType::TheCellType(),
                         StateDescriptor::Point,ngrow_state,NUM_STATE,
                         interp,state_data_extrap,store_in_checkpoint);

#ifdef MHD
  store_in_checkpoint = true;
  IndexType xface(IntVect{AMREX_D_DECL(1,0,0)});
  desc_lst.addDescriptor(Mag_Type_x, xface,
                         StateDescriptor::Point, 0, 1,
                         interp, state_data_extrap,
                         store_in_checkpoint);
  IndexType yface(IntVect{AMREX_D_DECL(0,1,0)});
  desc_lst.addDescriptor(Mag_Type_y, yface,
                         StateDescriptor::Point, 0, 1,
                         interp, state_data_extrap,
                         store_in_checkpoint);
  IndexType zface(IntVect{AMREX_D_DECL(0,0,1)});
  desc_lst.addDescriptor(Mag_Type_z, zface,
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
  desc_lst.addDescriptor(Gravity_Type,IndexType::TheCellType(),
                         StateDescriptor::Point,NUM_GROW_SRC,3,
                         interp,state_data_extrap,store_in_checkpoint);
#endif

  // Source terms -- for the CTU method, because we do characteristic
  // tracing on the source terms, we need NUM_GROW_SRC ghost cells to do
  // the reconstruction.  For SDC, on the other hand, we only
  // need 1 (for the fourth-order stuff). Simplified SDC uses the CTU
  // advance, so it behaves the same way as CTU here.

  store_in_checkpoint = true;
  int source_ng = 0;
  if (time_integration_method == CornerTransportUpwind ||
      time_integration_method == SimplifiedSpectralDeferredCorrections) {  // NOLINT(bugprone-branch-clone)
      source_ng = NUM_GROW_SRC;
  } else if (time_integration_method == SpectralDeferredCorrections) {
#ifdef SHOCK_VAR
      source_ng = NUM_GROW_SRC;
#else
      if (sdc_order == 2 && use_pslope) {
          source_ng = NUM_GROW_SRC;
      } else {
          source_ng = 1;
      }
#endif
  } else {
      amrex::Error("Unknown time_integration_method");
  }
  desc_lst.addDescriptor(Source_Type, IndexType::TheCellType(),
                         StateDescriptor::Point, source_ng, NSRC,
                         interp, state_data_extrap, store_in_checkpoint);


#ifdef REACTIONS
  // Component is  rho_enuc = rho * (eout-ein)
  // next NumSpec are rho * omegadot_i
  // next NumAux are rho * auxdot_i
  // next is nse status indication
  store_in_checkpoint = false;

  int num_react = 1;

  if (store_omegadot == 1) {
      num_react += NumSpec + NumAux;
  }
#ifdef NSE
  num_react += 1;
#endif
  desc_lst.addDescriptor(Reactions_Type,IndexType::TheCellType(),
                         StateDescriptor::Point, 0, num_react,
                         interp, state_data_extrap, store_in_checkpoint);
#endif

#ifdef REACTIONS
  // For simplified SDC, we want to store the reactions source.
  // these are not traced, so we only need a single ghost cell

  // note: this is of size NQ (and not NQSRC) because the species are
  // always included

  if (time_integration_method == SimplifiedSpectralDeferredCorrections) {

      store_in_checkpoint = true;
      desc_lst.addDescriptor(SDC_React_Source_Type, IndexType::TheCellType(),
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

  set_vel_bc<0>(bc, phys_bc);
  bcs[UMX] = bc;
  name[UMX] = "xmom";

  set_vel_bc<1>(bc, phys_bc);
  bcs[UMY] = bc;
  name[UMY] = "ymom";

  set_vel_bc<2>(bc, phys_bc);
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

  for (int i=0; i<NumAdv; ++i)
    {
      set_scalar_bc(bc, phys_bc);
      bcs[UADV+i] = bc;
      name[UADV+i] = std::format("adv_{}", i);
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
      bcs[USPEC+i] = bc;
      name[USPEC+i] = "rho_" + short_spec_names_cxx[i];
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
      bcs[UAUXV+i] = bc;
      name[UAUXV+i] = "rho_" + short_aux_names_cxx[i];
    }
#endif

#ifdef SHOCK_VAR
  set_scalar_bc(bc, phys_bc);
  bcs[USHOCK] = bc;
  name[USHOCK] = "Shock";
#endif

#ifdef NSE_NET
  set_scalar_bc(bc, phys_bc);
  bcs[UMU_P] = bc;
  name[UMU_P] = "mu_p";

  set_scalar_bc(bc, phys_bc);
  bcs[UMU_N] = bc;
  name[UMU_N] = "mu_n";
#endif

  BndryFunc stateBndryFunc(fill_state_bc);
  stateBndryFunc.setRunOnGPU(true);

  desc_lst.setComponent(State_Type,
                        URHO,
                        name,
                        bcs,
                        stateBndryFunc);

  BndryFunc defaultBndryFunc(default_bc_fill);
  defaultBndryFunc.setRunOnGPU(true);

#ifdef MHD
  set_mag_field_bc(bc, phys_bc);
  desc_lst.setComponent(Mag_Type_x, 0, "b_x", bc, defaultBndryFunc);
  desc_lst.setComponent(Mag_Type_y, 0, "b_y", bc, defaultBndryFunc);
  desc_lst.setComponent(Mag_Type_z, 0, "b_z", bc, defaultBndryFunc);
#endif



#ifdef GRAVITY
  set_scalar_bc(bc,phys_bc);
  replace_inflow_bc(bc);
  desc_lst.setComponent(Potential_Type,0,"gravPotential",bc,defaultBndryFunc);
  set_vel_bc<0>(bc,phys_bc);
  replace_inflow_bc(bc);
  desc_lst.setComponent(Gravity_Type,0,"grav_x",bc,defaultBndryFunc);
  set_vel_bc<1>(bc,phys_bc);
  replace_inflow_bc(bc);
  desc_lst.setComponent(Gravity_Type,1,"grav_y",bc,defaultBndryFunc);
  set_vel_bc<2>(bc,phys_bc);
  replace_inflow_bc(bc);
  desc_lst.setComponent(Gravity_Type,2,"grav_z",bc,defaultBndryFunc);
#endif

  // Source term array will use source fill

  Vector<BCRec> source_bcs(NSRC);
  Vector<std::string> state_type_source_names(NSRC);

  for (int i = 0; i < NSRC; ++i) {
    state_type_source_names[i] = name[i] + "_source";
    source_bcs[i] = bcs[i];

    // Replace any instances of inflow boundaries with first-order extrapolation.
    replace_inflow_bc(source_bcs[i]);
  }

  desc_lst.setComponent(Source_Type, URHO, state_type_source_names, source_bcs, defaultBndryFunc);

#ifdef REACTIONS
  desc_lst.setComponent(Reactions_Type, 0, "rho_enuc", bc, defaultBndryFunc);

  if (store_omegadot == 1) {

      // Reactions_Type includes the species -- we put those after rho_enuc
      std::string name_react;
      for (int i = 0; i < NumSpec; ++i)
      {
          set_scalar_bc(bc,phys_bc);
          replace_inflow_bc(bc);
          name_react = "rho_omegadot_" + short_spec_names_cxx[i];
          desc_lst.setComponent(Reactions_Type, 1+i, name_react, bc,defaultBndryFunc);
      }
#if NAUX_NET > 0
      std::string name_aux;
      for (int i = 0; i < NumAux; ++i) {
          set_scalar_bc(bc,phys_bc);
          replace_inflow_bc(bc);
          name_aux = "rho_auxdot_" + short_aux_names_cxx[i];
          desc_lst.setComponent(Reactions_Type, 1+NumSpec+i, name_aux, bc, defaultBndryFunc);
      }
#endif
  }
#ifdef NSE
  set_scalar_bc(bc,phys_bc);
  replace_inflow_bc(bc);
  if (store_omegadot == 1) {
    desc_lst.setComponent(Reactions_Type, NumSpec+NumAux+1, "in_nse", bc, defaultBndryFunc);
  }
  else {
    desc_lst.setComponent(Reactions_Type, 1, "in_nse", bc, defaultBndryFunc);
  }
#endif
  // names for the burn_weights that are manually added to the plotfile

  if (store_burn_weights) {

      if (hydrosim::time_integration_method == CornerTransportUpwind) {
          burn_weight_names.emplace_back("burn_weights_firsthalf");
          burn_weight_names.emplace_back("burn_weights_secondhalf");
      }
      else if (hydrosim::time_integration_method == SimplifiedSpectralDeferredCorrections) {
          for (int n = 0; n < sdc_iters+1; n++) {
              burn_weight_names.emplace_back(std::format("burn_weights_iter_{}", n + 1));
          }
      }
  }
#endif

#ifdef REACTIONS
  if (time_integration_method == SimplifiedSpectralDeferredCorrections) {
      for (int i = 0; i < NQ; ++i) {
          set_scalar_bc(bc,phys_bc);
          replace_inflow_bc(bc);

          desc_lst.setComponent(SDC_React_Source_Type, i,
                                std::format("sdc_react_source_{}", i),
                                bc, defaultBndryFunc);
      }
  }
#endif

#ifdef RADIATION
  int ngrow = 1;
  int ncomp = Radiation::nGroups;
  desc_lst.addDescriptor(Rad_Type, IndexType::TheCellType(),
                         StateDescriptor::Point, ngrow, ncomp,
                         interp);
  set_scalar_bc(bc,phys_bc);
  replace_inflow_bc(bc);

  if (ParallelDescriptor::IOProcessor()) {
    std::cout << "Radiation::nGroups = " << Radiation::nGroups << std::endl;
  }

  std::string rad_name;
  if (!Radiation::do_multigroup) {
    desc_lst
      .setComponent(Rad_Type, Rad, "rad", bc,
                    defaultBndryFunc);
  }
  else {
      for (int i = 0; i < Radiation::nGroups; i++) {
        rad_name = std::format("rad{}", i);
        desc_lst
          .setComponent(Rad_Type, i, rad_name, bc,
                        defaultBndryFunc);
      }
  }
#endif

  // some optional State_Type's -- since these depend on the value of
  // runtime parameters, we don't add these to the enum, but instead
  // add them to the count of State_Type's if we will use them

#ifdef REACTIONS
  if (time_integration_method == SpectralDeferredCorrections && sdc_order == 4) {

    // we are doing 4th order reactive SDC.  We need 2 ghost cells here
    SDC_Source_Type = desc_lst.size();

    store_in_checkpoint = false;
    desc_lst.addDescriptor(SDC_Source_Type, IndexType::TheCellType(),
                           StateDescriptor::Point, 2, NUM_STATE,
                           interp, state_data_extrap, store_in_checkpoint);

    // this is the same thing we do for the sources, but now we use the generic fill
    Vector<BCRec> sdc_source_bcs(NUM_STATE);
    Vector<std::string> sdc_source_names(NUM_STATE);

    for (int i = 0; i < NUM_STATE; ++i) {
      sdc_source_names[i] = name[i] + "_sdc_source";
      sdc_source_bcs[i] = bcs[i];

      // Replace any instances of inflow boundaries with first-order extrapolation.
      replace_inflow_bc(sdc_source_bcs[i]);
    }

    desc_lst.setComponent(SDC_Source_Type, URHO, sdc_source_names, sdc_source_bcs,
                          defaultBndryFunc);
  }
#endif

  n_state_types = desc_lst.size();

  //
  // DEFINE DERIVED QUANTITIES
  //
  // Pressure
  //
  derive_lst.add("pressure",IndexType::TheCellType(),1,derived_fields::derive_pressure,the_same_box);
  derive_lst.addComponent("pressure",desc_lst,State_Type, URHO, NUM_STATE);

  //
  // Kinetic energy
  //
  derive_lst.add("kineng",IndexType::TheCellType(),1,derived_fields::derive_kinetic_energy,the_same_box);
  derive_lst.addComponent("kineng",desc_lst,State_Type, URHO, 1);
  derive_lst.addComponent("kineng",desc_lst,State_Type, UMX, 3);

  //
  // Sound speed (c)
  //
  derive_lst.add("soundspeed",IndexType::TheCellType(),1,derived_fields::derive_sound_speed,the_same_box);
  derive_lst.addComponent("soundspeed",desc_lst,State_Type, URHO, NUM_STATE);

  //
  // Gamma_1
  //
  derive_lst.add("Gamma_1",IndexType::TheCellType(),1,derived_fields::derive_gamma1,the_same_box);
  derive_lst.addComponent("Gamma_1",desc_lst,State_Type, URHO, NUM_STATE);

  //
  // Mach number(M)
  //
  derive_lst.add("LocalMach",IndexType::TheCellType(),1,derived_fields::derive_mach,the_same_box);
  derive_lst.addComponent("LocalMach",desc_lst,State_Type, URHO, NUM_STATE);

#if (AMREX_SPACEDIM == 1)
  //
  // Wave speed u+c
  //
  derive_lst.add("u_plus_cs",IndexType::TheCellType(),1,derived_fields::derive_u_plus_cs,the_same_box);
  derive_lst.addComponent("u_plus_cs",desc_lst,State_Type, URHO, NUM_STATE);

  //
  // Wave speed u-c
  //
  derive_lst.add("u_minus_cs",IndexType::TheCellType(),1,derived_fields::derive_u_minus_cs,the_same_box);
  derive_lst.addComponent("u_minus_cs",desc_lst,State_Type, URHO, NUM_STATE);
#endif

  //
  // Entropy (S)
  //
  derive_lst.add("entropy",IndexType::TheCellType(),1,derived_fields::derive_entropy,the_same_box);
  derive_lst.addComponent("entropy",desc_lst,State_Type,URHO,NUM_STATE);

#ifdef DIFFUSION
  if (diffuse_temp) {
    //
    // thermal conductivity (k_th)
    //
    derive_lst.add("thermal_cond",IndexType::TheCellType(),1,derived_fields::derive_conductivity,the_same_box);
    derive_lst.addComponent("thermal_cond",desc_lst,State_Type,URHO,NUM_STATE);


    //
    // thermal diffusivity (k_th/(rho c_v))
    //
    derive_lst.add("diff_coeff",IndexType::TheCellType(),1,derived_fields::derive_diffusivity,the_same_box);
    derive_lst.addComponent("diff_coeff",desc_lst,State_Type,URHO,NUM_STATE);


    //
    // diffusion term (the divergence of thermal flux)
    //
    derive_lst.add("diff_term",IndexType::TheCellType(),1,derived_fields::derive_conduction_term,grow_box_by_one);
    derive_lst.addComponent("diff_term",desc_lst,State_Type,URHO,NUM_STATE);


  }
#endif

  //
  // Vorticity
  //
  derive_lst.add("vorticity_norm",IndexType::TheCellType(),1,derived_fields::derive_vorticity,grow_box_by_one);
  // Here we exploit the fact that UMX = URHO + 1
  //   in order to use the correct interpolation.
  if (UMX != URHO+1) {
    amrex::Error("We are assuming UMX = URHO + 1 in HydroSim_setup.cpp");
  }
  derive_lst.addComponent("vorticity_norm",desc_lst,State_Type,URHO,4);

  //
  // Div(u)
  //
  derive_lst.add("divu",IndexType::TheCellType(),1,derived_fields::derive_div_velocity,grow_box_by_one);
  derive_lst.addComponent("divu",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("divu",desc_lst,State_Type,UMX,3);

  //
  // Internal energy as derived from rho*E, part of the state
  //
  derive_lst.add("eint_from_rhoE",IndexType::TheCellType(),1,derived_fields::derive_eint_from_total,the_same_box);
  derive_lst.addComponent("eint_from_rhoE",desc_lst,State_Type,URHO,NUM_STATE);

  //
  // Internal energy as derived from rho*e, part of the state
  //
  derive_lst.add("eint_from_rhoe",IndexType::TheCellType(),1,derived_fields::derive_eint_from_rhoe,the_same_box);
  derive_lst.addComponent("eint_from_rhoe",desc_lst,State_Type,URHO,NUM_STATE);

  //
  // Log(density)
  //
  derive_lst.add("log10_density",IndexType::TheCellType(),1,derived_fields::derive_log_density,the_same_box);
  derive_lst.addComponent("log10_density",desc_lst,State_Type,URHO,NUM_STATE);

  derive_lst.add("TagState",IndexType::TheCellType(),3,derived_fields::derive_state_copy,grow_box_by_one);
  derive_lst.addComponent("TagState",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("TagState",desc_lst,State_Type,UTEMP,1);
  derive_lst.addComponent("TagState",desc_lst,State_Type,USPEC,1);

  //
  // X from rhoX
  //
  for (int i = 0; i < NumSpec; i++){
    std::string spec_string = "X(" + short_spec_names_cxx[i] + ")";
    derive_lst.add(spec_string,IndexType::TheCellType(),1,derived_fields::derive_species,the_same_box);
    derive_lst.addComponent(spec_string,desc_lst,State_Type,URHO,1);
    derive_lst.addComponent(spec_string,desc_lst,State_Type,USPEC+i,1);
  }

#ifndef AUX_THERMO
  //
  // Abar and Ye
  // note: if we are using aux thermodynamics, then abar is already an aux quantity
  //
  derive_lst.add("abar",IndexType::TheCellType(),1,derived_fields::derive_mean_mass_number,the_same_box);
  derive_lst.addComponent("abar",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("abar",desc_lst,State_Type,USPEC,NumSpec);

#ifdef REACTIONS
  derive_lst.add("Ye",IndexType::TheCellType(),1,derived_fields::derive_electron_fraction,the_same_box);
  derive_lst.addComponent("Ye",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("Ye",desc_lst,State_Type,USPEC,NumSpec);
#endif
#endif

  //
  // Velocities
  //
  derive_lst.add("x_velocity",IndexType::TheCellType(),1,derived_fields::derive_velocity,the_same_box);
  derive_lst.addComponent("x_velocity",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("x_velocity",desc_lst,State_Type,UMX,1);

  derive_lst.add("y_velocity",IndexType::TheCellType(),1,derived_fields::derive_velocity,the_same_box);
  derive_lst.addComponent("y_velocity",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("y_velocity",desc_lst,State_Type,UMY,1);

  derive_lst.add("z_velocity",IndexType::TheCellType(),1,derived_fields::derive_velocity,the_same_box);
  derive_lst.addComponent("z_velocity",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("z_velocity",desc_lst,State_Type,UMZ,1);

#ifdef REACTIONS
  //
  // Nuclear energy generation timescale t_e == e / edot
  // Sound-crossing time t_s == dx / c_s
  // Ratio of these is t_s_t_e == t_s / t_e
  //
  derive_lst.add("t_sound_t_enuc", IndexType::TheCellType(), 1, derived_fields::derive_burn_time_ratio, the_same_box);
  derive_lst.addComponent("t_sound_t_enuc", desc_lst, State_Type, URHO, NUM_STATE);
  derive_lst.addComponent("t_sound_t_enuc", desc_lst, Reactions_Type, 0, 1);

  //
  // Nuclear energy generation rate
  //
  derive_lst.add("enuc", IndexType::TheCellType(), 1, derived_fields::derive_energy_rate, the_same_box);
  derive_lst.addComponent("enuc", desc_lst, State_Type, URHO, 1);
  derive_lst.addComponent("enuc", desc_lst, Reactions_Type, 0, 1);
#endif

  derive_lst.add("flow_speed",IndexType::TheCellType(),1,derived_fields::derive_speed,the_same_box);
  derive_lst.addComponent("flow_speed",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("flow_speed",desc_lst,State_Type,UMX,3);

  derive_lst.add("radial_velocity",IndexType::TheCellType(),1,derived_fields::derive_radial_velocity,the_same_box);
  derive_lst.addComponent("radial_velocity",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("radial_velocity",desc_lst,State_Type,UMX,3);

  derive_lst.add("circular_velocity",IndexType::TheCellType(),1,derived_fields::derive_circular_velocity,the_same_box);
  derive_lst.addComponent("circular_velocity",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("circular_velocity",desc_lst,State_Type,UMX,3);

  derive_lst.add("momentum_norm",IndexType::TheCellType(),1,derived_fields::derive_momentum_magnitude,the_same_box);
  derive_lst.addComponent("momentum_norm",desc_lst,State_Type,UMX,3);

  derive_lst.add("angular_momentum_x",IndexType::TheCellType(),1,derived_fields::derive_ang_mom_x,the_same_box);
  derive_lst.addComponent("angular_momentum_x",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("angular_momentum_x",desc_lst,State_Type,UMX,3);

  derive_lst.add("angular_momentum_y",IndexType::TheCellType(),1,derived_fields::derive_ang_mom_y,the_same_box);
  derive_lst.addComponent("angular_momentum_y",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("angular_momentum_y",desc_lst,State_Type,UMX,3);

  derive_lst.add("angular_momentum_z",IndexType::TheCellType(),1,derived_fields::derive_ang_mom_z,the_same_box);
  derive_lst.addComponent("angular_momentum_z",desc_lst,State_Type,URHO,1);
  derive_lst.addComponent("angular_momentum_z",desc_lst,State_Type,UMX,3);

#ifdef GRAVITY
  derive_lst.add("gravity_norm",IndexType::TheCellType(),1,derived_fields::derive_grav_magnitude,the_same_box);
  derive_lst.addComponent("gravity_norm",desc_lst,Gravity_Type,0,3);
#endif

#ifdef AMREX_PARTICLES
  //
  // We want a derived type that corresponds to the number of particles
  // in each cell.  We only intend to use it in plotfiles for debugging
  // purposes.  We'll actually set the values in writePlotFile().
  //
  derive_lst.add("particle_count",IndexType::TheCellType(),1,derived_fields::derive_nothing,the_same_box);
  derive_lst.addComponent("particle_count",desc_lst,State_Type,URHO,1);

  derive_lst.add("total_particle_count",IndexType::TheCellType(),1,derived_fields::derive_nothing,the_same_box);
  derive_lst.addComponent("total_particle_count",desc_lst,State_Type,URHO,1);
#endif

#ifdef RADIATION
  if (Radiation::do_multigroup) {
    derive_lst.add("Ertot", IndexType::TheCellType(),1,derived_fields::derive_total_rad_energy,the_same_box);
    derive_lst.addComponent("Ertot",desc_lst,Rad_Type,0,Radiation::nGroups);
  }
#endif

#ifdef MHD
//Electric Field at the face
//
//Magentic Field Cell Centered
//x component
  derive_lst.add("B_x", IndexType::TheCellType(), 1, derived_fields::derive_b_center_x, the_same_box);
  derive_lst.addComponent("B_x", desc_lst, Mag_Type_x, 0, 1);
//y component
  derive_lst.add("B_y", IndexType::TheCellType(), 1, derived_fields::derive_b_center_y, the_same_box);
  derive_lst.addComponent("B_y", desc_lst, Mag_Type_y, 0, 1);
//z component
  derive_lst.add("B_z", IndexType::TheCellType(), 1, derived_fields::derive_b_center_z, the_same_box);
  derive_lst.addComponent("B_z", desc_lst, Mag_Type_z, 0, 1);

//Divergence of B
  derive_lst.add("Div_B", IndexType::TheCellType(), 1, derived_fields::derive_div_b, the_same_box);
  derive_lst.addComponent("Div_B", desc_lst, Mag_Type_x, 0, 1);
  derive_lst.addComponent("Div_B", desc_lst, Mag_Type_y, 0, 1);
  derive_lst.addComponent("Div_B", desc_lst, Mag_Type_z, 0, 1);

#endif


#if NAUX_NET > 0
  for (int i = 0; i < NumAux; i++)  {
    derive_lst.add(short_aux_names_cxx[i], IndexType::TheCellType(), 1, derived_fields::derive_species, the_same_box);
    derive_lst.addComponent(short_aux_names_cxx[i], desc_lst, State_Type, URHO, 1);
    derive_lst.addComponent(short_aux_names_cxx[i], desc_lst, State_Type, UAUXV+i, 1);
  }
#endif

  //
  // Problem-specific adds
#include <Problem_Derives.H>

  //
  // DEFINE ERROR ESTIMATION QUANTITIES
  //
  tag_field_names.emplace_back("density");
  tag_field_ngrow.push_back(1);

  tag_field_names.emplace_back("Temp");
  tag_field_ngrow.push_back(1);

  tag_field_names.emplace_back("pressure");
  tag_field_ngrow.push_back(1);

  tag_field_names.emplace_back("x_velocity");
  tag_field_ngrow.push_back(1);

#if (AMREX_SPACEDIM >= 2)
  tag_field_names.emplace_back("y_velocity");
  tag_field_ngrow.push_back(1);
#endif

#if (AMREX_SPACEDIM == 3)
  tag_field_names.emplace_back("z_velocity");
  tag_field_ngrow.push_back(1);
#endif

#ifdef REACTIONS
  tag_field_names.emplace_back("t_sound_t_enuc");
  tag_field_ngrow.push_back(0);

  tag_field_names.emplace_back("enuc");
  tag_field_ngrow.push_back(0);
#endif

#ifdef RADIATION
  if (do_radiation && !Radiation::do_multigroup) {
      tag_field_names.emplace_back("rad");
      tag_field_ngrow.push_back(1);
  }
#endif

  // Save the number of built-in functions; this will help us
  // distinguish between those, and the ones the user is about to add.

  n_default_tag_fields = static_cast<int>(tag_field_names.size());

  //
  // Construct an array holding the names of the source terms.
  //

  source_names.resize(num_src);

  // Fill with an empty string to initialize.

  std::fill(source_names.begin(), source_names.end(), "");

  source_names[ext_src] = "user-defined external";
  source_names[thermo_src] = "pdivU source";
#ifdef SPONGE
  source_names[sponge_src] = "sponge";
#endif

#ifdef DIFFUSION
  source_names[diff_src] = "diffusion";
#endif

#ifdef HYBRID_MOMENTUM
  source_names[hybrid_src] = "hybrid";
#endif

#ifdef GRAVITY
  source_names[grav_src] = "gravity";
#endif

#ifdef ROTATION
  source_names[rot_src] = "rotation";
#endif



#ifdef TRUE_SDC
  if (sdc_quadrature == 0) {
    // Gauss-Lobatto

    if (sdc_order == 2) {
      // trapezoid
      SDC_NODES = 2;

      dt_sdc.resize(SDC_NODES);
      dt_sdc = {0.0, 1.0};

      node_weights.resize(SDC_NODES);
      node_weights = {0.5, 0.5};

    } else if (sdc_order == 4) {
      // Simpsons
      SDC_NODES = 3;

      dt_sdc.resize(SDC_NODES);
      dt_sdc = {0.0, 0.5, 1.0};

      node_weights.resize(SDC_NODES);
      node_weights = {1.0/6.0, 4.0/6.0, 1.0/6.0};

    } else {
      amrex::Error("invalid value of sdc_order");
    }

  } else if (sdc_quadrature == 1) {
    // Radau

    if (sdc_order == 2) {
      SDC_NODES = 3;

      dt_sdc.resize(SDC_NODES);
      dt_sdc = {0.0, 1.0/3.0, 1.0};

      node_weights.resize(SDC_NODES);
      node_weights = {0.0, 3.0/4.0, 1.0/4.0};

    } else if (sdc_order == 4) {
      SDC_NODES = 4;

      dt_sdc.resize(SDC_NODES);
      dt_sdc = {0.0, (4.0 - std::sqrt(6.0))/10.0, (4.0 + std::sqrt(6.0))/10.0, 1.0};

      node_weights.resize(SDC_NODES);
      node_weights = {0.0, (16.0 - std::sqrt(6.0))/36.0, (16.0 + std::sqrt(6.0))/36.0, 1.0/9.0};

    } else {
      amrex::Error("invalid value of sdc_order");
    }

  } else {
    amrex::Error("invalid value of sdc_quadrature");
  }
#endif

}
