# Compile check for the module sources, with the flags of the real build.
# Each source is compiled for 1-D, 2-D, 3-D. Nothing is linked or run.
#
#   make -C build            every dimension
#   make -C build DIM=2      one dimension
#   make -C build clean

ROOT := ..
DIM ?= 1 2 3

CXX := g++

CXXFLAGS_1 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=1 -DAMREX_SPACEDIM=1 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DRADIATION -DRAD_INTERP -DNGROUPS=1 -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DNAUX_NET=0 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE -DAMREX_USE_HYPRE
INCLUDES_1 := -Iinclude/1d -Iinclude -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/external/amrex/Src/LinearSolvers/MLMG -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/problems -I$(ROOT)/Source/sources -I$(ROOT)/Source/radiation -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/gamma_law -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/opacity -I$(ROOT)/external/Microphysics/opacity/rad_power_law -I$(ROOT)/external/Microphysics/constants -isystem $(ROOT)/external/hypre/include
CXXFLAGS_2 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=2 -DAMREX_SPACEDIM=2 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DRADIATION -DRAD_INTERP -DNGROUPS=1 -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DNAUX_NET=0 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE -DAMREX_USE_HYPRE
INCLUDES_2 := -Iinclude/2d -Iinclude -I$(ROOT)/external/amrex/Src/Extern/HYPRE -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/external/amrex/Src/LinearSolvers/MLMG -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/problems -I$(ROOT)/Source/sources -I$(ROOT)/Source/radiation -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/gamma_law -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/opacity -I$(ROOT)/external/Microphysics/opacity/rad_power_law -I$(ROOT)/external/Microphysics/constants -isystem $(ROOT)/external/hypre/include
CXXFLAGS_3 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=3 -DAMREX_SPACEDIM=3 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DRADIATION -DRAD_INTERP -DNGROUPS=1 -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DNAUX_NET=0 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE -DAMREX_USE_HYPRE
INCLUDES_3 := -Iinclude/3d -Iinclude -I$(ROOT)/external/amrex/Src/Extern/HYPRE -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/external/amrex/Src/LinearSolvers/MLMG -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/problems -I$(ROOT)/Source/sources -I$(ROOT)/Source/radiation -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/gamma_law -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/opacity -I$(ROOT)/external/Microphysics/opacity/rad_power_law -I$(ROOT)/external/Microphysics/constants -isystem $(ROOT)/external/hypre/include

OBJECTS_1 := obj/1d/MultiLevelEllipticExt.o obj/1d/MultiLevelEllipticSolver.o obj/1d/aux_coupling.o obj/1d/multilevel_matrix_fill.o obj/1d/AlphaBetaSolver.o obj/1d/alpha_beta_bc_fill.o obj/1d/Radiation.o obj/1d/rad_energy_coupling.o obj/1d/rad_boundary_fill.o obj/1d/rad_coefficients.o obj/1d/RadTransportSolver.o obj/1d/RadLevelTerms.o obj/1d/rad_boundary_values.o obj/1d/rad_groups.o obj/1d/GrayUpdate.o obj/1d/rad_output.o obj/1d/MultigroupUpdate.o obj/1d/rad_hydro_terms.o obj/1d/RadInterpBoundaryData.o
OBJECTS_2 := obj/2d/MultiLevelEllipticExt.o obj/2d/MultiLevelEllipticSolver.o obj/2d/aux_coupling.o obj/2d/multilevel_matrix_fill.o obj/2d/AlphaBetaSolver.o obj/2d/alpha_beta_bc_fill.o obj/2d/Radiation.o obj/2d/rad_energy_coupling.o obj/2d/rad_boundary_fill.o obj/2d/rad_coefficients.o obj/2d/RadTransportSolver.o obj/2d/RadLevelTerms.o obj/2d/rad_boundary_values.o obj/2d/rad_groups.o obj/2d/GrayUpdate.o obj/2d/rad_output.o obj/2d/MultigroupUpdate.o obj/2d/rad_hydro_terms.o obj/2d/RadInterpBoundaryData.o
OBJECTS_3 := obj/3d/MultiLevelEllipticExt.o obj/3d/MultiLevelEllipticSolver.o obj/3d/aux_coupling.o obj/3d/multilevel_matrix_fill.o obj/3d/AlphaBetaSolver.o obj/3d/alpha_beta_bc_fill.o obj/3d/Radiation.o obj/3d/rad_energy_coupling.o obj/3d/rad_boundary_fill.o obj/3d/rad_coefficients.o obj/3d/RadTransportSolver.o obj/3d/RadLevelTerms.o obj/3d/rad_boundary_values.o obj/3d/rad_groups.o obj/3d/GrayUpdate.o obj/3d/rad_output.o obj/3d/MultigroupUpdate.o obj/3d/rad_hydro_terms.o obj/3d/RadInterpBoundaryData.o

all: $(foreach d,$(DIM),$(OBJECTS_$(d)))

obj/1d/MultiLevelEllipticExt.o: $(ROOT)/Source/radiation/MultiLevelEllipticExt.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/MultiLevelEllipticSolver.o: $(ROOT)/Source/radiation/MultiLevelEllipticSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/aux_coupling.o: $(ROOT)/Source/radiation/aux_coupling.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/multilevel_matrix_fill.o: $(ROOT)/Source/radiation/multilevel_matrix_fill.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/AlphaBetaSolver.o: $(ROOT)/Source/radiation/AlphaBetaSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/alpha_beta_bc_fill.o: $(ROOT)/Source/radiation/alpha_beta_bc_fill.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/Radiation.o: $(ROOT)/Source/radiation/Radiation.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/rad_energy_coupling.o: $(ROOT)/Source/radiation/rad_energy_coupling.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/rad_boundary_fill.o: $(ROOT)/Source/radiation/rad_boundary_fill.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/rad_coefficients.o: $(ROOT)/Source/radiation/rad_coefficients.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/RadTransportSolver.o: $(ROOT)/Source/radiation/RadTransportSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/RadLevelTerms.o: $(ROOT)/Source/radiation/RadLevelTerms.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/rad_boundary_values.o: $(ROOT)/Source/radiation/rad_boundary_values.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/rad_groups.o: $(ROOT)/Source/radiation/rad_groups.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/GrayUpdate.o: $(ROOT)/Source/radiation/GrayUpdate.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/rad_output.o: $(ROOT)/Source/radiation/rad_output.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/MultigroupUpdate.o: $(ROOT)/Source/radiation/MultigroupUpdate.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/rad_hydro_terms.o: $(ROOT)/Source/radiation/rad_hydro_terms.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/RadInterpBoundaryData.o: $(ROOT)/Source/radiation/RadInterpBoundaryData.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/2d/MultiLevelEllipticExt.o: $(ROOT)/Source/radiation/MultiLevelEllipticExt.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/MultiLevelEllipticSolver.o: $(ROOT)/Source/radiation/MultiLevelEllipticSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/aux_coupling.o: $(ROOT)/Source/radiation/aux_coupling.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/multilevel_matrix_fill.o: $(ROOT)/Source/radiation/multilevel_matrix_fill.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/AlphaBetaSolver.o: $(ROOT)/Source/radiation/AlphaBetaSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/alpha_beta_bc_fill.o: $(ROOT)/Source/radiation/alpha_beta_bc_fill.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/Radiation.o: $(ROOT)/Source/radiation/Radiation.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/rad_energy_coupling.o: $(ROOT)/Source/radiation/rad_energy_coupling.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/rad_boundary_fill.o: $(ROOT)/Source/radiation/rad_boundary_fill.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/rad_coefficients.o: $(ROOT)/Source/radiation/rad_coefficients.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/RadTransportSolver.o: $(ROOT)/Source/radiation/RadTransportSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/RadLevelTerms.o: $(ROOT)/Source/radiation/RadLevelTerms.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/rad_boundary_values.o: $(ROOT)/Source/radiation/rad_boundary_values.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/rad_groups.o: $(ROOT)/Source/radiation/rad_groups.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/GrayUpdate.o: $(ROOT)/Source/radiation/GrayUpdate.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/rad_output.o: $(ROOT)/Source/radiation/rad_output.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/MultigroupUpdate.o: $(ROOT)/Source/radiation/MultigroupUpdate.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/rad_hydro_terms.o: $(ROOT)/Source/radiation/rad_hydro_terms.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/RadInterpBoundaryData.o: $(ROOT)/Source/radiation/RadInterpBoundaryData.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/3d/MultiLevelEllipticExt.o: $(ROOT)/Source/radiation/MultiLevelEllipticExt.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/MultiLevelEllipticSolver.o: $(ROOT)/Source/radiation/MultiLevelEllipticSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/aux_coupling.o: $(ROOT)/Source/radiation/aux_coupling.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/multilevel_matrix_fill.o: $(ROOT)/Source/radiation/multilevel_matrix_fill.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/AlphaBetaSolver.o: $(ROOT)/Source/radiation/AlphaBetaSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/alpha_beta_bc_fill.o: $(ROOT)/Source/radiation/alpha_beta_bc_fill.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/Radiation.o: $(ROOT)/Source/radiation/Radiation.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/rad_energy_coupling.o: $(ROOT)/Source/radiation/rad_energy_coupling.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/rad_boundary_fill.o: $(ROOT)/Source/radiation/rad_boundary_fill.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/rad_coefficients.o: $(ROOT)/Source/radiation/rad_coefficients.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/RadTransportSolver.o: $(ROOT)/Source/radiation/RadTransportSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/RadLevelTerms.o: $(ROOT)/Source/radiation/RadLevelTerms.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/rad_boundary_values.o: $(ROOT)/Source/radiation/rad_boundary_values.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/rad_groups.o: $(ROOT)/Source/radiation/rad_groups.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/GrayUpdate.o: $(ROOT)/Source/radiation/GrayUpdate.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/rad_output.o: $(ROOT)/Source/radiation/rad_output.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/MultigroupUpdate.o: $(ROOT)/Source/radiation/MultigroupUpdate.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/rad_hydro_terms.o: $(ROOT)/Source/radiation/rad_hydro_terms.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/RadInterpBoundaryData.o: $(ROOT)/Source/radiation/RadInterpBoundaryData.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

clean:
	rm -rf obj

.PHONY: all clean

-include $(wildcard obj/*/*.d)
