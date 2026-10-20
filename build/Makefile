# Compile check for the module sources, with the flags of the real build.
# Each source is compiled for 1-D, 2-D, 3-D. Nothing is linked or run.
#
#   make -C build            every dimension
#   make -C build DIM=2      one dimension
#   make -C build clean

ROOT := ..
DIM ?= 1 2 3

CXX := g++

CXXFLAGS_1 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=1 -DAMREX_SPACEDIM=1 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DGRAVITY -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DNAUX_NET=0 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE
INCLUDES_1 := -Iinclude/1d -Iinclude -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/external/amrex/Src/LinearSolvers/MLMG -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/sources -I$(ROOT)/Source/gravity -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/gamma_law -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/constants
CXXFLAGS_2 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=2 -DAMREX_SPACEDIM=2 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DGRAVITY -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DNAUX_NET=0 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE
INCLUDES_2 := -Iinclude/2d -Iinclude -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/external/amrex/Src/LinearSolvers/MLMG -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/sources -I$(ROOT)/Source/gravity -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/gamma_law -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/constants
CXXFLAGS_3 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=3 -DAMREX_SPACEDIM=3 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DGRAVITY -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DNAUX_NET=0 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE
INCLUDES_3 := -Iinclude/3d -Iinclude -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/external/amrex/Src/LinearSolvers/MLMG -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/sources -I$(ROOT)/Source/gravity -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/gamma_law -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/constants

OBJECTS_1 := obj/1d/Gravity.o obj/1d/multipole_boundary.o obj/1d/poisson_solve.o obj/1d/radial_gravity.o obj/1d/gravity_coupling.o obj/1d/gravity_source_terms.o obj/1d/hydrostatic_boundary.o
OBJECTS_2 := obj/2d/Gravity.o obj/2d/multipole_boundary.o obj/2d/poisson_solve.o obj/2d/radial_gravity.o obj/2d/gravity_coupling.o obj/2d/gravity_source_terms.o obj/2d/hydrostatic_boundary.o
OBJECTS_3 := obj/3d/Gravity.o obj/3d/multipole_boundary.o obj/3d/poisson_solve.o obj/3d/radial_gravity.o obj/3d/gravity_coupling.o obj/3d/gravity_source_terms.o obj/3d/hydrostatic_boundary.o

all: $(foreach d,$(DIM),$(OBJECTS_$(d)))

obj/1d/Gravity.o: $(ROOT)/Source/gravity/Gravity.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/multipole_boundary.o: $(ROOT)/Source/gravity/multipole_boundary.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/poisson_solve.o: $(ROOT)/Source/gravity/poisson_solve.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/radial_gravity.o: $(ROOT)/Source/gravity/radial_gravity.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/gravity_coupling.o: $(ROOT)/Source/gravity/gravity_coupling.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/gravity_source_terms.o: $(ROOT)/Source/gravity/gravity_source_terms.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/hydrostatic_boundary.o: $(ROOT)/Source/gravity/hydrostatic_boundary.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/2d/Gravity.o: $(ROOT)/Source/gravity/Gravity.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/multipole_boundary.o: $(ROOT)/Source/gravity/multipole_boundary.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/poisson_solve.o: $(ROOT)/Source/gravity/poisson_solve.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/radial_gravity.o: $(ROOT)/Source/gravity/radial_gravity.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/gravity_coupling.o: $(ROOT)/Source/gravity/gravity_coupling.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/gravity_source_terms.o: $(ROOT)/Source/gravity/gravity_source_terms.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/hydrostatic_boundary.o: $(ROOT)/Source/gravity/hydrostatic_boundary.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/3d/Gravity.o: $(ROOT)/Source/gravity/Gravity.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/multipole_boundary.o: $(ROOT)/Source/gravity/multipole_boundary.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/poisson_solve.o: $(ROOT)/Source/gravity/poisson_solve.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/radial_gravity.o: $(ROOT)/Source/gravity/radial_gravity.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/gravity_coupling.o: $(ROOT)/Source/gravity/gravity_coupling.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/gravity_source_terms.o: $(ROOT)/Source/gravity/gravity_source_terms.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/hydrostatic_boundary.o: $(ROOT)/Source/gravity/hydrostatic_boundary.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

clean:
	rm -rf obj

.PHONY: all clean

-include $(wildcard obj/*/*.d)
