# Compile check for the module sources, with the flags of the real build.
# Each source is compiled for 1-D, 2-D, 3-D. Nothing is linked or run.
#
#   make -C build            every dimension
#   make -C build DIM=2      one dimension
#   make -C build clean

ROOT := ..
DIM ?= 1 2 3

CXX := g++

CXXFLAGS_1 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=1 -DAMREX_SPACEDIM=1 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DREACTIONS -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE -DALLOW_JACOBIAN_CACHING -DRATES -DSCREENING -DNAUX_NET=0
INCLUDES_1 := -Iinclude/1d -Iinclude -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/external/Microphysics/util -I$(ROOT)/external/Microphysics/util/autodiff -I$(ROOT)/external/Microphysics/util/gcem/include -I$(ROOT)/external/Microphysics/integration/VODE -I$(ROOT)/external/Microphysics/integration/utils -I$(ROOT)/external/Microphysics/integration -I$(ROOT)/external/Microphysics/rates -I$(ROOT)/external/Microphysics/screening -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/problems -I$(ROOT)/Source/sources -I$(ROOT)/Source/reactions -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/helmholtz -I$(ROOT)/external/Microphysics/networks/triple_alpha_plus_cago -I$(ROOT)/external/Microphysics/networks -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/constants -I$(ROOT)/external/Microphysics/util/approx_math
CXXFLAGS_2 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=2 -DAMREX_SPACEDIM=2 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DREACTIONS -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE -DALLOW_JACOBIAN_CACHING -DRATES -DSCREENING -DNAUX_NET=0
INCLUDES_2 := -Iinclude/2d -Iinclude -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/external/Microphysics/util -I$(ROOT)/external/Microphysics/util/autodiff -I$(ROOT)/external/Microphysics/util/gcem/include -I$(ROOT)/external/Microphysics/integration/VODE -I$(ROOT)/external/Microphysics/integration/utils -I$(ROOT)/external/Microphysics/integration -I$(ROOT)/external/Microphysics/rates -I$(ROOT)/external/Microphysics/screening -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/problems -I$(ROOT)/Source/sources -I$(ROOT)/Source/reactions -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/helmholtz -I$(ROOT)/external/Microphysics/networks/triple_alpha_plus_cago -I$(ROOT)/external/Microphysics/networks -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/constants -I$(ROOT)/external/Microphysics/util/approx_math
CXXFLAGS_3 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=3 -DAMREX_SPACEDIM=3 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DREACTIONS -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE -DALLOW_JACOBIAN_CACHING -DRATES -DSCREENING -DNAUX_NET=0
INCLUDES_3 := -Iinclude/3d -Iinclude -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/external/Microphysics/util -I$(ROOT)/external/Microphysics/util/autodiff -I$(ROOT)/external/Microphysics/util/gcem/include -I$(ROOT)/external/Microphysics/integration/VODE -I$(ROOT)/external/Microphysics/integration/utils -I$(ROOT)/external/Microphysics/integration -I$(ROOT)/external/Microphysics/rates -I$(ROOT)/external/Microphysics/screening -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/problems -I$(ROOT)/Source/sources -I$(ROOT)/Source/reactions -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/helmholtz -I$(ROOT)/external/Microphysics/networks/triple_alpha_plus_cago -I$(ROOT)/external/Microphysics/networks -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/constants -I$(ROOT)/external/Microphysics/util/approx_math

OBJECTS_1 := obj/1d/HydroSim_react.o
OBJECTS_2 := obj/2d/HydroSim_react.o
OBJECTS_3 := obj/3d/HydroSim_react.o

all: $(foreach d,$(DIM),$(OBJECTS_$(d)))

obj/1d/HydroSim_react.o: $(ROOT)/Source/reactions/HydroSim_react.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/2d/HydroSim_react.o: $(ROOT)/Source/reactions/HydroSim_react.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/3d/HydroSim_react.o: $(ROOT)/Source/reactions/HydroSim_react.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

clean:
	rm -rf obj

.PHONY: all clean

-include $(wildcard obj/*/*.d)
