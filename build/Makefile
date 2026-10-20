# Compile check for the module sources, with the flags of the real build.
# Each source is compiled for 1-D, 2-D, 3-D. Nothing is linked or run.
#
#   make -C build            every dimension
#   make -C build DIM=2      one dimension
#   make -C build clean

ROOT := ..
DIM ?= 1 2 3

CXX := g++

CXXFLAGS_1 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=1 -DAMREX_SPACEDIM=1 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DNAUX_NET=0 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE
INCLUDES_1 := -Iinclude/1d -Iinclude -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/problems -I$(ROOT)/Source/sources -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/gamma_law -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/constants
CXXFLAGS_2 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=2 -DAMREX_SPACEDIM=2 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DNAUX_NET=0 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE
INCLUDES_2 := -Iinclude/2d -Iinclude -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/problems -I$(ROOT)/Source/sources -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/gamma_law -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/constants
CXXFLAGS_3 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=3 -DAMREX_SPACEDIM=3 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DNAUX_NET=0 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE
INCLUDES_3 := -Iinclude/3d -Iinclude -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/problems -I$(ROOT)/Source/sources -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/gamma_law -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/constants

OBJECTS_1 := obj/1d/ctu_hydro_update.o obj/1d/primitive_state.o obj/1d/face_states.o obj/1d/flux_update.o obj/1d/hydro_cell_ops.o obj/1d/godunov_flux.o obj/1d/edge_tracing.o obj/1d/curvilinear_source.o obj/1d/pointwise_sources.o
OBJECTS_2 := obj/2d/ctu_hydro_update.o obj/2d/primitive_state.o obj/2d/face_states.o obj/2d/flux_update.o obj/2d/hydro_cell_ops.o obj/2d/godunov_flux.o obj/2d/edge_tracing.o obj/2d/transverse_terms.o obj/2d/curvilinear_source.o obj/2d/pointwise_sources.o
OBJECTS_3 := obj/3d/ctu_hydro_update.o obj/3d/primitive_state.o obj/3d/face_states.o obj/3d/flux_update.o obj/3d/hydro_cell_ops.o obj/3d/godunov_flux.o obj/3d/edge_tracing.o obj/3d/transverse_terms.o obj/3d/curvilinear_source.o obj/3d/pointwise_sources.o

all: $(foreach d,$(DIM),$(OBJECTS_$(d)))

obj/1d/ctu_hydro_update.o: $(ROOT)/Source/hydro/ctu_hydro_update.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/primitive_state.o: $(ROOT)/Source/hydro/primitive_state.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/face_states.o: $(ROOT)/Source/hydro/face_states.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/flux_update.o: $(ROOT)/Source/hydro/flux_update.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/hydro_cell_ops.o: $(ROOT)/Source/hydro/hydro_cell_ops.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/godunov_flux.o: $(ROOT)/Source/hydro/godunov_flux.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/edge_tracing.o: $(ROOT)/Source/hydro/edge_tracing.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/curvilinear_source.o: $(ROOT)/Source/sources/curvilinear_source.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/pointwise_sources.o: $(ROOT)/Source/sources/pointwise_sources.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/2d/ctu_hydro_update.o: $(ROOT)/Source/hydro/ctu_hydro_update.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/primitive_state.o: $(ROOT)/Source/hydro/primitive_state.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/face_states.o: $(ROOT)/Source/hydro/face_states.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/flux_update.o: $(ROOT)/Source/hydro/flux_update.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/hydro_cell_ops.o: $(ROOT)/Source/hydro/hydro_cell_ops.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/godunov_flux.o: $(ROOT)/Source/hydro/godunov_flux.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/edge_tracing.o: $(ROOT)/Source/hydro/edge_tracing.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/transverse_terms.o: $(ROOT)/Source/hydro/transverse_terms.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/curvilinear_source.o: $(ROOT)/Source/sources/curvilinear_source.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/pointwise_sources.o: $(ROOT)/Source/sources/pointwise_sources.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/3d/ctu_hydro_update.o: $(ROOT)/Source/hydro/ctu_hydro_update.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/primitive_state.o: $(ROOT)/Source/hydro/primitive_state.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/face_states.o: $(ROOT)/Source/hydro/face_states.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/flux_update.o: $(ROOT)/Source/hydro/flux_update.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/hydro_cell_ops.o: $(ROOT)/Source/hydro/hydro_cell_ops.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/godunov_flux.o: $(ROOT)/Source/hydro/godunov_flux.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/edge_tracing.o: $(ROOT)/Source/hydro/edge_tracing.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/transverse_terms.o: $(ROOT)/Source/hydro/transverse_terms.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/curvilinear_source.o: $(ROOT)/Source/sources/curvilinear_source.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/pointwise_sources.o: $(ROOT)/Source/sources/pointwise_sources.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

clean:
	rm -rf obj

.PHONY: all clean

-include $(wildcard obj/*/*.d)
