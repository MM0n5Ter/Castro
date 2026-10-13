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
INCLUDES_1 := -Iinclude/1d -Iinclude -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/external/amrex/Src/LinearSolvers/MLMG -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/problems -I$(ROOT)/Source/sources -I$(ROOT)/Source/radiation -I$(ROOT)/Source/radiation/_interpbndry -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/gamma_law -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/opacity -I$(ROOT)/external/Microphysics/opacity/rad_power_law -I$(ROOT)/external/Microphysics/constants -isystem $(ROOT)/external/hypre/include
CXXFLAGS_2 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=2 -DAMREX_SPACEDIM=2 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DRADIATION -DRAD_INTERP -DNGROUPS=1 -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DNAUX_NET=0 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE -DAMREX_USE_HYPRE
INCLUDES_2 := -Iinclude/2d -Iinclude -I$(ROOT)/external/amrex/Src/Extern/HYPRE -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/external/amrex/Src/LinearSolvers/MLMG -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/problems -I$(ROOT)/Source/sources -I$(ROOT)/Source/radiation -I$(ROOT)/Source/radiation/_interpbndry -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/gamma_law -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/opacity -I$(ROOT)/external/Microphysics/opacity/rad_power_law -I$(ROOT)/external/Microphysics/constants -isystem $(ROOT)/external/hypre/include
CXXFLAGS_3 := -Werror=return-type -g1 -O3 -finline-limit=43210 -std=c++20 -pthread -DBL_LAZY -DAMREX_LAZY -DBL_NO_FORT -DAMREX_GPU_MAX_THREADS=0 -DBL_SPACEDIM=3 -DAMREX_SPACEDIM=3 -DBL_FORT_USE_UNDERSCORE -DAMREX_FORT_USE_UNDERSCORE -DBL_Linux -DAMREX_Linux -DAMREX_DIMENSION_AGNOSTIC -DNDEBUG -DAMREX_NO_PROBINIT -DSPONGE -DRADIATION -DRAD_INTERP -DNGROUPS=1 -DSTRANG -DNET_LOOP_UNROLL_LEN=4 -DSCREEN_METHOD=SCREEN_METHOD_screen5 -DNEUTRINO_METHOD=NEUTRINO_METHOD_sneut5 -DNAUX_NET=0 -DAUTODIFF_DEVICE_FUNC=AMREX_GPU_HOST_DEVICE -DAMREX_USE_HYPRE
INCLUDES_3 := -Iinclude/3d -Iinclude -I$(ROOT)/external/amrex/Src/Extern/HYPRE -I$(ROOT)/external/amrex/Src/Base -I$(ROOT)/external/amrex/Src/Base/Parser -I$(ROOT)/external/amrex/Src/AmrCore -I$(ROOT)/external/amrex/Src/Amr -I$(ROOT)/external/amrex/Src/Boundary -I$(ROOT)/external/amrex/Src/LinearSolvers/MLMG -I$(ROOT)/Source/driver -I$(ROOT)/Source/hydro -I$(ROOT)/Source/problems -I$(ROOT)/Source/sources -I$(ROOT)/Source/radiation -I$(ROOT)/Source/radiation/_interpbndry -I$(ROOT)/external/Microphysics/EOS -I$(ROOT)/external/Microphysics/EOS/gamma_law -I$(ROOT)/external/Microphysics/interfaces -I$(ROOT)/external/Microphysics/opacity -I$(ROOT)/external/Microphysics/opacity/rad_power_law -I$(ROOT)/external/Microphysics/constants -isystem $(ROOT)/external/hypre/include

OBJECTS_1 := obj/1d/HypreExtMultiABec.o obj/1d/HypreMultiABec.o obj/1d/HypreABec.o obj/1d/Radiation.o obj/1d/RadSolve.o obj/1d/RadBndry.o obj/1d/RadMultiGroup.o obj/1d/MGRadBndry.o obj/1d/SGRadSolver.o obj/1d/RadPlotvar.o obj/1d/MGFLD.o obj/1d/MGFLDRadSolver.o obj/1d/Castro_radiation.o obj/1d/energy_diagnostics.o obj/1d/trace_ppm_rad.o obj/1d/RadDerive.o obj/1d/RadInterpBndryData.o obj/1d/RadBndryData.o obj/1d/Castro_ctu_rad.o
OBJECTS_2 := obj/2d/HypreExtMultiABec.o obj/2d/HypreMultiABec.o obj/2d/HypreABec.o obj/2d/Radiation.o obj/2d/RadSolve.o obj/2d/RadBndry.o obj/2d/RadMultiGroup.o obj/2d/MGRadBndry.o obj/2d/SGRadSolver.o obj/2d/RadPlotvar.o obj/2d/MGFLD.o obj/2d/MGFLDRadSolver.o obj/2d/Castro_radiation.o obj/2d/energy_diagnostics.o obj/2d/trace_ppm_rad.o obj/2d/RadDerive.o obj/2d/RadInterpBndryData.o obj/2d/RadBndryData.o obj/2d/Castro_ctu_rad.o
OBJECTS_3 := obj/3d/HypreExtMultiABec.o obj/3d/HypreMultiABec.o obj/3d/HypreABec.o obj/3d/Radiation.o obj/3d/RadSolve.o obj/3d/RadBndry.o obj/3d/RadMultiGroup.o obj/3d/MGRadBndry.o obj/3d/SGRadSolver.o obj/3d/RadPlotvar.o obj/3d/MGFLD.o obj/3d/MGFLDRadSolver.o obj/3d/Castro_radiation.o obj/3d/energy_diagnostics.o obj/3d/trace_ppm_rad.o obj/3d/RadDerive.o obj/3d/RadInterpBndryData.o obj/3d/RadBndryData.o obj/3d/Castro_ctu_rad.o

all: $(foreach d,$(DIM),$(OBJECTS_$(d)))

obj/1d/HypreExtMultiABec.o: $(ROOT)/Source/radiation/HypreExtMultiABec.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/HypreMultiABec.o: $(ROOT)/Source/radiation/HypreMultiABec.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/HypreABec.o: $(ROOT)/Source/radiation/HypreABec.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/Radiation.o: $(ROOT)/Source/radiation/Radiation.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/RadSolve.o: $(ROOT)/Source/radiation/RadSolve.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/RadBndry.o: $(ROOT)/Source/radiation/RadBndry.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/RadMultiGroup.o: $(ROOT)/Source/radiation/RadMultiGroup.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/MGRadBndry.o: $(ROOT)/Source/radiation/MGRadBndry.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/SGRadSolver.o: $(ROOT)/Source/radiation/SGRadSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/RadPlotvar.o: $(ROOT)/Source/radiation/RadPlotvar.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/MGFLD.o: $(ROOT)/Source/radiation/MGFLD.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/MGFLDRadSolver.o: $(ROOT)/Source/radiation/MGFLDRadSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/Castro_radiation.o: $(ROOT)/Source/radiation/Castro_radiation.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/energy_diagnostics.o: $(ROOT)/Source/radiation/energy_diagnostics.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/trace_ppm_rad.o: $(ROOT)/Source/radiation/trace_ppm_rad.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/RadDerive.o: $(ROOT)/Source/radiation/RadDerive.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/RadInterpBndryData.o: $(ROOT)/Source/radiation/_interpbndry/RadInterpBndryData.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/RadBndryData.o: $(ROOT)/Source/radiation/_interpbndry/RadBndryData.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/1d/Castro_ctu_rad.o: $(ROOT)/Source/hydro/Castro_ctu_rad.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_1) -MMD -MP $(INCLUDES_1) -c $< -o $@

obj/2d/HypreExtMultiABec.o: $(ROOT)/Source/radiation/HypreExtMultiABec.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/HypreMultiABec.o: $(ROOT)/Source/radiation/HypreMultiABec.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/HypreABec.o: $(ROOT)/Source/radiation/HypreABec.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/Radiation.o: $(ROOT)/Source/radiation/Radiation.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/RadSolve.o: $(ROOT)/Source/radiation/RadSolve.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/RadBndry.o: $(ROOT)/Source/radiation/RadBndry.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/RadMultiGroup.o: $(ROOT)/Source/radiation/RadMultiGroup.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/MGRadBndry.o: $(ROOT)/Source/radiation/MGRadBndry.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/SGRadSolver.o: $(ROOT)/Source/radiation/SGRadSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/RadPlotvar.o: $(ROOT)/Source/radiation/RadPlotvar.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/MGFLD.o: $(ROOT)/Source/radiation/MGFLD.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/MGFLDRadSolver.o: $(ROOT)/Source/radiation/MGFLDRadSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/Castro_radiation.o: $(ROOT)/Source/radiation/Castro_radiation.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/energy_diagnostics.o: $(ROOT)/Source/radiation/energy_diagnostics.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/trace_ppm_rad.o: $(ROOT)/Source/radiation/trace_ppm_rad.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/RadDerive.o: $(ROOT)/Source/radiation/RadDerive.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/RadInterpBndryData.o: $(ROOT)/Source/radiation/_interpbndry/RadInterpBndryData.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/RadBndryData.o: $(ROOT)/Source/radiation/_interpbndry/RadBndryData.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/2d/Castro_ctu_rad.o: $(ROOT)/Source/hydro/Castro_ctu_rad.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_2) -MMD -MP $(INCLUDES_2) -c $< -o $@

obj/3d/HypreExtMultiABec.o: $(ROOT)/Source/radiation/HypreExtMultiABec.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/HypreMultiABec.o: $(ROOT)/Source/radiation/HypreMultiABec.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/HypreABec.o: $(ROOT)/Source/radiation/HypreABec.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/Radiation.o: $(ROOT)/Source/radiation/Radiation.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/RadSolve.o: $(ROOT)/Source/radiation/RadSolve.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/RadBndry.o: $(ROOT)/Source/radiation/RadBndry.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/RadMultiGroup.o: $(ROOT)/Source/radiation/RadMultiGroup.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/MGRadBndry.o: $(ROOT)/Source/radiation/MGRadBndry.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/SGRadSolver.o: $(ROOT)/Source/radiation/SGRadSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/RadPlotvar.o: $(ROOT)/Source/radiation/RadPlotvar.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/MGFLD.o: $(ROOT)/Source/radiation/MGFLD.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/MGFLDRadSolver.o: $(ROOT)/Source/radiation/MGFLDRadSolver.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/Castro_radiation.o: $(ROOT)/Source/radiation/Castro_radiation.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/energy_diagnostics.o: $(ROOT)/Source/radiation/energy_diagnostics.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/trace_ppm_rad.o: $(ROOT)/Source/radiation/trace_ppm_rad.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/RadDerive.o: $(ROOT)/Source/radiation/RadDerive.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/RadInterpBndryData.o: $(ROOT)/Source/radiation/_interpbndry/RadInterpBndryData.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/RadBndryData.o: $(ROOT)/Source/radiation/_interpbndry/RadBndryData.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

obj/3d/Castro_ctu_rad.o: $(ROOT)/Source/hydro/Castro_ctu_rad.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_3) -MMD -MP $(INCLUDES_3) -c $< -o $@

clean:
	rm -rf obj

.PHONY: all clean

-include $(wildcard obj/*/*.d)
