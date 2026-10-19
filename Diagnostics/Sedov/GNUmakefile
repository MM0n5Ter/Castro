PRECISION = DOUBLE
PROFILE = FALSE
DEBUG = TRUE
DIM = 3

COMP = gnu

USE_MPI = FALSE
USE_OMP = FALSE

USE_REACT = FALSE

ALL: sedov_$(DIM)d.ex

EOS_DIR := helmholtz

NETWORK_DIR := aprox13

Bpack   := ./Make.package
Blocs   := .

HYDROSIM_HOME ?= ../..


include $(HYDROSIM_HOME)/Exec/Make.HydroSim

sedov_$(DIM)d.ex: $(objForExecs)
	@echo Linking $@ ...
	$(SILENT) $(PRELINK) $(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(libraries)
