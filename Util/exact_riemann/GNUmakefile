PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 3

COMP	   = gnu

USE_MPI    = FALSE
USE_OMP    = FALSE

USE_ALL_HYDROSIM = FALSE
USE_AMR_CORE = FALSE

HYDROSIM_HOME  ?= ../..

EOS_DIR     := helmholtz

NETWORK_DIR := general_null
NETWORK_INPUTS = ignition.net

STRUCT_USE_HYDROSIM_CLASS := FALSE

EXTERN_SEARCH += .

Bpack   := ./Make.package
Blocs   := .

Blocs += $(HYDROSIM_HOME)/Source/driver $(HYDROSIM_HOME)/Source/hydro

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
