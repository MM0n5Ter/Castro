PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 3

COMP	   = gnu

USE_MPI    = FALSE
USE_OMP    = FALSE

USE_ALL_HYDROSIM = FALSE
USE_AMR_CORE = FALSE

# define the location of the HYDROSIM top directory
HYDROSIM_HOME  ?= ../..

# This sets the EOS directory in HydroSim/EOS
EOS_DIR     := helmholtz

# This sets the network directory in HydroSim/Networks
NETWORK_DIR := general_null
NETWORK_INPUTS = ignition.net

# for the HydroSim runtime parameters, we don't want to use HydroSim::
STRUCT_USE_HYDROSIM_CLASS := FALSE

EXTERN_SEARCH += .

Bpack   := ./Make.package
Blocs   := .

# we explicitly want runtime_parameters.H so we have access to
# HydroSim's runtime parameter system
Blocs += $(HYDROSIM_HOME)/Source/driver $(HYDROSIM_HOME)/Source/hydro

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
