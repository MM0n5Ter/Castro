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
HYDROSIM_HOME  ?= ../../..

USE_MODEL_PARSER = TRUE
NUM_MODELS := 2

# This sets the EOS directory in HydroSim/EOS
EOS_DIR     := helmholtz

# This sets the network directory in HydroSim/Networks
NETWORK_DIR := he-burn/he-burn-19am

EXTERN_SEARCH += .

Bpack   := ./Make.package
Blocs   := . .. $(HYDROSIM_HOME)/Source/problems

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
