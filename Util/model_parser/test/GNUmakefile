PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 3

COMP	   = gnu

USE_MPI    = FALSE
USE_OMP    = FALSE

USE_ALL_HYDROSIM = FALSE
USE_AMR_CORE = FALSE

HYDROSIM_HOME  ?= ../../..

USE_MODEL_PARSER = TRUE
NUM_MODELS := 2

EOS_DIR     := helmholtz

NETWORK_DIR := he-burn/he-burn-19am

EXTERN_SEARCH += .

Bpack   := ./Make.package
Blocs   := . .. $(HYDROSIM_HOME)/Source/problems

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
