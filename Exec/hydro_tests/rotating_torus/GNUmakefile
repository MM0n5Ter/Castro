PRECISION  = DOUBLE

DEBUG      = FALSE

DIM        = 3

COMP	   = gnu

USE_MPI    = TRUE
USE_OMP    = FALSE

USE_GRAV      = TRUE
USE_ROTATION  = TRUE

USE_HYBRID_MOMENTUM = TRUE

HYDROSIM_HOME  ?= ../../..

EOS_DIR     := polytrope

NETWORK_DIR := general_null
NETWORK_INPUTS = ignition.net

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
