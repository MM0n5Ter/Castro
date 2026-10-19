PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 2

COMP	   = gnu

USE_MPI    = TRUE
USE_GRAV   = TRUE

USE_MODEL_PARSER = TRUE

HYDROSIM_HOME  ?= ../../..

EOS_DIR     := multigamma

NETWORK_DIR := general_null
NETWORK_INPUTS = simple3.net

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
