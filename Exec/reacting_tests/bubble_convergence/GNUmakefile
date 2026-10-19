PRECISION        = DOUBLE
PROFILE          = FALSE
DEBUG            = FALSE
DIM              = 2

COMP	         = gnu

USE_MPI          = TRUE
USE_GRAV         = TRUE
USE_REACT        = TRUE

USE_MODEL_PARSER = TRUE

USE_LOWMACH_INIT = FALSE

HYDROSIM_HOME ?= ../../..

EOS_DIR     := helmholtz

NETWORK_DIR := triple_alpha_plus_cago

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
