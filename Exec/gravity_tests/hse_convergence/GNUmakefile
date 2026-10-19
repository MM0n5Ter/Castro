PRECISION        = DOUBLE
PROFILE          = FALSE
DEBUG            = FALSE
DIM              = 1

COMP	         = gnu

USE_MPI          = TRUE
USE_GRAV         = TRUE
USE_REACT        = FALSE

USE_MODEL_PARSER = TRUE

USE_LOWMACH_INIT = FALSE

HYDROSIM_HOME ?= ../../..

# This sets the EOS directory in $(MICROPHYSICS_HOME)/EOS
EOS_DIR     := helmholtz

# This sets the network directory in $(MICROPHYSICS_HOME)/networks
NETWORK_DIR := general_null
NETWORK_INPUTS := triple_alpha_plus_o.net

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
