PRECISION        = DOUBLE
PROFILE          = FALSE
DEBUG            = FALSE
DIM              = 2

COMP	         = gnu

USE_MPI          = TRUE
USE_OMP          = FALSE

USE_GRAV         = TRUE
USE_REACT        = TRUE

USE_SHOCK_VAR 	 = TRUE

MAX_NPTS_MODEL 	 = 32678

USE_MODEL_PARSER = TRUE

USE_SIMPLIFIED_SDC = TRUE

HYDROSIM_HOME ?= ../../..

EOS_DIR     := helmholtz

NETWORK_DIR := he-burn/he-burn-22a

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
