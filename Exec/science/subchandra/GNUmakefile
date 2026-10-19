HYDROSIM_HOME ?= ../../..

PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 2

COMP	   = gnu

USE_MPI    = TRUE

USE_GRAV   = TRUE
USE_REACT  = TRUE
USE_SHOCK_VAR = TRUE

USE_MODEL_PARSER = TRUE

USE_SIMPLIFIED_SDC = TRUE

USE_JACOBIAN_CACHING = TRUE

EOS_DIR     := helmholtz

NETWORK_DIR := he-burn/he-burn-19am

INTEGRATOR_DIR := Rosenbrock

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
