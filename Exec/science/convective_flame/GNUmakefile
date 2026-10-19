PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 2

COMP	   = gnu

USE_MPI    = TRUE

USE_GRAV   = TRUE
USE_REACT = TRUE

USE_ROTATION = TRUE
USE_DIFFUSION = TRUE

USE_MODEL_PARSER = TRUE
NUM_MODELS := 2

HYDROSIM_HOME ?= ../../..

EOS_DIR     := gamma_law

NETWORK_DIR := powerlaw

CONDUCTIVITY_DIR := constant

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
