PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 2

COMP	   = gnu

USE_MPI    = TRUE

USE_REACT  = TRUE

USE_GRAV   = FALSE
USE_PARTICLES = FALSE

USE_MHD = FALSE


USE_MODEL_PARSER = TRUE

HYDROSIM_HOME ?= ../../..


EOS_DIR     := helmholtz

NETWORK_DIR := aprox13

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
