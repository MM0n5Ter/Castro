COMP	         = gnu

DIM              = 2

PRECISION        = DOUBLE

USE_MPI          = TRUE
USE_OMP          = FALSE

DEBUG            = FALSE

PROFILE          = FALSE


USE_RAD = TRUE

USE_GRAV         = TRUE
USE_REACT        = FALSE

USE_MODEL_PARSER = TRUE

HYDROSIM_HOME ?= ../../..

EOS_DIR     := gamma_law

NETWORK_DIR := general_null
NETWORK_INPUTS := H_He.net

OPACITY_DIR := rad_power_law

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
