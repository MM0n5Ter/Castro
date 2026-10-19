PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 2

COMP       = gnu

USE_MPI    = TRUE
USE_OMP    = FALSE

USE_DIFFUSION = TRUE

USE_POST_SIM = TRUE

HYDROSIM_HOME ?= ../../..

EOS_DIR     := gamma_law

CONDUCTIVITY_DIR = constant

NETWORK_DIR := general_null
NETWORK_INPUTS = gammalaw.net

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
