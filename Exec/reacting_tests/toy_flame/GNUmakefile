PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 1

COMP	   = gnu

USE_MPI    = TRUE

USE_REACT = TRUE

USE_DIFFUSION = TRUE


HYDROSIM_HOME ?= ../../..

EOS_DIR     := gamma_law

NETWORK_DIR := powerlaw

CONDUCTIVITY_DIR := constant

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
