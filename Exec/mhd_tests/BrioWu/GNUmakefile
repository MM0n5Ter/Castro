PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 3

COMP	   = gnu

USE_MPI    = TRUE
USE_OMP    = FALSE

USE_MHD    = TRUE

HYDROSIM_HOME  ?= ../../..

EOS_DIR     := gamma_law

NETWORK_DIR := general_null
NETWORK_INPUTS = gammalaw.net

PROBLEM_DIR ?= ./

Blocs   := $(PROBLEM_DIR)
Bpack   := $(PROBLEM_DIR)/Make.package

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
