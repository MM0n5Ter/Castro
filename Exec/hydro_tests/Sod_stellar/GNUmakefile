PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE


DIM        = 2

COMP	   = gnu

USE_MPI    = TRUE
USE_OMP    = FALSE

USE_MHD = FALSE


HYDROSIM_HOME ?= ../../..

EOS_DIR     := helmholtz

NETWORK_DIR := general_null
NETWORK_INPUTS = ignition.net

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
