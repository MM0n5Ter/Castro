PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 1
DIM        = 3

COMP	   = gnu

USE_MPI    = TRUE

USE_GRAV   = TRUE

USE_MHD = FALSE


HYDROSIM_HOME  ?= ../../..

EOS_DIR     := gamma_law

NETWORK_DIR := general_null
NETWORK_INPUTS = ignition.net

PROBLEM_DIR ?= ./

Bpack	:= $(PROBLEM_DIR)/Make.package
Blocs	:= $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
