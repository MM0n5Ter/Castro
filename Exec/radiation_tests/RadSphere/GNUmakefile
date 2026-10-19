COMP	   = gnu

DIM        = 1

USE_MPI    = TRUE
DEBUG      = FALSE

PROFILE    = FALSE

USE_RAD = TRUE


HYDROSIM_HOME ?= ../../..

EOS_DIR     := rad_power_law

NETWORK_DIR := general_null
NETWORK_INPUTS := gammalaw.net

OPACITY_DIR := rad_power_law

NGROUPS := 60

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
