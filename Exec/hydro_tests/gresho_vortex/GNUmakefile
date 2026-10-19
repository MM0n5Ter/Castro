
PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 2

ifneq ($(DIM),2)
  $(error Error: this problem is only supported in 2D)
endif

COMP	   = gnu

USE_MPI    = TRUE
USE_OMP    = FALSE

HYDROSIM_HOME  ?= ../../..

EOS_DIR     := gamma_law

NETWORK_DIR := general_null
NETWORK_INPUTS = gammalaw.net

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
