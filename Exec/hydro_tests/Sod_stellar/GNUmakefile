PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE


#DIM        = 1
DIM        = 2

COMP	   = gnu

USE_MPI    = TRUE
USE_OMP    = FALSE

USE_MHD = FALSE


# define the location of the HYDROSIM top directory
HYDROSIM_HOME ?= ../../..

# This sets the EOS directory in $(MICROPHYSICS_HOME)/EOS
EOS_DIR     := helmholtz

# This sets the network directory in $(MICROPHYSICS_HOME)/networks
NETWORK_DIR := general_null
NETWORK_INPUTS = ignition.net

PROBLEM_DIR ?= ./

Bpack   := $(PROBLEM_DIR)/Make.package
#$(HYDROSIM_HOME)/Util/exact_riemann/Make.package
#Blocs   := $(PROBLEM_DIR) $(HYDROSIM_HOME)/Util/exact_riemann/
#EXTERN_CORE_ADD := $(HYDROSIM_HOME)/Util/exact_riemann/

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
