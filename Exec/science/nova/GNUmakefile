PRECISION        = DOUBLE
PROFILE          = FALSE
DEBUG            = FALSE
DIM              = 3

COMP	         = gnu

USE_MPI          = TRUE
USE_OMP          = FALSE
USE_CUDA         = FALSE

USE_REACT        = TRUE
USE_DIFFUSION    = FALSE
USE_GRAV         = TRUE

HYDROSIM_HOME := ../../..

USE_MODEL_PARSER = TRUE
MAX_NPTS_MODEL = 20000

EOS_DIR     := helmholtz

NETWORK_DIR := nova

CONDUCTIVITY_DIR := stellar

INTEGRATOR_DIR := RKC

Bpack   := ./Make.package
Blocs   := .

include $(HYDROSIM_HOME)/Exec/Make.HydroSim
