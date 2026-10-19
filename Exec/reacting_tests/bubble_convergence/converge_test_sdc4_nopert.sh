#!/bin/bash

### NOTE: build with USE_REACT=FALSE USE_TRUE_SDC=TRUE

EXEC=./HydroSim2d.gnu.MPI.TRUESDC.ex

RUNPARAMS="
hydrosim.sdc_accuracy_order=4
hydrosim.time_scheme=2
hydrosim.clamp_fourth_order=1
hydrosim.use_recon_gamma1=1"

mpiexec -n 8 ${EXEC} inputs_2d.64 ${RUNPARAMS} hydrosim.do_react=0 problem.do_pert=0 >& 64.out
mpiexec -n 8 ${EXEC} inputs_2d.128 ${RUNPARAMS} hydrosim.do_react=0 problem.do_pert=0 >& 128.out
mpiexec -n 16 ${EXEC} inputs_2d.256 ${RUNPARAMS} hydrosim.do_react=0 problem.do_pert=0 >& 256.out


