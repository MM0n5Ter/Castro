#!/bin/bash
#SBATCH --job-name=bubble_convergence
#SBATCH --nodes=2
#SBATCH --tasks-per-node=32
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --constraint=haswell

echo "starting..."

export OMP_NUM_THREADS=1

export HYDROSIM_EXEC=./HydroSim2d.intel.haswell.MPI.TRUESDC.ex

# Cori Haswell has 2 Intel "Haswell" processors each with 16 cores.
# number of nodes = # of MPI * number of threads / 32 (cores / node)

RUNPARAMS="
hydrosim.sdc_accuracy_order=4
hydrosim.time_scheme=2
hydrosim.clamp_fourth_order=1
hydrosim.use_recon_gamma1=1
hydrosim.sdc_nonlin_method=1"

srun ${HYDROSIM_EXEC} inputs_2d.64 ${RUNPARAMS} amr.plot_file=bubble_64_sdc4_plt &> 64.out
srun ${HYDROSIM_EXEC} inputs_2d.128 ${RUNPARAMS} amr.plot_file=bubble_128_sdc4_plt &> 128.out
srun ${HYDROSIM_EXEC} inputs_2d.256 ${RUNPARAMS} amr.plot_file=bubble_256_sdc4_plt &> 256.out
srun ${HYDROSIM_EXEC} inputs_2d.512 ${RUNPARAMS} amr.plot_file=bubble_512_sdc4_plt &> 512.out
