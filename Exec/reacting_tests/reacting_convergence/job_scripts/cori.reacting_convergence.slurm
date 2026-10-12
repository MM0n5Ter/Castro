#!/bin/bash
#SBATCH --job-name=react_conv
#SBATCH --account=m3018
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --constraint=haswell

echo "starting..."

export OMP_NUM_THREADS=1


slack_job_start.py "starting reacting_convergence problem @ NERSC" @michael

export HYDROSIM_EXEC=./HydroSim2d.intel.haswell.MPI.ex

# Cori Haswell has 2 Intel "Haswell" processors each with 16 cores.
# number of nodes = # of MPI * number of threads / 32 (cores / node)


# SDC-4
RUNPARAMS="
hydrosim.sdc_order=4
hydrosim.time_integration_method=2
hydrosim.limit_fourth_order=1
hydrosim.use_reconstructed_gamma1=1
hydrosim.sdc_solver=1"

srun ${HYDROSIM_EXEC} inputs.64 amr.plot_file=react_converge_64_sdc4_plt ${RUNPARAMS}
srun ${HYDROSIM_EXEC} inputs.128 amr.plot_file=react_converge_128_sdc4_plt ${RUNPARAMS}
srun ${HYDROSIM_EXEC} inputs.256 amr.plot_file=react_converge_256_sdc4_plt ${RUNPARAMS}
srun ${HYDROSIM_EXEC} inputs.512 amr.plot_file=react_converge_512_sdc4_plt ${RUNPARAMS}

# SDC
RUNPARAMS="
hydrosim.sdc_order=2
hydrosim.time_integration_method=2
hydrosim.ppm_type=0
hydrosim.sdc_solve_for_rhoe=1
hydrosim.sdc_solver_tol=1.e-10
hydrosim.sdc_solver=1"

srun ${HYDROSIM_EXEC} inputs.64 amr.plot_file=react_converge_64_sdc_plt ${RUNPARAMS}
srun ${HYDROSIM_EXEC} inputs.128 amr.plot_file=react_converge_128_sdc_plt ${RUNPARAMS}
srun ${HYDROSIM_EXEC} inputs.256 amr.plot_file=react_converge_256_sdc_plt ${RUNPARAMS}
srun ${HYDROSIM_EXEC} inputs.512 amr.plot_file=react_converge_512_sdc_plt ${RUNPARAMS}

# Strang
RUNPARAMS="hydrosim.time_integration_method=0"

srun ${HYDROSIM_EXEC} inputs.64 amr.plot_file=react_converge_64_strang_plt ${RUNPARAMS}
srun ${HYDROSIM_EXEC} inputs.128 amr.plot_file=react_converge_128_strang_plt ${RUNPARAMS}
srun ${HYDROSIM_EXEC} inputs.256 amr.plot_file=react_converge_256_strang_plt ${RUNPARAMS}
srun ${HYDROSIM_EXEC} inputs.512 amr.plot_file=react_converge_512_strang_plt ${RUNPARAMS}
