#!/bin/bash


EXEC=./HydroSim1d.gnu.MPI.ex


## plm

ofile=plm.converge.out

RUNPARAMS="
hydrosim.ppm_recon_kind=0
hydrosim.use_pressure_slope=1
"""

${EXEC} inputs.ppm.64 ${RUNPARAMS} >& 64.out
pfile=`ls -t | grep -i hse_64_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel > ${ofile}

${EXEC} inputs.ppm.128 ${RUNPARAMS} >& 128.out
pfile=`ls -t | grep -i hse_128_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel >> ${ofile}

${EXEC} inputs.ppm.256 ${RUNPARAMS} >& 256.out
pfile=`ls -t | grep -i hse_256_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel >> ${ofile}

${EXEC} inputs.ppm.512 ${RUNPARAMS} >& 512.out
pfile=`ls -t | grep -i hse_512_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel >> ${ofile}


## plm + hse reflect

ofile=plm-hsereflect.converge.out

RUNPARAMS="
hydrosim.ppm_recon_kind=0
hydrosim.use_pressure_slope=1
hydrosim.hse_temp_extrap=1
hydrosim.hse_mirror_vel=1
"""

${EXEC} inputs.ppm.64 ${RUNPARAMS} >& 64.out
pfile=`ls -t | grep -i hse_64_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel > ${ofile}

${EXEC} inputs.ppm.128 ${RUNPARAMS} >& 128.out
pfile=`ls -t | grep -i hse_128_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel >> ${ofile}

${EXEC} inputs.ppm.256 ${RUNPARAMS} >& 256.out
pfile=`ls -t | grep -i hse_256_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel >> ${ofile}

${EXEC} inputs.ppm.512 ${RUNPARAMS} >& 512.out
pfile=`ls -t | grep -i hse_512_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel >> ${ofile}


## plm + reflect + nopslope

ofile=plm-reflect-nopslope.converge.out

RUNPARAMS="
hydrosim.ppm_recon_kind=0
hydrosim.lo_bc=3
hydrosim.hi_bc=3
hydrosim.use_pressure_slope=0
"""

${EXEC} inputs.ppm.64 ${RUNPARAMS} >& 64.out
pfile=`ls -t | grep -i hse_64_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel > ${ofile}

${EXEC} inputs.ppm.128 ${RUNPARAMS} >& 128.out
pfile=`ls -t | grep -i hse_128_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel >> ${ofile}

${EXEC} inputs.ppm.256 ${RUNPARAMS} >& 256.out
pfile=`ls -t | grep -i hse_256_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel >> ${ofile}

${EXEC} inputs.ppm.512 ${RUNPARAMS} >& 512.out
pfile=`ls -t | grep -i hse_512_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel >> ${ofile}


## plm + reflect + pressure_slope

ofile=plm-reflect-pressure_slope.converge.out

RUNPARAMS="
hydrosim.ppm_recon_kind=0
hydrosim.lo_bc=3
hydrosim.hi_bc=3
hydrosim.use_pressure_slope=1
"""

${EXEC} inputs.ppm.64 ${RUNPARAMS} >& 64.out
pfile=`ls -t | grep -i hse_64_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel > ${ofile}

${EXEC} inputs.ppm.128 ${RUNPARAMS} >& 128.out
pfile=`ls -t | grep -i hse_128_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel >> ${ofile}

${EXEC} inputs.ppm.256 ${RUNPARAMS} >& 256.out
pfile=`ls -t | grep -i hse_256_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel >> ${ofile}

${EXEC} inputs.ppm.512 ${RUNPARAMS} >& 512.out
pfile=`ls -t | grep -i hse_512_plt | head -1`
fextrema.gnu.ex -v magvel ${pfile} | grep -i magvel >> ${ofile}







