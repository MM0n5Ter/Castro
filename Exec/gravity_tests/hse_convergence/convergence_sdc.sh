#!/bin/bash


EXEC=./HydroSim1d.gnu.MPI.TRUESDC.ex


## sdc-2 + reflect

ofile=sdc2-reflect.converge.out

RUNPARAMS="
hydrosim.time_integration_method=2
hydrosim.sdc_order=2
hydrosim.ppm_type=0
hydrosim.use_pslope=1
hydrosim.limit_fourth_order=1
hydrosim.use_reconstructed_gamma1=1
hydrosim.lo_bc=3
hydrosim.hi_bc=3
"""

${EXEC} inputs.ppm.64 ${RUNPARAMS} >& 64.out
pfile=`ls -t | grep -i hse_64_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed > ${ofile}

${EXEC} inputs.ppm.128 ${RUNPARAMS} >& 128.out
pfile=`ls -t | grep -i hse_128_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}

${EXEC} inputs.ppm.256 ${RUNPARAMS} >& 256.out
pfile=`ls -t | grep -i hse_256_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}

${EXEC} inputs.ppm.512 ${RUNPARAMS} >& 512.out
pfile=`ls -t | grep -i hse_512_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}


## sdc-2 + ppm + reflect

ofile=sdc2-ppm-reflect.converge.out

RUNPARAMS="
hydrosim.time_integration_method=2
hydrosim.sdc_order=2
hydrosim.ppm_type=1
hydrosim.limit_fourth_order=1
hydrosim.use_reconstructed_gamma1=1
hydrosim.lo_bc=3
hydrosim.hi_bc=3
"""

${EXEC} inputs.ppm.64 ${RUNPARAMS} >& 64.out
pfile=`ls -t | grep -i hse_64_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed > ${ofile}

${EXEC} inputs.ppm.128 ${RUNPARAMS} >& 128.out
pfile=`ls -t | grep -i hse_128_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}

${EXEC} inputs.ppm.256 ${RUNPARAMS} >& 256.out
pfile=`ls -t | grep -i hse_256_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}

${EXEC} inputs.ppm.512 ${RUNPARAMS} >& 512.out
pfile=`ls -t | grep -i hse_512_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}


## sdc-2 + HSE

ofile=sdc2.converge.out

RUNPARAMS="
hydrosim.time_integration_method=2
hydrosim.sdc_order=2
hydrosim.ppm_type=0
hydrosim.use_pslope=1
hydrosim.limit_fourth_order=1
hydrosim.use_reconstructed_gamma1=1
"""

${EXEC} inputs.ppm.64 ${RUNPARAMS} >& 64.out
pfile=`ls -t | grep -i hse_64_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed > ${ofile}

${EXEC} inputs.ppm.128 ${RUNPARAMS} >& 128.out
pfile=`ls -t | grep -i hse_128_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}

${EXEC} inputs.ppm.256 ${RUNPARAMS} >& 256.out
pfile=`ls -t | grep -i hse_256_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}

${EXEC} inputs.ppm.512 ${RUNPARAMS} >& 512.out
pfile=`ls -t | grep -i hse_512_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}


## sdc-2 + ppm

ofile=sdc2-ppm.converge.out

RUNPARAMS="
hydrosim.time_integration_method=2
hydrosim.sdc_order=2
hydrosim.ppm_type=1
hydrosim.limit_fourth_order=1
hydrosim.use_reconstructed_gamma1=1
"""

${EXEC} inputs.ppm.64 ${RUNPARAMS} >& 64.out
pfile=`ls -t | grep -i hse_64_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed > ${ofile}

${EXEC} inputs.ppm.128 ${RUNPARAMS} >& 128.out
pfile=`ls -t | grep -i hse_128_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}

${EXEC} inputs.ppm.256 ${RUNPARAMS} >& 256.out
pfile=`ls -t | grep -i hse_256_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}

${EXEC} inputs.ppm.512 ${RUNPARAMS} >& 512.out
pfile=`ls -t | grep -i hse_512_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}


## sdc-4 + reflect

ofile=sdc4-reflect.converge.out

RUNPARAMS="
hydrosim.time_integration_method=2
hydrosim.sdc_order=4
hydrosim.limit_fourth_order=1
hydrosim.use_reconstructed_gamma1=1
hydrosim.lo_bc=3
hydrosim.hi_bc=3
"""

${EXEC} inputs.ppm.64 ${RUNPARAMS} >& 64.out
pfile=`ls -t | grep -i hse_64_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed > ${ofile}

${EXEC} inputs.ppm.128 ${RUNPARAMS} >& 128.out
pfile=`ls -t | grep -i hse_128_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}

${EXEC} inputs.ppm.256 ${RUNPARAMS} >& 256.out
pfile=`ls -t | grep -i hse_256_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}

${EXEC} inputs.ppm.512 ${RUNPARAMS} >& 512.out
pfile=`ls -t | grep -i hse_512_plt | head -1`
fextrema.gnu.ex -v flow_speed ${pfile} | grep -i flow_speed >> ${ofile}

