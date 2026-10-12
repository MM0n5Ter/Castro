#!/bin/sh

# clone HydroSim and all associated repos, then output the necessary
# environment variables to build with this source.

branch=master

amrex_server=https://github.com/AMReX-Codes
astro_server=https://github.com/AMReX-Astro
micro_server=https://github.com/StarKiller-astro


#-----------------------------------------------------------------------------

pwd=`pwd`

# HydroSim
echo "cloning HydroSim"
git clone ${astro_server}/HydroSim.git
cd HydroSim
git checkout ${branch}
cd ..


# AMReX
echo " "
echo "cloning amrex"
git clone ${amrex_server}/amrex.git
cd amrex
git checkout ${branch}
cd ..


# Microphysics
echo " "
echo "cloning Microphysics"
git clone ${micro_server}/Microphysics.git
cd Microphysics
git checkout ${branch}
cd ..


# output the necessary environment changes
if [ -f hydrosim_exports.sh ]; then
    echo "removing old hydrosim_exports.sh"
    rm -f hydrosim_exports.sh
fi

cat >> hydrosim_exports.sh << EOF
export HYDROSIM_HOME="${pwd}/HydroSim"
export MICROPHYSICS_HOME="${pwd}/Microphysics"
export AMREX_HOME="${pwd}/amrex"
EOF

echo " "
echo "source hydrosim_exports.sh to setup the environment for building"




