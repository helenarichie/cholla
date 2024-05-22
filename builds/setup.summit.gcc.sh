#!/bin/bash

#-- This script needs to be source-d in the terminal, e.g.
#   source ./setup.summit.gcc.sh

#module load gcc/10.2.0 cuda/11.4.0 fftw hdf5 python
#module load gcc cuda fftw hdf5 python googletest/1.11.0
# module load gcc cuda/12.2.0 fftw hdf5 python
module load gcc cuda/12.2.0 hdf5 python ums/default ums-ompix/default openmpi/5.0.3

export CHOLLA_ENVSET=1
