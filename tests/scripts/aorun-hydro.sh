#!/bin/bash

module use /home/users/twhite/share/modulefiles
module load ompi-cray hdf5

export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
OUTDIR="run/out.hydro-amd-ompi.$(date +%m%d.%H%M%S)"
set -x
mkdir -p ${OUTDIR}
cd ${OUTDIR}
export OMP_NUM_THREADS=16
srun -n1 -c$OMP_NUM_THREADS -N1 --exclusive -C MI60 ../../bin/cholla.hydro-amd-ompi ../../tests/3D/sod.txt |& tee tee.mi60
srun -n1 -c$OMP_NUM_THREADS -N1 --exclusive -p amdMI100 ../../bin/cholla.hydro-amd-ompi ../../tests/3D/sod.txt |& tee tee.mi100
