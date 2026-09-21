#!/bin/bash

module --force purge

module load modules/2.4-20250724

module load gcc
module load cuda/12.8.0
module load openmpi
module load hdf5
