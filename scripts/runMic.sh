#!/bin/bash

# script used to run C versions (ie. non-OpenCL) on Xeon phi by ssh'ing into mic

BIN=../../bin
ITERATIONS=10

# these must be called
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/lib/mic/
export KMP_AFFINITY=DISABLED

for ITER in $(seq 1 $ITERATIONS); do
    for I in 236  # the optimal number, but others were tested
    do
        export OMP_NUM_THREADS="$I"
        /home-hydra/h022/s1147290/workspace/phd/room_code/bin/BasicRoom # must be compiled correctly beforehand
    done
done
