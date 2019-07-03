#!/bin/bash

# script for running all versions on NVIDIA platforms - also runs the CPU version 

# setup directory strings
BIN=../../bin
SCRIPTS=../../scripts
NPROC=`nproc` # get number of processors available for CPU run

# loop over each version 10 times 
ITERATIONS=10
for ITER in $(seq 1 $ITERATIONS); do
    for DIR in ../src/*; do
    
         cd $DIR
         make clean 
         rm -f $BIN/*.cl
        
         if [[ $DIR == *"targetDP" ]]; then
             # for targetDP C version, run number of OMP threads = number of processors 
             export OMP_NUM_THREADS=$NPROC
             echo "OMP_NUM_THREADS="$OMP_NUM_THREADS
             make cc
             $BIN/BasicRoom
             $SCRIPTS/runTestLatest.sh
             make clean
             make cuda
             $BIN/BasicRoom
             $SCRIPTS/runTestLatest.sh
         elif [[ $DIR != "openclxx" ]];then # skip invalid opencl version
             make
             ln -s $(pwd)/*.cl $BIN
             $BIN/BasicRoom
             $SCRIPTS/runTestLatest.sh
         fi
         cd -
    done
done

