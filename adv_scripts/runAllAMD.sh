#!/bin/bash

# script for running all advanced versions on AMD platforms  

# setup directories
BIN=../../adv_bin
SCRIPTS=../../adv_scripts
NPROC=`nproc`

ITERATIONS=10

# loop over number of iterations
for ITER in $(seq 1 $ITERATIONS); do
	for DIR in ../adv_src/*; do
	    
	    cd $DIR
	    make clean 
	   
            # run all directories except the abstract and cuda versions
            if [[ $DIR != *"abstract" ]] && [[ $DIR != *"cuda" ]];then
	        make
                $BIN/AdvRoom
	    fi
	    cd -
	done
done
