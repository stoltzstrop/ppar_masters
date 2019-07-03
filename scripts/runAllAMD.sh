#!/bin/bash

# script for running all versions on AMD platforms  

# setup directories
BIN=../../bin
SCRIPTS=../../scripts

ITERATIONS=10

# loop over number of iterations
for ITER in $(seq 1 $ITERATIONS); do
	for DIR in ../src/*; do
	    
	    cd $DIR
	    make clean 
	   
    	    if [[ $DIR == *"targetDP" ]] || [[ $DIR == *"original_cuda" ]]; then # skip cuda versions
		echo "Skipping..."$DIR
	    else
                # make and run other versions 
	        make
	        $BIN/BasicRoom
                # run the unit tests
                $SCRIPTS/runTestLatest.sh
	    fi
	    cd -
	done
done
