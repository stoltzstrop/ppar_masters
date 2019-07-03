#!/bin/bash

# script for running all advanced versions on NVIDIA platforms 

# setup directory strings
BIN=../../adv_bin
SCRIPTS=../../adv_scripts
NPROC=`nproc`


ITERATIONS=10
for ITER in $(seq 1 $ITERATIONS); do
    for DIR in ../adv_src/*; do

         echo $DIR
         cd $DIR
         # run all but abstract version
         if [[ $DIR != *"abstract" ]];then

             make clean 
             make
             $BIN/AdvRoom
         fi
         cd -
    done
done

