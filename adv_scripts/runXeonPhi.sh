#!/bin/bash

# script for running advanced OpenCL versions on the Xeon Phi

# setup some string variables 
GPU_STR=CL_DEVICE_TYPE_GPU
ACC_STR=CL_DEVICE_TYPE_ACCELERATOR
CL_UTILS_FILE=../adv_include/cl_utils.h
BIN=../../adv_bin
SCRIPTS=../../adv_scripts

# always forget to do this, so just swap in that this is an accelerator run at the start .... 
sed -i "s/$GPU_STR/$ACC_STR/g" "$CL_UTILS_FILE" 

ITERATIONS=10

for ITER in $(seq 1 $ITERATIONS); do 
    for DIR in ../adv_src/*; do
        if [[ $DIR == *"abstract" ]] || [[ $DIR == *"cuda" ]];then
            echo "Skipping..."$DIR
        else
            cd $DIR
            make clean 
            rm -f $BIN/*.cl
            echo $DIR
            make
            ln -s $(pwd)/*.cl $BIN
            $BIN/AdvRoom
            cd -
        fi
    done
done

#...and swap it back at the end
sed -i "s/$ACC_STR/$GPU_STR/g" "$CL_UTILS_FILE" 
