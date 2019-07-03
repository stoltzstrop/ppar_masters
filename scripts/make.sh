#!/bin/bash

# this file serves mostly as a reminder of how to compile each of the versions
# user inputs the version to compile and the case statement calls the makefile appropriately

PROJ=$1

case "$PROJ" in

    targetDP-c )
        cd ../src/targetDP/
        make clean
        make cc
        ;;
   targetDP-cuda )
        cd ../src/targetDP/
        make clean
        make cuda
        ;;
   opencl-c )
        cd ../src/opencl/
        make clean
        make
        ;;
   opencl-cpp )
        cd ../src/opencl_cpp/
        make clean
        make
        ;;
   abstract )
        cd ../src/abstract
        make clean
        make
        ;;
   original-cuda )
        cd ../src/original_cuda
        make clean
        make
        ;;
esac
