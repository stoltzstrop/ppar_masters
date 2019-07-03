#!/bin/bash

# simple script to run a single version ITERATIONS number of times 

BIN=../../bin  # the executable directory
SCRIPTS=../../scripts # this directory 
DIR=../src/abstract # source directory 

ITERATIONS=10
cd $DIR
make clean 
rm -f $BIN/*.cl  # on some platforms, linking does not work from makefiles, so force it 
make 
ln -s $(pwd)/*.cl $BIN
for ITER in $(seq 1 $ITERATIONS); do
             $BIN/BasicRoom
done
cd -

