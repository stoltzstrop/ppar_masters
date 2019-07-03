#!/bin/bash

# simple script to run a single advanced version ITERATIONS number of times 

BIN=../../adv_bin
SCRIPTS=../../adv_scripts
DIR=../adv_src/leggy_cuda

ITERATIONS=10
cd $DIR
make clean 
rm -f $BIN/*.cl # on some platforms, linking does not work from makefiles, so force it 
make
ln -s $(pwd)/*.cl $BIN
for ITER in $(seq 1 $ITERATIONS); do
             $BIN/AdvRoom
done
cd -

