#!/bin/bash

#profilers like to output lots of data that you don't care about - this script pulls out the memory bandwidth values from an nvprof output file and outputs the memory bandwidths for both kernels formatted to GB/s

FILE=$1 

cat $FILE | grep UpDateScheme | awk 'BEGIN { total = 0.0; } { total+=$5 } END { printf total/1000000 "\n"}'

cat $FILE | grep inout | awk 'BEGIN { total = 0.0; } { total+=$5 } END { printf total/1000000 "\n"}'
