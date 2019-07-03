#!/bin/bash

#script for pulling out data layout results and putting it in a more friendly comparison format

#directory containins struct data
DIR=~/workspace/phd/room_code/data/struct_compare

# list files in the directory, find all appropriate lines in those files, paste the lines on the same line then pull out the data of interest and print it in an R-friendly way
find $DIR -type f | xargs egrep "time:|Kernel1:" | paste -d " " - - | awk -F'/|:' '{ printf $9":"$10":"$13":"$NF"\n" }'
