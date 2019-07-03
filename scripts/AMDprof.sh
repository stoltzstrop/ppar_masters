#!/bin/bash

# a teeny script for pulling out only the data of interest from the AMD APP profiler 

cat $1 | egrep "UpdateScheme|Method" | awk -F',' '{ print $7","$23","$29","$24","$16","$18","$28 }'

