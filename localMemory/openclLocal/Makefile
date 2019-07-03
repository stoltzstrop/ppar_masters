PROJ=BasicRoom
BIN=room
CC=gcc
NVCC=nvcc
C_PHI_CC=/opt/intel/bin/icc

CFLAGS=-std=c99 -w -lm 
CUDAARCHFLAGS=sm_35
MMIC_FLAG=-mmic

CLLIBS=-lOpenCL
INC_DIRS=../../include
PERF_INC=../../../../logistics/GPUPerfAPI/Include
PERF_LIB=../../../../logistics/GPUPerfAPI/Bin/Linx64

OPENCL_INC=-I/opt/AMDAPPSDK-3.0/include
#OPENCL_INC=-I/usr/local/cuda/targets/x86_64-linux/include

ifneq (,$(findstring casper, $(MYHOSTNAME)))
CUDAARCHFLAGS=sm_50
endif

CFLAGS_CUDA=-DCUDA -O3 -arch=$(CUDAARCHFLAGS) -dc -x cu #-save-temps

# architecture specific #

MYHOSTNAME=$(shell hostname)
ifneq (,$(findstring supersonic, $(MYHOSTNAME)))
CC=clang
else ifneq (,$(findstring phi, $(MYHOSTNAME)))
CFLAGS=-std=c99 -w -I../include
OPENCL_INC=-I/usr/include
LIB_DIRS=/usr/lib64/
else ifneq (,$(findstring spa, $(MYHOSTNAME)))
CFLAGS=-std=c99 -w -I../include -lm
OPENCL_INC=-I/usr/include
LIB_DIRS=/usr/lib64/
else ifeq ($(loc),hydra)
OPENCL_INC=-I/opt/cuda/include
else
#INC_DIRS+=/usr/local/cuda/targets/x86_64-linux/include
#CLLIBS+=-lm -L/usr/local/cuda/lib64
CLLIBS+=-lm -L/opt/AMDAPPSDK-3.0/lib/x86_64
endif 

# functions #

define cp-kernels =
rm -f $(BIN)/*.cl
ln -s $(shell pwd)/*.cl $(BIN)
endef

#define link-clgrid = 
#rm -f $(BIN)/CLGrid.h
#ln -s $(shell pwd)/CLGrid.h $(BIN)
#endef

define rmBin = 
rm $(PROJ)
endef

CC=g++

CFLAGS=-std=c++0x -w -g $(OPENCL_INC)

$(PROJ): $(PROJ).cpp
	$(CC) $(CFLAGS) -o $@ $^ $(INC_DIRS:%=-I%) $(PERF_INC:%=-I%) $(LIB_DIRS:%=-L%) $(PERF_LIB:%=-L%) $(CLLIBS)

.PHONY: clean

clean:
	rm $(PROJ)	
