PROJ=AdvRoom
BIN=../../adv_bin
CC=gcc
NVCC=nvcc
C_PHI_CC=/opt/intel/bin/icc

CFLAGS=-std=c99 -w -lm 
CUDAARCHFLAGS=sm_35
MMIC_FLAG=-mmic

CLLIBS=-lOpenCL
INC_DIRS=../../include

OPENCL_INC=-I/opt/AMDAPPSDK-3.0/include

ifneq (,$(findstring casper, $(MYHOSTNAME)))
CUDAARCHFLAGS=sm_50
endif

CFLAGS_CUDA=-DCUDA -O3 -arch=$(CUDAARCHFLAGS) -dc -x cu 

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
CLLIBS+=-lm -L/opt/AMDAPPSDK-3.0/lib/x86_64
endif 

# functions #
define cp-kernels =
rm -f $(BIN)/*.cl
ln -s $(shell pwd)/*.cl $(BIN)
endef

define rmBin = 
rm $(BIN)/*
endef
