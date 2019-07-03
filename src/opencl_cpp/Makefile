include ../../include/include.mk

# Makefile for opencl c++ version (mostly same as in include file)

CC=g++

CFLAGS=-std=c++0x -w -g $(OPENCL_INC)

$(PROJ): $(PROJ).cpp
	$(CC) $(CFLAGS) -lm  -o $(BIN)/$@ $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(CLLIBS)
	$(cp-kernels)	

.PHONY: clean

clean:
	$(rmBin)	
