include ../../include/include.mk

# Makefile for abstractCL version (mostly same as in include file)

CC=g++

INC=CLSetup.cpp CLDeviceData.hpp CLKernel.cpp SimpleGridData.hpp

CFLAGS=-std=c++0x -w -I../include $(OPENCL_INC) -g

$(PROJ): $(PROJ).cpp
	$(CC) $(CFLAGS) $^ $(INC) $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(CLLIBS) -o $(BIN)/$@
	$(cp-kernels)	
.PHONY: clean

clean:
	rm $(BIN)/$(PROJ)
