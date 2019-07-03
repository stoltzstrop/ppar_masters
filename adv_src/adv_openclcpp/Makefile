include ../../include/includeAdv.mk

CC=g++

CFLAGS=-std=c++0x -w -g $(OPENCL_INC)

$(PROJ): $(PROJ).cpp
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(CLLIBS)
	$(cp-kernels)	

.PHONY: clean

clean:
	$(rmBin)	
