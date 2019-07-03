include ../../include/include.mk

# Makefile for opencl version (mostly same as in include file)

CFLAGS=-std=c99 -Wall -DUNIX -g -DDEBUG -w $(OPENCL_INC)

$(PROJ): $(PROJ).c
	$(CC) $(CFLAGS)  $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(CLLIBS) -o $(BIN)/$@ $^
	$(cp-kernels)	
.PHONY: clean

clean:
	rm $(BIN)/$(PROJ)
