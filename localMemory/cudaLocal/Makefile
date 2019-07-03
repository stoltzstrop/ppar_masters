include ../../include/include.mk

OPENCL_LIB=/usr/local/cuda/targets/x86_64-linux/include

LIBS+=-lm -L/usr/local/cuda/lib64

BIN=../../bin
EXE=localRoom


$(PROJ): $(PROJ).cu
	$(NVCC) -arch=$(CUDAARCHFLAGS) -w -I$(INC_DIRS) -I./ $^ -o $(BIN)/$(EXE)

.PHONY: clean

clean:
	rm $(BIN)/$(EXE)
