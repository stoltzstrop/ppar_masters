include ../../include/include.mk
  
# Makefile for opencl cuda version (mostly same as in include file)

OPENCL_LIB=/usr/local/cuda/targets/x86_64-linux/include

LIBS+=-lm -L/usr/local/cuda/lib64

$(PROJ): $(PROJ).cu
	$(NVCC) -arch=$(CUDAARCHFLAGS) -I$(INC_DIRS) $^ -o $(BIN)/$@ 

.PHONY: clean

clean:
	rm $(BIN)/$(PROJ)
