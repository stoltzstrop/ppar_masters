include ../../include/include.mk

# Makefile for targetDP version 
# Includes the extra files necessary for targetDP library

CFLAGS+=-O3 -fopenmp #-qopt-report

ifneq (,$(findstring phi, $(MYHOSTNAME)))
CC=$(C_PHI_CC)
CFLAGS+=-mmic
endif

SRC_C=targetDP_C.c 
SRC_CUDA= targetDP_CUDA.c 
AR = ar -cru

NVCFLAGS= -DCUDA -arch=$(CUDAARCHFLAGS) -dc -x cu -lm#-save-temps
NVLFLAGS= -L/usr/local/cuda/lib64 -lcudart

TDP_INC=./

TDP_LIB=../../lib
TDP_C_AR=$(TDP_LIB)/libtargetDP_C.a 
TDP_CUDA_AR=$(TDP_LIB)/libtargetDP_CUDA.a 

.SUFFIXES:
.SUFFIXES: .c .cu  .o

OBJ=	$(SRC:.c=.o)
NVOBJ=	$(NVSRC:.cu=.o)
OCLOBJ=	$(OCLSRC:.c=.o)

# make c version using OMP 
cc: cc_lib
	$(CC) $(CFLAGS) -I$(INC_DIRS) -I$(TDP_INC) -c $(PROJ).c
	$(CC) $(CFLAGS) -I$(INC_DIRS) -I$(TDP_INC) $(PROJ).o $(TDP_C_AR) -o $(BIN)/$(PROJ)

#make cuda version 
cuda: cuda_lib
	$(NVCC) $(NVCFLAGS) -I$(INC_DIRS) -I$(TDP_INC)  -c $(PROJ).c
	$(NVCC) -arch=$(CUDAARCHFLAGS) $(PROJ).o $(TDP_CUDA_AR)  -o  $(BIN)/$(PROJ)
	
cuda_lib: 
	$(NVCC) $(CFLAGS_CUDA) -I$(INC_DIRS) -c $(SRC_CUDA)
	$(AR) $(TDP_CUDA_AR) $(SRC_CUDA:.c=.o)

cc_lib:
	$(CC) -I$(INC_DIRS) $(CFLAGS) -c $(SRC_C)
	$(AR) $(TDP_C_AR) $(SRC_C:.c=.o)
clean:
	rm -f  $(BIN)/$(PROJ) $(TDP_LIB)/*.a
