#ifndef INITIAL_H
#define INITIAL_H
#include "constants.h"

// header file for some data output and OpenCL related stuff

// names of OpenCL kernels and kernel file(s) 
#define STENCIL_KERNEL_FUNC "UpdateScheme"
#define INOUT_KERNEL_FUNC "inout"
#define PROGRAM_NAME "kernelsGrid.cl"

// hard-coded directories to make saving data easier 
#define data_directory "/workspace/phd/room_code/data/"
#define mic_directory "/home-hydra/h022/s1147290/workspace/phd/"

// strings for filenames for saving data 
#define CUDA_STR "cuda"
#define UNOPT_CUDA_STR "unopt_cuda"
#define OPENCL_STR "opencl"
#define OPENCLCPP_STR  "openclcpp" 
#define ABSTRACT_STR "abstract"
#define TARGETDP_STR "targetDP" 

// test macros for OpenCL kernels 
#define GET_NEIGHBOR_SUM(u, cp) u[cp-1]+u[cp+1]+u[cp-Nx]+u[cp+Nx]+u[cp-AREA]+u[cp+AREA]
#define CALCULATE_INDEX(x,y,z) z*AREA+(y*Nx+x);



#endif
