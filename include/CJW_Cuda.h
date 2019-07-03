/*
    CJW_Cuda.h
    Generic Cuda functions

    Craig J. Webb
    2/10/11


*/

#ifndef CJW_Cuda_h
#define CJW_Cuda_h

#define cuErr(err) __checkCudaErrors (err, __FILE__, __LINE__)

// ----------------------------------------------------------------
// FUNCTION PROTOTYPES
// ----------------------------------------------------------------

void printDevDetails();

inline void __checkCudaErrors( cudaError err, const char *file, const int line );

void checkLastCUDAError(const char *msg);

// ----------------------------------------------------------------
// FUNCTION IMPLEMENTATIONS
// ----------------------------------------------------------------

inline void __checkCudaErrors( cudaError err, const char *file, const int line ){
        if( cudaSuccess != err) {
		    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                    file, line, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }
    

void checkLastCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
	{
	    fprintf(stderr, "\nCuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
	    exit(EXIT_FAILURE);
	}                         
}

void printDevDetails(){
	
	// Get name of device being used...
	int devy;
	cudaGetDevice(&devy);
	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, devy);
	printf("\n");
	printf("RUNNING ON GPU        : %d - %s, compute %d.%d\n", devy,devProp.name,devProp.major,devProp.minor);
    printf("Total global memory   : %u\n", devProp.totalGlobalMem);
    printf("Maximum memory pitch  : %u\n", devProp.memPitch);
    printf("Shared mem per block  : %u\n", devProp.sharedMemPerBlock);
    printf("Total constant memory : %u\n",  devProp.totalConstMem);
    printf("Registers per block   : %d\n", devProp.regsPerBlock);
    printf("Warp size             : %d\n",  devProp.warpSize);
    printf("Max threads per block : %d\n",  devProp.maxThreadsPerBlock);
    printf("Max dims of block     : %dx%dx%d\n", devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
    printf("Max dims of grid      : %dx%dx%d\n", devProp.maxGridSize[0],devProp.maxGridSize[1],devProp.maxGridSize[2]);
	printf("Number of multiprocs  : %d\n\n",  devProp.multiProcessorCount);
}


#endif
