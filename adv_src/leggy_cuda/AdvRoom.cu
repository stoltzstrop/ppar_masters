#include "Audio.h" 
#include "timing.h"
#include "timing_macros.h"
#include "constants.h"
#include "CJW_Cuda.h"
#include <stdio.h>
#include <string.h>
enum BOOLEAN useOptimisation = FALSE;

// kernel methods
__global__ void UpDateScheme(double *u,const double* __restrict__ u1);
__global__ void inout(double *u,double *out,double ins,int n);

// struct for coeffs
typedef struct
{
	double l2;
	double loss1;
	double loss2;

} coeffs_type;

// constant memory
__constant__ coeffs_type cf_d[1];

// ----------------------------------------------

int main(int argc, char *argv[]){

    if( argc > 1 )
    {
        if(strcmp(argv[1],"off") == 0)
        {
            printf("turning off optimisations...\n");
            useOptimisation = FALSE; 
        }
    }

    // Simulation parameters					      
    int NF            = numberSamples;
    double SR         = (double) numberSamples;             // Sample Rate
    double alpha      = 0.005;               // Boundary loss
    double c          = 344.0;               // Speed of sound in air (m/s)
    double k          = 1/SR;                                    
    double h          = sqrt(3.0)*c*k;
    double lambda     = c*k/h;                        

    // Set constant memory coeffs
    coeffs_type cf_h[1];
	cf_h[0].l2      = (c*c*k*k)/(h*h);
	cf_h[0].loss1   = 1.0/(1.0+lambda*alpha);
	cf_h[0].loss2   = 1.0-lambda*alpha;
     cudaMemcpyToSymbol(cf_d,cf_h,sizeof(coeffs_type)) ;

    char cudaStr[100];
    {
        sprintf(cudaStr,"%s","cuda_leggy");
    }

     printToTimingFileName(cudaStr);

    startTime = getTime(); 
    //-------------------------------------------
    // Initialise input
    int n;
    int alength = dur;
    double *si_h = (double *)calloc(NF,sizeof(double));
    for(n=0;n<dur;n++){
      si_h[n] = 0.5*(1.0-cos(2.0*pi*n/(double)dur));
    }
    if(si_h == NULL) {
    	printf("\nFailed to open input file...\n\n"); exit(EXIT_FAILURE);
    }
    
	// Set up grid and blocks
	int Gx          = Nx/Bx;
	int Gy          = Ny/By;
	int Gz          = Nz/Bz;
	
	dim3 dimBlockInt(Bx, By, Bz);
	dim3 dimGridInt(Gx, Gy, Gz);
	dim3 dimBlockIO(1, 1, 1);
	dim3 dimGridIO(1, 1, 1);

	size_t pr_size  = sizeof(double);
	size_t mem_size = AREA*Nz*pr_size;
	double *out_d, *u_d, *u1_d, *dummy_ptr;                      
	double ins;             
    
        
        dataCopyInitStart = getTime();

	// Initialise memory on device
	 cudaMalloc(&u_d, mem_size) ;       cudaMemset(u_d, 0, mem_size) ;
	 cudaMalloc(&u1_d, mem_size) ;      cudaMemset(u1_d, 0, mem_size) ;
	 cudaMalloc(&out_d, NF*pr_size) ;   cudaMemset(out_d, 0, NF*pr_size) ;
	
        dataCopyInitEnd = getTime();
        dataCopyInitTotal = dataCopyInitEnd - dataCopyInitStart;

	// initialise memory on host
	double *out_h  = (double *)malloc(NF*pr_size);
	double *u_h  = (double *)malloc(AREA*Nz*pr_size);
	if((out_h == NULL)){
		printf("\nout_h memory alloc failed...\n");
		exit(EXIT_FAILURE);
	}
	
	
        if(useOptimisation)
        {
            printf("Switching on cache config\n");
	    cudaFuncSetCacheConfig(UpDateScheme,cudaFuncCachePreferL1);
	}

        kernel1Time = 0.0;
        kernel2Time = 0.0;
	dataCopyBtwTotal = 0.0;

        startKernels = getTime();

        for(n=0;n<NF;n++)
	{
		
                startKernel1 = getTime();
		UpDateScheme<<<dimGridInt,dimBlockInt>>>(u_d,u1_d);
	        cudaThreadSynchronize() ; 
                endKernel1 = getTime();
                kernel1Time += endKernel1-startKernel1;
	        checkLastCUDAError("1st kernel");
		
		// perform eead in out
		ins = 0.0;
		if(n<alength)
                {
                  ins = si_h[n];
                }
                startKernel2 = getTime();
                inout<<<dimGridIO,dimBlockIO>>>(u_d,out_d,ins,n);
	        cudaThreadSynchronize() ; 
                endKernel2 = getTime();
                kernel2Time += endKernel2-startKernel2;
	        checkLastCUDAError("2nd kernel");
	
                dataCopyBtwStart = getTime();
		// update pointers
		dummy_ptr = u1_d;
		u1_d = u_d;
		u_d = dummy_ptr;
                dataCopyBtwEnd = getTime();
                dataCopyBtwTotal += dataCopyBtwEnd - dataCopyBtwStart;
	}
	
        endKernels = getTime();
    
        kernelsTime = endKernels-startKernels;
	checkLastCUDAError("Kernel");
        cudaThreadSynchronize() ; 
    
        dataCopyBackStart = getTime();     
        // copy result back from device
        cudaMemcpy(out_h, out_d, NF*pr_size, cudaMemcpyDeviceToHost) ;
	cudaMemcpy(u_h, u_d, AREA*Nz*pr_size, cudaMemcpyDeviceToHost) ;
    
        dataCopyBackEnd = getTime();     
        dataCopyBackTotal = dataCopyBackEnd - dataCopyBackStart;
        dataCopyTotal = dataCopyInitTotal + dataCopyBtwTotal + dataCopyBackTotal;

        endTime = getTime();
        totalTime = (double) (endTime-startTime);
       

        writeBinaryDataToFile(u_h,getOutputFileName(cudaStr,"room","bin"),VOLUME);       
        writeBinaryDataToFile(out_h,getOutputFileName(cudaStr,"receiver","bin"),numberSamples);       

        for(int jj=NF-10;jj<NF;jj++)
        {
            printf("%.14lf\n",out_h[jj]);
        }

        printToString;
        printOutputs;
        writeTimingsToFile; 
	
        // Free memory
        free(si_h);free(out_h);
	cudaFree(out_d);cudaFree(u_d);cudaFree(u1_d);
	
	exit(EXIT_SUCCESS);
}

// Update Kernel
__global__ void UpDateScheme(double *u,const double* __restrict__ uz)
{
	
   int X = blockIdx.x * Bx + threadIdx.x;
   int Y = blockIdx.y * By + threadIdx.y;
   int Z = blockIdx.z * Bz + threadIdx.z;


      //get center-point index

      //interior
      if( (X>2) && (X<(Nx-3)) && (Y>2) && (Y<(Ny-3)) && (Z>2) && (Z<(Nz-3)) ){
	// get X,Y,Z from thread and block Id's
		// get linear position
		
		int cp   = Z*AREA+(Y*Nx+X);
         u[cp] =  15.0f/68.0f/180.0f*(
                 270.0f*(uz[cp+1] + uz[cp-1] + uz[cp+Nx]     + uz[cp-Nx]     + uz[cp+AREA]     + uz[cp-AREA])
                 -27.0f*(uz[cp+2] + uz[cp-2] + uz[cp+(2*Nx)] + uz[cp-(2*Nx)] + uz[cp+(2*AREA)] + uz[cp-(2*AREA)])
                  +2.0f*(uz[cp+3] + uz[cp-3] + uz[cp+(3*Nx)] + uz[cp-(3*Nx)] + uz[cp+(3*AREA)] + uz[cp-(3*AREA)])
                -3.0f*490.0f*uz[cp]) +2.0f*uz[cp] - u[cp];
        
       }
	
}


//-----------------------------------------------
// read output and sum in input
__global__ void inout(double *u,double *out,double ins,int n)
{	
	// sum in source
	u[(Sz*AREA)+(Sy*Nx+Sx)] += ins;
	
	// non-interp read out
	out[n]  = u[(Rz*AREA)+(Ry*Nx+Rx)];
	
}

