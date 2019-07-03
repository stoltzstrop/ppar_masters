#include "Audio.h" 
#include "timing.h"
#include "timing_macros.h"
#include "constants.h"
#include <stdio.h>
#include <string.h>
#include "CJW_Cuda.h"
#define inputfname  "Godin44100.wav"
#define writetoWav 1
#define outputfname "BasicRoom.wav"

enum BOOLEAN useOptimisation = FALSE;

// kernel methods
__global__ void UpDateScheme(double *u,double *u1, double* u2, double l2, double lvah);
__global__ void inout(double *u,double *out,double ins,int n);

// struct for coeffs
typedef struct
{
	double l2;
	double loss1;
	double loss2;
        double q;

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
    double valpha     = 0.0000018;           // Viscosity   
    double c          = 344.0;               // Speed of sound in air (m/s)
    double k          = 1/SR;                                    
    double h          = sqrt((3.0*c*c*k*k)+(6.0*c*valpha*k));
    double lambda     = c*k/h;                        
    double l2      = lambda * lambda;//(c*c*k*k)/(h*h);
    double va         = 2e-6;           // Viscosity   
    double vah        = va/h;
    double lvah       = lambda * vah;

    // Set constant memory coeffs
    coeffs_type cf_h[1];
	cf_h[0].l2      = (c*c*k*k)/(h*h);
	cf_h[0].loss1   = 1.0/(1.0+lambda*alpha);
	cf_h[0].loss2   = 1.0-lambda*alpha;
	cf_h[0].q       = c*valpha*k/(h*h);
     cudaMemcpyToSymbol(cf_d,cf_h,sizeof(coeffs_type)) ;


    char cudaStr[100];
    {
        sprintf(cudaStr,"%s","cuda_advanced");
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
	int Gz          = (Nz-2)/Bz;
	
	dim3 dimBlockInt(Bx, By, Bz);
	dim3 dimGridInt(Gx, Gy, Gz);
	dim3 dimBlockIO(1, 1, 1);
	dim3 dimGridIO(1, 1, 1);

	size_t pr_size  = sizeof(double);
	size_t mem_size = AREA*Nz*pr_size;
	double *out_d, *u_d, *u1_d, *u2_d, *dummy_ptr;                      
	double ins;             
    
        
        dataCopyInitStart = getTime();

	// Initialise memory on device
	 cudaMalloc(&u_d, mem_size) ;       cudaMemset(u_d, 0, mem_size) ;
	 cudaMalloc(&u1_d, mem_size) ;      cudaMemset(u1_d, 0, mem_size) ;
	 cudaMalloc(&u2_d, mem_size) ;      cudaMemset(u2_d, 0, mem_size) ;
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
    		UpDateScheme<<<dimGridInt,dimBlockInt>>>(u_d,u1_d,u2_d,l2,lvah);
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
		dummy_ptr = u2_d;
		u2_d = u1_d;
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

//-----------------------------------------------
// Update Kernel
__global__ void UpDateScheme(double *u,double *u1, double *u2, double l2, double lvah)
{
	
	// get X,Y,Z from thread and block Id's
	int X = blockIdx.x * Bx + threadIdx.x;                                              
	int Y = blockIdx.y * By + threadIdx.y;
	int Z = blockIdx.z * Bz + threadIdx.z + 1;
	
	// Test that not at halo, Z block excludes Z halo
	if( (X>0) && (X<(Nx-1)) && (Y>0) && (Y<(Ny-1)) ){
		// get linear position
		
		int cp   = Z*AREA+(Y*Nx+X);
		// local variables
		double ls1 = 1.0;
		double ls2 = 1.0;
		double lm2 = cf_d[0].l2;
		
		int K    = (0||(X-1)) + (0||(X-(Nx-2))) + (0||(Y-1)) + (0||(Y-(Ny-2))) + (0||(Z-1)) + (0||(Z-(Nz-2)));
		
        double N1  = u1[cp-1]+u1[cp+1]+u1[cp-Nx]+u1[cp+Nx]+u1[cp-AREA]+u1[cp+AREA];
        double N2  = u2[cp-1]+u2[cp+1]+u2[cp-Nx]+u2[cp+Nx]+u2[cp-AREA]+u2[cp+AREA];
        double C1  = u1[cp];
        double C2  = u2[cp];
        
        double qv  = cf_d[0].q*( (N1-6.0*C1)-(N2-6.0*C2) );
        
        // Calc update
        u[cp] = (2.0-K*l2-K*lvah)*u1[cp] +(K*lvah -1.0)*u2[cp]+(l2+lvah)*N1-lvah*N2;
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

