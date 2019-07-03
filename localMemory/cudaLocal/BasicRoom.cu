/*
-------------------------------------------------
File      : BasicRoom.cu
Author    : Craig J. Webb
Date      : 18/03/12
Desc      : 3D wave eq test
-------------------------------------------------
*/

// Set precision
#include "blocks.h"
#include "timing.h"
#include "cuda_profiler_api.h"

#define area (Nx*Ny)

// Define Source and Read
#define Sx 120
#define Sy 120
#define Sz 60

#define Nx 512
#define Ny 512
#define Nz 404 

#define Rx 50
#define Ry 50
#define Rz 50
#define numberSamples 4410
#define dim         3

#define nano        1.e-09

#define pi 3.1415926535897932384626433832795

// kernel methods
__global__ void UpdateRoom(double *u,  const double* __restrict__  u1);

__global__ void inout(double *u,double *out,double ins,int n);

typedef struct
{
	double l2;
	double loss1;
	double loss2;

} coeffs_type;
__constant__ coeffs_type cf_d[1];
// ----------------------------------------------


// some of Craig's definitions
#define cuErr(err) __checkCudaErrors (err, __FILE__, __LINE__)

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


void printLastSamples(double *audio, int NF, int N) {
	
    int n;
    double maxy;
    // Test that N > 0
    if(N<1){
        printf("Must display 1 or more samples...\n");
    }
    else{
        //print last N samples
        printf("\n");
        for(n=NF-N;n<NF;n++)
        {
            printf("Sample %d : %.20f\n",n,audio[n]);
        }
        
        // find max
        maxy = 0.0;
        for(n=0;n<NF;n++)
        {
            if(fabs(audio[n])>maxy) maxy = fabs(audio[n]);
            
        }
        printf("\nMax sample : %.20f\n",maxy);
    }
}

int main(){
	
    // Simulation parameters					      
    double SR         = 1000.0;
    int NF          = 1000;
    double c          = 344.0;
    double k          = 1/SR;
    double h          = sqrt(3.0)*c*k;
	double l2         = 1.0/3.0;
    double alpha      = 0.005;               // Boundary loss
    double lambda     = c*k/h;                        
    
    coeffs_type cf_h[1];
	cf_h[0].l2      = (c*c*k*k)/(h*h);
	cf_h[0].loss1   = 1.0/(1.0+lambda*alpha);
	cf_h[0].loss2   = 1.0-lambda*alpha;
     cudaMemcpyToSymbol(cf_d,cf_h,sizeof(coeffs_type)) ;
    //-------------------------------------------
    // Initialise input
    int n;
    size_t pr_size  = sizeof(double);
	int dur         = 20;
    double *si_h      = (double *)calloc(NF,pr_size);
    for(n=0;n<dur;n++){
        si_h[n] = 0.5*(1.0-cos(2.0*pi*n/(double)dur));
    }
    
	// ------------------------------------------
	// Set up grid and blocks
	int Gx          = Nx/BLOCK_X;
	int Gy          = Ny/BLOCK_Y;
	int Gz          = (Nz-2)/BLOCK_Z;
	int GxL         = Nx/BLOCK_X;
	int GyL         = Ny/BLOCK_Y;
	int GzS         = Nz-2;
	
	dim3 dimBlockInt(BLOCK_X, BLOCK_Y, BLOCK_Z);
	dim3 dimGridInt(Gx, Gy, Gz);
	
	dim3 dimBlockIntL(BLOCK_X, BLOCK_Y, 1);
	dim3 dimGridIntL(GxL, GyL, 1);
	
	dim3 dimBlockIO(1, 1, 1);
	dim3 dimGridIO(1, 1, 1);

	size_t mem_size = area*Nz*pr_size;
	double *out_d, *u_d, *u1_d, *dummy_ptr;                      
	double ins;             
	
	//-------------------------------------------
	// Initialise memory on device
	cuErr( cudaMalloc(&u_d, mem_size) );      cuErr( cudaMemset(u_d, 0, mem_size) );
	cuErr( cudaMalloc(&u1_d, mem_size) );     cuErr( cudaMemset(u1_d, 0, mem_size) );
	cuErr( cudaMalloc(&out_d, NF*pr_size) );  cuErr( cudaMemset(out_d, 0, NF*pr_size) );
	
	//-------------------------------------------
	// initialise memory on host
	double *out_h  = (double *)calloc(NF,pr_size);
	if((out_h == NULL)){
		printf("\nout_h memory alloc failed...\n");
		exit(EXIT_FAILURE);
	}
	

        // using only very basic timings here 
	double start = getTime();
	
	for(n=0;n<NF;n++)
	{
		
		UpdateRoom<<<dimGridIntL,dimBlockIntL>>>(u_d,u1_d);
		
		// perform read in out
		ins = si_h[n];
		inout<<<dimGridIO,dimBlockIO>>>(u_d,out_d,ins,n);
		
		// update pointers
		dummy_ptr = u1_d;
		u1_d = u_d;
		u_d = dummy_ptr;

	}
        double end = getTime();
	
	// print process time
	checkLastCUDAError("Kernel");
	cuErr( cudaDeviceSynchronize() ); 
	
        double totalTime = end - start;
    
    // copy result back from device
	cuErr( cudaMemcpy(out_h, out_d, NF*pr_size, cudaMemcpyDeviceToHost) );
    
    // print last samples, and write output file
    printLastSamples(out_h, NF, 5);
	

    double bandwidth = (area*Nz*sizeof(double)*1e-9*2)/(totalTime/NF); 
    printf("\nProcess time : %4.4lf seconds\nBandwidth: %4.4lf", (end-start), bandwidth );

    // Free memory
    free(si_h);free(out_h);
	cudaFree(out_d);cudaFree(u_d);cudaFree(u1_d);
	
	printf("\nPut down that cocktail... Simulation complete.\n\n");
	exit(EXIT_SUCCESS);
}

// Standard 3D update scheme
__global__ void UpdateRoom(double *u, const double* __restrict__ u1)
{
	
	__shared__ double uS1[BLOCK_X+2][BLOCK_Y+2];
	
	// get thread indices
	int tdx = threadIdx.x;
	int tdy = threadIdx.y;
	// get X,Y,Z from thread and block Id's
	int X = blockIdx.x * BLOCK_X + tdx;                                              
	int Y = blockIdx.y * BLOCK_Y + tdy;
	
	int Z,cp;
	
	// Set Z=0, Get Z=1 cp value
	double u1cpm = 0.0;
	double u1cp  = u1[area+(Y*Nx+X)];
	double u1cpp;
	
	tdx++;
	tdy++;

	for(Z=1;Z<(Nz-1);Z++){

	// Test that not at halo, Z block excludes Z halo
		// get linear position
		
		cp   = Z*area+(Y*Nx+X);
		u1cpp  = u1[cp+area];
		uS1[tdx][tdy] = u1cp;
		
		if ( (tdy==1) && !(Y==0) ){
		uS1[tdx][tdy-1] = u1[cp-Nx];
		}
		if ( (tdy==BLOCK_Y) && !(Y==(Ny-1)) ){
			uS1[tdx][tdy+1] = u1[cp+Nx];
		}
		if ( (tdx==1) && !(X==0) ){
			uS1[tdx-1][tdy] = u1[cp-1];
		}
		if ( (tdx==BLOCK_X) && !(X==(Nx-1)) ){
			uS1[tdx+1][tdy] = u1[cp+1];
		}
		
		__syncthreads();
		// local variables
		double cf  = 1.0;
		double cf2 = 1.0;
		
        	if( (X>0) && (X<(Nx-1)) && (Y>0) && (Y<(Ny-1)) ){
		    int K    = (0||(X-1)) + (0||(X-(Nx-2))) + (0||(Y-1)) + (0||(Y-(Ny-2))) + (0||(Z-1)) + (0||(Z-(Nz-2)));
		
		// set loss coeffs at walls
    		    if(K<6){
    		    	    cf   = cf_d[0].loss1;
    			    cf2  = cf_d[0].loss2;
		    }
        
		        // Get sum of neighbours
                        double S   = uS1[tdx-1][tdy]+uS1[tdx+1][tdy]+uS1[tdx][tdy-1]+uS1[tdx][tdy+1]+u1cpm+u1cpp;
        
                        // Calc update
                        u[cp]    = cf*( (2.0-K*cf_d[0].l2)*u1cp + cf_d[0].l2*S - cf2*u[cp] );
			
			// Shift cps
			u1cpm = u1cp;
			u1cp  = u1cpp;
			__syncthreads();
	        }
        }
	
}

// read output and sum in input
__global__ void inout(double *u,double *out,double ins,int n)
{	
	// sum in source
	u[(Sz*area)+(Sy*Nx+Sx)] += ins;
	
	// non-interp read out
	out[n]  = u[(Rz*area)+(Ry*Nx+Rx)];
	
}


