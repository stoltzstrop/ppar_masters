#include "header.h"
#include "constants.h"
#include "Audio.h" 
#include "timing.h"
#include "timing_macros.h"

/*
 * attempt to combine leggy and advanced versions of room code
 * does not output correct values, but no clear reason why
 */


#define def_s 3.5       // "samples per wavelength"
#define fname "out.bin" // output binary filename

__constant__ int Ni,Nj,Nk,Si,Sj,Sk,Ri,Rj,Rk;
#define area (Ni*Nj)    

__global__ void UpdateScheme(double *grid, const double * __restrict__ grid1, const double * __restrict__ grid2);
__global__ void UpdateSchemeShared(double *u, const double * __restrict__ uz);
__global__ void inout(double *u,double *out,int n, double ins);

typedef struct
{
	double lambda;
	double lvah;
	double beta30;
	double beta31;
	double beta32;
	double beta33;

} coeffs_type;

__constant__ coeffs_type cf_d[1];


int main() {

    char cudaStr[100];
    {
        sprintf(cudaStr,"%s","cuda_leggyadv");
    }

     printToTimingFileName(cudaStr);

    startTime = getTime(); 

   // Parameters to set
    int NF            = numberSamples;
   double c      = 340.0;      //sound speed
   double vmax   = 4000.0;     //max frequency of interest  
   double s      = def_s;      //"samples per wavelength"
   double h      = c/vmax/s;   //grid spacing = .01;
   double alpha = 2e-6;
   double dt     = sqrt(((h*h)/(c*c))*15/68 + (alpha/c)*(alpha/c)) - alpha/c; 
   double beta30_h = -490.0/180.0;
   double beta31_h = 270.0/180.0;
   double beta32_h = -27.0/180.0;
   double beta33_h = 2.0/180.0;

   double lambda_h = (c*dt)/h;
   double lvah_h = ( lambda_h * alpha ) / h;

    coeffs_type cf_h[1];
	cf_h[0].lambda      = lambda_h; 
	cf_h[0].lvah   = lvah_h;
	cf_h[0].beta30 = beta30_h; 
	cf_h[0].beta31 = beta31_h; 
	cf_h[0].beta32 = beta32_h; 
	cf_h[0].beta33 = beta33_h; 
     cudaMemcpyToSymbol(cf_d,cf_h,sizeof(coeffs_type)) ;

   // Calculated parameters
   int N          = (int)(dur/dt);       //number of samples to compute
   double FT       = c*c*dt*dt/h/h/h;     //forcing term

   int Ni_h          = Nx;//ceil(6.09/h)+7; // x-dim + 2*halo +1
   int Nj_h          = Ny;//ceil(6.09/h)+7;  // y-dim + 2*halo +1
   int Nk_h          = Nz;//ceil(4.78/h)+7;  // z-dim + 2*halo +1
   int Si_h          = Sx;//round(2.5/h)+3; // x-source + halo
   int Sj_h          = Sy;//round(2.5/h)+3; // y-source + halo
   int Sk_h          = Sz;//round(2.5/h)+3; // z-source + halo
   int Ri_h          = Rx;//round(17.5/h)+3;// x-receiver + halo
   int Rj_h          = Ry;//round(2.5/h)+3; // y-receiver + halo
   int Rk_h          = Rz;//round(2.5/h)+3; // z-receiver + halo

   // Set up grid and blocks
   int Gx          = (Ni_h-1)/Bx+1;
   int Gy          = (Nj_h-1)/By+1;
   int Gz          = (Nk_h-1)/Bz+1;

   dim3 dimBlockInt(Bx, By, Bz);
   dim3 dimGridInt(Gx, Gy, Gz);

   dim3 dimBlockIO(1, 1, 1);
   dim3 dimGridIO(1, 1, 1);

   size_t pr_size  = sizeof(double);
   size_t mem_size = Ni_h*Nj_h*Nk_h*pr_size;
   double *out_d, *u_d, *uz_d, *uzz_d,*dummy_ptr;  //output, two grids, dummy pointer
   int n;
   double ins;

        dataCopyInitStart = getTime();
   // ------------------------------------------
   // Set constant memory          
   // ------------------------------------------
   cuErr( cudaMemcpyToSymbol(Ni, &Ni_h, sizeof(int)) );
   cuErr( cudaMemcpyToSymbol(Nj, &Nj_h, sizeof(int)) );
   cuErr( cudaMemcpyToSymbol(Nk, &Nk_h, sizeof(int)) );
   cuErr( cudaMemcpyToSymbol(Si, &Si_h, sizeof(int)) );
   cuErr( cudaMemcpyToSymbol(Sj, &Sj_h, sizeof(int)) );
   cuErr( cudaMemcpyToSymbol(Sk, &Sk_h, sizeof(int)) );
   cuErr( cudaMemcpyToSymbol(Ri, &Ri_h, sizeof(int)) );
   cuErr( cudaMemcpyToSymbol(Rj, &Rj_h, sizeof(int)) );
   cuErr( cudaMemcpyToSymbol(Rk, &Rk_h, sizeof(int)) );

        dataCopyInitEnd = getTime();
        dataCopyInitTotal = dataCopyInitEnd - dataCopyInitStart;
   //-------------------------------------------
   // Initialise memory on device
   //-------------------------------------------
   cuErr( cudaMalloc(&u_d, mem_size) );     cuErr( cudaMemset(u_d, 0, mem_size) );
   cuErr( cudaMalloc(&uz_d, mem_size) );    cuErr( cudaMemset(uz_d, 0, mem_size) );
   cuErr( cudaMalloc(&uzz_d, mem_size) );    cuErr( cudaMemset(uzz_d, 0, mem_size) );
   cuErr( cudaMalloc(&out_d, N*pr_size) );  cuErr( cudaMemset(out_d, 0, N*pr_size) );

   cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
   //-------------------------------------------
   // initialise memory on host
   //-------------------------------------------
   double *out_h  = (double *)calloc(N,pr_size);
   if((out_h == NULL)){
      printf("\nout_h memory alloc failed...\n");
      exit(EXIT_FAILURE);
   }

        kernel1Time = 0.0;
        kernel2Time = 0.0;
	dataCopyBtwTotal = 0.0;

        startKernels = getTime();
       for(n=0;n<NF;n++)
       {
            //kernel launch
            startKernel1 = getTime();
            UpdateScheme<<<dimGridInt,dimBlockInt>>>(u_d,uz_d,uzz_d);
    	    cudaThreadSynchronize() ; 
            endKernel1 = getTime();
            kernel1Time += endKernel1-startKernel1;

      //input, output
            ins = 0.0f;
            if (n==0) ins=FT; //delta forcing term
            startKernel2 = getTime();
            inout<<<dimGridIO,dimBlockIO>>>(u_d,out_d,n,ins);
 	    cudaThreadSynchronize() ; 
            endKernel2 = getTime();
            kernel2Time += endKernel2-startKernel2;

      // update pointers (time-stepping)
            dataCopyBtwStart = getTime();
            dummy_ptr = uzz_d;
            uzz_d  = uz_d;
            uz_d  = u_d;
            u_d   = dummy_ptr;
            dataCopyBtwEnd = getTime();
            dataCopyBtwTotal += dataCopyBtwEnd - dataCopyBtwStart;
   }

    endKernels = getTime();
    kernelsTime = endKernels-startKernels;
    checkLastCUDAError("Kernel");
    cuErr( cudaThreadSynchronize() );

    dataCopyBackStart = getTime();     
    // copy result back from device
    cuErr( cudaMemcpy(out_h, out_d, N*pr_size, cudaMemcpyDeviceToHost) );
   
    dataCopyBackEnd = getTime();     
    dataCopyBackTotal = dataCopyBackEnd - dataCopyBackStart;
    dataCopyTotal = dataCopyInitTotal + dataCopyBtwTotal + dataCopyBackTotal;

    endTime = getTime();
    totalTime = (double) (endTime-startTime);

    for(int jj=NF-10;jj<NF;jj++)
    {
        printf("%.14lf\n",out_h[jj]);
    }
    printToString;
    printOutputs;
    writeTimingsToFile; 

   // free memory
   free(out_h);
   cudaFree(out_d);cudaFree(u_d);cudaFree(uz_d);

   exit(EXIT_SUCCESS);
}


// Update Kernels combining advanced and leggy stencils 

__global__ void UpdateScheme(double *grid, const double * __restrict__ grid1, const double * __restrict__ grid2)
// this kernel is provided to illustrate the scheme
// without the use of shared memory (slower implementation)
{
   // get X,Y,Z from thread and block Id's
   int X = blockIdx.x * Bx + threadIdx.x;
   int Y = blockIdx.y * By + threadIdx.y;
   int Z = blockIdx.z * Bz + threadIdx.z;
   int cp;

   double lvah = cf_d[0].lvah;
   double lambda = cf_d[0].lambda;
   double beta30 = cf_d[0].beta30;
   double beta31 = cf_d[0].beta31;
   double beta32 = cf_d[0].beta32;
   double beta33 = cf_d[0].beta33;
   //iterate through z-planes

      //get center-point index
   cp  = Z*area+(Y*Ni+X);

      //interior
   if( (X>2) && (X<(Ni-3)) && (Y>2) && (Y<(Nj-3)) && (Z>2) && (Z<(Nk-3)) ){
         //update
        double N1 = beta31*(grid1[cp+1] + grid1[cp-1] + grid1[cp+Ni]+grid1[cp-Ni]+grid1[cp+area]+grid1[cp-area]) +  beta32*(grid1[cp+2] + grid1[cp-2] + grid1[cp+(2*Ni)] + grid1[cp-(2*Ni)] + grid1[cp+(2*area)] + grid1[cp-(2*area)]) + beta33*(grid1[cp+3] + grid1[cp-3] + grid1[cp+(3*Ni)] + grid1[cp-(3*Ni)] + grid1[cp+(3*area)] + grid1[cp-(3*area)]);


        double N2 = beta31*(grid2[cp+1] + grid2[cp-1] + grid2[cp+Ni]+grid2[cp-Ni]+grid2[cp+area]+grid2[cp-area]) + beta32*(grid2[cp+2] + grid2[cp-2] + grid2[cp+(2*Ni)] + grid2[cp-(2*Ni)] + grid2[cp+(2*area)] + grid2[cp-(2*area)]) +  beta33*(grid2[cp+3] + grid2[cp-3] + grid2[cp+(3*Ni)] + grid2[cp-(3*Ni)] + grid2[cp+(3*area)] + grid2[cp-(3*area)]);

        grid[cp] = (2.0 - (lambda*lambda + lvah)*3*beta30)*grid1[cp]+(3*beta30*lvah-1)*grid2[cp]+(lambda*lambda+lvah)*N1-lvah*N2;

      }
}


//-----------------------------------------------
// read output and sum in input
//-----------------------------------------------
__global__ void inout(double *u,double *out,int n, double ins)
{	
   // sum in source
   u[(Sk*area)+(Sj*Ni+Si)] += ins;
   // Read out
   out[n] = u[(Rk*area)+(Rj*Ni+Ri)];
}
