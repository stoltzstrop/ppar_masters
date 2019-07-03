#include <time.h>
#include "timing.h"
#include "initial.h"
#include "targetDP.h"
#include "timing_macros.h"
#include "Audio.h"
#include "grid_structs.h"

#include <stdio.h>
#define inputfname  "Godin44100.wav"
#define writetoWav 1
#define outputfname "BasicRoom.wav"

/*
 * Main room simulation for targetDP version
 */


// kernel methods
__targetEntry__ void UpDateScheme(double *u,double *u1);
__targetEntry__ void inout(double *u,double *out,double ins,int n);

// boundary conditions  
__targetConst__ coeffs_type cf_d[1];


int main(){

    char* programType = TARGETDP_STR;

// just check what number of threads available for use 
#if defined(_OPENMP)
   printOMPThreads();
   int nthreads = 0;
    char cname[10];
    #pragma omp parallel private(nthreads)
    {
        if(omp_get_thread_num() == 0)
        {
            sprintf(cname,"_C_%d",omp_get_num_threads());
        }
    }
    programType = concatStrings(programType,cname);
#else
    programType = concatStrings(programType,"_CUDA");
#endif
    
        // Simulation parameters (generally untouched from Craig Webb's code) 

        double SR         = (double)numberSamples;             // Sample Rate
        double alpha      = 0.005;               // Boundary loss
        double tx         = 0.10;                // Run time after audio length (sec)    
        double c          = 344.0;               // Speed of sound in air (m/s)
        double k          = 1/SR;                                    
        double h          = sqrt(3.0)*c*k;
        double lambda     = c*k/h;                        

       // setup boundary conditions
        coeffs_type cf_h[1];
            cf_h[0].l2      = (c*c*k*k)/(h*h);
            cf_h[0].loss1   = 1.0/(1.0+lambda*alpha);
	    cf_h[0].loss2   = 1.0-lambda*alpha;
         copyConstToTarget(cf_d,cf_h,sizeof(coeffs_type)) ;
    
        //setup version specific output file name 
        printToTimingFileName(programType);

        // Initialise source input (just a cosine wave)
        int SRw;
        int n;
        int alength = 20;
        double *si_h = (double *)calloc(numberSamples,sizeof(double));
        for(n=0;n<dur;n++){
           si_h[n] = 0.5*(1.0-cos(2.0*pi*n/(double)dur));
        }

        startTime = getTime(); // start the total time clock 
    
	// Set up grid and blocks
	int Gx          = Nx/Bx;
	int Gy          = Ny/By;
	int Gz          = (Nz-2)/Bz;
	
	size_t pr_size  = sizeof(double);
	size_t mem_size = AREA*Nz*pr_size;
	double *out_d, *u_d, *u1_d, *dummy_ptr;                      
	double ins;             
	
        dataCopyInitStart = getTime();

	// Initialise memory on device
	 targetCalloc((void**)&u_d, mem_size);
	 targetCalloc((void**)&u1_d, mem_size);
	 targetCalloc((void**)&out_d, numberSamples*pr_size);

        dataCopyInitEnd = getTime();
        dataCopyInitTotal = dataCopyInitEnd - dataCopyInitStart;
	
	// initialise memory on host
	double *out_h  = (double *)malloc(numberSamples*pr_size);
	double *u_h  = (double *)malloc(AREA*Nz*pr_size);
	if((out_h == NULL)){
		printf("\nout_h memory alloc failed...\n");
		exit(EXIT_FAILURE);
	}

        // initialise kernel run times 
        kernel1Time = 0.0;
        kernel2Time = 0.0;
	dataCopyBtwTotal = 0.0;
        startKernels = getTime();

        // this is the main loop for iterating over the room simulation - number of samples
        // is the total timesteps run for 
	for(n=0;n<numberSamples;n++)
	{
                startKernel1 = getTime();

                //run the room update kernel 
		UpDateScheme __targetLaunch__(VOLUME) (u_d,u1_d);

                // synchronise afterwards 
        	targetSynchronize() ; 
                endKernel1 = getTime();
                kernel1Time += endKernel1-startKernel1;
	        checkTargetError("1st kernel");
	
               
                // the "sound" only plays for a certain amount of time, then propogates through the room
		ins = 0.0;
		if(n<alength)
                {
                  ins = si_h[n];
                }
                startKernel2 = getTime();

                // run source and receiver update kernel
                inout __targetLaunch__(1) (u_d,out_d,ins,n);
	         targetSynchronize() ; 
                endKernel2 = getTime();
                kernel2Time += endKernel2-startKernel2;
	        checkTargetError("2nd kernel");
		
                dataCopyBtwStart = getTime();

		// swap grid pointers
		dummy_ptr = u1_d;
		u1_d = u_d;
		u_d = dummy_ptr;
                dataCopyBtwEnd = getTime();
                dataCopyBtwTotal += dataCopyBtwEnd - dataCopyBtwStart;
	}
	
        endKernels = getTime();
    

        kernelsTime = endKernels-startKernels;
	checkTargetError("Kernel");
	targetSynchronize() ; 
    
        dataCopyBackStart = getTime();     

        // copy result back from device
        copyFromTarget(out_h, out_d, numberSamples*pr_size);
        copyFromTarget(u_h, u_d, AREA*Nz*pr_size);
        dataCopyBackEnd = getTime();     
        dataCopyBackTotal = dataCopyBackEnd - dataCopyBackStart;
        dataCopyTotal = dataCopyInitTotal + dataCopyBtwTotal + dataCopyBackTotal;
    
	
        endTime = getTime();
        totalTime = (double) (endTime-startTime);
        
        // write out final data to file 
        writeBinaryDataToFile(u_h,getOutputFileName(programType,"room","bin"),VOLUME);       
        writeBinaryDataToFile(out_h,getOutputFileName(programType,"receiver","bin"),numberSamples);       

        // sanity check the output values 
        for(int jj=numberSamples-10;jj<numberSamples;jj++)
        {
            printf("%.14lf\n",out_h[jj]);
        }

        printf("%s\n",timingString);
        printf("%s\n",infoString);
        printf("%s\n",getTimingFileName());

        // coordinate timing output 
        printToString;
        printOutputs;
        writeTimingsToFile; 

        // Free memory
        free(si_h);free(out_h);
	targetFree(out_d);targetFree(u_d);targetFree(u1_d);
	
	exit(EXIT_SUCCESS);
}

// Room update Kernel
__targetEntry__ void UpDateScheme(double *u,double *u1)
{
	
        int idx;
     __targetTLP__(idx, VOLUME)
    {

                // data variables - set up in arrays to take advantage of vectorisation 
                // (where applicable - ie. the xeon phi)
                int vi; //vector index
                int revisedIdx[VVL]; // VVL defined in targetDP header file - vector instruction size 
                int X[VVL];
                int Y[VVL];
                int Z[VVL];
                int onBoundary[VVL];
                int K[VVL];
        	double cf[VVL];
		double cf2[VVL];
        	double S[VVL];
                
                // vectorised for the xeon phi
               __targetILP__(vi) revisedIdx[vi] = idx + vi; 
	
        
                // get X,Y,Z from IDX
                __targetILP__(vi) X[vi] = revisedIdx[vi] % Nx;
	        __targetILP__(vi) Y[vi] = (revisedIdx[vi] / Nx) % Ny;
	        __targetILP__(vi) Z[vi] = (revisedIdx[vi] / (Nx*Ny));

                __targetILP__(vi) cf[vi] = 1.0;
                __targetILP__(vi) cf2[vi] = 1.0;

                	
                //check if pt is on boundary 
	        __targetILP__(vi) onBoundary[vi] = (X[vi]>0) && (X[vi]<(Nx-1)) && (Y[vi]>0) && (Y[vi]<(Ny-1)) && (Z[vi]>0) && (Z[vi]<(Nz-1));
	
                // get number of neighbour points 
                __targetILP__(vi) K[vi] = (0||(X[vi]-1)) + (0||(X[vi]-(Nx-2))) + (0||(Y[vi]-1)) + (0||(Y[vi]-(Ny-2))) + (0||(Z[vi]-1)) + (0||(Z[vi]-(Nz-2)));
		

		// set loss coeffs at walls
                __targetILP__(vi)
                {
                    if(K[vi]<6)
                    {
		        cf[vi]   = cf_d[0].loss1;
                        cf2[vi]  = cf_d[0].loss2;
                     }
                }
        
	        // Get sum of neighbours
                __targetILP__(vi){ if(onBoundary[vi]){ S[vi] = u1[revisedIdx[vi]-1]+u1[revisedIdx[vi]+1]+u1[revisedIdx[vi]-Nx]+u1[revisedIdx[vi]+Nx]+u1[revisedIdx[vi]-AREA]+u1[revisedIdx[vi]+AREA];}}
        
                // Calculate grid point  update
                 __targetILP__(vi){ if(onBoundary[vi]){ u[revisedIdx[vi]] = cf[vi]*( (2.0-K[vi]*cf_d[0].l2)*u1[revisedIdx[vi]] + cf_d[0].l2*S[vi] - cf2[vi]*u[revisedIdx[vi]] );} }
    }
}

// sum in input and set receiver data
__targetEntry__ void inout(double *u,double *out,double ins,int n)
{	
	// sum in source
	u[(Sz*AREA)+(Sy*Nx+Sx)] += ins;
	
	// non-interp read out
	out[n]  = u[(Rz*AREA)+(Ry*Nx+Rx)];
	
}

