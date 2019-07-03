#include<CL/cl.h>
#include "Audio.h"
#include "timing_macros.h"
#include "timing.h"
#include "grid_structs.h"
#include <sys/time.h>
#ifndef cl_utils_h
#include "cl_utils.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

//some useful strings 
#define STENCIL_PROGRAM_NAME "stencil_kernel.cl"
#define STENCIL_KERNEL_FUNC "UpdateScheme"

#define INOUT_PROGRAM_NAME "inout_kernel.cl"
#define INOUT_KERNEL_FUNC "inout"

// timing for openCL must be done through "events"
// these are collected for each kernel run, stored in an array 
// and dealt with later
double getTimeForAllEvents(int numEvents, cl_event* events)
{
    double time = 0.0;
    cl_int err;

    err = clWaitForEvents(numEvents,events);
    
    cl_ulong start,end;
    
    for(int k=0; k<numEvents; k++)
    {
        err = clGetEventProfilingInfo(events[k],CL_PROFILING_COMMAND_END, 
                    sizeof(cl_ulong), &end, NULL);
        err = clGetEventProfilingInfo(events[k],CL_PROFILING_COMMAND_START, 
                    sizeof(cl_ulong), &start, NULL);
        time += (nano * ((double)end-(double)start));
    }

    return time;
}

/*
 * Main room simulation for opencl version
 */

int main(){
	
    // Simulation parameters					      
    double SR         = (double) numberSamples;             // Sample Rate
    double alpha      = 0.005;               // Boundary loss
    double c          = 344.0;               // Speed of sound in air (m/s)
    double k          = 1/SR;                                    
    double h          = sqrt(3.0)*c*k;
    double lambda     = c*k/h;                        
   
    // Setup boundary conditions
    coeffs_type cf_h[1];
    cf_h[0].l2      = (c*c*k*k)/(h*h);
    cf_h[0].loss1   = 1.0/(1.0+lambda*alpha);
    cf_h[0].loss2   = 1.0-lambda*alpha;
    
    // Initialise sound input
    int n;
    int duration = 20;
    int alength = duration;
    double *si_h = (double *)calloc(numberSamples,sizeof(double));
    for(n=0;n<duration;n++)
    {
      si_h[n] = 0.5*(1.0-cos(2.0*pi*n/(double)duration));
    }
    
    // start the clock
    startTime = getTime(); 

	// Set up grid and blocks
	int Gx          = Nx/Bx;
	int Gy          = Ny/By;
	int Gz          = Nz/Bz;
	
	size_t pr_size  = sizeof(double);
	size_t mem_size = AREA*Nz*pr_size;
        
        cl_mem out_d, u_d, u1_d;    
        cl_mem cf_d, dummy_ptr;
        double ins;             
	
        // Setup OpenCL stuff

        cl_platform_id cpPlatform;        // OpenCL platform
        cl_device_id device_id;           // device ID
        cl_context context;               // context
        cl_command_queue queue;           // command queue
        cl_program program;               // program
        cl_kernel stencil_kernel;                 // kernel1
        cl_kernel inout_kernel;                 // kernel2

        cl_int err;
	double *out_h  = (double *)malloc(numberSamples*sizeof(double));
	double *u_h  = (double *)malloc(VOLUME*sizeof(double));
	double *u1_h  = (double *)malloc(VOLUME*sizeof(double));
	if((out_h == NULL) || (u_h == NULL)  || (u1_h == NULL) )
        {
		printf("\host memory alloc failed...\n");
		exit(EXIT_FAILURE);
	}

        // initialize host memory (will be written on GPU memory as well)
        int xi;
        for(xi=0;xi<VOLUME;xi++)
        {
            u_h[xi] = 0.0;
            u1_h[xi] = 0.0;
            if(xi<numberSamples)
            {
              out_h[xi] = 0.0;
            }
        }
    
        printToTimingFileName(OPENCL_STR);
       

        // Number of work items in each local work group
        size_t localSize[3] = { Bx, By, Bz };
        size_t localSizeIO[3] = { 1, 1, 1 };
    
        // Number of total work items - localSize must be divisor
        size_t globalSize[3] = { Nx, Ny, Nz };
        size_t globalSizeIO[3] = { 1, 1, 1 };
     
        programBuildStart = getTime();

        device_id = create_device(infoString);
        
        // Create a context  
        context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
        err_check(err);
     
        // Create a command queue 
        queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
        printConstants();
	puts(infoString);

       // Create the compute program from the source buffer
        FILE* stencil_handle = fopen(STENCIL_PROGRAM_NAME,"r");
        char* program_buffer[2];
    
        fseek(stencil_handle,0,SEEK_END);
        size_t program_size[2];
        program_size[0] = ftell(stencil_handle);
        rewind(stencil_handle);
    
        program_buffer[0] = (char*)malloc(program_size[0]+1);
        fread(program_buffer[0], sizeof(char), program_size[0], stencil_handle);
        fclose(stencil_handle);
    
       
        FILE* inout_handle = fopen(INOUT_PROGRAM_NAME,"r");
    
        fseek(inout_handle,0,SEEK_END);
        program_size[1] = ftell(inout_handle);
        rewind(inout_handle);
    
        program_buffer[1] = (char*)malloc(program_size[1]+1);
        fread(program_buffer[1], sizeof(char), program_size[1], inout_handle);
        fclose(inout_handle);
   

        program = clCreateProgramWithSource(context, 2,
                             &program_buffer, (size_t*)&program_size, &err);
        err_check(err);
    
        free(program_buffer[0]);
        free(program_buffer[1]);

        char options[100];
        sprintf(options,"-DNx=%d -DNy=%d -DNz=%d",Nx,Ny,Nz);

        err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
        
        if(err != CL_SUCCESS)
        {
          size_t length;
          char buffer[2048];
          clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
          printf("--- Build log --- %s\n ",buffer);
          exit(1);
        }
       
        // create kernel objects 
        stencil_kernel = clCreateKernel(program, STENCIL_KERNEL_FUNC, &err);
        err_check(err);
        inout_kernel = clCreateKernel(program, INOUT_KERNEL_FUNC, &err);
        err_check(err);
        programBuildEnd = getTime();
        programBuildTotal = programBuildEnd-programBuildStart;
        

        dataCopyInitStart = getTime();
        
	// Initialise memory on device
        u_d = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, NULL);
        u1_d = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, NULL);
        out_d = clCreateBuffer(context, CL_MEM_READ_WRITE, numberSamples*sizeof(double), NULL, NULL);
        cf_d = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(coeffs_type), NULL, NULL);

        cl_mem *u_ptr = &u_d;
        cl_mem *u1_ptr = &u1_d;
        cl_mem *dummy;


        err |= clEnqueueWriteBuffer(queue, u_d, CL_TRUE, 0,
                                  mem_size, u_h, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, u1_d, CL_TRUE, 0,
                                  mem_size, u1_h, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, out_d, CL_TRUE, 0,
                                  numberSamples*sizeof(double), out_h, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, cf_d, CL_TRUE, 0,
                                   sizeof(coeffs_type), cf_h, 0, NULL, NULL);

        // set OpenCL kernel arguments 
        err  = clSetKernelArg(stencil_kernel, 0, sizeof(cl_mem), &u_d);
        err_check(err);
        err |= clSetKernelArg(stencil_kernel, 1, sizeof(cl_mem), &u1_d);
        err_check(err);
        err |= clSetKernelArg(stencil_kernel, 2, sizeof(cl_mem), &cf_d);
        err_check(err);
 
        err |= clSetKernelArg(inout_kernel, 0, sizeof(cl_mem), &u_d);
        err_check(err);
        err |= clSetKernelArg(inout_kernel, 1, sizeof(cl_mem), &out_d);
        err_check(err);
        err |= clSetKernelArg(inout_kernel, 2, sizeof(double), &ins);
        err_check(err);
        err |= clSetKernelArg(inout_kernel, 3, sizeof(int), &n);
        err_check(err);
        dataCopyInitEnd = getTime();
        dataCopyInitTotal = dataCopyInitEnd - dataCopyInitStart;

        int jj;
        startKernels = getTime();
        kernel1Time = 0.0;
        kernel2Time = 0.0;
	dataCopyBtwTotal = 0.0;
	
        cl_event kernel1Events[numberSamples];
        cl_event kernel2Events[numberSamples];
	// loop over the number of timesteps
        for(n=0;n<numberSamples;n++)
	{
            // cal main room update kernel
            err = clEnqueueNDRangeKernel(queue, stencil_kernel, 3, NULL, &globalSize, &localSize,
                                                              0, NULL, &kernel1Events[n]);
            err_check(err);
            ins = 0.0;
	    if(n<alength)
            {
              ins = si_h[n];
            }
            
            err |= clSetKernelArg(inout_kernel, 2, sizeof(double), &ins);
            err |= clSetKernelArg(inout_kernel, 3, sizeof(int), &n);
            
            // call secondary  kernel for updating grid and receiver 
            err = clEnqueueNDRangeKernel(queue, inout_kernel, 3, NULL, &globalSizeIO, &localSizeIO,
                                                              0, NULL, &kernel2Events[n]);
                                                                                
            dummy = u1_ptr;
            u1_ptr = u_ptr;
            u_ptr = dummy;


            err  = clSetKernelArg(stencil_kernel, 0, sizeof(cl_mem), u_ptr);
             err |= clSetKernelArg(stencil_kernel, 1, sizeof(cl_mem), u1_ptr);
            err |= clSetKernelArg(inout_kernel, 0, sizeof(cl_mem), u_ptr);
	
	}

    
    // Finish up OpenCL    
    err = clFinish(queue);
    endKernels = getTime();
    kernelsTime = endKernels-startKernels;
    
    dataCopyBackStart = getTime();     

    // Read the results from the device
    clEnqueueReadBuffer(queue, out_d, CL_TRUE, 0,
                               numberSamples*sizeof(double), out_h, 0, NULL, NULL );
    clEnqueueReadBuffer(queue, u_d, CL_TRUE, 0,
                              VOLUME*sizeof(double), u_h, 0, NULL, NULL );
    dataCopyBackEnd = getTime();     
    dataCopyBackTotal = dataCopyBackEnd - dataCopyBackStart;


    endTime = getTime();
    

    totalTime = (double) (endTime-startTime);
    
    writeBinaryDataToFile(u_h,getOutputFileName(OPENCL_STR,"room","bin"),VOLUME);       
    writeBinaryDataToFile(out_h,getOutputFileName(OPENCL_STR,"receiver","bin"),numberSamples);       

    // sanity check results
    for(jj=numberSamples-10;jj<numberSamples;jj++)
    {
        printf("%.14lf\n",out_h[jj]);
    }
    
    kernel1Time = getTimeForAllEvents(numberSamples,kernel1Events);
                      kernel2Time = getTimeForAllEvents(numberSamples,kernel2Events);
                      dataCopyBtwTotal = 0.0; 
                      dataCopyTotal = dataCopyInitTotal + dataCopyBtwTotal + dataCopyBackTotal;


    printToString;
    printOutputs;
    writeTimingsToFile; 

    // Free memory
    clReleaseMemObject(out_d);
    clReleaseMemObject(u_d);
    clReleaseMemObject(u1_d);

    clReleaseProgram(program);
    clReleaseKernel(stencil_kernel);
    clReleaseKernel(inout_kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    

    free(si_h);free(out_h);free(u_h);free(u1_h);
	
    exit(EXIT_SUCCESS);
}

