#ifndef cl_utils_h
#include "cl_utils.h"
#endif

// let openCL throw exceptions (c++ ftw)
#define __CL_ENABLE_EXCEPTIONS

#include "timing.h"
#include "Audio.h"
#include "timing_macros.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include<CL/cl.hpp>
#include<CL/opencl.h>

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "CLUtils.hpp"
#include "constants.h"
using namespace std;

// some useful string names 
#define STENCIL_KERNEL_FUNC "UpdateScheme"
#define INOUT_KERNEL_FUNC "inout"
#define PROGRAM_NAME "kernels.cl"


// struct for coeffs
typedef struct
{
	double l2;
	double loss1;
	double loss2;

} coeffs_type;

// debug info for OpenCL
void printDeviceInfo(cl::Device device, char* infoString)
{
    string deviceStr;
    string versionStr;
    device.getInfo<string>(CL_DEVICE_NAME,&deviceStr);
    cout<<deviceStr<<"\n";
    device.getInfo<string>(CL_DEVICE_OPENCL_C_VERSION,&versionStr);
    cout<<versionStr<<"\n";
    sprintf(infoString,"%s\n%s\n",deviceStr.c_str(),versionStr.c_str());
}


// timing for openCL must be done through "events"
// these are collected for each kernel run, stored in an array 
// and dealt with later 
double getTimeForAllEvents(int numEvents, vector<cl::Event> events)
{
    double time = 0.0;
    cl_int err;

    cl_ulong start,end;
    cl::Event::waitForEvents(events);
    
    for(int k=0; k<numEvents; k++)
    {
        time += CLUtils::GetEventTiming(events[k]); 
    }

    return time;
}

/*
 * Main room simulation for opencl C++ version
 */

int main(){
	
	
    // Simulation parameters					      
    double SR         = (double)numberSamples;             // Sample Rate
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
    int alength = dur;
    double *si_h = (double *)calloc(numberSamples,sizeof(double));
    for(n=0;n<dur;n++)
    {
      si_h[n] = 0.5*(1.0-cos(2.0*pi*n/(double)dur));
    }
    
    startTime = getTime(); 
    size_t pr_size  = sizeof(double);
    size_t mem_size = VOLUME*pr_size;
    double ins;             

    // setup host memory
    double *out_h  = (double *)malloc(numberSamples*sizeof(double));
    double *u_h  = (double *)malloc(mem_size);
    double *u1_h  = (double *)malloc(mem_size);
    if((out_h == NULL) || (u_h == NULL)  || (u1_h == NULL))
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
    
    // initialise output filename 
    printToTimingFileName(OPENCLCPP_STR);
    
    programBuildStart = getTime();

    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    vector<cl::Kernel> kernels;

    cl_int err;
    cl::Event event;

    // boilerplate for setting up devices, platforms, programs..
    cl::Platform::get(&platforms);
    int platformIdx = CLUtils::getRightPlatform(platforms);
    platforms[platformIdx].getDevices(PROCESSOR_VERSION,&devices);
    cl::Context context2(devices); 
    printConstants();
    printDeviceInfo(devices[0],infoString);
    puts(infoString);
    cl::CommandQueue queue2(context2, devices[0],CL_QUEUE_PROFILING_ENABLE);

    //setup to read in kernel
    ifstream cl_file(PROGRAM_NAME);
    string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
    cl::Program::Sources source(1, make_pair(cl_string.c_str(), 
                            cl_string.length() + 1));

                            
    cl::Program program2(context2,source);
    char options[100];
    sprintf(options,"-DNx=%d -DNy=%d -DNz=%d",Nx,Ny,Nz); // pass in grid sizes to opencl kernel

    if(program2.build(devices, options) != CL_SUCCESS)
    {
          std::cout<<"Error building program"<<endl;
    }

    programBuildEnd = getTime();
    programBuildTotal = programBuildEnd-programBuildStart;
        
    cl::Kernel stencil_kernel2(program2,STENCIL_KERNEL_FUNC);
    cl::Kernel inout_kernel2(program2,INOUT_KERNEL_FUNC);

    // must use opencl data types for arrays 
    cl::Buffer u_d2 = cl::Buffer(context2, CL_MEM_READ_WRITE,mem_size,NULL,&err);
    err_check(err);
    cl::Buffer u1_d2 = cl::Buffer(context2, CL_MEM_READ_WRITE,mem_size,NULL,&err);
    err_check(err);
    err_check(err);
    cl::Buffer out_d2 = cl::Buffer(context2, CL_MEM_READ_WRITE,numberSamples*sizeof(double),NULL,&err);
    err_check(err);

    cl::Buffer *u_ptr = &u_d2;
    cl::Buffer *u1_ptr = &u1_d2;
    cl::Buffer *dummy;

    cl::Buffer cf_d2 = cl::Buffer(context2, CL_MEM_READ_ONLY,sizeof(coeffs_type),NULL,&err);
    err_check(err);
    cl::Buffer ins_d = cl::Buffer(context2, CL_MEM_READ_ONLY,sizeof(double),NULL,&err);
    err_check(err);
    cl::Buffer n_d = cl::Buffer(context2, CL_MEM_READ_ONLY,sizeof(int),NULL,&err);
    err_check(err);
     
    cl::Event uWriteEvent, u1WriteEvent, dummyWriteEvent, outWriteEvent, cfWriteEvent;
    err = queue2.enqueueWriteBuffer(u_d2,CL_TRUE,0,mem_size,u_h, NULL, &uWriteEvent);
    err_check(err);
    err = queue2.enqueueWriteBuffer(u1_d2,CL_TRUE,0,mem_size,u1_h, NULL, &u1WriteEvent);
    err_check(err);
    err = queue2.enqueueWriteBuffer(out_d2,CL_TRUE,0,numberSamples*sizeof(double),out_h, NULL, &outWriteEvent);
    err_check(err);
    err = queue2.enqueueWriteBuffer(cf_d2,CL_TRUE,0,sizeof(coeffs_type),cf_h, NULL, &cfWriteEvent);
    err_check(err);


    // set OpenCL kernel arguments 
    err = stencil_kernel2.setArg(0,sizeof(cl::Buffer),&u_d2);
    err_check(err);
    err = stencil_kernel2.setArg(1,sizeof(cl::Buffer),&u1_d2);
    err_check(err);
    err = stencil_kernel2.setArg(2,sizeof(cl::Buffer),&cf_d2);
    err_check(err);

    err = inout_kernel2.setArg(0,sizeof(cl::Buffer),&u_d2);
    err_check(err);
    err = inout_kernel2.setArg(1,sizeof(cl::Buffer),&out_d2);
    err_check(err);
    err = inout_kernel2.setArg(2,sizeof(double),&ins);
    err_check(err);
    err = inout_kernel2.setArg(3,sizeof(int),&n);
    err_check(err);

	
    // these vectors keep track of timings of the kernels using opencl event objects 
    std::vector<cl::Event> kernel1Events(numberSamples);
    std::vector<cl::Event> kernel2Events(numberSamples);
        
    startKernels = getTime();
    kernel1Time = 0.0;
    kernel2Time = 0.0;
    dataCopyBtwTotal = 0.0;
    int jj;

    // loop over the number of timesteps
    for(n=0;n<numberSamples;n++)
    {
        //run the room update kernel 
        err = queue2.enqueueNDRangeKernel(stencil_kernel2, cl::NullRange, cl::NDRange(Nx,Ny,Nz), 
                                                        cl::NDRange(Bx,By,Bz), NULL, &kernel1Events[n]);
        err_check(err);
        
        ins = 0.0;
	if(n<alength)
        {
            ins = si_h[n];
        }
            
        err = inout_kernel2.setArg(2,sizeof(double),&ins);
        err_check(err);
        err = inout_kernel2.setArg(3,sizeof(int),&n);
        err_check(err);
           
            
        // run source and receiver update kernel
        err = queue2.enqueueNDRangeKernel(inout_kernel2, cl::NullRange, cl::NDRange(1,1,1), 
                                                        cl::NDRange(1,1,1), NULL, &kernel2Events[n]);
        err_check(err);
		
        // update pointers
        dummy = u1_ptr;
        u1_ptr = u_ptr;
        u_ptr = dummy;

        // reset the arguments (may not be necessary..)
        err = stencil_kernel2.setArg(0,sizeof(cl::Buffer),u_ptr);
        err_check(err);
        err = stencil_kernel2.setArg(1,sizeof(cl::Buffer),u1_ptr);
        err_check(err);
        err = inout_kernel2.setArg(0,sizeof(cl::Buffer),u_ptr);
        err_check(err);


    }

    // Finish up OpenCL    
    err = queue2.finish();
    endKernels = getTime();
    kernelsTime = endKernels-startKernels;
    
    
    // Read the results from the device
    cl::Event uReadEvent, outReadEvent;
    err = queue2.enqueueReadBuffer(out_d2, CL_TRUE, 0,
                               numberSamples*sizeof(double), out_h, NULL, &outReadEvent );
    err = queue2.enqueueReadBuffer(u_d2, CL_TRUE, 0,
                               VOLUME*sizeof(double), u_h, NULL, &uReadEvent );
    err_check(err);
    
    
    endTime = getTime();
    totalTime = (double) (endTime-startTime);


    // write out final data to file 
    writeBinaryDataToFile(u_h,getOutputFileName(OPENCLCPP_STR,"room","bin"),VOLUME);       
    writeBinaryDataToFile(out_h,getOutputFileName(OPENCLCPP_STR,"receiver","bin"),numberSamples);       
    for(jj=numberSamples-10;jj<numberSamples;jj++)
    {
            printf("%.14lf\n",out_h[jj]);
    }

    //collect all those kernel timings 
    kernel1Time = getTimeForAllEvents(numberSamples, kernel1Events);
                      kernel2Time = getTimeForAllEvents(numberSamples, kernel2Events);
                      dataCopyTotal = dataCopyInitTotal + dataCopyBtwTotal + dataCopyBackTotal;
    // coordinate timing output 
    printToString;
    printOutputs;
    writeTimingsToFile; 

    
    // Free memory
    free(si_h);free(out_h);free(u_h);free(u1_h);
	
    exit(EXIT_SUCCESS);
}

