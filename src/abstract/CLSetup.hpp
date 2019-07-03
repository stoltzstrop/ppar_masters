#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include<CL/cl.hpp>
#include<CL/opencl.h>
#include "CLKernel.hpp"
#include "GridData.hpp"

using namespace std;

/*
 * wrapper function for keeping track of all the OpenCL boilerplate
 */

class CLSetup 
{

public:


    CLSetup(string programName, int device_type, string directives, string str=""); 

    ~CLSetup();

    // helpful debugging / awareness function
    void printDeviceInfo(char* infoString);

    template<typename T> int SetKernelParameter(int kernelIdx, T parameter, size_t size, int argN);

    int AddKernel(char* kernelName, int numberRuns); // store in "kernels" vector with idx value returned
    
    int RunKernel(int kernelIdx, std::vector<int> globalSize, std::vector<int> localSize);

    double GetKernelTiming(int kernelIdx);

    int FinishRun();

    // function to build kernel header file on the fly
    static string ReadInHeaderFile(string fileName, bool cutOffName);

    cl::Program getProgram() { return program;  }
    cl::Context getContext() { return context;  }
    cl::CommandQueue getQueue() { return queue;  }

    /** ERROR HANDLING **/
    static void err_check( int err );

    /** OpenCL kernel header builder **/
    static string buildHeaderFile(GridData* dataObject); 

private:

     // keep track of devices 
     vector<cl::Device> devices;
     cl::Device defaultDevice;

     // keep track of the kernels 
     vector<CLKernel> kernels;
     int numKernels;
    
     // keep track of other boilerplate params 
     // 
     cl::Context context; 
     cl::CommandQueue queue;
     cl::Program program;
     cl_int err;

    //helpful debugging function
    static const char *getErrorString(cl_int error);
};
