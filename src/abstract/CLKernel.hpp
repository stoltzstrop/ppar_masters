#include<CL/cl.hpp>
#include<CL/opencl.h>
#include "initial.h"
#include "CLUtils.hpp"

/*
 *  Wrapper class for OpenCL kernel
 */ 

class CLKernel
{

    public:
    
        CLKernel()
        {
        }

        CLKernel(cl::Program program, char* kernelName, int numberRuns) : 
                     kernelEvents(numberRuns), iterations(0) 
        {
           Create(program, kernelName);
        }
        
        ~CLKernel()
        {
        }

        cl::Kernel GetKernel()
        {
            return kernel;
        }
       
        // wrappers for the basic OpenCL funcionality
        template<typename T> int SetParameter(T parameter, size_t size, int argN);
    
        void Create(cl::Program program, char* kernelName);    
    
        int Run(cl::CommandQueue* queue, cl::NDRange global, cl::NDRange local);

        // this simple call masks the hardship one would normally have to go through to find 
        // out how long a kernel took to run 
        double getTimings();

    protected:

    private:
        
        // the OpenCL nugget
        cl::Kernel kernel;
        
        // all kernel timings stored here 
        std::vector<cl::Event> kernelEvents;
        
        // number of arguments stored here 
        int numParameters;

        // track the number of times a kernel gets called (for timing purposes) 
        int iterations;
};


