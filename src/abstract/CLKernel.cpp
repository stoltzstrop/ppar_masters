#include "CLKernel.hpp"

// wrapper implementations of OpenCL functionality 
void CLKernel::Create(cl::Program program, char* kernelName)
{ 
    kernel =  cl::Kernel(program,kernelName);
} // store in "kernels" with idx value returned

template<typename T> int CLKernel::SetParameter(T parameter, size_t size, int argN)
{
    kernel.setArg(argN, size, &parameter);
}

int CLKernel::Run(cl::CommandQueue* queue, cl::NDRange global, cl::NDRange local)
{
    return queue->enqueueNDRangeKernel(kernel, cl::NullRange, global,local, NULL, &kernelEvents[iterations++]);
}

// get how long a kernel took to run
double CLKernel::getTimings()
{

    cl_int err;
    double time = 0.0;

    cl::Event::waitForEvents(kernelEvents);

    for(int k=0; k<iterations; k++)
    {
          time += CLUtils::GetEventTiming(kernelEvents[k]);
    }

    return time;
}

// calls required by C++ when setting up specific templates 
template int CLKernel::SetParameter<cl::Buffer>(cl::Buffer, size_t,  int);
template int CLKernel::SetParameter<double>(double, size_t, int);
template int CLKernel::SetParameter<float>(float, size_t, int);
template int CLKernel::SetParameter<int>(int, size_t, int);
