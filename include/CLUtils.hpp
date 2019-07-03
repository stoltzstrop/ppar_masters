#ifndef CLUTILS_h
#define CLUTILS_h

#include<CL/cl.hpp>
#include<CL/opencl.h>

// utility helper functions for opencl C++ versions 

using namespace std;

class CLUtils{

public:

// in opencl timing calls get stored in 'events' - these must be stored up and pulled out at the end 
static double GetEventTiming(cl::Event event)
{
    cl_int err;
    cl_ulong start,end;
    end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>(&err);
    start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>(&err); 
    return (nano * (end-start)); 
}

// return which device to use, some architectures have more than one
// on fuji: nvidia = 0 & amd = 1
static int getRightPlatform(std::vector<cl::Platform> platforms)
{
  //  return platforms.size()-1;
    return 0;
}

};
#endif
