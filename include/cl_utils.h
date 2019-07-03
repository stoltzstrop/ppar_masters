#ifndef cl_utils_h
#define cl_utils_h

#include<CL/cl.h>
#include <stdio.h>

/*
 * OpenCL helper and debugging function file 
 */


// which type of platform are we running on? 

//#define PROCESSOR_VERSION CL_DEVICE_TYPE_ACCELERATOR 
//#define PROCESSOR_VERSION CL_DEVICE_TYPE_CPU 
#define PROCESSOR_VERSION CL_DEVICE_TYPE_GPU 

// OpenCL is known for its obscure error messages, this case statement helps decipher them
// from StackOverflow: http://stackoverflow.com/questions/9464190/error-code-11-what-are-all-possible-reasons-of-getting-error-cl-build-prog
const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

// check for an error
void err_check( int err ) {
    if ( err != CL_SUCCESS ) {
            printf("Error: %d ( %s )\n",err,getErrorString(err));
    }
}

// print some information about what the limits are for this device
void print_thread_info(cl_device_id device_id)
{
        char buffer[5000];
        size_t bufi;
        size_t workitem_size[3];
        
     clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(bufi), &bufi, NULL);
        printf("MAX_WORKGROUP_SIZE: %zu\n",bufi);
        printf("MAX_WORK_ITEM_SIZES: %d\n",bufi);
         clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
             printf(  " CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%zu / %zu / %zu \n", workitem_size[0], workitem_size[1], workitem_size[2]);
}

// for multiple devices, this is not always 0
int getPlatformID(int platformCount)
{
    return 0;
    //return platformCount - 1;
}

// select the appropriate device (ie. which chip to run OpenCL on)
cl_device_id create_device() {

   cl_platform_id platform;
   cl_platform_id *platforms;
   cl_device_id dev;
   cl_uint platformCount;
   int err;
   char* info;
   size_t infoSize;
   char buffer[10245];


    // information we're going to print out 
   const char* attributeNames[5] = { "Name", "Vendor",
             "Version", "Profile", "Extensions" };
   const cl_platform_info attributeTypes[5] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
             CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };
   const int attributeCount = sizeof(attributeNames) / sizeof(char*);
   
    // get platform
    clGetPlatformIDs(5, NULL, &platformCount);

    int platformID = getPlatformID(platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    // print out some useful information about devices available
    printf("Platforms: %d\n",platformCount);
    for (int i = 0; i < platformCount; i++) 
    {
        printf("n %d. Platform \n", i+1);
        for (int j = 0; j < attributeCount; j++) 
        {
          clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
          info = (char*) malloc(infoSize);
          clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);
          printf("  %d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);
          free(info);
        }
    }
   

   if(err < 0) {
      perror("Couldn't identify a platform \n");
      err_check(err);
      exit(1);
   } 

   // access the device
   err = clGetDeviceIDs(platforms[platformID], PROCESSOR_VERSION, 1, &dev, NULL);
   clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
   printf("driver: %s\n",buffer);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platforms[platformID], PROCESSOR_VERSION, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   free(platforms);
   return dev;
}

#endif

