#define __CL_ENABLE_EXCEPTIONS
#include<CL/cl.hpp>
#include<CL/opencl.h>
#include "DeviceData.hpp"
#include "CLSetup.hpp"

/*
 *  This class is built as a wrapper class for an OpenCL data object
 */


// DeviceType -- type of OpenCL object (ie. cl::Buffer)
// CreateType -- data type (ie. double) 
template<class DeviceType, class CreateType>
class CLDeviceData : virtual public DeviceData<DeviceType, CreateType>
{
    using DeviceData<DeviceType,CreateType>::data;

    public:

        // initialise data object
        // requires pointer to CLSetup for other things in main code
        // memory type is ie. read-only
        // size is the size of the dataIn being passed to the wrapper 
       CLDeviceData(CreateType dataIn, CLSetup* setup, int memoryType, size_t size)
       {
            cl_mem_flags memFlag = (cl_mem_flags) memoryType; 
            clSetup = setup;
            SetData(memFlag, size);
            ReadFromHost(dataIn, size);
       }

       ~CLDeviceData()
       {
       
       }

        int SetData(size_t size)
        {
            return SetData(CL_MEM_READ_WRITE,size);
        }

        int SetData(cl_mem_flags memory_type, size_t size)
        {
            cl_int err;
            this->data = cl::Buffer(clSetup->getContext(), memory_type, size, NULL, &err);
        }

        // interface to read/write to and copy data object without directly accessing it 
        int ReadFromHost(CreateType hostData, size_t size)
        {
            return clSetup->getQueue().enqueueWriteBuffer(this->data, CL_TRUE, 0, size, hostData, NULL, &readEvent);
            
        }

        int WriteToHost(CreateType hostData, size_t size)
        {
            return clSetup->getQueue().enqueueReadBuffer(this->data, CL_TRUE, 0, size, hostData, NULL, &writeEvent);
        }

        int CopyToDevice(DeviceType deviceArray, size_t size) //data -> deviceArray
        {
             return clSetup->getQueue().enqueueCopyBuffer(data, deviceArray, 0, 0, size, NULL, &copyEvent );
        }

        // track time for IO through the wrapper 
        double GetReadTiming()
        {
            readEvent.wait();
            return CLUtils::GetEventTiming(readEvent);

        }
        
        double GetWriteTiming()
        {
            writeEvent.wait();
            return CLUtils::GetEventTiming(writeEvent);
        }


    private:
   
        cl::Event readEvent;

        cl::Event writeEvent;
        
        cl::Event copyEvent;

        CLSetup* clSetup;
        
        cl::Event event;

};

