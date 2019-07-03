#include <stdlib.h>

/*
 * Wrapper parent class for data types 
 * Currently only implemented for OpenCL data 
 */

// DeviceType -- type of device object 
// CreateType -- data type (ie. double) -- could be the same thing 
template<class DeviceType, class CreateType>
class DeviceData
{
    public:
       DeviceData()
       {
       }

       ~DeviceData()
       {
       
       }

        // child classes must implement! 
       virtual int SetData(size_t size) = 0; 

       virtual int ReadFromHost(CreateType hostData, size_t size) = 0; 

       virtual int WriteToHost(CreateType hostData, size_t size) = 0; 
       
       virtual int CopyToDevice(DeviceType deviceArray, size_t size) = 0; 

       virtual double GetReadTiming() = 0;
       
       virtual double GetWriteTiming() = 0;

       void CopyFromDevice(DeviceType dataArray)
       {
            data = dataArray;
       }

       DeviceType GetData()
       {
            return data;
       }

    protected:

        DeviceType data;
};
