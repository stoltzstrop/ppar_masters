#include "CLSetup.hpp"


    
    //get correct device, setup the program, initialise kernels size, ensure everything is set to go 
    // limited to 100 kernels in the program 
    CLSetup::CLSetup(string programName, int device_type, string directives, string options) // try to caste
    : kernels(100), numKernels(0)
    {
        cl_int err;
        vector<cl::Platform> platforms;
        int defaultDeviceIdx = 0;
        cl_device_type deviceType = (cl_device_type) device_type;
        cl::Platform::get(&platforms);
        int defaultPlatformIdx = CLUtils::getRightPlatform(platforms);
        platforms[defaultPlatformIdx].getDevices(deviceType,&devices);
        defaultDevice = devices[defaultDeviceIdx];
        context = cl::Context(devices); 
        queue = cl::CommandQueue(context, devices[0],CL_QUEUE_PROFILING_ENABLE);
        
        ifstream cl_file(programName);
        string program_string = directives;
        string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
        program_string+="\n";
        program_string+=cl_string;
        cl::Program::Sources source(1, make_pair(program_string.c_str(), 
                            program_string.length() + 1));

        program = cl::Program(context,source,&err);
        err_check(err);   
        const char* op_str = options.c_str();
        err = program.build(devices,options.c_str());
        err_check(err);
        if(getErrorString(err) == "CL_BUILD_PROGRAM_FAILURE") 
        {
              size_t log_size;
              cl::STRING_CLASS buildLog;
              program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &buildLog);

              cout<<"Build: "<<buildLog<<endl;
        }
    } 

    CLSetup::~CLSetup()
    {
        // free up what we don't need anymore 
        clReleaseProgram(program());
        clReleaseCommandQueue(queue());
        clReleaseContext(context());
    }

    void CLSetup::printDeviceInfo(char* infoString)
    {
        string defaultDeviceStr;
        string versionStr;
        defaultDevice.getInfo<string>(CL_DEVICE_NAME,&defaultDeviceStr);
        cout<<defaultDeviceStr<<"\n";
        defaultDevice.getInfo<string>(CL_DEVICE_OPENCL_C_VERSION,&versionStr);
        cout<<versionStr<<"\n";
        sprintf(infoString,"%s\n%s\n",defaultDeviceStr.c_str(),versionStr.c_str());
    }

    // read in a header file as a string to concatenate with kernel file 
    string CLSetup::ReadInHeaderFile(string fileName, bool cutOffName)
    {
       // copy same directives and constants used to kernels 
        ifstream cl_file(fileName);
        string init_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
        if(cutOffName)
        {
             int idxSecond = init_string.find("\n",init_string.find("\n")+1);
             init_string.erase(0,idxSecond+1);
             init_string.erase(init_string.find_last_of("#endif")-6,init_string.length()-1);
        }
        return init_string;
    }

    template<typename T> int CLSetup::SetKernelParameter(int kernelIdx, T parameter, size_t size, int argN)
    {
        return kernels[kernelIdx].SetParameter(parameter, size, argN);
    }

    int CLSetup::FinishRun()
    {
        return queue.finish();
    }
    
    // pop a new kernel in the kernel list, return index of that kernel  
    int CLSetup::AddKernel(char* kernelName, int numberRuns)
    { 
        kernels[numKernels++] =  CLKernel(program,kernelName, numberRuns);
        return numKernels-1; 
    } // store in "kernels" with idx value returned
    
    int CLSetup::RunKernel(int kernelIdx, std::vector<int> globalSize, std::vector<int> localSize)
    {
        int dimension = globalSize.size();

        if(dimension != localSize.size())
        {
            printf("mismatched workgroup dimensions!\n");
            return -99;
        }
         
        cl::NDRange global;
        cl::NDRange local;
        
        // return correctly sizes workitem  & workgroup arrays
       switch(dimension)
       {
            case 1:
                global = cl::NDRange(globalSize[0]);
                local = cl::NDRange(localSize[0]);
                break;
            case 2:
                global = cl::NDRange(globalSize[0],globalSize[1]);
                local = cl::NDRange(localSize[0],localSize[1]);
                break;
            case 3:
                global = cl::NDRange(globalSize[0],globalSize[1],globalSize[2]);
                local = cl::NDRange(localSize[0],localSize[1],localSize[2]);
                break;

       }
        
        return kernels[kernelIdx].Run(&queue, global, local);

    }

double CLSetup::GetKernelTiming(int kernelIdx)
{
    return kernels[kernelIdx].getTimings();
}

// ripped off StackOverflow: 
// http://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes 
//

const char* CLSetup::getErrorString(cl_int error)
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


void CLSetup::err_check( int err ) {
    if ( err != CL_SUCCESS ) {
        printf("Error: %d ( %s )\n",err,getErrorString(err));
    }
}

// build header file for the kernel on the fly, based on user decisions
string CLSetup::buildHeaderFile(GridData* dataObject)
{
    
          string headerString = "";
          
          // these functions get called based on what headers are included for the GridData object 
          headerString += dataObject->getInitialDeclaration();
       
          headerString += "\n\n";

	headerString += dataObject->getXDimFunction(); // the ordering here is necessary for functions that are actually defined!
	        
          headerString += "\n\n";

	headerString += dataObject->getYDimFunction();
	        
          headerString += "\n\n";

	headerString += dataObject->getZDimFunction();
	        
          headerString += "\n\n";

	headerString += dataObject->getAreaFunction();
	        
          headerString += "\n\n";
          
	headerString += dataObject->getSourceIndexFunction();
	        
          headerString += "\n\n";
          
	headerString += dataObject->getReceiverIndexFunction();
	        
          headerString += "\n\n";

	headerString += dataObject->setReceiverPointFunction();
	        
          headerString += "\n\n";

	headerString += dataObject->getKernelXIdxFunction();
	        
          headerString += "\n\n";

	headerString += dataObject->getKernelYIdxFunction();
	        
          headerString += "\n\n";

	headerString += dataObject->getKernelZIdxFunction();

          headerString += "\n\n";

          headerString += dataObject->CalculateIndexFunction();

          headerString += "\n\n";

          headerString += dataObject->getValueAtPointWithIndexFunction();

          headerString += "\n\n";

          headerString += dataObject->getValueAtPointFunction();

          headerString += "\n\n";

          headerString += dataObject->addToPointWithIndexFunction();

          headerString += "\n\n";

          headerString += dataObject->addToPointFunction();
        
          headerString += "\n\n";

          headerString += dataObject->setPointWithIndexFunction();

          headerString += "\n\n";

          headerString += dataObject->setPointFunction();
        
          headerString += "\n\n";

          headerString += dataObject->isOnGridFunction();

          headerString += "\n\n";

          headerString += dataObject->getNeighbourSumFunction();

          headerString += "\n\n";

          headerString += dataObject->getNumberOfNeighboursFunction();
        
          headerString += "\n\n";

          headerString += dataObject->isIndexOnBoundaryFunction();
        
          headerString += "\n\n";

	headerString += dataObject->calculateBoundaryConditionChangeFunction();
	        
          headerString += "\n\n";
	
	headerString += dataObject->InitialiseBoundaryConditionsFunction();
	        
          headerString += "\n\n";

	headerString += dataObject->UpdateBoundaryConditionsFunction();
	        
          headerString += "\n\n";

	headerString += dataObject->calculateBoundaryCondition1Function();
	        
          headerString += "\n\n";

	headerString += dataObject->calculateBoundaryCondition2Function();
	        
          headerString += "\n\n";

	headerString += dataObject->UpdateStencilFunction();
	        
          headerString += "\n\n";

	headerString += dataObject->ApplyStencilUpdateFunction();
	        
          headerString += "\n\n";

	headerString += dataObject->SetupFunction();
	        
          headerString += "\n\n";

	headerString += dataObject->BreakdownFunction();
	        
          headerString += "\n\n";

	headerString += dataObject->OptimiseFunction();
	        
          headerString += "\n\n";


        return headerString;
}

// calls required by C++ when setting up specific templates 
template int CLSetup::SetKernelParameter<cl::Buffer>(int, cl::Buffer, size_t,  int);
template int CLSetup::SetKernelParameter<double>(int, double, size_t, int);
template int CLSetup::SetKernelParameter<float>(int, float, size_t, int);
template int CLSetup::SetKernelParameter<int>(int, int, size_t, int);
