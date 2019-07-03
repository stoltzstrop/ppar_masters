#include "initial.h" 
#include "cl_utils.h"
#include "CLDeviceData.hpp"
#include "Audio.h" 
#include "timing.h"
#include "SimpleGridData.hpp"
#include "CLSimpleGrid.h"
#include "timing_macros.h"
#include "Audio.h"

// debugging to print out header file for on-the-fly kernel
#define PRINT_HEADER FALSE
#define TEST_HEADER "testHeader.h"


/*
 * Main room simulation for abstractCL version
 */

int main()
{
    
        // simulation parameters
        double SR         = (double)numberSamples;             // Sample Rate   
        double alpha      = 0.005;               // Boundary loss
        double c          = 344.0;               // Speed of sound in air (m/s)
        double k          = 1/SR;
        double h          = sqrt(3.0)*c*k;
        double lambda     = c*k/h;

        // Set constant memory coeffs
        coeffs_type cf_h[1];
        cf_h[0].l2      = (c*c*k*k)/(h*h);
        cf_h[0].loss1   = 1.0/(1.0+lambda*alpha);
        cf_h[0].loss2   = 1.0-lambda*alpha;
  
        coeffs_type cf_hG;
        cf_hG.l2      = (c*c*k*k)/(h*h);
        cf_hG.loss1   = 1.0/(1.0+lambda*alpha);
        cf_hG.loss2   = 1.0-lambda*alpha;


        // Initialise input
        int alength = dur;
        int n = 0;
        value *source_h = (value *)calloc(numberSamples,sizeof(value));
        for(n=0;n<dur;n++)
        {
          source_h[n] = 0.5*(1.0-cos(2.0*pi*n/(value)dur));
        }
        startTime = getTime(); 
	size_t pr_size  = sizeof(value);
	size_t mem_size = VOLUME*pr_size;
        value ins;             
	

        //initialise host data
        value *receiver_h  = (value *)malloc(numberSamples*sizeof(value));
	value *roomGrid_h  = (value *)malloc(mem_size);


        grid_data *roomGridG_h  = (grid_data *)malloc(grid_data_size);
        grid_data *roomGridGTS1_h  = (grid_data *)malloc(grid_data_size);
        grid_data *dummyPointerG_h = (grid_data *)malloc(grid_data_size);
        
	
        if((receiver_h == NULL) || (roomGrid_h == NULL) || (roomGridG_h == NULL))
        {
		printf("host memory alloc failed...\n");
		exit(EXIT_FAILURE);
	}

        // initialize host memory (will be written on GPU memory as well)
        int xi;
        for(xi=0;xi<VOLUME;xi++)
        {
            roomGrid_h[xi] = 0.0;
            setPointWithIndex(xi,roomGridG_h,0.0); // grid_data functionality 
            setPointWithIndex(xi,roomGridGTS1_h,0.0);
            setPointWithIndex(xi,dummyPointerG_h,0.0);

            if(xi<numberSamples)
            {
              receiver_h[xi] = 0.0;
            }
        }


        printToTimingFileName(ABSTRACT_STR);

        // build the on-the-fly kernel
        programBuildStart = getTime();
        SimpleGridData testGridData; 
        string input = CLSetup::buildHeaderFile(&testGridData);
     
        // include this header too
        string init_string = CLSetup::ReadInHeaderFile("../../include/constants.h",true);
        string struct_string = CLSetup::ReadInHeaderFile(includeToRead,true);

        init_string+=struct_string;
        init_string+=input;

        // debug header  
        if(PRINT_HEADER)
        {
             std::ofstream outFile(TEST_HEADER);
             outFile << init_string;
             outFile.close();
        }

        string program_name(PROGRAM_NAME);
        CLSetup clSetup(program_name,PROCESSOR_VERSION,init_string); 
        printf("Nx: %d Ny: %d Nz: %d\n",Nx,Ny,Nz);
        clSetup.printDeviceInfo(infoString);
        
        cl_int err;


        // masked boilerplate - use C++ classes instead of openCL calls to set up 
        // data, kernels, etc
        int inoutKernelIdx = clSetup.AddKernel(INOUT_KERNEL_FUNC, numberSamples);
        int stencilKernelIdx = clSetup.AddKernel(STENCIL_KERNEL_FUNC, numberSamples);
        programBuildEnd = getTime();
        programBuildTotal = programBuildEnd-programBuildStart;

        CLSetup::err_check(err);
      
        // setup device data
        CLDeviceData<cl::Buffer, grid_data*> roomGridG_d(roomGridG_h, &clSetup,CL_MEM_READ_WRITE,grid_data_size);
        size_t roomSize = sizeof(roomGridG_h);
        size_t bufSize = sizeof(roomGridG_d); 
        size_t structsize = sizeof(grid_data); 

        CLDeviceData<cl::Buffer, grid_data*> roomGridGTS1_d(roomGridGTS1_h, &clSetup,CL_MEM_READ_WRITE,grid_data_size);
        CLDeviceData<cl::Buffer, grid_data*> dummyPointerG_d(dummyPointerG_h, &clSetup,CL_MEM_READ_WRITE,grid_data_size);

        CLDeviceData<cl::Buffer, value*> receiver_d(receiver_h, &clSetup,CL_MEM_READ_WRITE,numberSamples*sizeof(value));
        CLDeviceData<cl::Buffer, coeffs_type*> cf_d(cf_h, &clSetup,CL_MEM_READ_ONLY,sizeof(coeffs_type));
      
        err = clSetup.SetKernelParameter<cl::Buffer>(stencilKernelIdx,roomGridG_d.GetData(),sizeof(cl::Buffer),0);
        CLSetup::err_check(err);
        err = clSetup.SetKernelParameter<cl::Buffer>(stencilKernelIdx,roomGridGTS1_d.GetData(),sizeof(cl::Buffer),1);
        CLSetup::err_check(err);
        err = clSetup.SetKernelParameter<cl::Buffer>(stencilKernelIdx,cf_d.GetData(),sizeof(cl::Buffer),2);
        CLSetup::err_check(err);

        err = clSetup.SetKernelParameter<cl::Buffer>(inoutKernelIdx,roomGridG_d.GetData(),sizeof(cl::Buffer),0);
        CLSetup::err_check(err);
        err = clSetup.SetKernelParameter<cl::Buffer>(inoutKernelIdx,receiver_d.GetData(),sizeof(cl::Buffer),1);
        CLSetup::err_check(err);
        err = clSetup.SetKernelParameter<value>(inoutKernelIdx,ins,sizeof(value),2);
        CLSetup::err_check(err);
        err = clSetup.SetKernelParameter<int>(inoutKernelIdx,n,sizeof(int),3);
        CLSetup::err_check(err);

        int jj;

        std::vector<int> globalSizeStencil(3);
        std::vector<int> localSizeStencil(3);
    
        globalSizeStencil[0] = Nx;
        globalSizeStencil[1] = Ny;
        // do not need Z dimension if local memory is being run (this should be set somewhere better!)
        globalSizeStencil[2] = 1;//Nz;  
        
        localSizeStencil[0] = Bx;
        localSizeStencil[1] = By;
        localSizeStencil[2] = Bz;

        std::vector<int> globalSizeInout(3);
        std::vector<int> localSizeInout(3);

        globalSizeInout[0] = 1;
        globalSizeInout[1] = 1;
        globalSizeInout[2] = 1;
        
        localSizeInout[0] = 1;
        localSizeInout[1] = 1;
        localSizeInout[2] = 1;


        startKernels = getTime();

        // main loop over timesteps 
        for(n=0;n<numberSamples;n++)
	{
            // run room update kernel
            clSetup.RunKernel(stencilKernelIdx,globalSizeStencil, localSizeStencil);
            CLSetup::err_check(err);
            
            ins = 0.0;
	    if(n<alength)
            {
              ins = source_h[n];
            }
            
            err = clSetup.SetKernelParameter<value>(inoutKernelIdx,ins,sizeof(value),2);
            CLSetup::err_check(err);
            err = clSetup.SetKernelParameter<int>(inoutKernelIdx,n,sizeof(int),3);
            CLSetup::err_check(err);
           
            // run source and receiver update kernel 
            clSetup.RunKernel(inoutKernelIdx,globalSizeInout, localSizeInout);
            CLSetup::err_check(err);

            dummyPointerG_d.CopyFromDevice(roomGridGTS1_d.GetData());
            roomGridGTS1_d.CopyFromDevice(roomGridG_d.GetData());
            roomGridG_d.CopyFromDevice(dummyPointerG_d.GetData());

            err = clSetup.SetKernelParameter<cl::Buffer>(stencilKernelIdx, roomGridG_d.GetData(),sizeof(cl::Buffer),0);
            CLSetup::err_check(err);
            err = clSetup.SetKernelParameter<cl::Buffer>(stencilKernelIdx, roomGridGTS1_d.GetData(),sizeof(cl::Buffer),1);
            CLSetup::err_check(err);
            err = clSetup.SetKernelParameter<cl::Buffer>(inoutKernelIdx, roomGridG_d.GetData(),sizeof(cl::Buffer),0);
            CLSetup::err_check(err);

	}


    // Finish up OpenCL    
    err = clSetup.FinishRun();
    CLSetup::err_check(err);
    endKernels = getTime();
   
    kernelsTime = endKernels-startKernels;
   
    kernel1Time = clSetup.GetKernelTiming(stencilKernelIdx);
    kernel2Time = clSetup.GetKernelTiming(inoutKernelIdx);
    dataCopyBtwTotal = kernelsTime - kernel1Time - kernel2Time;

    // Read the results from the device
    receiver_d.WriteToHost(receiver_h, numberSamples*sizeof(value));
    roomGridG_d.WriteToHost(roomGridG_h, grid_data_size);
    CLSetup::err_check(err);
   
    double dataTo = 0.0;
    double dataFrom = 0.0;
    dataTo += roomGridG_d.GetReadTiming();
    dataTo += roomGridGTS1_d.GetReadTiming();
    dataTo += dummyPointerG_d.GetReadTiming();
    dataTo += receiver_d.GetReadTiming();
    dataTo += cf_d.GetReadTiming();
    dataCopyInitTotal = dataTo;

    dataFrom += roomGridG_d.GetWriteTiming();
    dataFrom += receiver_d.GetWriteTiming();
    dataCopyBackTotal = dataFrom;
    
    dataCopyTotal = dataCopyInitTotal + dataCopyBtwTotal + dataCopyBackTotal;

    endTime = getTime();
    totalTime = (double) (endTime-startTime);
    
    // sanity check 
    for(jj=0;jj<VOLUME;jj++)
    {
        roomGrid_h[jj] = getValueAtPointWithIndex(jj,roomGridG_h);
    }

    // write data to files  
    writeBinaryDataToFile(roomGrid_h,getOutputFileName(ABSTRACT_STR,"room","bin"),VOLUME);       
    writeBinaryDataToFile(receiver_h,getOutputFileName(ABSTRACT_STR,"receiver","bin"),numberSamples);       
    
    for(jj=numberSamples-10;jj<numberSamples;jj++)
    {
        printf("%.14lf\n",receiver_h[jj]);
    }


    printToString;
    printOutputs;
    writeTimingsToFile; 
 
    // Free memory
    free(source_h);free(receiver_h);free(roomGrid_h);
    exit(EXIT_SUCCESS);
}

