#ifndef header_h
#define header_h

#define cuErr(err) __checkCudaErrors (err, __FILE__, __LINE__)

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdint.h>


/*
 * WRITTEN BY CRAIG WEBB, LEFT AS IS 
 */


// ----------------------------------------------------------------
// FUNCTION PROTOTYPES
// ----------------------------------------------------------------

void printDevDetails();

inline void __checkCudaErrors( cudaError err, const char *file, const int line );

void checkLastCUDAError(const char *msg);

// ----------------------------------------------------------------
// writeOutputAudio
// ----------------------------------------------------------------
// Write mono audio buffer to .bin file
void writeOutputAudio(double *audio,                 // Pointer to buffer
      int NF,                      // Lenght of buffer, samples
      const char outputFile[]);    // Full Name of output file


// ----------------------------------------------------------------
// printLastSamples
// ----------------------------------------------------------------
// Prints last N samples, plus max(abs) of buffer
void printLastSamples(double *audio,           // Pointer to buffer
      int NF,                // Length of buffer, samples
      int N);                // Number of last samples to display, > 0


// ----------------------------------------------------------------
// FUNCTION IMPLEMENTATIONS
// ----------------------------------------------------------------

inline void __checkCudaErrors( cudaError err, const char *file, const int line ){
   if( cudaSuccess != err) {
      fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
            file, line, (int)err, cudaGetErrorString( err ) );
      exit(-1);
   }
}


void checkLastCUDAError(const char *msg)
{
   cudaError_t err = cudaGetLastError();
   if( cudaSuccess != err) 
   {
      fprintf(stderr, "\nCuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
      exit(EXIT_FAILURE);
   }                         
}

void printDevDetails(){

   // Get name of device being used...
   int devy;
   cudaGetDevice(&devy);
   cudaDeviceProp devProp;
   cudaGetDeviceProperties(&devProp, devy);
   printf("\n");
   printf("RUNNING ON GPU        : %d - %s, compute %d.%d\n", devy,devProp.name,devProp.major,devProp.minor);
   printf("Total global memory   : %u\n", devProp.totalGlobalMem);
   printf("Maximum memory pitch  : %u\n", devProp.memPitch);
   printf("Shared mem per block  : %u\n", devProp.sharedMemPerBlock);
   printf("Total constant memory : %u\n",  devProp.totalConstMem);
   printf("Registers per block   : %d\n", devProp.regsPerBlock);
   printf("Warp size             : %d\n",  devProp.warpSize);
   printf("Max threads per block : %d\n",  devProp.maxThreadsPerBlock);
   printf("Max dims of block     : %dx%dx%d\n", devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
   printf("Max dims of grid      : %dx%dx%d\n", devProp.maxGridSize[0],devProp.maxGridSize[1],devProp.maxGridSize[2]);
   printf("Number of multiprocs  : %d\n\n",  devProp.multiProcessorCount);
}


// ----------------------------------------------------------------
void printLastSamples(double *audio, int NF, int N) {

   int n;
   double maxy;
   // Test that N > 0
   if(N<1){
      printf("Must display 1 or more samples...\n");
   }
   else{
      //print last N samples
      printf("\n");
      for(n=NF-N;n<NF;n++)
      {
         if (audio[n]<0.0) printf("Sample %d : %.15f\n",n,audio[n]);
         else              printf("Sample %d :  %.15f\n",n,audio[n]);
      }

      // find max
      maxy = 0.0;
      for(n=0;n<NF;n++)
      {
         if(fabs(audio[n])>maxy) maxy = fabs(audio[n]);

      }
      printf("\nMax sample : %.15f\n",maxy);
   }
}

// ----------------------------------------------------------------
void writeOutputAudio(double *audio, int NF, const char outputFile[]) {

   // write to file
   FILE *file_ptr;
   size_t bytes_written;

   file_ptr = fopen(outputFile,"wb");
   if(file_ptr != NULL)
   {
      bytes_written = fwrite(audio,sizeof(double),(size_t)NF,file_ptr);
      (void)fclose(file_ptr);
      printf("\n%d samples written to output %s\n",(int)bytes_written,outputFile);
   }
   else
   {
      printf("\nOutput Audio File open failed...\n");
   }

}

#endif

