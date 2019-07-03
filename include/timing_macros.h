#ifndef TIMING_MACROS_H
#define TIMING_MACROS_H

// this file is basically a big hack to force all versions to use the same timing variables and calls

// these arrays hold names of files, titles and directories...
char timingOutputFile[MAX_STR_LEN];
char timingString[MAX_STR_LEN] = "";
char infoString[MAX_STR_LEN] = "";
char gitDir[MAX_STR_LEN] = "";

// print to stdout the string output to the timing file (sanity check, basically)
#define printOutputs puts(timingString); 

// standard timing printout for the different versions
#define printToString sprintf(timingString,"Program Build: %.14f\nKernel1: %.14f\nKernel2: %.14f\nKernels: %.14f\nData Copy To: %.14f\nData Copy Between: %.14f\nData Copy Back: %.14f\nData Copy Total: %.14f\nTotal time: %5.6lf \nNF: %d\nBandwidth: %4.6f\nGFLOPS: %4.6f\n",programBuildTotal,kernel1Time,kernel2Time,kernelsTime,dataCopyInitTotal, dataCopyBtwTotal, dataCopyBackTotal, dataCopyTotal, totalTime,numberSamples,getBandWidth(kernel1Time), getGFLOPS(kernel1Time));

// macro to write the timings to file 
#define writeTimingsToFile writeStringsToFile(timingString,infoString,getTimingFileName());

// timing variables 
double startKernel1, endKernel1, startKernel2, endKernel2, startKernels, endKernels, kernel1Time, kernel2Time, kernelsTime, startTime, endTime, totalTime;
double dataCopyInitStart, dataCopyInitEnd, dataCopyInitTotal, dataCopyBtwStart, dataCopyBtwEnd, dataCopyBtwTotal, dataCopyBackStart, dataCopyBackEnd, dataCopyBackTotal, dataCopyTotal, programBuildStart, programBuildEnd, programBuildTotal;



#endif
