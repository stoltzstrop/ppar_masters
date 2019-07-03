#ifndef AUDIO_H
#define AUDIO_H

// only include the omp header file for OMP runs, otherwise other versions throw a fit
#if defined(_OPENMP)
#include <omp.h>
#endif

#include <math.h>
#include <pwd.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include "initial.h"
#include "timing_macros.h"

/*
 * Helper functions for calculating values, outputting values, naming files and directories 
 */

enum BOOLEAN { FALSE, TRUE };
char dataOutputFile[MAX_STR_LEN];

// calculate bandwidth values 
value getBandWidth(double kernelTime)
{
    // 9 memory accesses in kernel 1, 2 grids
    value bandWidth = (VOLUME*sizeof(double)*1e-9*3)/(kernelTime/numberSamples); 
    return bandWidth;
}

// calculated gflops value 
value getGFLOPS(double kernelTime)
{
    // 13 floating point operations in kernel 1
    value gflops = (VOLUME*1e-9*13)/(kernelTime/numberSamples); 
    return gflops;
}

// print reminder of what run this is 
void printConstants()
{
    printf("Room size: %d %d %d \n",Nx,Ny,Nz);
    printf("Work Group size: %d %d %d \n",Bx,By,Bz);
    printf("Number of samples: %d \n",numberSamples);
}

// check if data directory exists already, if not then make it
void checkMakeDir(char* outputDir)
{
    struct stat st = {0};

    if (stat(outputDir, &st) == -1) 
    {
        mkdir(outputDir, 0777);
    }
}

// get home directory 
const char* getHomeDirectory()
{
    struct passwd *pw = getpwuid(getuid());
    return pw->pw_dir;
}

// get hostname of machine running on 
void getHostName(char* hostname, int size)
{
    gethostname(hostname,size);
}

// get the fullname of the data directory 
void getDataDirectory(char* dataDir)
{
    int size = 100;
    char hostname[size];
    getHostName(hostname,size);

    if(strstr(hostname,"mic0") || strstr(hostname,"phi-mic1")) // on the xeon phi, the file structure
                                                               // is markedly different
    {
        sprintf(dataDir,"%sroom_code/data/",mic_directory);
    }
    else
    {
        sprintf(dataDir,"%s%s",getHomeDirectory(),data_directory); 
    }
}

// get the current timestamp for timing runs - this ensures files with the same name
// don't get overwritten 
void getTimeStamp(char* stamp, int size)
{
    char cmd[150];
    sprintf(cmd,"date +%%s");
    FILE* fp;
    fp = popen(cmd,"r");
    fgets(stamp,size-1,fp);
    pclose(fp);
    stamp[strcspn(stamp, "\n")] = 0;
}

// format the data repository with the gitID
void printGitRepo()
{
    char cmd[150];
    char repo[150];
    int lengthOfGitRepo = 11; // actually 10!
    char endArr[] = "/";
    int size = 100;
    char hostname[size];
    getHostName(hostname,size);

    // again, the xeon phi likes to be different 
    if(strstr(hostname,"mic0") || strstr(hostname,"phi-mic1"))
    {
        sprintf(cmd,"cat %s.git/refs/heads/master",mic_directory);
    }
    else
    {
        sprintf(cmd,"git rev-parse HEAD");
    }

    FILE* fp;
    fp = popen(cmd,"r");
    if(fp == NULL)
    {
        printf("Problem getting git repo!\n");
    }
    fgets(gitDir,lengthOfGitRepo,fp);
    pclose(fp);
    repo[strcspn(gitDir, "\n")] = 0;
    strcat(gitDir,endArr);
}

// get the git ID for debugging purposes ..
void getGitRepo(char* repo)
{
    char cmd[150];
    int lengthOfGitRepo = 11; // actually 10!
    char endArr[] = "/";
    int size = 100;
    char hostname[size];
    getHostName(hostname,size);
    if(strstr(hostname,"mic0") || strstr(hostname,"phi-mic1"))
    {
        sprintf(repo,"%s",gitDir);
    }
    else
    {
        sprintf(cmd,"git rev-parse HEAD");
        FILE* fp;
        fp = popen(cmd,"r");
        if(fp == NULL)
        {
            printf("Problem getting git repo!\n");
        }
        fgets(repo,lengthOfGitRepo,fp);
        pclose(fp);
        repo[strcspn(repo, "\n")] = 0;
        strcat(repo,endArr);
    }
}

// translate the hostname to a more familiar name 
char* getPlatform(char* hostname)
{
    if(strstr(hostname, "kepler"))
    {
        return "nvidia_kepler";
    }
    else if(strstr(hostname,"casper"))
    {
        return "nvidia_maxwell";
    }
    else if(strstr(hostname,"phi"))
    {
        return "xeon_phi";
    }
    else if(strstr(hostname,"supersonic"))
    {
        return "AMD_E5530";
    }
    else if(strstr(hostname,"suzuka"))
    {
        return "AMD_R9_295X2";
    }
    else if(strstr(hostname,"fuji"))
    {
        return "AMD_R280";
    }
    else if(strstr(hostname,"monza"))
    {
        return "AMD_7970";
    }
    else if(strstr(hostname,"spa"))
    {
        return "xeon_phi";
    }
    else if(strstr(hostname,"monaco"))
    {
        return "nvidia_tesla";
    }
    else
    {
        return hostname;
    }
}

// instead of using real sound, knock up some from a cosine wave - ORIGINALLY WRITTEN BY CRAIG WEBB
void CreateSineWave(int duration, value* si_h)
{
    int n;

    for(n=0;n<duration;n++)
    {
        si_h[n] = 0.5*(1.0-cos(2.0*pi*n/(value)duration));
    }
}

// format the timing output filename 
// (lots of global variables used)
char* getOutputFileName(char* program_type, char* fileType, char* ext)
{
    int size = 100;
    char hostname[size];
    getHostName(hostname,size);
    char* platform = getPlatform(hostname);
    dataOutputFile[0] = 0;
    sprintf(dataOutputFile,"%s-%s-%s-A%dx%dx%d-S%dx%dx%dx-R%dx%dx%d-NF%d.%s",program_type,platform,fileType,Nx,Ny,Nz,Sx,Sy,Sz,Rx,Ry,Rz,numberSamples,ext);
    return dataOutputFile;
}

// print the timing information to the timing output file 
// (lots of global variables used)
void printToTimingFileName(char* program_type)
{
    int size = 100;
    char hostname[size];
    char timestamp[size];
    getHostName(hostname,size);
    getTimeStamp(timestamp,size);
    
    char* platform = getPlatform(hostname);
    sprintf(timingOutputFile,"%s-%s-timings-A%dx%dx%d-S%dx%dx%dx-R%dx%dx%d-NF%d-%s.txt",program_type,platform,Nx,Ny,Nz,Sx,Sy,Sz,Rx,Ry,Rz,numberSamples,timestamp);
}

// return the name of the timing output file 
char* getTimingFileName()
{
    return timingOutputFile;
}


// helper function for concatenating directories for outputting data
void getFullFilePath(char* path, char* fileString)
{

    char gitDir[500];
    getGitRepo(gitDir);
    getDataDirectory(path);
    strcat(path,gitDir);
    checkMakeDir(path);
    strcat(path,fileString);

}

char* concatStrings(char* str1, char* str2)
{
    int len1 = strlen(str1);
    int len2 = strlen(str2);
    
    char* wholeStr=(char*)malloc(len1+len2+1);
    strcpy(wholeStr,str1);
    strcat(wholeStr,str2);
    return wholeStr;
}

// write timing strings to file for data analysis
void writeStringsToFile(char* timingStr, char* infoStr, char* fileString)
{
    char wholeFile[1024];
    getFullFilePath(wholeFile,fileString);

    FILE * file;
    file = fopen(wholeFile, "w");
    
    if(file == NULL)
    {
        printf("Error opening file: %s!",wholeFile);
        exit(1);
    }
    if(infoStr != "")
    {   
        fprintf(file, "%s\n", infoStr);
    }
    fprintf(file, "%s\n", timingStr);

    fclose(file);
}

// data output calls for old  unit tests or sanity checks 
void writeDoublesToFile(double *data, char* output, int ndata)
{
    char wholeFile[1024];
    getFullFilePath(wholeFile,output);
    
    
    FILE * file;
    file = fopen(wholeFile, "w");
    
    if(file == NULL)
    {
        printf("Error opening file: %s!",wholeFile);
        exit(1);
    }

    int i;
    for(i=0; i<ndata; i++)
    {
        fprintf(file, "%f\n", data[i]);
    }

    fclose(file);
}

// data output calls for unit tests 
void writeBinaryDataToFile(value *data, char* output, int ndata)
{
    char wholeFile[1024];
    getFullFilePath(wholeFile,output);

    FILE * file;
    file = fopen(wholeFile, "wb");
    
    if(file == NULL)
    {
        printf("Error opening file: %s!",wholeFile);
        exit(1);
    }

    int i;
    for(i=0; i<ndata; i++)
    {
        fwrite(&data[i], sizeof(value),1,file);
    }

    fclose(file);
}

// print out debugging information for OMP runs 
#if defined(_OPENMP)
int printOMPThreads()
{

    int tid, nthreads;
    #pragma omp parallel private(nthreads, tid)
    {
        tid = omp_get_thread_num();
        printf("Hello World from thread = %d\n", tid);

        if (tid == 0) 
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }    

    }
    return nthreads;
}
#endif

#endif
