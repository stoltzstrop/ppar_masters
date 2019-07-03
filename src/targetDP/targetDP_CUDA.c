/*
 * targetDP_CUDA.c: API Implementation for targetDP: CUDA version
 * Alan Gray
 *
 * Copyright 2015 The University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <omp.h>
#include <math.h>

#include "targetDP.h"

//pointers to internal work space
static double* dwork;
static double* dwork_d;
static int* iwork;
static int* iwork_d;

static int* b3Dedgemap;
static int* b3Dedgemap_d;

static int* b3Dhalomap;
static int* b3Dhalomap_d;

static char* cwork;

static int b3Dedgesize_=0;
static int b3Dhalosize_=0;

int haloEdge(int index, int extents[3],int offset, int depth);

//The targetMalloc function allocates memory on the target.
__targetHost__ void targetMalloc(void **address_of_ptr,size_t size){

 
  cudaMalloc(address_of_ptr,size);
  checkTargetError("targetMalloc");

  return;
}


// The targetMalloc function allocates unified memory that can be accessed
// on the host or the target.
__targetHost__ void targetMallocUnified(void **address_of_ptr,size_t size){

 
  cudaMallocManaged(address_of_ptr,size);
  checkTargetError("targetMallocUnified");

  return;
}


//The targetCalloc function allocates, and initialises to zero, memory on the target.
__targetHost__ void targetCalloc(void **address_of_ptr,size_t size){

 
  cudaMalloc(address_of_ptr,size);
  double ZERO=0.;
  cudaMemset(*address_of_ptr, ZERO, size);
  checkTargetError("targetCalloc");

  return;
}


// The targetCalloc function allocates unified memory that can be accessed
// on the host or the target, and is initialised to 0
__targetHost__ void targetCallocUnified(void **address_of_ptr,size_t size){

 
  cudaMallocManaged(address_of_ptr,size);
  double ZERO=0.;
  cudaMemset(*address_of_ptr, ZERO, size);
  checkTargetError("targetCallocUnified");

  return;
}


//The copyToTarget function copies data from the host to the target.
__targetHost__ void copyToTarget(void *targetData,const void* data,size_t size){

  cudaMemcpy(targetData,data,size,cudaMemcpyHostToDevice);
  checkTargetError("copyToTarget");
  return;
}

//The copyFromTarget function copies data from the target to the host.
__targetHost__ void copyFromTarget(void *data,const void* targetData,size_t size){

  cudaMemcpy(data,targetData,size,cudaMemcpyDeviceToHost);
  checkTargetError("copyFromTarget");
  return;

}


// The targetInit3D initialises the environment required to perform any of the
// “3D” operations defined below.
__targetHost__ void targetInit3D(int extents[3], size_t nfieldsmax, int nhalo){

  int nsites=extents[0]*extents[1]*extents[2];
  // allocate internal work space

  dwork = (double*) malloc (nsites*nfieldsmax*sizeof(double));
  
  cudaMalloc(&dwork_d,nsites*nfieldsmax*sizeof(double));
  checkTargetError("malloc dwork_d");


  iwork = (int*) malloc (nsites*sizeof(int));
  
  cudaMalloc(&iwork_d,nsites*sizeof(int));
  checkTargetError("malloc iwork_d");


  cwork = (char*) malloc (nsites*sizeof(char));


  b3Dedgemap = (int*) malloc (nsites*sizeof(int));
  
  cudaMalloc(&b3Dedgemap_d,nsites*sizeof(int));

  checkTargetError("malloc b3Dedgemap_d ");


  b3Dhalomap = (int*) malloc (nsites*sizeof(int));
  
  cudaMalloc(&b3Dhalomap_d,nsites*sizeof(int));

  checkTargetError("malloc b3Dhalomap_d ");


  //get 3D boundary compression mapping

  int i;

  int offset=nhalo;
  int depth=nhalo;
  int j=0;
  for (i=0; i<nsites; i++){
    if(haloEdge(i,extents,offset,depth)){
      b3Dedgemap[j]=i;
      j++;
    }
    
  }

  b3Dedgesize_=j;


  offset=0;
  depth=nhalo;
  j=0;
  for (i=0; i<nsites; i++){
    if(haloEdge(i,extents,offset,depth)){
      b3Dhalomap[j]=i;
      j++;
    }
    
  }

  b3Dhalosize_=j;


  //copy compresssion info to GPU
  cudaMemcpy(b3Dedgemap_d, b3Dedgemap, b3Dedgesize_*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(b3Dhalomap_d, b3Dhalomap, b3Dhalosize_*sizeof(int), cudaMemcpyHostToDevice);


  return;
}

//deprecated
__targetHost__ void targetInit(int extents[3], size_t nfieldsmax, int nhalo){
  targetInit3D(extents,nfieldsmax,nhalo);
  return;
}

// The targetFinalize3D finalises the targetDP 3D environment.
__targetHost__ void targetFinalize3D(){

  free(iwork);
  free(dwork);
  free(cwork);
  free(b3Dedgemap);
  free(b3Dhalomap);

  cudaFree(iwork_d);
  cudaFree(dwork_d);
  cudaFree(b3Dedgemap_d);
  cudaFree(b3Dhalomap_d);

}

//deprecated
__targetHost__ void targetFinalize(){

  targetFinalize3D();
  return;
}

//
__global__ static void copy_field_partial_gpu_d(double* f_out, const double* f_in, int nsites, int nfields, int *fullindex_d, int packedsize, int inpack) {

  int threadIndex;
  int i;


    threadIndex = blockIdx.x*blockDim.x+threadIdx.x;


  if ((threadIndex < packedsize))
    {


      for (i=0;i<nfields;i++)
	{
	    
	  if (inpack)
	    f_out[i*nsites+fullindex_d[threadIndex]]
	    =f_in[i*packedsize+threadIndex];
	  else
	   f_out[i*packedsize+threadIndex]
	      =f_in[i*nsites+fullindex_d[threadIndex]];
	  
	}
    }


  return;
}

//
__global__ static void copy_field_partial_gpu_AoS_d(double* f_out, const double* f_in, int nsites, int nfields, int *fullindex_d, int packedsize, int inpack) {

  int threadIndex;
  int i;


    threadIndex = blockIdx.x*blockDim.x+threadIdx.x;


  if ((threadIndex < packedsize))
    {

      for (i=0;i<nfields;i++)
	{
	    
	  /* if (inpack) */
	  /*   f_out[i*nsites+fullindex_d[threadIndex]] */
	  /*   =f_in[i*packedsize+threadIndex]; */
	  /* else */
	  /*  f_out[i*packedsize+threadIndex] */
	  /*     =f_in[i*nsites+fullindex_d[threadIndex]]; */


	  if (inpack)
	    f_out[fullindex_d[threadIndex]*nfields+i]
	    =f_in[threadIndex*nfields+i];
	  else
	   f_out[threadIndex*nfields+i]
	     =f_in[fullindex_d[threadIndex]*nfields+i];
	  
	}
    }


  return;
}


//
__targetHost__ void copyToTargetMasked(double *targetData,const double* data,size_t nsites,
			size_t nfields,char* siteMask){


  int i;
  int index;

  int* fullindex = iwork;
  int* fullindex_d = iwork_d;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;

  //get compression mapping
  int j=0;
  for (i=0; i<nsites; i++){
    if(siteMask[i]){
      fullindex[j]=i;
      j++;
    }
    
  }

  int packedsize=j;


  //copy compresssion info to GPU
  cudaMemcpy(fullindex_d, fullindex, packedsize*sizeof(int), cudaMemcpyHostToDevice);


    
  //compress grid
  for (i=0;i<nfields;i++)
    {
      
      j=0;
      for (index=0; index<nsites; index++){
	if(siteMask[index]){
	  tmpGrid[i*packedsize+j]=data[i*nsites+index];
	  j++;
	}
      }
      
    }


  //put compressed grid on GPU
  cudaMemcpy(tmpGrid_d, tmpGrid, packedsize*nfields*sizeof(double), cudaMemcpyHostToDevice); 



  //uncompress grid on GPU
  int nblocks=(packedsize+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(targetData,tmpGrid_d,nsites,
						    nfields,
						    fullindex_d, packedsize, 1);
  cudaThreadSynchronize();

  
  return;
  
}


//
__targetHost__ void copyFromTargetMasked(double *data,const double* targetData,size_t nsites,
			size_t nfields,char* siteMask){




  int i;
  int index;


  int* fullindex = iwork;
  int* fullindex_d = iwork_d;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;


  //get compression mapping
  int j=0;
  for (i=0; i<nsites; i++){
    if(siteMask[i]){
      fullindex[j]=i;
      j++;
    }
    
  }

  int packedsize=j;

  //copy compresssion info to GPU
  cudaMemcpy(fullindex_d, fullindex, packedsize*sizeof(int), cudaMemcpyHostToDevice);

  
  //compress grid on GPU
  int nblocks=(packedsize+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(tmpGrid_d,targetData,nsites,
						    nfields,
						    fullindex_d, packedsize, 0);
  cudaThreadSynchronize();

  //get compressed grid from GPU
  cudaMemcpy(tmpGrid, tmpGrid_d, packedsize*nfields*sizeof(double), cudaMemcpyDeviceToHost); 

    
  //expand into final grid
  for (i=0;i<nfields;i++)
    {
      
      j=0;
      for (index=0; index<nsites; index++){
	if(siteMask[index]){
	  data[i*nsites+index]=tmpGrid[i*packedsize+j];	
	  j++;
	}
      }
      
    }

  return;

}


//
int haloEdge(int index, int extents[3],int offset, int depth){

  int coords[3];


  targetCoords3D(coords,extents,index);

  int returncode=0;

  int i;



    for (i=0;i<3;i++){
      if ( 
	  (coords[i]>=(offset)) && 
	  (coords[i]<(offset+depth))  
	  
	   ) returncode=1;
      
      if ( 
	  (coords[i] >= (extents[i]-offset-depth) ) && 
	  (coords[i] < (extents[i]-offset) )   
	   ) returncode=1;


  }

  
  
    return returncode;


}


//
__targetHost__ void copyFromTarget3DEdge(double *data,const double* targetData,int extents[3], size_t nfields){


  size_t nsites=extents[0]*extents[1]*extents[2];


  int i;
  int index;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;


  int j;

  
  //compress grid on GPU
  int nblocks=(b3Dedgesize_+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(tmpGrid_d,targetData,nsites,
						    nfields,
						    b3Dedgemap_d, b3Dedgesize_, 0);
  cudaThreadSynchronize();

  //get compressed grid from GPU
  cudaMemcpy(tmpGrid, tmpGrid_d, b3Dedgesize_*nfields*sizeof(double), cudaMemcpyDeviceToHost); 

    
  //expand into final grid


  for (j=0; j<b3Dedgesize_; j++){
    index=b3Dedgemap[j];
    for (i=0;i<nfields;i++)
	data[i*nsites+index]=tmpGrid[i*b3Dedgesize_+j];	
  
  }

  checkTargetError("copyFromTarget3DEdge");
  return;

}

//
__targetHost__ void copyToTarget3DHalo(double *targetData,const double* data, int extents[3], size_t nfields){

  size_t nsites=extents[0]*extents[1]*extents[2];

  int i,j;
  int index;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;

    
  //  compress grid
        
  for (j=0; j<b3Dhalosize_; j++){
    index=b3Dhalomap[j];
    for (i=0;i<nfields;i++)
      tmpGrid[i*b3Dedgesize_+j]=data[i*nsites+index];
  
  }

  

  //put compressed grid on GPU
  cudaMemcpy(tmpGrid_d, tmpGrid, b3Dhalosize_*nfields*sizeof(double), cudaMemcpyHostToDevice); 

  //uncompress grid on GPU
  int nblocks=(b3Dhalosize_+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(targetData,tmpGrid_d,nsites,
						    nfields,
						    b3Dhalomap_d, b3Dhalosize_, 1);
  cudaThreadSynchronize();

  checkTargetError("copyToTarget3DHalo");
  
  return;
  
}

static int neighb3D[19][3] = {{ 0,  0,  0},
		 { 1,  1,  0}, { 1,  0,  1}, { 1,  0,  0},
		 { 1,  0, -1}, { 1, -1,  0}, { 0,  1,  1},
		 { 0,  1,  0}, { 0,  1, -1}, { 0,  0,  1},
		 { 0,  0, -1}, { 0, -1,  1}, { 0, -1,  0},
		 { 0, -1, -1}, {-1,  1,  0}, {-1,  0,  1},
		 {-1,  0,  0}, {-1,  0, -1}, {-1, -1,  0}};



//
__targetHost__ void copyFromTargetPointerMap3D(double *data,const double* targetData,int extents[3], size_t nfields, int includeNeighbours,  void** ptrarray){


  size_t nsites=extents[0]*extents[1]*extents[2];


  int i;
  int index;


  int* fullindex = iwork;
  int* fullindex_d = iwork_d;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;

  char* siteMap = cwork; 
  char zero=0;
  memset(siteMap,zero,nsites);

  int coords[3];

  //get compression mapping

    if (includeNeighbours){
    int p;
    for (i=0; i<nsites; i++){
      if(ptrarray[i]){
	
	
	targetCoords3D(coords,extents,i);
	for (p=0;p<19;p++){
	  
	  int shiftIndex=targetIndex3D(coords[0]-neighb3D[p][0],coords[1]-neighb3D[p][1],coords[2]-neighb3D[p][2],extents);
	  
	  if ((shiftIndex >= 0) &&   (shiftIndex < nsites)){
	    siteMap[shiftIndex]=1;
	 
	  }
	  
	}
	
      }
      
    }
    }
    else{
      for (i=0; i<nsites; i++){
	if(ptrarray[i]){ 
	  siteMap[i]=1;
	  
	}
      }
    }

  int j=0;
  for (i=0; i<nsites; i++){
    if(siteMap[i]){ 
       j++;
    }
  }
  
  int packedsize=j;
  
  j=0;
  for (i=0; i<nsites; i++){
    if(siteMap[i]){

      fullindex[j]=i;
      j++;

    }
    
  }



  //copy compresssion info to GPU
  cudaMemcpy(fullindex_d, fullindex, packedsize*sizeof(int), cudaMemcpyHostToDevice);

  
  //compress grid on GPU
  int nblocks=(packedsize+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(tmpGrid_d,targetData,nsites,
						    nfields,
						    fullindex_d, packedsize, 0);
  cudaThreadSynchronize();

  //get compressed grid from GPU
  cudaMemcpy(tmpGrid, tmpGrid_d, packedsize*nfields*sizeof(double), cudaMemcpyDeviceToHost); 

    

  //expand into final grid       
  for (index=0; index<packedsize; index++){
    for (i=0;i<nfields;i++)  
      data[i*nsites+fullindex[index]] = tmpGrid[i*packedsize+index];
  }

  checkTargetError("copyFromTargetPointerMap");
  return;

}
//
__targetHost__ void copyToTargetPointerMap3D(double *targetData,const double* data, int extents[3], size_t nfields, int includeNeighbours, void** ptrarray){

  size_t nsites=extents[0]*extents[1]*extents[2];

  int i;
  int index;

  int* fullindex = iwork;
  int* fullindex_d = iwork_d;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;

  char* siteMap = cwork; 
  char zero=0;
  memset(siteMap,zero,nsites);

  int coords[3];

  //get compression mapping

    if (includeNeighbours){
    int p;
    for (i=0; i<nsites; i++){
      if(ptrarray[i]){
	
	
	targetCoords3D(coords,extents,i);
	for (p=0;p<19;p++){
	  
	  int shiftIndex=targetIndex3D(coords[0]-neighb3D[p][0],coords[1]-neighb3D[p][1],coords[2]-neighb3D[p][2],extents);
	  
	  if ((shiftIndex >= 0) &&   (shiftIndex < nsites)){
	    siteMap[shiftIndex]=1;
	  }
	  
	}
	
      }
      
    }
    }
    else{
      for (i=0; i<nsites; i++){
	if(ptrarray[i]){ 
	  siteMap[i]=1;
	}
      }
    }

  int j=0;
  for (i=0; i<nsites; i++){
    if(siteMap[i]){ 
       j++;
    }
  }

  
  int packedsize=j;
  
  j=0;
  for (i=0; i<nsites; i++){
    if(siteMap[i]){

      fullindex[j]=i;
      j++;

    }
    
  }


  //copy compresssion info to GPU
  cudaMemcpy(fullindex_d, fullindex, packedsize*sizeof(int), cudaMemcpyHostToDevice);



    
  //  compress grid
        
  for (index=0; index<packedsize; index++){
    for (i=0;i<nfields;i++)  
      tmpGrid[i*packedsize+index]=data[i*nsites+fullindex[index]];
  }
  
  

  //put compressed grid on GPU
  cudaMemcpy(tmpGrid_d, tmpGrid, packedsize*nfields*sizeof(double), cudaMemcpyHostToDevice); 

  //uncompress grid on GPU
  int nblocks=(packedsize+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(targetData,tmpGrid_d,nsites,
						    nfields,
						    fullindex_d, packedsize, 1);
  cudaThreadSynchronize();

  checkTargetError("copyToTargetPointerMap");
  
  return;
  
}




//
__targetHost__ void copyToTargetMaskedAoS(double *targetData,const double* data,size_t nsites,
			size_t nfields,char* siteMask){


  int i;
  int index;

  int* fullindex = iwork;
  int* fullindex_d = iwork_d;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;

  //get compression mapping
  int j=0;
  for (i=0; i<nsites; i++){
    if(siteMask[i]){
      fullindex[j]=i;
      j++;
    }
    
  }

  int packedsize=j;


  //copy compresssion info to GPU
  cudaMemcpy(fullindex_d, fullindex, packedsize*sizeof(int), cudaMemcpyHostToDevice);


    
  //compress grid
  for (i=0;i<nfields;i++)
    {
      
      j=0;
      for (index=0; index<nsites; index++){
	if(siteMask[index]){
	  //tmpGrid[i*packedsize+j]=data[i*nsites+index];
	  tmpGrid[j*nfields+i]=data[index*nfields+i];
	  j++;
	}
      }
      
    }


  //put compressed grid on GPU
  cudaMemcpy(tmpGrid_d, tmpGrid, packedsize*nfields*sizeof(double), cudaMemcpyHostToDevice); 



  //uncompress grid on GPU
  int nblocks=(packedsize+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_AoS_d<<<nblocks,DEFAULT_TPB>>>(targetData,tmpGrid_d,nsites,
						    nfields,
						    fullindex_d, packedsize, 1);
  cudaThreadSynchronize();

  
  return;
  
}

//
__targetHost__ void copyFromTargetMaskedAoS(double *data,const double* targetData,size_t nsites,
			size_t nfields,char* siteMask){




  int i;
  int index;


  int* fullindex = iwork;
  int* fullindex_d = iwork_d;

  double* tmpGrid = dwork;
  double* tmpGrid_d = dwork_d;


  //get compression mapping
  int j=0;
  for (i=0; i<nsites; i++){
    if(siteMask[i]){
      fullindex[j]=i;
      j++;
    }
    
  }

  int packedsize=j;

  //copy compresssion info to GPU
  cudaMemcpy(fullindex_d, fullindex, packedsize*sizeof(int), cudaMemcpyHostToDevice);

  
  //compress grid on GPU
  int nblocks=(packedsize+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_AoS_d<<<nblocks,DEFAULT_TPB>>>(tmpGrid_d,targetData,nsites,
						    nfields,
						    fullindex_d, packedsize, 0);
  cudaThreadSynchronize();

  //get compressed grid from GPU
  cudaMemcpy(tmpGrid, tmpGrid_d, packedsize*nfields*sizeof(double), cudaMemcpyDeviceToHost); 

    
  //expand into final grid
  for (i=0;i<nfields;i++)
    {
      
      j=0;
      for (index=0; index<nsites; index++){
	if(siteMask[index]){
	  //data[i*nsites+index]=tmpGrid[i*packedsize+j];	
	  data[index*nfields+i]=tmpGrid[j*nfields+i];	
	  j++;
	}
      }
      
    }


  return;

}


// The targetSynchronize function is used to block until 
// the preceding __targetLaunch__ has completed.
__targetHost__ void targetSynchronize(){
  cudaThreadSynchronize();
  checkTargetError("syncTarget");
  return;
}

//The targetFree function deallocates memory on the target.
__targetHost__ void targetFree(void *ptr){
  
  cudaFree(ptr);
  checkTargetError("targetFree");
  return;
  
}


//
__global__ void zero_array(double* array,size_t size){

  int threadIndex;


  threadIndex = blockIdx.x*blockDim.x+threadIdx.x;


  if (threadIndex < size)
    array[threadIndex]=0.;
  


  return;

}

//
void targetZero(double* array,size_t size){

  int nblocks=(size+DEFAULT_TPB-1)/DEFAULT_TPB;
  zero_array<<<nblocks,DEFAULT_TPB>>>(array,size);
  cudaThreadSynchronize();



}


//
__targetHost__ void checkTargetError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) 
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
				cudaGetErrorString( err) );
		fflush(stdout);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}                         
}
