#ifndef CONSTANTS_H
#define CONSTANTS_H

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef double value;

// Dimensions
/*
*/
#define Nx 256
#define Ny 256
#define Nz 202

/*
#define Nx 512
#define Ny 512
#define Nz 404 

*/
/*
#define Nx 1024
#define Ny 512
#define Nz 256
*/

#define AREA (Nx*Ny)
#define VOLUME (AREA*Nz)

// Define Thread block size
#define Bx 16
#define By 16
#define Bz 1

// Define Source and Receiver
#define Sx 120
#define Sy 120
#define Sz 60

#define Rx 50
#define Ry 50
#define Rz 50


#define MAX_STR_LEN 1024

// ------------------------------------------
// Simulation parameters					      

#define numberSamples 4410
#define dim         3
#define dur         20

#define nano        1.e-09

#define pi 3.1415926535897932384626433832795



#endif
