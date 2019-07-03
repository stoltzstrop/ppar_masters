#pragma OPENCL EXTENSION cl_khr_fp64 : enable


#define area (Nx*Ny)

// Define Source and Read
#define Sx 120
#define Sy 120
#define Sz 60

#define Rx 50
#define Ry 50
#define Rz 50


typedef struct coeffs_type
{
	double l2;
	double loss1;
	double loss2;

} coeffs_type;

__kernel void UpdateScheme(__global double* restrict u,
                           __global double* restrict u1, 
                           __constant struct coeffs_type* cf_d)
{
	
	// get X,Y,Z from thread and block Id's
	int X = get_global_id(0); 
	int Y = get_global_id(1); 
	int Z = get_global_id(2);

	
	// Test that not at halo, Z block excludes Z halo
        if( (X>2) && (X<(Nx-3)) && (Y>2) && (Y<(Ny-3)) && (Z>2) && (Z<(Nz-3)) ){
		
		// get linear position
		int cp   = Z*area+(Y*Nx+X);
		
		
            u[cp] =  15.0f/68.0f/180.0f*(270.0f*(u1[cp+1]+u1[cp-1]+u1[cp+Nx]+u1[cp-Nx]+u1[cp+area]+u1[cp-area])-27.0f*(u1[cp+2]+u1[cp-2]+u1[cp+(2*Nx)]+u1[cp-(2*Nx)]+u1[cp+(2*area)]+u1[cp-(2*area)])+2.0*(u1[cp+3]+u1[cp-3]+u1[cp+(3*Nx)]+u1[cp-(3*Nx)]+u1[cp+(3*area)]+u1[cp-(3*area)])-3.0f*490.0*u1[cp])+2.0f*u1[cp]-u[cp];
	}
	
}

__kernel void inout(__global double *u,
                    __global double *out,
                    const double ins,
                    const int n)
{	
	// sum in source
	u[(Sz*area)+(Sy*Nx+Sx)] += ins;

	// non-interp read out
	out[n]  = u[(Rz*area)+(Ry*Nx+Rx)];
	
}
