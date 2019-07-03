#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define area (Nx*Ny)


// Define Source and Read
#define Sx 120
#define Sy 120
#define Sz 60

#define Rx 50
#define Ry 50
#define Rz 50


// must define same boundary conditions struct 
typedef struct coeffs_type
{
	double l2;
	double loss1;
	double loss2;

} coeffs_type;

__kernel void UpdateScheme(__global double*  u,
                           __global double*  u1, 
                           __constant struct coeffs_type* cf_d)
{
	
	// get X,Y,Z from thread and block Id's
	int X = get_global_id(0); 
	int Y = get_global_id(1); 
	int Z = get_global_id(2);

	
	// Test that not at halo, Z block excludes Z halo
	if( (X>0) && (X<(Nx-1)) && (Y>0) && (Y<(Ny-1)) && (Z<(Nz-1)) && (Z>0) ){
		
		// get linear position
		int cp   = Z*area+(Y*Nx+X);
		
		// local variables
		double cf  = 1.0;
		double cf2 = 1.0;
		
		int K    = (0||(X-1)) + (0||(X-(Nx-2))) + (0||(Y-1)) + (0||(Y-(Ny-2))) + (0||(Z-1)) + (0||(Z-(Nz-2)));
		
		// set loss coeffs at walls
		if(K<6){
			cf   = cf_d[0].loss1;
			cf2  = cf_d[0].loss2;
		}
        
		// Get sum of neighbours
        double S   = u1[cp-1]+u1[cp+1]+u1[cp-Nx]+u1[cp+Nx]+u1[cp-area]+u1[cp+area];
        
        // Calculate room update
        u[cp]    = cf*( (2.0-K*cf_d[0].l2)*u1[cp] + cf_d[0].l2*S - cf2*u[cp] );
		
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
