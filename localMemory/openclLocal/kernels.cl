#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define Nx 256
#define Ny 256
#define Nz 202

/*
#define Nx 512
#define Ny 512
#define Nz 404 
*/
#define area (Nx*Ny)


// Define Source and Read
#define Sx 120
#define Sy 120
#define Sz 60

#define Rx 50
#define Ry 50
#define Rz 50

#define BLOCK_X 64
#define BLOCK_Y 8
#define Bz 1


typedef struct coeffs_type
{
	double l2;
	double loss1;
	double loss2;

} coeffs_type;

__kernel void UpdateScheme(__global double *u,
                           __global double *u1, 
                           __constant struct coeffs_type* cf_d)
{
	
	// get X,Y from thread and block Id - tiling along Z
	int X = get_global_id(0); 
	int Y = get_global_id(1); 


	int cp = area+(Y*Nx+X);


        double sum = 0.0; 
        double l2 = cf_d[0].l2;
        double loss1 = cf_d[0].loss1;
        double loss2 = cf_d[0].loss2;
        double cf = 1.0;
        double cf2 = 1.0; 

        
	int Z;
        __local double localGrid1[BLOCK_X+2][BLOCK_Y+2];
        double cpAreaP = 0.0; 
        double cpArea = u1[cp]; 
        double cpAreaM = 0.0; 
        
        // loop over the Z index 
        for(Z=1;Z<(Nz-1);Z++)
        {

           cp = Z*area+(Y*Nx+X);
                
                cpAreaP=u1[cp+area];

                // setup local memory XY grid 
       	        int localX = get_local_id(0); 
                int localY = get_local_id(1); 
    
                localX++;
                localY++;

                localGrid1[localX][localY] = cpArea;  

                if((localX==1) && !(X==0)){ 
                    localGrid1[localX-1][localY]=u1[cp-1];
                }
                if((localX==BLOCK_X) && !(X==(Nx-1))){ 
                    localGrid1[localX+1][localY]=u1[cp+1];
                }

                if((localY==1) && !(Y==0)){ 
                    localGrid1[localX][localY-1]=u1[cp-Nx];
                }

                if((localY==BLOCK_Y) && !(Y==(Ny-1))){ 
                    localGrid1[localX][localY+1]=u1[cp+Nx];
                }

               barrier(CLK_LOCAL_MEM_FENCE);

                if( (X>0) && (X<(Nx-1)) && (Y>0) && (Y<(Ny-1))){
		// get linear position
                    sum = localGrid1[localX-1][localY]+localGrid1[localX+1][localY]+localGrid1[localX][localY-1]+localGrid1[localX][localY+1]+cpAreaP+cpAreaM;
		
		    // local variables

        	    int K    = (0||(X-1)) + (0||(X-(Nx-2))) + (0||(Y-1)) + (0||(Y-(Ny-2))) + (0||(Z-1)) + (0||(Z-(Nz-2)));
		
                    cf  = 1.0;
                    cf2 = 1.0;
                    // set loss coeffs at walls
                    if(K<6){
			cf   = loss1; 
			cf2  = loss2;
		    } 
        
                    // Calc update
                    u[cp]    = cf*( (2.0-K*cf_d[0].l2)*cpArea + cf_d[0].l2*sum - cf2*u[cp] );

                    // swap values for next iteration
                    cpAreaM = cpArea;
                    cpArea = cpAreaP;
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
	    	
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
