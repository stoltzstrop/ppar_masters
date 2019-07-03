// sum source and update at receiver kernel 

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
