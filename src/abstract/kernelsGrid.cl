__kernel void UpdateScheme(__global grid_data *grid,
                           __global grid_data *gridTS1, 
                           __constant coeffs_type* cf_d)
{
	
	// get X,Y,Z from thread and block Id's
	int X = getKernelXIdx(); 
	int Y = getKernelYIdx(); 
	int Z = getKernelZIdx(); 
	value cfL2 = cf_d[0].l2;

	int idx; 
    
        //macro for local variable setup 
        SETUP(idx,gridTS1) 

	// get linear position
        idx = CalculateIndex(X,Y,Z);

        // local variables
	coeffs_type wallCoeffs; 
	InitialiseBoundaryConditions(&wallCoeffs,cf_d);	
		
        value S = 0.0;
    
        //macro for local optimisations
        OPTIMISE
	// set loss coeffs at walls
	if(isOnGrid(X,Y,Z)) 
        {

            //macro for pulling out neighbours
            GETNEIGHBOURSUM(idx,gridTS1,S)
    	    if(isIndexOnBoundary(X,Y,Z))
	    {
		UpdateBoundaryConditions(&wallCoeffs,cf_d);	
	    }
        
            // Get sum of neighbours
            int K = getNumberOfNeighbours(X,Y,Z); 
        
    	    // Calc update
	    UpdateStencil(idx, K, S, grid, gridTS1, &wallCoeffs);
            //macro for cleanup 
            BREAKDOWN
        }
	
}

__kernel void inout(__global grid_data *grid,
                    __global value *out,
                    const value ins,
                    const int numSample)
{	
	// sum in source
        int sourceIdx = getSourceIndex(); 
        addToPointWithIndex(sourceIdx, grid, ins);

	// update receiver 
        int outIdx = getReceiverIndex(); 
	setReceiverPoint(numSample, out, getValueAtPointWithIndex(outIdx,grid));
}
