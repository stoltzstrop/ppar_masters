/*
 * Definitions in this file are specific to the LOCAL implementation 
 * As defined in the SimpleGridData class 
 */


std::string GetSetupFunction(){ return "#define SETUP(idx,gridInput) \ 
	idx = Nx*Ny +(Y*Nx+X);\
        __local double localGrid1[Bx+2][By+2];\
        double cpAreaP = 0.0; \
        double cpArea = getValueAtPointWithIndex(idx,gridInput);\ 
        double cpAreaM = 0.0;\
        for(Z=1;Z<(Nz-1);Z++){\
"; }


std::string GetBreakdownFunction(){ return "#define BREAKDOWN \ 
    cpAreaM = cpArea; cpArea = cpAreaP; barrier(CLK_LOCAL_MEM_FENCE); }\
"; }

std::string GetZIdxFunction(){ return "int getKernelZIdx() { return 0; }"; }

std::string GetIsOnGridFunction(){ return "\
    int isOnGrid(int idxX, int idxY, int idxZ)\
    {\
        return ( (idxX>0) && (idxX<(getXDim()-1)) && (idxY>0) && (idxY<(getYDim()-1)) );\
    }"; }

std::string GetOptimiseFunction(){ return "#define OPTIMISE\
                cpAreaP=getValueAtPointWithIndex(idx+getArea(),gridTS1);\
       	        int localX = get_local_id(0); \
                int localY = get_local_id(1); \
                localX++;\
                localY++;\
                localGrid1[localX][localY] = cpArea; \ 
                if((localX==1) && !(X==0)){ \
                    localGrid1[localX-1][localY]=getValueAtPointWithIndex(idx-1,gridTS1);\
                }\
                if((localX==Bx) && !(X==(Nx-1))){ \
                    localGrid1[localX+1][localY]=getValueAtPointWithIndex(idx+1,gridTS1);\
                }\
                if((localY==1) && !(Y==0)){ \
                    localGrid1[localX][localY-1]=getValueAtPointWithIndex(idx-Nx,gridTS1);\
                }\
                if((localY==By) && !(Y==(Ny-1))){ \
                    localGrid1[localX][localY+1]=getValueAtPointWithIndex(idx+Nx,gridTS1);\
                }\
           barrier(CLK_LOCAL_MEM_FENCE);\
";}

std::string GetNeighbourSumFunctionMacro(){ return "#define GETNEIGHBOURSUM(idx, gridInput, sum)\
{\
                sum = localGrid1[localX-1][localY]+localGrid1[localX+1][localY]+localGrid1[localX][localY-1]+localGrid1[localX][localY+1]+cpAreaP+cpAreaM;\
}"; }
