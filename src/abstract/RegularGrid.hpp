/*
 * Definitions in this file are specific to the DEFAULT implementation
 * As defined in the SimpleGridData class 
 *
 * So if no optimisation is defined to pull in another file, then
 * this one is used by default
 * 
 */

std::string GetSetupFunction(){ return "#define SETUP(idx,gridInput)"; }

std::string GetBreakdownFunction(){ return "#define BREAKDOWN";}

std::string GetZIdxFunction(){ return "int getKernelZIdx() { return get_global_id(2); }"; }

std::string GetIsOnGridFunction(){ return "\
    int isOnGrid(int idxX, int idxY, int idxZ)\
    {\
        return ( (idxX>0) && (idxX<(getXDim()-1)) && (idxY>0) && (idxY<(getYDim()-1)) && idxZ>0 && (idxZ<(getZDim()-1)) );\
}"; }


std::string GetOptimiseFunction(){ return "#define OPTIMISE";}

std::string GetNeighbourSumFunctionMacro(){ return "#define GETNEIGHBOURSUM(idx, gridInput, sum)\
{\
    sum = getValueAtPointWithIndex(idx-1,gridInput)+getValueAtPointWithIndex(idx+1,gridInput)+getValueAtPointWithIndex(idx-getXDim(),gridInput)+getValueAtPointWithIndex(idx+getXDim(),gridInput)+getValueAtPointWithIndex(idx-getArea(),gridInput)+getValueAtPointWithIndex(idx+getArea(),gridInput);  \
}"; }
