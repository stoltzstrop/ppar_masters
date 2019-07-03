#include "GridData.hpp"

// here the layout definitions direct decisions

#ifdef NONE
#include "GridGetSetNone.hpp"
#endif
#ifdef FULL
#include "GridGetSetFull.hpp"
#endif
#ifdef ONELAYER
#include "GridGetSetOneLayer.hpp"
#endif
#ifdef TWOLAYER
#include "GridGetSetTwoLayer.hpp"
#endif
#ifdef TWOTWOLAYER
#include "GridGetSetTwoTwoLayer.hpp"
#endif
#ifdef STRUCTARR
#include "GridGetSetStructArr.hpp"
#endif

// only one optimisation so far 
#ifdef LOCAL
#include "LocalGrid.hpp"
#else
#include "RegularGrid.hpp"
#endif


/*
 * "Simple" grid implementation of grid data. an advanced one could be made in the same manner.
 *
 * All functions output strings because they are concatenated together to build a header file
 * for the kernel.
 * Then they get compiled as code
 * This does make it quite difficult to debug! 
 */


using namespace std;

// inherits from the parent class to ensure that necessary definitions get included 
// ie. cannot run the abstracted kernel without certain functions being defined 
class SimpleGridData : virtual public GridData
{
    public:

        SimpleGridData(){}
        
        ~SimpleGridData(){}

        std::string getInitialDeclaration()
        { return 
"value GetNeighbourSum(int idx, __global grid_data *grid);\n\n\
void CalculateStencilUpdate(int idxX, int idxY, int idxZ, __global grid_data *grid, __global grid_data *gridTS1);\n\n\
void addToPoint(int idxX, int idxY, int idxZ, __global grid_data *grid, value gVal);\n\n\
void addToPointWithIndex(int idx, __global grid_data *grid, value gVal);\n\n\
void setPoint(int idxX, int idxY, int idxZ, __global grid_data *grid, value gVal);\n\n\
void setPointWithIndex(int idx, __global grid_data *grid, value gVal);\n\n\
void setReceiverPoint(int idx, __global value *receiver, value val);\n\n\
value getValueAtPoint(int idxX, int idxY, int idxZ, __global grid_data *grid);\n\n\
value getValueAtPointWithIndex(int idx, __global grid_data *grid);\n\n\
int CalculateIndex(int x, int y, int z);\n\n\
int getSourceIndex();\n\n\
int getReceiverIndex();\n\n\
int getNumberOfNeighbours(int idxX, int idxY, int idxZ);\n\n\
int isIndexOnBoundary(int idxX, int idxY, int idxZ);\n\n\
double calculateBoundaryCondition1(__constant coeffs_type* cf_d);\n\n\
double calculateBoundaryCondition2(__constant coeffs_type* cf_d);\n\n\
double calculateBoundaryConditionChange(__constant coeffs_type* cf_d);\n\n\
void InitialiseBoundaryConditions(coeffs_type* wallCoeffs, __constant coeffs_type* cf_d);\n\n\
void UpdateBoundaryConditions(coeffs_type* wallCoeffs, __constant coeffs_type* cf_d);\n\n\
void UpdateStencil(int idx, int numNeighbours, value neighbourSum, __global grid_data *grid, __global grid_data *gridTS1, coeffs_type* wallCoeffs);\n\n\
void ApplyStencilUpdate(int idx, int numNeighbours, value neighbourSum, __global grid_data *grid, __global grid_data *gridTS1, double boundaryChange);\n\n\
int isOnGrid(int idxX, int idxY, int idxZ);\n\n";}

        
std::string getSourceIndexFunction(){ return 
"int getSourceIndex()\
{\
    return (Sz*AREA)+(Sy*Nx+Sx);\
}"; }

std::string getReceiverIndexFunction(){ return 
"int getReceiverIndex()\
{\
    return (Rz*AREA)+(Ry*Nx+Rx);\
}"; }

std::string setReceiverPointFunction(){ return 
"void setReceiverPoint(int idx, __global value *receiver, value val)\
{\
    receiver[idx] = val;\
}"; }

        std::string getNeighbourSumFunction(){ return GetNeighbourSumFunctionMacro(); }

        std::string addToPointFunction(){ return "\
void addToPoint(int idxX, int idxY, int idxZ, __global grid_data *grid, value gVal)\
{\
    int idx = CalculateIndex(idxX, idxY, idxZ);\
    addToPointWithIndex(idx, grid, gVal);\
}"; }

std::string addToPointWithIndexFunction()
{ 
            return GetAddString();
}

std::string setPointFunction(){ return "\
void setPoint(int idxX, int idxY, int idxZ, __global grid_data *grid, value gVal)\
{\
    int idx = CalculateIndex(idxX, idxY, idxZ);\
    setPointWithIndex(idx, grid, gVal);\
}"; }

std::string setPointWithIndexFunction()
{ 
    return GetSetString(); 
}
        
        
        std::string getValueAtPointFunction(){ return "\
value getValueAtPoint(int idxX, int idxY, int idxZ, __global grid_data *grid)\
{\
    int idx = CalculateIndex(idxX, idxY, idxZ);\
    return getValueAtPointWithIndex(idx, grid);\
}"; }
       
std::string getValueAtPointWithIndexFunction()
{ 
    return GetGetString(); 
}
        
        std::string CalculateIndexFunction(){ return "\
int CalculateIndex(int x, int y, int z)\
{\
    return z*getArea()+(y*getXDim()+x);\
}"; }
        
        std::string getNumberOfNeighboursFunction(){ return "\
int getNumberOfNeighbours(int idxX, int idxY, int idxZ)\
{\
    return (0||(idxX-1)) + (0||(idxX-(getXDim()-2))) + (0||(idxY-1)) + (0||(idxY-(getYDim()-2))) + (0||(idxZ-1)) + (0||(idxZ-(getZDim()-2)));\
}"; }
        
        std::string isIndexOnBoundaryFunction(){ return "\
int isIndexOnBoundary(int idxX, int idxY, int idxZ)\
{\
    int K = getNumberOfNeighbours(idxX, idxY, idxZ);\
    return (K<6);\
}"; }
        
        std::string calculateBoundaryCondition1Function(){ return "\
double calculateBoundaryCondition1(__constant coeffs_type *cf_d)\
{\
    return cf_d->loss1;\
}"; }
        
        std::string calculateBoundaryCondition2Function(){ return "\
double calculateBoundaryCondition2(__constant coeffs_type *cf_d)\
{\
    return cf_d->loss2;\
}"; }
        
        std::string calculateBoundaryConditionChangeFunction(){ return "\
double calculateBoundaryConditionChange(__constant coeffs_type *cf_d)\
{\
    return cf_d->l2;\
}"; }

        std::string InitialiseBoundaryConditionsFunction(){ return "\
void InitialiseBoundaryConditions( coeffs_type* wallCoeffs, __constant coeffs_type* originalCoeffs )\
{\
	wallCoeffs->loss1 = 1.0;\
	wallCoeffs->loss2 = 1.0;\
	wallCoeffs->l2 = originalCoeffs[0].l2;\
}"; }

        std::string UpdateBoundaryConditionsFunction(){ return "\
void UpdateBoundaryConditions( coeffs_type* wallCoeffs , __constant coeffs_type* originalCoeffs)\
{\
	wallCoeffs->loss1 = originalCoeffs->loss1;\
	wallCoeffs->loss2 = originalCoeffs->loss2;\
}"; }
        
        std::string UpdateStencilFunction(){ return "\
void UpdateStencil(int idx, int numNeighbours, value neighbourSum, __global grid_data *grid, __global grid_data *gridTS1, coeffs_type *wallCoeffs)\
{\
	value boundary1 = wallCoeffs->loss1;\
	value boundary2 = wallCoeffs->loss2;\
	value boundaryChange = wallCoeffs->l2;\
        setPointWithIndex(idx, grid, boundary1 *\
        ((2.0-numNeighbours*boundaryChange)*getValueAtPointWithIndex(idx,gridTS1) +\
        boundaryChange*neighbourSum - boundary2*getValueAtPointWithIndex(idx,grid)));\
}"; }
        
        std::string ApplyStencilUpdateFunction(){ return "\
void ApplyStencilUpdate(int idx, int numNeighbours, value neighbourSum, __global grid_data *grid, __global grid_data *gridTS1, double boundaryChange)\
{\
        setPointWithIndex(idx,grid, \
                ((2.0-numNeighbours*boundaryChange)*getValueAtPointWithIndex(idx,gridTS1) + \
                        boundaryChange*neighbourSum - getValueAtPointWithIndex(idx,grid)));\
}"; }
        
        std::string isOnGridFunction(){ return GetIsOnGridFunction(); }

        std::string getXDimFunction(){ return "int getXDim() { return Nx; }"; }
        
        std::string getYDimFunction(){ return "int getYDim() { return Ny; }"; }
        
        std::string getZDimFunction(){ return "int getZDim() { return Nz;  }"; }
        
        std::string getAreaFunction(){ return "int getArea() { return AREA; }"; }
        
        std::string getKernelXIdxFunction(){ return "int getKernelXIdx() { return get_global_id(0); }"; }
        
        std::string getKernelYIdxFunction(){ return "int getKernelYIdx() { return get_global_id(1); }"; }
        
        std::string getKernelZIdxFunction(){ return GetZIdxFunction(); }

        std::string SetupFunction(){ return GetSetupFunction(); }

        std::string BreakdownFunction(){ return GetBreakdownFunction(); }

        std::string OptimiseFunction(){ return GetOptimiseFunction(); }

};
