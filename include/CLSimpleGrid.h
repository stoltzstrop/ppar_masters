#ifndef CLSIMPLEGRID_H
#define CLSIMPLEGRID_H

#include "select.h"


value GetNeighbourSum(int idx,  grid_data *grid);
void CalculateStencilUpdate(int idxX, int idxY, int idxZ,  grid_data *grid,  grid_data *gridTS1);
void addToPoint(int idxX, int idxY, int idxZ,  grid_data *grid, value gVal);
void addToPointWithIndex(int idx,  grid_data *grid, value gVal);
void setPoint(int idxX, int idxY, int idxZ,  grid_data *grid, value gVal);
void setPointWithIndex(int idx,  grid_data *grid, value gVal);
void setReceiverPoint(int idx,  value *receiver, value val);
value getValueAtPoint(int idxX, int idxY, int idxZ,  grid_data *grid);
value getValueAtPointWithIndex(int idx,  grid_data *grid);
int CalculateIndex(int x, int y, int z);
int getSourceIndex();
int getReceiverIndex();
int getNumberOfNeighbours(int idxX, int idxY, int idxZ);
int isIndexOnBoundary(int idxX, int idxY, int idxZ);
double calculateBoundaryCondition1( grid_data *grid);
double calculateBoundaryCondition2( grid_data *grid);
double calculateBoundaryConditionChange( grid_data *grid);
double InitialiseBoundaryConditions( coeffs_type* wallCoeffs );
double UpdateBoundaryConditions( coeffs_type* wallCoeffs );
void UpdateStencil(int idx, int numNeighbours, value neighbourSum,  grid_data *grid,  grid_data *gridTS1, coeffs_type wallCoeffs);
void ApplyStencilUpdate(int idx, int numNeighbours, value neighbourSum,  grid_data *grid,  grid_data *gridTS1, double boundaryChange);
int isOnGrid(int idxX, int idxY, int idxZ);

int getXDim() { return Nx; }
int getYDim() { return Ny; }
int getZDim() { return Nz;  }
int getArea() { return AREA; }

int getSourceIndex()
{
    return (Sz*AREA)+(Sy*Nx+Sx);
}

int getReceiverIndex()
{
    return (Rz*AREA)+(Ry*Nx+Rx);
}

void setReceiverPoint(int idx,  value *receiver, value val)
{
    receiver[idx] = val;
}
value GetNeighbourSum(int idx,  grid_data *grid)
{
    value neighborSum =  0.0 ;

    neighborSum = getValueAtPointWithIndex(idx-1,grid)+getValueAtPointWithIndex(idx+1,grid)+getValueAtPointWithIndex(idx-getXDim(),grid)+getValueAtPointWithIndex(idx+getXDim(),grid)+getValueAtPointWithIndex(idx-getArea(),grid)+getValueAtPointWithIndex(idx+getArea(),grid);  
    return neighborSum;
}

void addToPoint(int idxX, int idxY, int idxZ,  grid_data *grid, value gVal)
{
    int idx = CalculateIndex(idxX, idxY, idxZ);
    addToPointWithIndex(idx, grid, gVal);
}

void setPoint(int idxX, int idxY, int idxZ,  grid_data *grid, value gVal)
{
    int idx = CalculateIndex(idxX, idxY, idxZ);
    setPointWithIndex(idx, grid, gVal);
}

value getValueAtPoint(int idxX, int idxY, int idxZ,  grid_data *grid)
{
    int idx = CalculateIndex(idxX, idxY, idxZ);
    return getValueAtPointWithIndex(idx, grid);
}
       
int CalculateIndex(int x, int y, int z)
{
    return z*getArea()+(y*getXDim()+x);
}
        
int getNumberOfNeighbours(int idxX, int idxY, int idxZ)
{
    return (0||(idxX-1)) + (0||(idxX-(getXDim()-2))) + (0||(idxY-1)) + (0||(idxY-(getYDim()-2))) + (0||(idxZ-1)) + (0||(idxZ-(getZDim()-2)));
}
        
int isIndexOnBoundary(int idxX, int idxY, int idxZ)
{
    int K = getNumberOfNeighbours(idxX, idxY, idxZ);
    return (K<6);
}
        
double calculateBoundaryCondition1(coeffs_type *cf_d)
{
    return cf_d->loss1;
}
        
double calculateBoundaryCondition2(coeffs_type *cf_d)
{
    return cf_d->loss2;
}
        
double calculateBoundaryConditionChange(coeffs_type *cf_d)
{
    return cf_d->l2;
}

void InitialiseBoundaryConditions( coeffs_type* wallCoeffs, coeffs_type* originalCoeffs )
{
	wallCoeffs->loss1 = 1.0;
	wallCoeffs->loss2 = 1.0;
	wallCoeffs->l2 = originalCoeffs[0].l2;
}

void UpdateBoundaryConditions( coeffs_type* wallCoeffs , coeffs_type* originalCoeffs)
{
	wallCoeffs->loss1 = originalCoeffs->loss1;
	wallCoeffs->loss2 = originalCoeffs->loss2;
}
        
void UpdateStencil(int idx, int numNeighbours, value neighbourSum,  grid_data *grid,  grid_data *gridTS1, coeffs_type wallCoeffs)
{
	value boundary1 = wallCoeffs.loss1;
	value boundary2 = wallCoeffs.loss2;
	value boundaryChange = wallCoeffs.l2;
        grid[idx] = boundary1 *
        ((2.0-numNeighbours*boundaryChange)*getValueAtPointWithIndex(idx,gridTS1) +
        boundaryChange*neighbourSum - boundary2*getValueAtPointWithIndex(idx,grid));
}
        
void ApplyStencilUpdate(int idx, int numNeighbours, value neighbourSum,  grid_data *grid,  grid_data *gridTS1, double boundaryChange)
{
        setPointWithIndex(idx,grid, 
                ((2.0-numNeighbours*boundaryChange)*getValueAtPointWithIndex(idx,gridTS1) + 
                        boundaryChange*neighbourSum - getValueAtPointWithIndex(idx,grid)));
}
        
int isOnGrid(int idxX, int idxY, int idxZ)
{
    return ( (idxX>0) && (idxX<(getXDim()-1)) && (idxY>0) && (idxY<(getYDim()-1)) && idxZ>0 && (idxZ<(getZDim()-1)) );
}
#endif
