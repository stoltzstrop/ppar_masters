/*
 * Definitions in this file are specific to the NONE data layout  
 * as defined in grid_structs_none.h
 *
 * Provides the default definition for 
 * - getting a point from the grid_data object
 * - setting a point on a grid_data object 
 * - adding to a point on a grid_data object
 *
 * all take in an index of the grid point and a pointer to the grid_data object
 */

std::string GetAddString()
{ 
    return "\
void addToPointWithIndex(int idx, __global grid_data *grid, value gVal)\
{\
    grid[idx] += gVal;\
}"; 
}

std::string GetSetString()
{ 
    return "\
void setPointWithIndex(int idx, __global grid_data *grid, value gVal)\
{\
    grid[idx] = gVal;\
}"; 
}
        
        
std::string GetGetString()
{ 
    return "\
    value getValueAtPointWithIndex(int idx, __global grid_data *grid)\
    {\
        return grid[idx]; \
    }"; 
}
