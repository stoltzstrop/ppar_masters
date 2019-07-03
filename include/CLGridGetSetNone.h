
/*
 * Companion file to GridGetSetNone.hpp
 *
 * Same functionality, real functions for including in main 
 */

size_t grid_data_size = VOLUME*sizeof(grid_data); 
const char* includeToRead = "../../include/grid_structs_none.h";

void addToPointWithIndex(int idx,  grid_data *grid, value gVal)
{
    grid[idx] += gVal;
}

value getValueAtPointWithIndex(int idx,  grid_data *grid)
{
    return grid[idx]; 
}

void setPointWithIndex(int idx,  grid_data *grid, value gVal)
{
    grid[idx] = gVal;
}
