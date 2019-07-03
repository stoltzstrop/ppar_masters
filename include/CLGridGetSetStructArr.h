
/*
 * Companion file to GridGetSetStructarr.hpp
 *
 * Same functionality, real functions for including in main 
 */

size_t grid_data_size = VOLUME*sizeof(grid_data_value);
const char* includeToRead = "../../include/grid_structs_structarr.h";

void addToPointWithIndex(int idx,  grid_data *grid, value gVal)
{
    grid[idx].val += gVal;
}

value getValueAtPointWithIndex(int idx,  grid_data *grid)
{
    return grid[idx].val; 
}

void setPointWithIndex(int idx,  grid_data *grid, value gVal)
{
    grid[idx].val = gVal;
}
