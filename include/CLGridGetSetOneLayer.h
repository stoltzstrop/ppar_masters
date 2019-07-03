
/*
 * Companion file to GridGetSetOneLayer.hpp
 *
 * Same functionality, real functions for including in main 
 */

size_t grid_data_size = sizeof(grid_data);
const char* includeToRead = "../../include/grid_structs_onelayer.h";

void addToPointWithIndex(int idx,  grid_data *grid, value gVal)
{
    grid->data[idx] += gVal;
}

value getValueAtPointWithIndex(int idx,  grid_data *grid)
{
    return grid->data[idx]; 
}

void setPointWithIndex(int idx,  grid_data *grid, value gVal)
{
    grid->data[idx] = gVal;
}
