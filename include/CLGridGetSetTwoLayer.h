
/*
 * Companion file to GridGetSetTwoLayer.hpp
 *
 * Same functionality, real functions for including in main 
 */

size_t grid_data_size = sizeof(grid_data);
const char* includeToRead = "../../include/grid_structs_twolayer.h";

void addToPointWithIndex(int idx,  grid_data *grid, value gVal)
{
    grid->data[idx].val += gVal;
}

value getValueAtPointWithIndex(int idx,  grid_data *grid)
{
    return grid->data[idx].val; 
}

void setPointWithIndex(int idx,  grid_data *grid, value gVal)
{
    grid->data[idx].val = gVal;
}
