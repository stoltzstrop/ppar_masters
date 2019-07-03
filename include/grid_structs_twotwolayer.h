#ifndef GRID_STRUCTS_H
#define GRID_STRUCTS_H
/*
 * grid_data and grid_data_value definitions for TWOTWOLAYER layout 
 */

typedef struct coeffs_type
{
	double l2;
	double loss1;
	double loss2;

} coeffs_type;

typedef struct grid_data_value // point in grid
{
    value val;
    value val2;
} grid_data_value;

typedef struct grid_data // the grid grid
{
        grid_data_value data[VOLUME];
} grid_data;

#endif
