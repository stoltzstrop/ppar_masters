#ifndef GRID_STRUCTS_H
#define GRID_STRUCTS_H
/*
 * grid_data and grid_data_value definitions for NONE layout 
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
} grid_data_value;

typedef value grid_data;


#endif
