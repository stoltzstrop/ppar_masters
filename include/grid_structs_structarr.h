#ifndef GRID_STRUCTS_H
#define GRID_STRUCTS_H
/*
 * grid_data and grid_data_value definitions for STRUCTARR layout 
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


typedef grid_data_value grid_data;

#endif
