#ifndef GRID_STRUCTS_H
#define GRID_STRUCTS_H

/*
 * original test file for various grid_data layouts
 */

typedef struct coeffs_type
{
	value l2;
	value loss1;
	value loss2;

} coeffs_type;

typedef struct grid_data_value // point in grid
{
    value val;
} grid_data_value;

typedef struct grid_data // the grid grid
{
        value data[VOLUME];
} grid_data;

#endif
