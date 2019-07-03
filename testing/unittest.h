#include <stdio.h>
#include "constants.h"

// outline helper unit test functions, defined in unittest.c

// make things easier with a boolean enum
typedef enum { false, true } bool;

int compareDouble(value val1, value val2, value delta);

int compareValues(value* data1, value* data2, int n1, int n2);

int getSize(const char filename[]);

void readDataIn(value* data, const char* filename, int lines);
