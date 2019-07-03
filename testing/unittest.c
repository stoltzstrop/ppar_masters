#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "unittest.h"
#include "constants.h"

// compare two doubles against a percentile
int compareDouble(value val1, value val2, value percentile)
{
    if(fabs(val2) > 0)
    {
        return (fabs(val1-val2)/val2 < percentile);
    }
    else
    {
        return (fabs(val1-val2) < percentile);
    }

}

// compare list of doubles
int compareValues(value* data1, value* data2, int n1, int n2)
{
    value percentile = 0.000001;
    if(n1 != n2)
    {
         printf("Sizes of data files not the same\n");
         return -1;
    }
   
    int i;

    // loop over whole array of doubles and check one by one
    for(i=0; i<n1; i++)
    {
        if(!compareDouble(data1[i],data2[i],percentile))
        {
           printf("Values don't match for line %d! (%.14f != %.14f)\n", i+1, data1[i], data2[i]); 
           return false;
        }
    }
    return true;
}

// get the number of lines from a filename
int getSize(const char filename[])
{
    FILE * file;
    file = fopen(filename, "rb");
    fseek(file, 0, SEEK_END);
    int len = ftell(file);
    rewind(file);
    fclose(file);
    return len;
}

// get the number of lines from a file
int getLines(FILE* infile)
{
    int lines = 0;
    int ch = 0;

    do 
    {
         ch = fgetc(infile);
         if(ch == '\n')
             lines++;

    } while (ch != EOF);

    rewind(infile);
    return lines;
}

// read data in a file into an array
void readDataIn(value* data, const char* filename, int lines)
{
    FILE *infile = fopen(filename, "rb");
   
    int i;
    for(i=0; i<lines; i++) 
    {
        fread(&data[i], sizeof(value), 1, infile);
    }

    fclose(infile);
}
