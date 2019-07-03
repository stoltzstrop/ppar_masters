#include <stdio.h>
#include <stdlib.h>
#include "unittest.h"
#include "constants.h"

// main function for binary executable unit test

int main(int argc, char *argv[])
{
    // reads in two files and compares their contents (should contain only lists of doubles) 
    if(argc <= 2 )
    {
    	printf("Not enough arguments!\n");
	exit(1);
    }

    // parse the arguments 
    char* file1 = argv[1];;
    char* file2 = argv[2];;

    // read the first file
    FILE *infile1 = fopen(file1,"rb");
    if(infile1 == NULL)
    {
       printf("Problem opening file \" %s \"!\n",file1);
       exit(-1);
    }
    fclose(infile1);


    // read the second file
    FILE *infile2 = fopen(file2,"rb");
    if(infile2 == NULL)
    {
       printf("Problem opening file \" %s \"!\n",file2);
       exit(-1);
    }
    fclose(infile2);

    // get sizes of files 
    int size1 = getSize(file1);
    int size2 = getSize(file2);

    // if sizes don't match, then something is wrong!
    if(size1 != size2)
    {
        printf("File sizes don't match!\n");
        exit(-1);
    }
   
   // read in data
    value* data1 = (value*)malloc(sizeof(value)*(size1+1));
    value* data2 = (value*)malloc(sizeof(value)*(size2+1));

    readDataIn(data1, file1, size1);
    readDataIn(data2, file2, size2);



    // compare the data 
    if(!compareValues(data1,data2,size1,size2))
    {
        // something isn't equal - exit
        exit(-1);
    }
    else
    {
    // otherwise test passed
      printf("Golden!\n");
    }

    // free memory used
    free(data1);
    free(data2);
    exit(0);
}


