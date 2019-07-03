#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

/*
 * same timing call as for stream benchmark
 * https://www.cs.virginia.edu/stream/
 */

double getTime() // same timing call as for stream
{
        struct timeval tp;
        int i;

        i = gettimeofday(&tp,NULL);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

