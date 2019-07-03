#!/bin/bash

# simple iteration script to run a batch of different block configurations for GPU codes 

#header file to change
FILE=blocks.h

#specify which values of blocks to run
BLOCKX=( 2 4 8 16 32 64 128 )
BLOCKY=( 2 4 8 16 32 64 128 )
BLOCKZ=( 2 202 )

# temporary file for outputs - copied to real filename if contains valid data
tmpFile=tmp.txt


for Z in "${BLOCKZ[@]}"; do # loop over Z values

    for Y in "${BLOCKY[@]}"; do # loop over Y values

            for X in "${BLOCKX[@]}"; do # loop over X values
            
            #tell the user what we're running
            echo -e "#define BLOCK_X" $X "\n#define BLOCK_Y" $Y "\n#define BLOCK_Z" $Z "\n"
            #update the file to reflect what we're running
            echo -e "#define BLOCK_X" $X "\n#define BLOCK_Y" $Y "\n#define BLOCK_Z" $Z "\n" > $FILE
            # clean and make file 
            make clean && make
            #configure a nice output name 
            outfile="blockx_"$X"_blocky_"$Y"_blockz_"$Z".txt"
            
            ../../bin/BasicRoom > $tmpFile # output results to temp file

            # check that the file ran for more than 0 seconds (ie. is valid) and copy to good filename if so
            value=$( grep -ic "time:" $tmpFile )
            if [ $value -gt 0 ]
            then
                cp $tmpFile $outfile 
            fi
        done
    done
done

#delete any leftover temp file
rm $tmpFile
