#!/bin/bash

# this script runs the unit tests on the last version of the code run (found in the data directory)

# setup directories  
DIR=~/workspace/phd/room_code
DATA_DIR=$DIR/data

TESTING_DIR=$DIR/testing
COMPARE_DIR=$TESTING_DIR/compareData

COMPARE_BIN=$TESTING_DIR/compareValuesBin

# the correct binary files for comparison are pulled out using these strings 
ROOM_STR="room"
REC_STR="receiver"
COMPARE_WITH_STR="cuda-"

# get information from the last runs done 
LATEST=$(ls -t $DATA_DIR | head -1)

LATEST_ROOM_RUN=$DATA_DIR/$LATEST/$(ls -t $DATA_DIR/$LATEST | grep bin$ | grep $ROOM_STR | head -1)
LATEST_REC_RUN=$DATA_DIR/$LATEST/$(ls -t $DATA_DIR/$LATEST | grep bin$ | grep $REC_STR | head -1)

# pull out room and sample sizes from latest run
LATEST_ROOM_SIZE=$(echo $LATEST_ROOM_RUN | awk -F'-' '{ print $4 }')
LATEST_REC_SIZE=$(echo $LATEST_REC_RUN | awk -F'-' '{ print $4 }')

LATEST_ROOM_SAMPLES=$(echo $LATEST_ROOM_RUN | awk -F'-' '{ print $NF }')
LATEST_REC_SAMPLES=$(echo $LATEST_REC_RUN | awk -F'-' '{ print $NF }')

# find the appropriate comparison data in the testing directory to match the room and sample size
LATEST_ROOM_COMPARE=$COMPARE_DIR/$(ls $COMPARE_DIR | grep $LATEST_ROOM_SIZE | grep $LATEST_ROOM_SAMPLES | grep $ROOM_STR | grep $COMPARE_WITH_STR | head -1)
LATEST_REC_COMPARE=$COMPARE_DIR/$(ls $COMPARE_DIR | grep $LATEST_REC_SIZE | grep $LATEST_REC_SAMPLES | grep $REC_STR | grep $COMPARE_WITH_STR | head -1)

# print output of running the unit tests (one run on the room data and one run on the receiver data)
printf "***COMPARING ROOM DATA *** \n %s \n WITH \n %s \n" $LATEST_ROOM_RUN  $LATEST_ROOM_COMPARE
echo "Result is....."
$COMPARE_BIN $LATEST_ROOM_RUN $LATEST_ROOM_COMPARE

printf "*** COMPARING RECEIVER DATA ***  \n %s \n WITH \n %s \n" $LATEST_REC_RUN  $LATEST_REC_COMPARE
echo "Result is....."
$COMPARE_BIN $LATEST_REC_RUN $LATEST_REC_COMPARE

