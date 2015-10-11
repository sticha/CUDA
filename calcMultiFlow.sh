#!/bin/bash

# Takes all images with the specified name convention and calculates the flow
# between every two following images.
# Images will be placed in the output folder in the current directory!

COUNTER=1
INPUT_DIR=$(readlink -f frames)

if [ $# -lt "1" ]
then
	echo "ERROR: Wrong amount of arguments"
	echo "Usage: $0 <amount input images> [path to input directory]"
	exit 1
fi

if [ $# -eq "2" ]
then
	echo "change input dir"
	INPUT_DIR=$(readlink -f $2)
fi

mkdir -p ./output/
cd output/

while [ $COUNTER -lt $1 ]
do
	../SuperResolution -path $INPUT_DIR/ -digits 4 -start $COUNTER -name img- -type png
	let COUNTER=COUNTER+1
done
