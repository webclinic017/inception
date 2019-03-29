#!/bin/bash
PATH=/home/$(whoami)/miniconda3/envs/inception/bin:$PATH
p=/home/$(whoami)/inception
cd $p

model=$1
option=$2
if [ "$model" = "macro" ]; then
    time python3 $p/macro_ML.py $option
elif [ "$model" = "micro" ]; then
    time python3 $p/micro_ML.py $option
else 
    echo "Error: Model/option not found"
fi
