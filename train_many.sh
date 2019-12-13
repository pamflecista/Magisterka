#!/usr/bin/env bash

usage="usage: $(basename "$0") [-h] [-p DIR] [--run NUM] NUMBER DATASET

train many networks in a row

positional arguments:
   NUMBER               number of networks to train
   DATASET              name of the dataset which should be used for training

optional arguments:
   -h, --help           show this help message and exit
   -p, --path           working directory
   --run                number from which run numbers should start, default: 1
   "

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  echo "$usage"
  exit
fi

number=$1
dataset=$2
run=1
shift
shift
while [ "$1" != "" ]; do
    case $1 in
        -p | --path )           shift
                                path=$1
                                ;;
        --run )                 shift
                                run=$1
                                ;;
        -h | --help )           echo "$usage"
                                exit
                                ;;
        * )                     echo "$usage"
                                exit 1
    esac
    shift
done


j=1
for i in $(seq "$run" 1 $(("$run"+"$number"-1)))
do
  printf "\nTRAINING NETWORK %d / %d\n" "$j" "$number"
  python3 train.py $dataset --path $path --run $i
  j=$(("$j"+1))
done
