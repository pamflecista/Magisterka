#!/usr/bin/env bash

usage="usage: $(basename "$0") [-h] [--run NUM] NUMBER *PARAMS

train many networks in a row

positional arguments:
   NUMBER               number of networks to train
   *PARAMS              parameters for the train.py script

optional arguments:
   -h, --help           show this help message and exit
   --run                number from which run numbers should start, default: 1
   "

if [ "$1" != '--help' ] | [ "$1" != '-h' ]; then
  number=$1
  shift
fi
run=1
arguments=()

while [ "$1" != "" ]; do
    case $1 in
        --run )           shift
                          run=$1 ;;
        -h | --help )     echo "$usage"
                          python3 train.py --help
                          exit ;;
        * )               arguments+=( $1 )
    esac
    shift
done

j=1
for i in $(seq "$run" 1 $(("$run"+"$number"-1)))
do
  printf "\nTRAINING NETWORK %d / %d\n" "$j" "$number"
  python3 train.py "${arguments[@]}" --run "$i"
  j=$(("$j"+1))
done
