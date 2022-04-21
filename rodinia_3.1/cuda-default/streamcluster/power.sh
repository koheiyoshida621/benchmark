#!/bin/bash
## Rodinia Benchmarkで電力測定をするためのプログラム
## usage : power.sh GPUnum Progname
FILENAME=`hostname`"_GPU"$2"_"$(basename `pwd`)"_"$3
export CUDA_VISIBLE_DEVICES=$2
nvidia-smi --query-gpu=uuid,timestamp,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,temperature.gpu --format=csv -i $2 > ../$FILENAME
nvidia-smi --query-gpu=uuid,timestamp,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,temperature.gpu --format=csv -i $2 -lms 5 >> ../$FILENAME &
sleep 1
$1
sleep 1
kill $!
echo "End $2"
