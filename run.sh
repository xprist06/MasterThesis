#!/bin/bash
#PBS -q qnvidia
#PBS -N nasprist
#PBS -J 1-2
#PBS -l select=1:ncpus=24:mpiprocs=1:ompthreads=24,walltime=24:00:00
#PBS -A OPEN-17-39

cd /home/mrazek/2021_pristas/src

ml Anaconda3/2019.10 CUDA
source activate tf-gpu-2.4.1-pristas

id=$(expr ${PBS_ARRAY_INDEX} - 1 )

cmd="python3 main.py -p 15 -g 15 -m 0.15 --phases=3 --modules=6"
for i in $(seq 0 3); do
    d=$(date +"%Y%d%m_%H%M")
    CUDA_VISIBLE_DEVICES=$i $cmd > log/log_${id}_${d}_${i}.std.log 2>log/log_${id}_${d}_${i}.err.log &
    pids[${i}]=$!
done

# wait for all pids
for pid in ${pids[*]}; do
    echo "wait for $pid" 
    wait $pid
done
