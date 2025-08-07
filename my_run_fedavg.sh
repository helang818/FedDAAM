#!/usr/bin/env bash

CLIENT_NUM=$1
WORKER_NUM=$2
ROUND=$3
EPOCH=$4

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

#hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./myMainFedavg.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_config_5_3" \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
