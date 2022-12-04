#!/bin/bash
#SBATCH --time=00-12:00:00          # max walltime, hh:mm:ss
#SBATCH --nodes 1                   # Number of nodes to request
#SBATCH --gpus-per-node=p100:4      # Number of GPUs per node to request
#SBATCH --tasks-per-node=4          # Number of processes to spawn per node
#SBATCH --cpus-per-task=6           # Number of CPUs per GPU
#SBATCH --mem=32G                   # Memory per node
#SBATCH --output../../logs/%x_%A-%a_%n-%t.out
                                    # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                                    # Note: You must manually create output directory "logs" before launching job.
#SBATCH --job-name=mae_complexity
#SBATCH --account=rrg-ttt			# Use default account

GPUS_PER_NODE=4

# Exit if any command hits an error
set -e

#Store the time at which the script was launched
start_time="$SECONDS"

# Multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export MASTER_ADDR=$(hostname -s)  # Store the master node’s IP address in the MASTER_ADDR environment variable.
export MAIN_HOST="$MASTER_ADDR"

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# Get the address of an open socket
source "../../slurm/get_socket.sh"

srun python ../../main.py \
    --method "mae" \
	--train_cfg "mae.cfg" \
	--aug_stack_cfg "mae_cifar.cfg" \
    --filter_cfg "machine_animal.cfg" \
	--nodes $SLURM_JOB_NUM_NODES \
	--gpus $GPUS_PER_NODE \
	--name "mae_pre-train"
	