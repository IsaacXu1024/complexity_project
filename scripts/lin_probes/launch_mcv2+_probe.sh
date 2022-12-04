#!/bin/bash
#SBATCH --time=00-02:30:00          # max walltime, hh:mm:ss
#SBATCH --nodes 1                   # Number of nodes to request
#SBATCH --gpus-per-node=p100:4      # Number of GPUs per node to request
#SBATCH --tasks-per-node=4          # Number of processes to spawn per node
#SBATCH --cpus-per-task=6           # Number of CPUs per GPU
#SBATCH --mem=32G                   # Memory per node
#SBATCH --output=../../logs/%x_%A-%a_%n-%t.out
                                    # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                                    # Note: You must manually create output directory "logs" before launching job.
#SBATCH --job-name=mcv2+_lp
#SBATCH --account=rrg-ttt			# Use default account

GPUS_PER_NODE=4

# Exit if any command hits an error
set -e

#Store the time at which the script was launched
start_time="$SECONDS"
echo $start_time

# Multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MASTER_ADDR=$(hostname -s)  # Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_ADDR
export MAIN_HOST="$MASTER_ADDR"

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# Get the address of an open socket
source "../../slurm/get_socket.sh"

declare -a filters_array=( \
	"machine_animal.cfg" \
	"only_airplane.cfg" \
	"only_automobile.cfg" \
	"only_cat.cfg" \
	"only_dog.cfg" \
	"airplane-cat_automobile-dog.cfg" \
	"airplane-dog_automobile-cat.cfg" \
	"individual.cfg" \
	)

for filter in "${filters_array[@]}"
do
	echo "Filter = ${filter}"
	srun python ../../main.py \
		--ckpt_path "../../checkpoints/mocov2+_pre-train/1dmqulbe/mocov2+_pre-train-1dmqulbe-ep=199.ckpt" \
		--train_cfg "lp_resnet18.cfg" \
		--aug_stack_cfg "lp.cfg" \
		--filter_cfg "$filter" \
		--nodes "$SLURM_JOB_NUM_NODES" \
		--gpus "$GPUS_PER_NODE" \
		--name "mcv2+_resnet_lp_${filter}" \
		--filter_train true \
		--filter_test true
done