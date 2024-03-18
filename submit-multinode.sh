#!/bin/bash

#SBATCH --job-name=multinode
#SBATCH -D /home/nct01/nct01328/transformers-in-supercomputers
#SBATCH --output=/home/nct01/nct01328/transformers-in-supercomputers/reports/R-%x.%j.out
#SBATCH --error=/home/nct01/nct01328/transformers-in-supercomputers/reports/R-%x.%j.err
#SBATCH --nodes=4                   # number of Nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=160         # number of cores per tasks
#SBATCH --time=01:59:00             # maximum execution time (HH:MM:SS)
#SBATCH --exclusive
#SBATCH --qos=debug

cat /home/nct01/nct01328/transformers-in-supercomputers/submit-multinode.sh

######################
### Set enviroment ###
######################
module purge
module load anaconda3/2020.02 cuda/10.2 cudnn/8.0.5 nccl/2.9.9 arrow/7.0.0 openmpi
source activate /home/nct01/nct01328/pytorch_antoni_local

export HF_HOME=/gpfs/projects/nct01/nct01328/
export HF_LOCAL_HOME=/gpfs/projects/nct01/nct01328/HF_LOCAL
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/nct01/nct01328/transformers-in-supercomputers:$PYTHONPATH 
export GPUS_PER_NODE=4
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "NODES: $SLURM_NNODES"
######################

export LAUNCHER=" \
  torchrun --nnodes $SLURM_NNODES \
  --nproc_per_node $GPUS_PER_NODE \
  --rdzv_id $RANDOM \
  --rdzv_backend c10d \
  --rdzv_endpoint $head_node_ip:29500 \
  "
export PYTHON_FILE="/home/nct01/nct01328/transformers-in-supercomputers/training_metric.py"

batchsizeList=64,128,256

  for bs in ${batchsizeList//,/ }
do  
    export PYTHON_ARGS=" \ 
        --mixed_precision fp16 \ 
        --batch_size $bs \
        --eval_batch_size 512 \
        --num_epochs 5 \
        --model_name distilbert-base-uncased \
        --learning_rate 5e-4 \
        --dataset emotion \
        "
    srun $LAUNCHER $PYTHON_FILE $PYTHON_ARGS
 
done

