#!/bin/bash

#SBATCH --job-name=multigpu
#SBATCH -D /home/nct01/nct01328/transformers-in-supercomputers
#SBATCH --output=/home/nct01/nct01328/transformers-in-supercomputers/reports/R-%x.%j.out
#SBATCH --error=/home/nct01/nct01328/transformers-in-supercomputers/reports/R-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=160         # number of cores per tasks
#SBATCH --time=01:59:00             # maximum execution time (HH:MM:SS)
#SBATCH --exclusive
#SBATCH --qos=debug

cat /home/nct01/nct01328/transformers-in-supercomputers/submit-multigpu.sh

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
######################

export PYTHON_FILE="/home/nct01/nct01328/transformers-in-supercomputers/training_metric.py"

ngpusList=1,2,4
batchsizeList=64,128,256

  for bs in ${batchsizeList//,/ }
do 
  for n_gpu in ${ngpusList//,/ }
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
    torchrun --nproc_per_node $n_gpu $PYTHON_FILE $PYTHON_ARGS
done