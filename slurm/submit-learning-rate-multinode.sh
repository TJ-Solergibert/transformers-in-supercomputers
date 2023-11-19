#!/bin/bash

#SBATCH --job-name=learning-rate-multinode
#SBATCH -D /home/nct01/nct01328/transformers-in-supercomputers
#SBATCH --output=/home/nct01/nct01328/transformers-in-supercomputers/reports/R-%x.%j.out
#SBATCH --error=/home/nct01/nct01328/transformers-in-supercomputers/reports/R-%x.%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=160         # number of cores per tasks
#SBATCH --time=01:59:00             # maximum execution time (HH:MM:SS)
#SBATCH --exclusive
#SBATCH --qos=debug

cat /home/nct01/nct01328/transformers-in-supercomputers/slurm/submit-learning-rate-multinode.sh

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

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "NODES: $SLURM_NNODES"
######################

echo "####################################################################################"
echo "################################ Learning Rate Test ################################"
echo "####################################################################################"

export PYTHON_FILE="/home/nct01/nct01328/transformers-in-supercomputers/experiments/metric/training_metric.py"

export LAUNCHER=" \
  torchrun --nnodes $SLURM_NNODES \
  --nproc_per_node 4 \
  --rdzv_id $RANDOM \
  --rdzv_backend c10d \
  --rdzv_endpoint $head_node_ip:29500 \
  "

learningRates=5e-5,1e-4,5e-4
precisionList=no,fp16
export bs=256
export ebs=512

  for lr in ${learningRates//,/ }
do
  for prec in ${precisionList//,/ }
do


    echo "####################################################################################"
    echo "################ LearningRate: $lr, Precision: $prec, GPUs: $n_gpu ###############"
    echo "####################################################################################"
    export PYTHON_ARGS=" \ 
        --mixed_precision $prec \ 
        --batch_size $bs \
        --eval_batch_size $ebs \
        --learning_rate $lr \
    "
    srun $LAUNCHER $PYTHON_FILE $PYTHON_ARGS
 
done
done

