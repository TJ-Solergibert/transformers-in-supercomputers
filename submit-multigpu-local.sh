export PYTHON_FILE="/workspace/transformers-in-supercomputers/training_metric_local.py"

ngpusList=1
batchsizeList=64,128,256
precisionList=fp16,bf16

  for prec in ${precisionList//,/ }
do
  for bs in ${batchsizeList//,/ }
do 
  for n_gpu in ${ngpusList//,/ }
do
    export PYTHON_ARGS=" \ 
        --mixed_precision $prec \ 
        --batch_size $bs \
        --eval_batch_size 512 \
        --num_epochs 5 \
        --model_name distilbert-base-uncased \
        --learning_rate 5e-4 \
        --dataset dair-ai/emotion \
        "
    torchrun --nproc_per_node $n_gpu $PYTHON_FILE $PYTHON_ARGS
done
done
done