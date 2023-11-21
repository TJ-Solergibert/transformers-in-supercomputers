<h3 align="center">
<p>Training Transformers in Supercomputers with 🤗 Transformers, 🚀 Accelerate and Slurm
</h3>

- [Introduction](#introduction)
- [Experiments](#experiments)
  - [Training batch size](#training-batch-size)
  - [Mixed precision](#mixed-precision)
  - [Evaluation batch size](#evaluation-batch-size)
- [Modifications](#modifications)
  - [Distributed evaluation with a shared file system](#distributed-evaluation-with-a-shared-file-system)
  - [Broadcast the metric](#broadcast-the-metric)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction
In this project, we have studied data parallelism to accelerate the training of transformers, both with multiple GPUs on the same node and across multiple nodes. We have primarily analyzed the effect of three variables: Training batch size, mixed precision and evaluation batch size.

All experiments were conducted using PyTorch, 🤗 Transformers for the models, 🤗 Datasets for preprocessing and injecting data into the model, 🤗 Evaluate for calculating model metrics, and 🤗 Accelerate for distributing the model training across multiple devices. We conducted the experiments on MareNostrum4 - CTE Power partition at the [Barcelona Supercomputing Center](https://www.bsc.es). This supercomputer utilized Slurm for job scheduling and resource management. The cluster is composed of 54 servers, each one containing:
- 2 x IBM Power9 8335-GTG @ 3.00GHz (20 cores and 4 threads/core, total 160 threads per node)
- 4 x GPU NVIDIA V100 (Volta) with 16GB HBM2.
- 512GB of main memory distributed in 16 dimms x 32GB @ 2666MHz
- 2 x SSD 1.9TB as local storage
- 2 x 3.2TB NVME
- Single Port Mellanox EDR
- GPFS via one fiber link 10 GBit

In the execution of this project, we have primarily focused on the performance of the distributed training of the models, leaving precision or their learning capacity in the background. Nevertheless, we aimed to validate the theory by training a Distil-BERT model on the emotion dataset for a multi-class classification task. Below, we present the training configurations along with the results.

## Experiments
To conduct the study, we have relied on the traditional scheme of training a model over several epochs, in which, for each epoch, the model consumes the entire training dataset to update it's parameters and calculates a metric using the evaluation dataset. Therefore, both datasets are synthetic, and we will not consider the model metrics as our focus is on the training and communication performance. More specifically, we will focus on the throughput per GPU and how the time decreases in the training and evaluation phases as we increase the number of devices.

To do this, we will train a multi-class classification model with a dataset containing 32,768 training samples and 131,072 for evaluation. The metric we will calculate based on the model's predictions will be the accuracy.

Note that the evaluation part will be divided into three components: The evaluation batch size used by the model for making predictions on the GPU, the metric calculation performed by only one process (the _main_ or _master_ process) on the CPU, and the broadcast of the calculated metric to the rest of the processes.
### Training batch size
The training batch size refers to the number of training samples utilized in one training step. During the training, the entire dataset is divided into smaller batches, and the model is updated based on the gradients calculated from each batch. In practical terms, the choice of the batch size is a hyperparameter that may require experimentation to find the optimal value for a specific task and dataset. 

In this study, we have considered batch sizes of 64, 128, and 256, as beyond 512 we encountered Out Of Memory (OOM) errors. Whenever we refer to batch size, we will be specifying the quantity per GPU involved. That is, in the case of 16 GPUs and a batch size of 256, we will have a global batch size of 4096 samples.
### Mixed precision
Another factor we have studied is the effect of mixed precision. We will work with fp32 and fp16, as, despite Accelerate offering the ability to work with bf16 and fp8, we cannot use them because V100 GPUs do not support them.
### Evaluation batch size
To make model predictions, we don't need to calculate gradients, so we will use the `torch.no_grad()` context manager, which will save us a significant amount of VRAM. This allows us to use larger evaluation batch sizes. We have studied evaluation batch sizes from 128, 256, 512, 1024, 2048 up to 4096, the maximum allowed by the GPU before incurring in Out Of Memory (OOM) errors.

## Modifications
### Distributed evaluation with a shared file system
For computing the metrics, the 🤗 Hugging Face stack relies on a lock system to prevent conflicts. The implemented mechanism works correctly when operating in a local file system but is not designed for use in a shared file system, as is our case with GPFS. To address this issue, it is sufficient to make the following modifications to the source code of the 🤗 Datasets and 🤗 Evaluate libraries:
- Modify the `UnixFileLock` class in [`datasets/utils/filelock.py`](https://github.com/huggingface/datasets/blob/344086a7a1707ef20b57399f813ef64ce679e956/src/datasets/utils/filelock.py#L393), in order to use the `lockf` system call which works in shared file systems.

```diff
class UnixFileLock(BaseFileLock):
    """
    Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems.
    """

    def __init__(self, lock_file, timeout=-1, max_filename_length=None):
        max_filename_length = os.statvfs(os.path.dirname(lock_file)).f_namemax
        super().__init__(lock_file, timeout=timeout, max_filename_length=max_filename_length)

    def _acquire(self):
        open_mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC
        fd = os.open(self._lock_file, open_mode)

        try:
-           fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
+           fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
        else:
            self._lock_file_fd = fd
        return None

    def _release(self):
        # Do not remove the lockfile:
        #
        #   https://github.com/benediktschmitt/py-filelock/issues/31
        #   https://stackoverflow.com/questions/17708885/flock-removing-locked-file-without-race-condition
        fd = self._lock_file_fd
        self._lock_file_fd = None
-       fcntl.flock(fd, fcntl.LOCK_UN)
+       fcntl.lockf(fd, fcntl.LOCK_UN)
        os.close(fd)
        return None
```
- Modify the `_check_all_processes_locks` method in [evaluate/module.py](https://github.com/huggingface/evaluate/blob/18932858570b9fa97ac478e1e6e709438e4d093b/src/evaluate/module.py#L340), to prevent the process 0 (which is the one that coordinates the rendezvous) from checking its own lock because it doesn't works with lockf.
```diff
def _check_all_processes_locks(self):
    expected_lock_file_names = [
        os.path.join(self.data_dir, f"{self.experiment_id}-{self.num_process}-{process_id}.arrow.lock")
        for process_id in range(self.num_process)
    ]
-   for expected_lock_file_name in expected_lock_file_names:
+   for expected_lock_file_name in expected_lock_file_names[1:]:
        nofilelock = FileFreeLock(expected_lock_file_name)
        try:
            nofilelock.acquire(timeout=self.timeout)
        except Timeout:
            raise ValueError(
                f"Expected to find locked file {expected_lock_file_name} from process {self.process_id} but it doesn't exist."
            ) from None
        else:
            nofilelock.release()
```

### Broadcast the metric
🤗 Evaluate only calculates the metrics on the main process. Therefore, if we want to use a callback based on this data, we will need to broadcast it to the rest of the processes. To do this, we simply have to create a tensor on all processes that we will move to the GPU where we will store the metric, and once calculated, broadcast it to the rest of the processes. 
```diff
+accuracy_tensor = torch.empty(1).cuda() # Tensor to allocate broadcasted accuracy 
for epoch in range(starting_epoch, num_epochs):
    ...
    ###################### Evaluation ######################
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    
    eval_metric = metric.compute()

    ###################### Broadcast #######################
+   if accelerator.is_main_process:
+       accuracy_tensor = torch.Tensor([eval_metric['accuracy']]).cuda()
    
+   broadcast(accuracy_tensor, 0)
```
## Results
## Acknowledgements