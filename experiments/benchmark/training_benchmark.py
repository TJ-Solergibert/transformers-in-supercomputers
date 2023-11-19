import logging
import sys

# Set logging configs
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)
logging.getLogger('numexpr').setLevel(logging.WARNING)

import argparse
import os
import time
from functools import partial

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.distributed import broadcast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset, concatenate_datasets
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import evaluate
from accelerate import Accelerator

from utils import shift_bit_length
from numpy import average

MAX_EVAL_BATCH_SIZE = 4096

train_times = []
eval_times = []
compute_metric_times = []
broadcast_times = []

def tokenize_function(examples, tokenizer):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["text"], truncation=True)
    return outputs

def collate_fn(examples, tokenizer, pad_to_multiple_of):
    return tokenizer.pad(
        examples,
        padding="longest",
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )


def training_function(args):
    ################ Initialize accelerator ################
    if args.with_tracking:
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision, log_with="all", logging_dir=args.logging_dir
        )
    else:
        accelerator = Accelerator(mixed_precision=args.mixed_precision)
    ########################################################

    ############# Resume training sanity check #############
    if hasattr(args.checkpointing_steps, "isdigit"):
        if args.checkpointing_steps == "epoch":
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{args.checkpointing_steps}` passed."
            )
    else:
        checkpointing_steps = None    
    ########################################################

    ######### Load tokenizer, datasets and metrics #########
    path_to_model = os.path.join(os.environ["HF_LOCAL_HOME"], "models", args.model_name)
    path_to_dataset = os.path.join(os.environ["HF_LOCAL_HOME"], "datasets", args.dataset)
    path_to_metric = os.path.join(os.environ["HF_LOCAL_HOME"], "metrics", "accuracy.py")    # TODO: Add more metrics. https://github.com/huggingface/evaluate/pull/150
    
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    datasets = load_dataset(path_to_dataset)
    metric = evaluate.load(path_to_metric, process_id = int(os.environ["RANK"]), num_process= int(os.environ["WORLD_SIZE"]), experiment_id="slurm")
    ########################################################

    ################# Preprocess datasets ##################
    # Starting with the main process first to exploit HF cache:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            fn_kwargs = {"tokenizer": tokenizer},
            batched=True,
            batch_size = None,
            remove_columns=["text"],
        )

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    # Multiply trainind and evaluation data.
   
    # Every device will have 4096 samples for training and 4096 samples for evaluation
    # tokenized_datasets["train"] = tokenized_datasets["train"].select(range(4096))
    # tokenized_datasets["train"] = concatenate_datasets([tokenized_datasets["train"] for _ in range(int(os.environ["WORLD_SIZE"]))])
    # tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(1024))
    # tokenized_datasets["validation"] = concatenate_datasets([tokenized_datasets["validation"] for _ in range(4*int(os.environ["WORLD_SIZE"]))])
    
    # For benchmarks we will have the same number of samples in all the runs
    # 32768 for train and 131072 for evaluation
    # So in the case with 256 batch size, each device of the 4 nodes will have 8 batches for training
    # and in a evaluation with eval batch size of 4096, every device will have 2 batches
    tokenized_datasets["train"] = tokenized_datasets["train"].select(range(8192))
    tokenized_datasets["train"] = concatenate_datasets([tokenized_datasets["train"] for _ in range(4)])
    tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(1024))
    tokenized_datasets["validation"] = concatenate_datasets([tokenized_datasets["validation"] for _ in range(2*64)])
    ########################################################

    ############### Training hyperparameters ###############
    lr = args.learning_rate  
    num_epochs = int(args.num_epochs)
    train_size = len(tokenized_datasets["train"])
    eval_size = len(tokenized_datasets["validation"])
    batch_size = int(args.batch_size)

    # If not specified, autocompute eval_batch_size. 1 batch per device
    if args.eval_batch_size is None: 
        eval_batch_size = shift_bit_length(eval_size/int(os.environ["WORLD_SIZE"]))
        if eval_batch_size > MAX_EVAL_BATCH_SIZE:
            eval_batch_size = MAX_EVAL_BATCH_SIZE
    else:
        eval_batch_size = int(args.eval_batch_size)
    ########################################################

    ############# Initialize trackers with HPs #############
    if args.with_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, args)
    ########################################################

    ########## Instantiate dataloaders, model, optimizers and schedulers ##########
    if accelerator.mixed_precision == "fp8":
        pad_to_multiple_of = 16
    elif accelerator.mixed_precision != "no":
        pad_to_multiple_of = 8
    else:
        pad_to_multiple_of = None

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, 
        collate_fn=partial(collate_fn, tokenizer = tokenizer, pad_to_multiple_of = pad_to_multiple_of), 
        batch_size=batch_size
    )

    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, 
        collate_fn=partial(collate_fn, tokenizer = tokenizer, pad_to_multiple_of = pad_to_multiple_of), 
        batch_size=eval_batch_size
    )

    num_labels = tokenized_datasets["train"].features["labels"].num_classes
    model = AutoModelForSequenceClassification.from_pretrained(path_to_model, num_labels = num_labels, return_dict=True)

    optimizer = AdamW(params=model.parameters(), lr=lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    ###############################################################################

    ########## Prepare everything with Accelerate ##########
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    ########################################################
    
    ########## Load weights and states from previous save ##########
    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the stating epoch so files are named properly
    starting_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]     # Sorts folders by date modified, most recent checkpoint is the last
                                # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
    ##############################################################
    
    
    ####################### Training #######################
    ft0 = time.time()
    accuracy_tensor = torch.empty(1).cuda() # Tensor to allocate broadcasted accuracy 
    for epoch in range(starting_epoch, num_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We need to skip steps until we reach the resumed step
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            overall_step += resume_step
        t0 = time.time() 
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            overall_step += 1
        
            if isinstance(checkpointing_steps, int):
                if overall_step % checkpointing_steps == 0:
                    output_dir = f"step_{overall_step}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
        
        train_times.append(time.time()-t0)

        ###################### Evaluation ######################
        model.eval()
        e0 = time.time()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        
        eval_times.append(time.time()-e0)
        c0 = time.time()
        eval_metric = metric.compute()
        compute_metric_times.append(time.time()-c0)
        ########################################################

        ###################### Broadcast #######################
        if accelerator.is_main_process:
            accuracy_tensor = torch.Tensor([eval_metric['accuracy']]).cuda()
        accelerator.wait_for_everyone()
        b0 = time.time()
        broadcast(accuracy_tensor, 0)
        broadcast_times.append(time.time()-b0)
        ########################################################

        if args.with_tracking:
            # TODO Handle multiple metrics dict
            accelerator.log(
                {
                    "accuracy": eval_metric["accuracy"],
                    # "f1": eval_metric["f1"],
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                },
                step=epoch,
            )

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()
    ########################################################
    
    full_training_time = time.time()-ft0
    if accelerator.is_main_process:
        logging.info(f"Training finished!")
        logging.info(f"Complete training Time: {full_training_time} s")
        logging.info(f"[{int(os.environ['RANK'])}] Training iteration: {average(train_times)} s")
        logging.info(f"[{int(os.environ['RANK'])}] Evaluation iteration: {average(eval_times)} s")
        logging.info(f"[{int(os.environ['RANK'])}] Compute metric: {average(compute_metric_times)} s")
        logging.info(f"[{int(os.environ['RANK'])}] Training Throughput: {(train_size*num_epochs/int(os.environ['WORLD_SIZE']))/sum(train_times):.2f} samples/second")
        logging.info(f"[{int(os.environ['RANK'])}] Eval Throughput: {(eval_size*num_epochs/int(os.environ['WORLD_SIZE']))/sum(eval_times):.2f} samples/second")

    accelerator.wait_for_everyone()
    logging.info(f"[{int(os.environ['RANK'])}] Broadcast: {average(broadcast_times)*1000} ms/broadcast")
def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs.",
    )
    parser.add_argument("--batch_size", 
        type=int, 
        default=256, 
        help="Batch size for training per device. Global batch size is batch_size * num_devices."
    )
    parser.add_argument("--eval_batch_size", 
        type=int, 
        default=512, 
        help="Batch size for evaluation per device. Global batch size is batch_size * num_devices."
    )
    parser.add_argument("--num_epochs", 
        type=int, 
        default=5, 
        help="Number of epochs"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Model name from Hugging Face.",
    )
    parser.add_argument("--learning_rate", 
        type=float, 
        default=5e-4, 
        help="Learning rate."
    )   
    parser.add_argument(
        "--dataset",
        type=str,
        default="emotion",
        help="Dataset name from Hugging Face.",
    )
    
    args, _ = parser.parse_known_args()
    training_function(args)


if __name__ == "__main__":
    main()