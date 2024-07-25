import os
import sys
from typing import Optional, List
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (HfArgumentParser, get_scheduler, set_seed, TrainerCallback,
                          Seq2SeqTrainingArguments, DataCollatorForSeq2Seq)

# 将目录设置到项目目录层再导入项目中的库
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
from data import get_dataset, preprocess_dataset, split_dataset
from data.pretrain_datasets import PretrainDataset
from data.sft_datasets import SFTDataset
from modeling import ModelingFactory
from train.hyperparams import TrainArgs, ModelArguments, DataArguments, GeneratingArguments, FinetuningArguments
from train.trainer.distribute.trainer_factory import TrainerFactory
from train.trainer.pt import PretrainSFTTrainer
from utils.logger import add_file_handler_to_logger, log_rank0
from train.trainer.common import load_model_and_tokenizer
from train.trainer.utils.constants import IGNORE_INDEX

def train():
    # 1. 训练的参数设置, 可以通过--传参(如:--modeling_name moss)，具体参数设置参考TrainingArgs
    args: TrainArgs = HfArgumentParser((TrainArgs)).parse_args_into_dataclasses()[0]
    set_seed(args.seed)  # Set seed before initializing model.

    # 2.初始化torch分布式参数
    torch.cuda.set_device(args.local_rank)  # 默认使用 gpu 0
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs.
    # torch.distributed帮助自动创建n个进程运行在n个gpu上
    torch.distributed.init_process_group(backend="nccl")  # 后端通信: gpu训练用nccl, cpu训练用gloo
    device_rank = torch.distributed.get_rank()  # 分布式任务进程组group中进程的序号，默认进程数等于gpu数

    if device_rank <= 0:
        add_file_handler_to_logger(dir_path=args.log_dir, name="train")  # 增加训练log输出

    # 3.模型加载，即加载模型参数和tokenizer两部分。本地没有时会自动下载到~/.cache/huggingface/
    log_rank0("loading model...", device_rank)
    # TODO: 确认模型是否有其他参数需求
    modeling = ModelingFactory.get(
        args.modeling_name, model_path=args.model_name_or_path
    )

    # 4.加载训练数据，第一次读取后会在原目录保存数据cache文件
    log_rank0("loading datasets...", device_rank)
    # train_dataset 为长度不等的tensor的集合\
    if args.resume_dataloader:
        log_rank0("resume training, loading dataloader...", device_rank)
        train_data_loader = torch.load(args.dataloader_state_path)
    else:
        if args.task_type == "sft":
            log_rank0("loading dataloader from SFTDataset", device_rank)
            train_dataset = SFTDataset(
                args.train_data_dir, modeling.tokenizer, max_length=args.max_seq_len
            )
        else:
            log_rank0("loading dataloader from PretrainDataset", device_rank)
            train_dataset = PretrainDataset(
                args.train_data_dir, modeling.tokenizer, max_length=args.max_seq_len
            )
        train_data_loader = DataLoader(
            train_dataset,
            sampler=DistributedSampler(train_dataset),
            batch_size=args.train_bsz_per_gpu,
            shuffle=False,
            drop_last=True,
            collate_fn=train_dataset.collate_fn,
        )
        # Todo add valdataset

    # 5.优化器选择
    optimizer = torch.optim.AdamW(
        modeling.model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95)
    )

    # 6.学习率设计
    num_training_steps = len(train_data_loader) * args.n_epochs
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_rates * num_training_steps,
        num_training_steps=num_training_steps,
    )

    # 7.训练器加载
    log_rank0("Initialising ddp_trainer...", device_rank)
    ddp_trainer = TrainerFactory.get_trainer(
        args.ddp_backend,
        model=modeling.model,
        tokenizer=modeling.tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        args=args,
    )
    trainer = PretrainSFTTrainer(ddp_trainer)

    if args.resume_training:
        log_rank0("resume training, loading checkpoints...", device_rank)
        ddp_trainer.load_checkpoint(args.zero_model_input)

    log_rank0("start training...", device_rank)
    trainer.train(train_data_loader, args.n_epochs)


def train_sft(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
    generating_args: GeneratingArguments,
    callback: Optional[List[TrainerCallback]] = None
):
    """
    基于PEFT框架的lora模式sft
    """
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage='sft')
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage='sft')
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    # 覆盖seq2seqTrainer的decoding参数
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.max_target_length,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # 初始化trainer
    trainer = Seq2SeqPeftTrainer(

    )


if __name__ == "__main__":
    train()
