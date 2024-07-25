# ==============================================
# 功能: 用于配置模型训练需要的输入参数，外部传参进行模型配置
# ==============================================
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from modeling import ModelingFactory


@dataclass
class TrainArgs:
    """为了区分transformers的TrainingArguments，使用名字TrainArgs
    模型训练相关参数
    """

    seed: int = field(
        default=os.getenv("SEED", default=42),
        metadata={"help": "设置种字点，消除训练中部分参数随机性，便于模型复现"},
    )
    stage: Optional[Literal["pt", "sft", "rm", "ppo"]] = field(
        default="sft", metadata={"help": "Which stage will be performed in training."}
    )
    task_type: str = field(
        default=os.getenv("TASK_TYPE", default="sft"),
        metadata={"help": "模型训练类型，预训练pretrain或对话训练sft", "choices": ["pretrain", "sft"]},
    )
    # 模型modeling，参考ModelingFactory
    modeling_name: Optional[str] = field(
        default=os.getenv("MODELING_NAME", default="baichuan"),
        metadata={"help": "modeling支持的模型名", "choices": ModelingFactory.list()},
    )
    # 模型参数地址: 由于huggingface有时候连接不上，建议走本地文件夹
    model_name_or_path: Optional[str] = field(
        default=os.getenv("MODEL_NAME_OR_PATH", default=None),
        metadata={"help": "模型参数地址或huggingface上模型名."},
    )
    # 分布式训练框架选择 TODO: 当前只支持了deepspeed， 其余还在开发中
    ddp_backend: str = field(
        default=os.getenv("DDP_BACKEND", "deepspeed"),
        metadata={
            "help": "分布式训练框架选择",
            "choices": ["deepspeed", "pytorch", "accelerate"],
        },
    )
    # 训练数据地址
    train_data_dir: str = field(
        default=os.getenv("TRAIN_DATA_DIR", default=None), metadata={"help": "训练数据地址"}
    )
    # 验证数据地址
    val_data_dir: str = field(
        default=os.getenv("VAL_DATA_DIR", default=None), metadata={"help": "验证数据地址"}
    )
    # 模型保存地址
    checkpoint_dir: str = field(
        default=os.getenv("CHECKPOINT_DIR", default="./checkpoint"),
        metadata={"help": "训练模型保存地址"},
    )
    # log输出地址
    log_dir: str = field(
        default=os.getenv("LOG_DIR", default="./log"), metadata={"help": "log输出地址"}
    )
    # 模型输入token最大长度设置
    max_seq_len: int = field(
        default=os.getenv("MAX_SEQ_LEN", default=2048),
        metadata={"help": "输入数据token最大长度，需要小于模型的最大输入长度"},
    )

    train_bsz_per_gpu: int = field(
        default=os.getenv("TRAIN_BSZ_PER_GPU", default=2),
        metadata={"help": "训练数据每张显卡的输入样本"},
    )
    eval_bsz_per_gpu: int = field(
        default=os.getenv("EVAL_BSZ_PER_GPU", default=2),
        metadata={"help": "验证数据每张显卡的输入样本"},
    )

    local_rank: int = field(
        default=os.getenv("LOCAL_RANK", default=0), metadata={"help": "local_rank"}
    )

    n_epochs: int = field(
        default=os.getenv("N_EPOCHS", default=3), metadata={"help": "训练轮数"}
    )
    learning_rate: float = field(
        default=os.getenv("LEARNING_RATE", default=1e-8), metadata={"help": "学习率"}
    )
    warmup_rates: float = field(
        default=os.getenv("WARMUP_RATES", default=0.05), metadata={"help": "学习率预热指数"}
    )
    lr_scheduler_type: str = field(
        default=os.getenv("LR_SCHEDULER_TYPE", default="cosine"),
        metadata={"help": "学习率调整方式"},
    )
    weight_decay: float = field(
        default=os.getenv("WEIGHT_DECAY", default=0.1), metadata={"help": "权重正则化"}
    )

    save_step_interval: int = field(
        default=os.getenv("SAVE_STEP_INTERVAL", default=50),
        metadata={"help": "训练到多少步保存模型参数"},
    )
    eval_step: int = field(
        default=os.getenv("EVAL_STEP", default=50), metadata={"help": "训练到多少步进行验证集测试"}
    )

    resume_training: bool = field(
        default=os.getenv("RESUME_TRAINING", default=True),
        metadata={"help": "是否从断点参数继续训练"},
    )
    zero_model_input: str = field(
        default=os.getenv("ZERO_MODEL_INPUT", default=None), metadata={"help": "断点参数地址"}
    )

    resume_dataloader: bool = field(
        default=os.getenv("RESUME_DATALOADER", default=True),
        metadata={"help": "是否从断点数据继续训练"},
    )
    dataloader_state_path: str = field(
        default=os.getenv("DATALOADER_STATE_PATH", default=None),
        metadata={"help": "断点数据地址"},
    )

    def __post_init__(self):  # 对输入进行输入检查
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.model_name_or_path is None:  # 模型地址必须输入
            raise ValueError("--model_name_or_path can't be None")
