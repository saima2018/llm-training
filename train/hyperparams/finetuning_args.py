import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


@dataclass
class FinetuningArguments:
    """各种微调训练类型的参数"""

    stage: Optional[Literal["pt", "sft", "rm", "ppo"]] = field(
        default="sft", metadata={"help": "Which stage will be performed in training."}
    )
    finetuning_type: Optional[Literal["none", "freeze", "lora", "full"]] = field(
        default="lora", metadata={"help": "指定微调类型"}
    )
    num_hidden_layers: Optional[int] = field(
        default=32,
        metadata={
            "help": '模型的解码器层数。 \
                  LLaMA choices: ["32", "40", "60", "80"], \
                  LLaMA-2 choices: ["32", "40", "80"], \
                  Baichuan choices: ["32", "40"]'
        },
    )
    num_layer_trainable: Optional[int] = field(
        default=3, metadata={"help": "Freeze微调的可训练层数。"}
    )
    name_module_trainable: Optional[
        Literal["mlp", "self_attn", "self_attention"]
    ] = field(
        default="mlp",
        metadata={
            "help": 'Freeze微调的可训练模块名。 \
                  LLaMA & LLaMA-2 choices: ["mlp", "self_attn"], \
                  Baichuan choices: ["mlp", "self_attn"]'
        },
    )
    lora_rank: Optional[int] = field(default=8, metadata={"help": "LoRA微调的维度"})
    lora_alpha: Optional[float] = field(
        default=32.0, metadata={"help": "LoRA微调的缩放因子，类似学习率"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "lora微调的Dropout rate"}
    )
    lora_target: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={
            "help": '使用LoRA的目标模块，用逗号分隔。 \
                  LLaMA & LLaMA-2 choices: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                  Baichuan choices: ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]'
        },
    )

    def __post_init__(self):
        if isinstance(self.lora_target, str):  # 处理参数为list，支持自定义LoRA训练的模型模块。
            self.lora_target = [
                target.strip() for target in self.lora_target.split(",")
            ]

        if self.num_layer_trainable > 0:  # 微调最后n层
            trainable_layer_ids = [
                self.num_hidden_layers - k - 1 for k in range(self.num_layer_trainable)
            ]
        else:  # 微调最前n层
            trainable_layer_ids = [k for k in range(-self.num_layer_trainable)]

        self.trainable_layers = [
            "{:d}.{}".format(idx, self.name_module_trainable)
            for idx in trainable_layer_ids
        ]

        assert self.finetuning_type in [
            "none",
            "freeze",
            "lora",
            "full",
        ], "Invalid fine-tuning method."

    def save_to_json(self, json_path: str):
        """保存微调参数到`json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """从json文件加载类实例"""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))
