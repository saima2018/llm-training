import os
from typing import Dict, Optional

import torch
from peft import PeftModel
from transformers import Seq2SeqTrainer
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer import TRAINING_ARGS_NAME

from train.hyperparams import FinetuningArguments
from train.trainer.utils.constants import FINETUNING_ARGS_NAME, VALUE_HEAD_FILE_NAME
from train.trainer.utils.save_and_load import (
    get_state_dict,
    load_trainable_params,
    load_valuehead_params,
)
from utils.logger import logger


class PeftTrainer(Seq2SeqTrainer):  # Seq2SeqTrainer继承Trainer类
    """
    Inherits Seq2SeqTrainer to support parameter-efficient checkpoints.
    """

    def __init__(self, finetuning_args: FinetuningArguments, **kwargs):
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self._remove_log()

    def _remove_log(self):
        if self.is_world_process_zero() and os.path.exists(
            os.path.join(self.args.output_dir, "trainer_log.jsonl")
        ):
            logger.warning("文件夹中的旧日志文件将被删除。")
            os.remove(os.path.join(self.args.output_dir, "trainer_log.jsonl"))

    def _save(
        self,
        output_dir: Optional[str] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        将可训练参数保存为 model checkpoint.
        该函数只在 process zero 为True时执行.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        model = unwrap_model(self.model)

        if hasattr(model, "pretrained_model"):  # 对于有valuehead的模型, 去掉valuehead并单独保存
            backbone_model = getattr(model, "pretrained_model")
            torch.save(
                get_state_dict(getattr(model, "v_head")),
                os.path.join(output_dir, VALUE_HEAD_FILE_NAME),
            )
        else:
            backbone_model = model

        if isinstance(backbone_model, PeftModel):  # lora tuning
            backbone_model.save_pretrained(
                output_dir, state_dict=get_state_dict(backbone_model)
            )
        elif isinstance(backbone_model, PreTrainedModel):  # freeze/full tuning
            print("save pretrained==============================")
            backbone_model.config.use_cache = True
            backbone_model.save_pretrained(
                output_dir,
                state_dict=get_state_dict(
                    backbone_model,
                    trainable_only=(self.finetuning_args.finetuning_type != "full"),
                ),
                safe_serialization=self.args.save_safetensors,
            )
            backbone_model.config.use_cache = False
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
        else:
            logger.warning("没有可保存的模型。")

        with open(
            os.path.join(output_dir, TRAINING_ARGS_NAME), "w", encoding="utf-8"
        ) as f:
            f.write(self.args.to_json_string() + "\n")
        self.finetuning_args.save_to_json(
            os.path.join(output_dir, FINETUNING_ARGS_NAME)
        )

    def _load_best_model(self):
        """从checkpoint加载可训练的参数"""
        logger.info(
            f"从{self.state.best_model_checkpoint}加载最佳模型, score: {self.state.best_metric}"
        )

        model = unwrap_model(self.model)
        backbone_model = (
            getattr(model, "pretrained_model")
            if hasattr(model, "pretrained_model")
            else model
        )

        if isinstance(backbone_model, PeftModel):  # lora tuning
            backbone_model.load_adapter(
                self.state.best_model_checkpoint, backbone_model.active_adapter
            )
            if hasattr(model, "v_head") and load_valuehead_params(
                model, self.state.best_model_checkpoint
            ):
                model.v_head.load_state_dict(
                    {
                        "summary.weight": getattr(model, "reward_head_weight"),
                        "summary.bias": getattr(model, "reward_head_bias"),
                    }
                )
        else:  # freeze/full tuning
            load_trainable_params(backbone_model, self.state.best_model_checkpoint)
