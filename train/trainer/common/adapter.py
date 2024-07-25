import os

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.modeling_utils import PreTrainedModel

from train.hyperparams import FinetuningArguments, ModelArguments
from train.trainer.utils.save_and_load import load_trainable_params
from utils.logger import logger


def init_adapter(
    model: PreTrainedModel,
    model_args: ModelArguments,
    finetuning_args: FinetuningArguments,
    is_trainable: bool,
    is_mergeable: bool,
) -> PreTrainedModel:
    """
    初始化 adapters. 支持 full, freeze, and lora 训练。
    可训练参数必须映射为float32.
    """

    if finetuning_args.finetuning_type == "none" and is_trainable:
        raise ValueError("训练时finetuning_type不能为none.")

    if finetuning_args.finetuning_type == "full":
        logger.info("finetuning method: full")
        model = model.float()

    if finetuning_args.finetuning_type == "freeze":
        logger.info("finetuning method: freeze")

        for name, param in model.named_parameters():
            if not any(
                trainable_layer in name
                for trainable_layer in finetuning_args.trainable_layers
            ):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

        if model_args.checkpoint_dir is not None:
            assert load_trainable_params(
                model, model_args.checkpoint_dir[0]
            ), "模型checkpoint未正确加载。"

    if finetuning_args.finetuning_type == "lora":
        logger.info("finetuning method: lora")
        latest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            assert os.path.exists(
                os.path.join(model_args.checkpoint_dir[0], WEIGHTS_NAME)
            ), "Provided path ({}) does not contain a LoRA weight.".format(
                model_args.checkpoint_dir[0]
            )
            assert os.path.exists(
                os.path.join(model_args.checkpoint_dir[0], CONFIG_NAME)
            ), "The given checkpoint may be not a LoRA checkpoint, please specify `--finetuning_type full/freeze` instead."
            if (is_trainable and model_args.resume_lora_training) or (
                not is_mergeable
            ):  # continually train on the lora weights
                checkpoints_to_merge, latest_checkpoint = (
                    model_args.checkpoint_dir[:-1],
                    model_args.checkpoint_dir[-1],
                )
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                logger.info(
                    "Merged {} model checkpoint(s).".format(len(checkpoints_to_merge))
                )

            if (
                latest_checkpoint is not None
            ):  # resume lora training or quantized inference
                model = PeftModel.from_pretrained(
                    model, latest_checkpoint, is_trainable=is_trainable
                )

        if is_trainable and latest_checkpoint is None:  # 训练时创建新的lora权重
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=finetuning_args.lora_target,
            )
            model = get_peft_model(model, lora_config)

    if model_args.checkpoint_dir is not None:
        logger.info(
            "Loaded fine-tuned model from checkpoint(s): {}".format(
                ",".join(model_args.checkpoint_dir)
            )
        )

    return model
