import os
from typing import Dict, Optional

import torch
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.trainer import WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from train.trainer.utils.constants import VALUE_HEAD_FILE_NAME
from utils.logger import logger


def get_state_dict(
    model: torch.nn.Module, trainable_only: Optional[bool] = True
) -> Dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    filtered_state_dict = {}

    for k, v in model.named_parameters():
        if (not trainable_only) or v.requires_grad:
            filtered_state_dict[k] = state_dict[k].cpu().clone().detach()

    return filtered_state_dict


def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    valuehead_file = os.path.join(checkpoint_dir, VALUE_HEAD_FILE_NAME)
    if not os.path.exists(valuehead_file):
        logger.warning("提供的路径{}中没有valuehead权重文件".format(checkpoint_dir))
        return False
    valuehead_state_dict = torch.load(valuehead_file, map_location="cpu")
    model.register_buffer("reward_head_weight", valuehead_state_dict["summary.weight"])
    model.register_buffer("reward_head_bias", valuehead_state_dict["summary.bias"])
    # 加载default value head, 作为加载不成功的后备。正常情况下会被覆盖。
    model.register_buffer(
        "default_head_weight", torch.zeros_like(valuehead_state_dict["summary.weight"])
    )
    model.register_buffer(
        "default_head_bias", torch.zeros_like(valuehead_state_dict["summary.bias"])
    )
    return True


def load_trainable_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    weights_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if os.path.exists(weights_file):
        model_state_dict = torch.load(weights_file, map_location="cpu")
        model.load_state_dict(
            model_state_dict, strict=False
        )  # 加载到当前模型，并skip missing keys
    elif os.path.exists(os.path.join(checkpoint_dir, WEIGHTS_INDEX_NAME)):
        load_sharded_checkpoint(model, checkpoint_dir, strict=False)
    else:
        logger.warning(
            "Provided path ({}) does not contain pre-trained weights.".format(
                checkpoint_dir
            )
        )
        return False
    return True
