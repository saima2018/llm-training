# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import math

# DeepSpeed Team
import os

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig

from .reward_model import RewardModel


def create_hf_model(
    model_class,
    model_name_or_path,
    tokenizer,
    ds_config=None,
    rlhf_training=False,
    disable_dropout=False
):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config, trust_remote_code=True)
    elif 'glm' in model_name_or_path:
        model = AutoModel.from_pretrained(
            model_name_or_path,
            config=model_config,
            # empty_init=False,
            trust_remote_code=True
        )
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            trust_remote_code=True
        ).half()

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    # model.resize_token_embeddings(
    #     int(8 * math.ceil(len(tokenizer) / 8.0))
    # )  # make the vocab size multiple of 8

    return model


def create_critic_model(
    model_name_or_path,
    tokenizer,
    ds_config,
    num_padding_at_beginning=0,
    rlhf_training=False,
    disable_dropout=False,
    device=None
):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule
    model_class = AutoModel
    if 'baichuan' in model_name_or_path:
        model_class = AutoModelForCausalLM
    critic_model = create_hf_model(
        model_class,
        model_name_or_path,
        tokenizer,
        ds_config,
        rlhf_training,
        disable_dropout,
    )
    critic_model = RewardModel(
        critic_model.base_model, tokenizer, num_padding_at_beginning=num_padding_at_beginning
    )

    if rlhf_training: # 加载参数
        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        # critic model needs to load the weight here
        model_ckpt_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"
        critic_model.load_state_dict(torch.load(model_ckpt_path,
                                                map_location=device))

    return critic_model
