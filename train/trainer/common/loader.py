import os
from typing import Literal, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from trl import AutoModelForCausalLMWithValueHead

from train.hyperparams import FinetuningArguments, ModelArguments
from train.trainer.common.adapter import init_adapter
from train.trainer.utils.miscellaneous import (
    prepare_model_for_training,
    print_trainable_params,
)
from train.trainer.utils.save_and_load import load_valuehead_params
from utils.logger import logger

check_min_version("4.29.1")
require_version("datasets>=2.12.0", "To fix: pip install datasets>=2.12.0")
require_version("accelerate>=0.21.0", "To fix: pip install accelerate>=0.21.0")
require_version("peft>=0.4.0", "To fix: pip install peft>=0.4.0")
require_version("trl>=0.4.7", "To fix: pip install trl>=0.4.7")


def load_model_and_tokenizer(
    model_args: ModelArguments,
    finetuning_args: FinetuningArguments,
    is_trainable: Optional[bool] = False,
    stage: Optional[Literal["pt", "sft", "rm", "ppo"]] = "sft",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    加载模型和tokenizer，支持训练和预测。
    """
    if (not is_trainable) and model_args.checkpoint_dir is None:
        logger.warning(
            "Checkpoint is not found at evaluation, load the original model."
        )
        finetuning_args = FinetuningArguments(finetuning_type="none")

    # assert (
    #     stage in ["pt", "sft"] or finetuning_args.finetuning_type == "lora"
    # ), "RM and PPO training can only be performed with the LoRA method."

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side=model_args.padding_side,
        **config_kwargs
    )
    if (
        tokenizer.pad_token_id is None or tokenizer.pad_token_id == 64000
    ):  # 百川模型用的是64000
        tokenizer.pad_token_id = 0  # 使用 <unk> token

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    is_mergeable = True

    # 使用bitsandbytes库进行量化操作配置
    if model_args.quantization_bit is not None:
        if model_args.quantization_bit == 8:
            require_version(
                "bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0"
            )
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_threshold=6.0
            )
        elif model_args.quantization_bit == 4:
            require_version(
                "bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0"
            )
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type,
            )

        is_mergeable = False
        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        logger.info("将模型量化为 {} bit.".format(model_args.quantization_bit))

    if not is_trainable:  # 只有预测时`device_map=auto`
        config_kwargs["device_map"] = "auto"

    if (
        model_args.checkpoint_dir is not None
        and finetuning_args.finetuning_type == "full"
    ):
        model_to_load = model_args.checkpoint_dir[0]
    else:
        model_to_load = model_args.model_name_or_path

    # 加载不含valuehead的pretrained模型
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        config=config,
        torch_dtype=torch.bfloat16
        if model_args.compute_dtype == torch.bfloat16
        else torch.float16,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs
    )
    # 自动实例化config, model, and tokenizer
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(
        config, "auto_map", {}
    ):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(
        config, "auto_map", {}
    ):
        model.__class__.register_for_auto_class()
    if isinstance(
        tokenizer, PreTrainedTokenizerBase
    ) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

    # 初始化adapters
    model = (
        prepare_model_for_training(model, finetuning_args.finetuning_type)
        if is_trainable
        else model
    )
    # 基于model和config, 返回一个peft model
    model = init_adapter(model, model_args, finetuning_args, is_trainable, is_mergeable)
    if stage == "rm" or stage == "ppo":  # 添加valuehead
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        if (
            stage == "rm" and model_args.checkpoint_dir is not None
        ):  # 加载reward模型最新checkpoint的valuehead权重
            if load_valuehead_params(model, model_args.checkpoint_dir[-1]):
                # 加载到模型参数里，才能进行训练更新
                model.v_head.load_state_dict(
                    {
                        "summary.weight": getattr(model, "reward_head_weight"),
                        "summary.bias": getattr(model, "reward_head_bias"),
                    }
                )

        if stage == "ppo":  # 加载reward模型
            assert is_trainable, "PPO训练阶段不能是evaluation模式。"
            assert model_args.reward_model is not None, "PPO训练需要有reward模型。"
            logger.info("Load reward model from {}".format(model_args.reward_model))
            model.pretrained_model.load_adapter(  # peft model定义了pretrained_model属性
                model_args.reward_model, "reward", is_trainable=False
            )  # 将rm作为adapter加入ppo模型
            assert load_valuehead_params(
                model, model_args.reward_model
            ), "Reward模型加载失败。"  # 加载rm的value head权重

    if not is_trainable:
        model.requires_grad_(False)  # 冻结模型所有参数
        model = (
            model.half() if model_args.quantization_bit is None else model
        )  # 从fp32映射到fp16

    print_trainable_params(model)

    return model, tokenizer
