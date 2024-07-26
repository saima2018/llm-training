import ast

import deepspeed
import torch

from utils import logger
from utils.jinja_template import BaseTemplate

from .base_trainer import BaseTrainer


class DeepspeedTrainer(BaseTrainer):
    def __init__(self, model, tokenizer, optimizer, lr_scheduler, **kwargs):
        self.global_rank = torch.distributed.get_rank()
        torch.cuda.set_device(self.global_rank)
        deepspeed.init_distributed()

        self.args = kwargs.get("args", None)
        logger.info(f"DeepspeedTrainer args: {self.args}")

        self.tokenizer = tokenizer
        params = {
            "train_batch_size": self.args.train_bsz_per_gpu
            * torch.distributed.get_world_size(),
            "train_micro_batch_size_per_gpu": 1,
            "offload": self.args.offload,
            "stage": kwargs.get("zero_stage", 2),
            "enable_hybrid_engine": False,
            "inference_tp_size": 1,
            "release_inference_cache": False,
            "pin_parameters": True,
            "tp_gather_partition_size": 8,
            "max_out_tokens": 512,
        }
        self.ds_config = get_train_ds_config(**params)

        self.ds_config["train_micro_batch_size_per_gpu"] = kwargs.get(
            "per_device_train_batch_size", 1
        )

        self.model_engine, *_ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config_params=self.ds_config,
        )

    def save_checkpoint(self, model_path, tag):
        # save_zero_three_model(self.model_engine,self.global_rank,model_path,self.ds_config["zero_optimization"]["stage"])
        self.model_engine.save_checkpoint(model_path, tag=tag)

    def load_checkpoint(self, model_path):
        self.model_engine.load_checkpoint(model_path, load_optimizer_states=True)


def get_train_ds_config(config_path="./config/deepspeed/ds_config.jinja", **kwargs):
    """
    读取jinja模板，配置deepspeed参数
    """
    device = {"device": "cpu"} if kwargs["offload"] else None
    kwargs["device"] = device
    config_template = BaseTemplate(config_path)
    configs = config_template.template.render(kwargs)
    return ast.literal_eval(configs)
