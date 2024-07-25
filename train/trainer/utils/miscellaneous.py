from typing import List, Optional

import torch
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from transformers.modeling_utils import PreTrainedModel

from train.trainer.utils.constants import LAYERNORM_NAMES


class AverageMeter:
    r"""
    计算和保存平均值和当前值
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Avoid runtime error in model.generate(do_sample=True).
class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 0] = 1.0
        return scores


def get_logits_processor() -> LogitsProcessorList:
    """
    计算loss或预测前，处理logits中的nan和inf值为0, 并将sample直接归为第一类
    """
    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    return logits_processor


def print_trainable_params(model: torch.nn.Module) -> None:
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )


def prepare_model_for_training(
    model: PreTrainedModel,
    finetuning_type: str,
    output_embedding_layer_name: Optional[str] = "lm_head",
    use_gradient_checkpointing: Optional[bool] = True,
    layer_norm_names: Optional[List[str]] = LAYERNORM_NAMES,
) -> PreTrainedModel:
    # 将layer norm映射为fp32
    for name, param in model.named_parameters():
        if param.ndim == 1 and any(
            layer_norm_name in name for layer_norm_name in layer_norm_names
        ):
            param.data = param.data.to(torch.float32)

    # 使输出层embedding计算梯度
    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()  # 可减少内存使用
        model.config.use_cache = (
            False  # turn off when gradient checkpointing is enabled
        )

    # 将lm_head映射为fp32
    if finetuning_type != "full" and hasattr(model, output_embedding_layer_name):
        output_embedding_layer: torch.nn.Linear = getattr(
            model, output_embedding_layer_name
        )
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(
            model,
            output_embedding_layer_name,
            CastOutputToFloat(output_embedding_layer),
        )

    return model


def torch_gc() -> None:
    """回收GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
