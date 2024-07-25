import json
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainingArguments
from transformers.modeling_utils import PreTrainedModel

from data.loader import get_dataset
from train.hyperparams import DataArguments, FinetuningArguments, ModelArguments
from train.trainer.common.trainer import PeftTrainer
from utils.logger import logger


class PairwisePeftTrainer(PeftTrainer):
    """
    继承PeftTrainer,计算reward模型的pairwise loss
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_accuracy = self.compute_accuracy
        self.can_return_loss = True  # Trainer类默认为False, 省去分布式训练时不必要的loss归集

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        计算pairwise loss. 前n个为接受样例，后n个为拒绝样例。
        使用eos token的score作为整个句子的reward score.
        """
        batch_size = inputs["input_ids"].size(0) // 2
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True)
        r_accept, r_reject = values[:, -1].split(batch_size, dim=0)
        loss = -torch.log(torch.sigmoid(r_accept - r_reject)).mean()
        return (loss, [loss, r_accept, r_reject]) if return_outputs else loss

    @staticmethod
    def compute_accuracy(
        eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]
    ) -> Dict[str, float]:
        preds, _ = eval_preds
        return {"accuracy": (preds[0] > preds[1]).sum() / len(preds[0])}
