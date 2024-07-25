from typing import Any, Dict, Sequence

import torch
from transformers import DataCollatorWithPadding


class PairwiseDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Data collator for pairwise data. 将被用作可调用的函数，以处理数据
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        将batch数据补全至同最长序列。
        生成2*n个样例，前n个为接受样例，后n个为拒绝样例。
        """
        features = [
            {"input_ids": feature[key]}
            for key in ("accept_ids", "reject_ids")
            for feature in features
        ]
        return super().__call__(features)
