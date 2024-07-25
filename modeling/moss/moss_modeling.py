# ----------------------------------
# 2023年复旦发版模型
# 网络结构: transformer decoder
# 模型大小: 16B
# token数: -
# ----------------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelMoss:
    def __init__(self, model_name_or_path):
        """
        :param model_name_or_path: 模型参数地址，包含模型权重文件和tokenizer相关文件
        """
        self.model_name_or_path = model_name_or_path
        self.model, self.tokenizer = self.load_model()
        return

    def load_model(self, torch_dtype=torch.float16, train_step="sft"):
        """
        :param torch_dtype: 模型加载类型
        :return:
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, trust_remote_code=True, use_cache=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        # The eos_token_id of base model is 106028. We need map the eos token to <eom> (its token id is 106068)
        if train_step == "sft":
            tokenizer.eos_token_id = 106068

        model.transformer.gradient_checkpointing = True
        assert model.transformer.gradient_checkpointing is True
        return model, tokenizer
