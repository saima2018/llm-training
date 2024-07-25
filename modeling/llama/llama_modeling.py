# ----------------------------------
# 2023年facebook发版模型
# 网络结构: transformer decoder
# 模型大小: 7B/13B/33B/65B
# token数: 32000
# ----------------------------------

# Todo 目前modeling 没有用，是直接用的模型的remote_code, 后续集成进来

import torch
from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer


class ModelLLaMa:
    def __init__(self, model_name_or_path):
        """
        :param model_name_or_path: 模型参数地址，包含模型权重文件和tokenizer相关文件
        """
        self.model_name_or_path = model_name_or_path
        self.model, self.tokenizer = self.load_model()
        return

    def load_model(self, torch_dtype=torch.float16):
        """
        :param torch_dtype: 模型加载类型
        :return:
        """
        config = AutoConfig.from_pretrained(self.model_name_or_path)
        model = LlamaForCausalLM.from_pretrained(
            self.model_name_or_path,
            from_tf=False,  # 是否走tensorflow
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        tokenizer = LlamaTokenizer.from_pretrained(self.model_name_or_path)
        return model, tokenizer
