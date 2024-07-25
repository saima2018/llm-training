# ----------------------------------
# 2019年openai发版模型
# 论文地址: https://paperswithcode.com/paper/language-models-are-unsupervised-multitask
# 网络结构: transformer decoder
# 模型大小: 124M
# 输入token长度: 1024
# token数: -
# ----------------------------------
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class ModelGPT2:
    def __init__(self, model_name_or_path):
        """
        :param model_name_or_path: 模型参数
        """
        self.model_name_or_path = model_name_or_path
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        """
        :return: 模型和对应的tokenizer
        """
        model = GPT2LMHeadModel.from_pretrained(self.model_name_or_path)
        tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name_or_path)
        return model, tokenizer
