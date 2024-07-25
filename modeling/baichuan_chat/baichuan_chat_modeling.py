# ----------------------------------
# 2023年百川发版模型
# 网络结构: transformer decoder
# 模型大小: 13B
# 最大输入: 4096
# token数: -
# ----------------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from modeling.config import ModelConfig
from modeling.interface import ChatData, ModelPredictRequest, ModelTokenEncodeRequest
from utils import logger


class ModelBaichuanChat:
    """
    1. 模型通过model_name_or_path传入进行初始化，_开始的方法不对外提供，其他方法对外提供
    2. 模型包含token_encode_sft和token_encode_pretrain方法，对外给dataloader提供训练编码，为加快读取速度，方法必须均为staticmethod.
    3. 模型包含predict方法，为torch.no_grad()，根据prompt、历史对话和当前提问 输出 本次回答和使用的历史对话
    """

    def __init__(self, model_name_or_path: str):
        """
        :param model_path: 模型参数地址，包含模型权重文件和tokenizer相关文件
        """
        self.model_name_or_path = model_name_or_path
        self.model, self.tokenizer = self._load_model()
        # self.model = self.model.quantize(8).cuda() # model.quantize(4).cuda()
        # 配置项config，用于staticmethod的token_encode_sft和token_encode_pretrain方法输入
        self.config = ModelConfig(
            max_model_input_length=4096,
            pad_token_id=self.tokenizer.unk_token_id,  # TODO：未开放训练代码,使用unk_token pad
            user_token="<reserved_102>",
            assistant_token="<reserved_103>",
            eos_token="</s>",
        )
        return

    def _load_model(self):
        """
        :return: 模型和tokenizer
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )  # 模型加载到内存
        model.generation_config = GenerationConfig.from_pretrained(
            self.model_name_or_path
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True, trust_remote_code=True
        )
        return model, tokenizer

    @staticmethod
    def token_encode_sft(tokenizer, conf: ModelConfig, req: ModelTokenEncodeRequest):
        # 1.prompt有输入则进行token否则返回[]，待对话构建完成后再将提示词加到前面
        prompt_ids = tokenizer.encode(req.prompt) if req.prompt else []
        # 2.当前对话构造
        input_txt = (
            conf.user_token
            + req.input_chat.input
            + conf.assistant_token
            + req.input_chat.output
            + conf.eos_token
        )
        input_ids = tokenizer.encode(input_txt)
        if len(input_ids) + len(prompt_ids) > req.max_input_length:  # prompt+当前对话就已经超了
            return None, None
        # 3.历史对话检查，从后向前遍历
        for i, chat in enumerate(req.chat_history[::-1]):
            # 构建倒数第i+1轮历史对话
            input_txt = (
                conf.user_token
                + chat.input
                + conf.assistant_token
                + chat.output
                + conf.eos_token
            )
            token_ids = tokenizer.encode(input_txt)
            if len(input_ids) + len(token_ids) > req.max_input_length:  # 长度已经超了
                break
            input_ids = token_ids + input_ids  # 历史对话加到前面

        # 4.增加提示词并构建labels
        input_ids = prompt_ids + input_ids  # 增加prompt_ids
        labels = input_ids.copy()  # TODO: labels是否需要和input一样
        return input_ids, labels

    @torch.no_grad()
    def predict(self, req: ModelPredictRequest):
        # 1.prompt有输入则进行token否则返回[]，待对话构建完成后再将提示词加到前面
        prompt_ids = self.tokenizer.encode(req.prompt) if req.prompt else []
        # 2.当前对话构造
        input_txt = (
            self.config.user_token + req.input_message + self.config.assistant_token
        )
        query_ids = self.tokenizer.encode(input_txt)

        token_len = len(query_ids) + len(prompt_ids)
        if token_len > self.config.max_model_input_length:  # prompt+当前对话就已经超了
            logger.error(
                f"prompt+input_message length:{token_len} > "
                f"max_model_input_length:{self.config.max_model_input_length}, return None"
            )
            return None, None

        # 3.构建的前置历史对话
        front_lock_ids = []
        for msg in req.front_lock_messages:
            input_txt = (
                self.config.user_token
                + msg.input
                + self.config.assistant_token
                + msg.output
                + self.config.eos_token
            )
            front_lock_ids += self.tokenizer.encode(input_txt)

        token_len += len(front_lock_ids)
        if token_len > self.config.max_model_input_length:
            logger.error(
                f"prompt+input_message+front_lock_messages length:{token_len} > "
                f"max_model_input_length:{self.config.max_model_input_length}, return None"
            )
            return None, None

        # 4.历史对话
        history_ids = []
        history_messages = req.history_messages.copy()
        cut_idx = 0  # 截取使用的历史消息索引
        for msg in history_messages[::-1]:  # 从后往前遍历
            input_txt = (
                self.config.user_token
                + msg.input
                + self.config.assistant_token
                + msg.output
                + self.config.eos_token
            )
            token_ids = self.tokenizer.encode(input_txt)
            if (
                len(token_ids) + len(history_ids) + token_len
                > self.config.max_model_input_length
            ):
                break
            history_ids = token_ids + history_ids  # token_ids
            cut_idx += 1

        history_len = len(history_messages)
        if history_len > 0:  # 去掉前面不需要的历史对话
            history_messages = history_messages[history_len - cut_idx :]

        input_ids = prompt_ids + front_lock_ids + history_ids + query_ids
        encoded_input = torch.tensor([input_ids]).to("cuda")

        output = self.model.generate(
            encoded_input,  ## tensor 不能用**encoded_input
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            max_length=req.max_new_tokens,
            do_sample=True,
        )
        resp = self.tokenizer.decode(output.cpu()[0], skip_special_tokens=True)
        logger.info(f"model predict:\n{resp}")
        resp = resp.split(self.config.assistant_token)[-1]
        history_messages.append(ChatData(input=req.input_message, output=resp))
        return resp, history_messages


if __name__ == '__main__':
    path = ''
    baichuan = ModelBaichuanChat(path)
    messages = []
    messages.append({'role':'user', 'content':'世界上第二高的山是什么？'})
    response = baichuan.model.chat(baichuan.tokenizer, messages)
    print(response)