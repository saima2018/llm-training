# ----------------------------------
# 2023年清华发版模型
# 网络结构: transformer
# 模型大小: 6B
# 最大输入: -
# token数: -
# ----------------------------------
import torch
from transformers import AutoModel, AutoTokenizer

from modeling.config import ModelConfig
from modeling.interface import ChatData, ModelPredictRequest, ModelTokenEncodeRequest
from utils import logger


class ModelChatGLMV2:
    def __init__(self, model_name_or_path):
        """
        :param model_name_or_path: 模型参数地址，包含模型权重文件和tokenizer相关文件
        """
        logger.info(f"start load model from {model_name_or_path}")
        self.model_name_or_path = model_name_or_path
        self.model, self.tokenizer = self.load_model()
        # 由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练，允许更多轮次的对话。
        # 但当前版本的 ChatGLM2-6B 对单轮超长文档的理解能力有限，我们会在后续迭代升级中着重进行优化
        # 所以当前先将input_len设置为8k 而不是config.seq_length
        self.config = ModelConfig(
            max_model_input_length=8000,
            pad_token_id=self.tokenizer.pad_token_id,
            user_token="问：",
            assistant_token="答：",
            eos_token=self.tokenizer.eos_token,
        )

        logger.info(
            f"finish load model, model's max_model_input_length is {self.config.max_model_input_length}"
        )
        return

    def load_model(self):
        """
        :param torch_dtype: 模型加载类型
        :return:
        """
        model = AutoModel.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        return model, tokenizer

    @staticmethod
    def token_encode_sft(
        tokenizer,
        conf: ModelConfig,
        req: ModelTokenEncodeRequest,
        ignore_pad_token_for_loss: bool = True,
    ):
        # 1.input_ids构建-在前面增加prefix_tokens和prompt_ids
        prefix_ids = tokenizer.get_prefix_tokens()  # prefix_tokens
        # prompt有输入则进行token否则返回
        prompt_ids = (
            tokenizer.encode(text=req.prompt, add_special_tokens=False)
            if req.prompt
            else []
        )
        input_ids = prefix_ids + prompt_ids

        # 2.对当前问题和答案编码，查看是否超过输入长度
        # 当前对话先没有加答案
        query_ids = tokenizer.encode(
            text=f"{conf.user_token}{req.input_chat.input}\n\n{conf.assistant_token}",
            add_special_tokens=False,
        )[
            1:
        ]  # TODO 处理多加的空格30910
        # 统计所有可能出现的Round，从1开始计数
        rounds = [f"[Round {i+1}]\n\n" for i in range(len(req.chat_history) + 1)]
        rounds_tokens = [
            tokenizer.encode(text=i, add_special_tokens=False) for i in rounds
        ]
        # answer id 需要加上结束符
        answer_ids = tokenizer.encode(
            text=req.input_chat.output, add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        # 如果当前问题+答案就已经超过限制，则放弃这条数据
        input_length = (
            len(input_ids) + len(rounds_tokens[0]) + len(query_ids) + len(answer_ids)
        )
        # 判断prompt+当前对话是否就已经超了
        if input_length > req.max_input_length:
            return None, None

        # 3.对历史对话进行构建，遍历历史对话，在token输入长度允许范围内尽量多的添加历史对话
        history_len = len(req.chat_history)
        history_ids_list = []
        for idx in range(history_len):
            chat = req.chat_history[history_len - 1 - idx]
            history_ids = tokenizer.encode(
                text=f"{conf.user_token}{chat.input}\n\n{conf.assistant_token}{chat.output}\n\n",
                add_special_tokens=False,
            )[
                1:
            ]  # TODO 处理多加的空格30910
            input_length += len(history_ids) + len(rounds_tokens[idx + 1])
            if input_length > req.max_input_length:
                break
            history_ids_list.insert(0, history_ids)  # 前向追加

        # 4.通过当前对话和历史对话，构建数据input_ids
        for i, history_ids in enumerate(history_ids_list):
            input_ids += rounds_tokens[i]  # round-第几轮对话标识
            input_ids += history_ids  # 添加历史对话
        input_ids += rounds_tokens[len(history_ids_list)]  # 添加当前对话的round
        input_ids += query_ids  # 添加当前对话
        # 拿到输入样本，将数据压缩成一个list
        # input_ids = list(functools.reduce(operator.concat, input_ids))

        labels = [tokenizer.pad_token_id] * len(input_ids) + answer_ids  # 构造labels
        input_ids += answer_ids  # 添加当前对话的答案,labels和input_ids的长度一致

        # TODO: 是否有必要按照源码在此时就全部pad到最大长度
        # max_seq_length = req.max_input_length
        # pad_len = max_seq_length - len(input_ids)
        # input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        # labels = labels + [tokenizer.pad_token_id] * pad_len
        if ignore_pad_token_for_loss:
            labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
        return input_ids, labels

    @torch.no_grad()
    def predict(self, req: ModelPredictRequest):
        # TODO:
        #  self.tokenizer.encode("[Round 1]\n\n问：", add_special_tokens=False) 和
        #  self.tokenizer.encode("[Round 1]\n\n", add_special_tokens=False) +
        #  self.tokenizer.encode("问：", add_special_tokens=False) 不一样
        prefix_ids = self.tokenizer.get_prefix_tokens()  # prefix_tokens
        # 1.prompt有输入则进行token否则返回[]，待对话构建完成后再将提示词加到前面
        prompt_ids = (
            prefix_ids + self.tokenizer.encode(req.prompt, add_special_tokens=False)
            if req.prompt
            else prefix_ids
        )
        # 2.当前对话构造，查看是否超过输入长度
        query_ids = self.tokenizer.encode(
            text=f"{self.config.user_token}{req.input_message}\n\n{self.config.assistant_token}",
            add_special_tokens=False,
        )[
            1:
        ]  # TODO 处理多加的空格30910

        # 统计所有可能出现的Round，从1开始计数
        rounds = [
            f"[Round {i + 1}]\n\n"
            for i in range(len(req.front_lock_messages) + len(req.history_messages) + 1)
        ]
        rounds_ids = [
            self.tokenizer.encode(text=i, add_special_tokens=False) for i in rounds
        ]

        token_len = len(query_ids) + len(prompt_ids) + len(rounds_ids[0])
        if token_len > self.config.max_model_input_length:  # prompt+当前对话就已经超了
            logger.error(
                f"prompt+input_message length:{token_len} > "
                f"max_model_input_length:{self.config.max_model_input_length}, return None"
            )
            return None, None

        # 3.构建的前置历史对话
        front_lock_ids = []
        for i, msg in enumerate(req.front_lock_messages):
            front_lock_ids += (
                rounds_ids[i]
                + self.tokenizer.encode(
                    text=f"{self.config.user_token}{msg.input}\n\n{self.config.assistant_token}{msg.output}\n\n",
                    add_special_tokens=False,
                )[1:]
            )  # TODO 处理多加的空格30910

        token_len += len(front_lock_ids)

        for rounds_id in rounds_ids[1 : len(req.front_lock_messages) + 1]:
            token_len += len(rounds_id)

        if token_len > self.config.max_model_input_length:
            logger.error(
                f"prompt+input_message+front_lock_messages length:{token_len} > "
                f"max_model_input_length:{self.config.max_model_input_length}, return None"
            )
            return None, None

        # 4.历史对话
        history_ids_list = []
        history_messages = req.history_messages.copy()
        cut_rounds_ids = rounds_ids[len(req.front_lock_messages) :]
        cut_idx = 0  # 截取使用的历史消息索引
        for i, msg in enumerate(history_messages[::-1]):  # 从后往前遍历
            token_ids = self.tokenizer.encode(
                text=f"{self.config.user_token}{msg.input}\n\n{self.config.assistant_token}{msg.output}\n\n",
                add_special_tokens=False,
            )[
                1:
            ]  # TODO 处理多加的空格30910

            token_len += len(token_ids) + len(cut_rounds_ids[i])
            # 数据达到最大历史长度
            if token_len > self.config.max_model_input_length:
                break
            history_ids_list.append(token_ids)
            cut_idx += 1
        history_len = len(history_messages)
        if history_len > 0:  # 去掉前面不需要的历史对话
            history_messages = history_messages[history_len - cut_idx :]

        # 5.将历史对话和当前对话合并
        input_ids = prompt_ids + front_lock_ids
        for i, history_ids in enumerate(history_ids_list[::-1]):  # 将历史对话按照对话时间先后进行遍历
            input_ids += cut_rounds_ids[i] + history_ids

        input_ids += cut_rounds_ids[len(history_ids_list)] + query_ids

        # inputs = self.model.build_inputs(tokenizer=self.tokenizer, query=req.input_message, history=[])
        # {'input_ids': tensor([[64790, 64792,  790, 30951, 39701,   13,    13, 55437, 31211]], device='cuda:0'),
        #          'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0'),
        #          'position_ids': tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8]],device='cuda:0')}
        input_ids = torch.tensor([input_ids]).to("cuda")
        position_ids = torch.arange(
            input_ids.size(-1), dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        attention_mask = torch.ones(input_ids.size(), dtype=torch.long).to("cuda")
        encoded_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        output = self.model.generate(
            **encoded_input,
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
