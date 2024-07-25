from typing import List

from pydantic import BaseModel


# 调用模型的predict方法需要输入的参数 dataclass只能用基础数据类型
class ModelPredictRequest(BaseModel):  # 引入BaseModel方便dataclass与json等结构的转化
    prompt: str = ""  # 提示词
    front_lock_messages: list = []  # prompt后紧跟的不消除的前置对话
    history_messages: list = []  # 历史对话  # TODO: dadaclass不能指定class类型，如何确认输入规范
    input_message: str = ""  # 当前提问
    temperature: float = 0.7  # 温度
    top_p: float = 0.9  # 保留概率从大到小之和在top_p范围内的值进行取值
    top_k: int = 5  # 保留预测概率从大到小排序的前top_k个进行取值
    repetition_penalty: float = 1.0  # 重复惩罚的参数。在 1.0 和无穷大之间。1.0 意味着没有惩罚。默认为 1.0
    max_new_tokens: int = 100  # 要生成的序列的最大长度


# 定义一条问答的数据结构
class ChatData:
    def __init__(self, input: str, output: str):
        """
        input: 用户提问
        output: AI回答
        """
        self.input = input
        self.output = output


# 调用模型的token_encode_sft和token_encode_pretrain需要输入的参数
class ModelTokenEncodeRequest:
    """
    modeling的encode请求需要的输入
      - 当前提供给dataset来调用
    """

    def __init__(
        self,
        max_input_length: int,
        prompt: str,
        input_chat: ChatData,
        chat_history: List[ChatData],
    ):
        """
        max_input_length: 构建的数据最大输入长度
        prompt: 提示词
        input_chat: 当前对话，格式example: {"input": "成都在哪里", "output": "四川"}
        chat_history: 历史对话，格式example: [{"input": "遵义在哪里", "output": "贵州"},{"input": "绵阳呢", "output": "四川"}]
        """
        self.max_input_length = max_input_length
        self.prompt = prompt
        self.input_chat = input_chat
        self.chat_history = chat_history
