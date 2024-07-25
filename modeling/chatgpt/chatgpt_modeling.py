# ----------------------------------
# openai
# 网络结构: transformer
# 模型大小: 175B
# 最大输入: -
# token数: -
# ----------------------------------

import json

import requests

from modeling.interface import ChatData, ModelPredictRequest
from utils import logger


class ModelChatGPT:
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        self.headers = {
            "Authorization": "Bearer xxx"
        }
        self.url = "https://x/v1/chat/completions"  # -stream'
        self.max_model_input_length = 4096
        return

    def predict(self, req: ModelPredictRequest):
        messages = (
            [{"role": "system", "content": req.prompt}] if req.prompt else []
        )  # 考虑prompt
        for msg in req.front_lock_messages:
            messages.append({"role": "user", "content": msg.input})
            messages.append({"role": "assistant", "content": msg.output})
        for msg in req.history_messages:  # TODO 保留多长的token避免越界
            messages.append({"role": "user", "content": msg.input})
            messages.append({"role": "assistant", "content": msg.output})
        messages.append({"role": "user", "content": req.input_message})

        # TODO: 确认是否req所有参数有效
        params = {
            "model": self.model_name_or_path,  # gpt-3.5-turbo or gpt-4
            "messages": messages,
            "max_tokens": req.max_new_tokens,
            "temperature": req.temperature,
            # "stream": True,
        }

        resp = requests.post(
            self.url, data=json.dumps(params), headers=self.headers, timeout=180
        ).json()
        logger.info(f"model predict:\n{resp}")

        result = resp["choices"][0]["message"]["content"]

        history_messages = req.history_messages.copy()
        history_messages.append(
            ChatData(input=req.input_message, output=result)
        )  # {'input': req.input_message, 'output': result})

        # token_len = resp['usage']['total_tokens']
        return result, history_messages
