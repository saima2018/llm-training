import json
import requests
import sys
import os
import httpx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from modeling.interface import ModelPredictRequest


class LLMInfer:
    def __init__(self, url: str, port: int):
        self.url = url
        self.port = port
        return

    def chat(self, request: ModelPredictRequest):
        req = {"inputs": request.input_message,
               "history_messages": request.history_messages,
               "temperature": request.temperature,
               "top_p": request.top_p,
               "top_k": request.top_k,
               "max_new_tokens": request.max_new_tokens,
               "repetition_penalty": request.repetition_penalty,
               "do_sample": True}
        timeout = 60
        timeout = httpx.Timeout(timeout)
        url = "http://36.111.142.249:8877/firefly"

        headers = {"Content-Type": "application/json", "Connection": "close"}
        session = httpx.Client(base_url="", headers=headers)
        response = session.request("POST", url, json=req, timeout=timeout)

        out = json.loads(response.text)['output']
        return out, []

    def is_ready(self):
        resp = requests.get(f"{self.url}:{self.port}/ready")
        out = resp.json()["status"]
        return out

    def is_health(self):
        resp = requests.get(f"{self.url}:{self.port}/healthz")
        out = resp.json()["status"]
        return out