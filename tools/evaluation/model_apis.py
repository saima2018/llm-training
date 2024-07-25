import json
import os

import dotenv

dotenv.load_dotenv()
MUXUE_KEY = os.environ.get("MUXUE_KEY")
headers = {"Authorization": f"Bearer {MUXUE_KEY}"}


async def gpt_api(question, url, llm_params, session):
    """
    openai接口，输入问题，返回字符串回答
    """
    params = {
        "model": llm_params.model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": llm_params.max_new_tokens,
        "temperature": llm_params.temperature,
        "top_p": llm_params.top_p,
        "n": llm_params.top_k,
        "stream": False,
    }
    async with session.post(url, json=params, headers=headers) as response:
        r = await response.read()
        r = json.loads(r.decode())
        answer = r["choices"][0]["message"]["content"]
    return answer


async def model_api(question, url, llm_params, session):
    """
    模型接口，输入问题，返回字符串回答
    """
    params = {
        "prompt": question.replace("\noutput: ", "") + "\n<|assistant|>: ",
        "max_new_tokens": llm_params.max_new_tokens,
        "temperature": llm_params.temperature,
        "top_p": llm_params.top_p,
        "top_k": llm_params.top_k,
    }
    async with session.post(url, json=params) as response:
        r = await response.read()
        r = json.loads(r.decode())
        answer = r["message"].split("<|assistant|>: ")[-1][:-8]
    # print('content：', answer)
    return answer
