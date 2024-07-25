"""
模型对比评测功能模块：
  1. 用自训模型和gpt3.5分别回答验证集的问题
  2. 用gpt4基于验证集的参考答案为两个模型生成的回答打分
  3. 验证集数据为与sft微调训练相同的jsonl文件
"""
import ast
import asyncio
from typing import Callable, List

from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio

from .configs import llm_params, prompts, rating_prompt
from .model_apis import gpt_api, model_api


class GPTEval:
    """
    读取现有的评测数据，分别调用gpt3.5和待测模型生成答案，并用gpt4对照参考数据分别打分
    """

    def __init__(
        self,
        eval_data_path="./tools/evaluation/materials/dany_eval.jsonl",
        openai_url="https://openai.muxuekeji.cn/v1/chat/completions",
        model_url="http://127.0.0.1:8000/llm/inference",
    ):
        self.path = eval_data_path
        self.openai_url = openai_url
        self.model_url = model_url
        self.roles = []
        self.individual_questions = []  # 评测集中的单条问题 List[List]
        self.questions = []  # 组合出对话历史的问题
        self.reference_answers = []  # 评测数据集的单条答案
        self.gpt_answers = []  # gpt3.5的单条答案
        self.model_answers = []  # 待测模型的单条答案

    def read_file(self):
        """
        读取jsonl评测数据，抽取角色list、答案list，组合多轮问题list
        """
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = ast.literal_eval(line)
                self.roles.append(line["role"])
                questions = []
                current_q = ""
                for qa_pair in line["chat"]:
                    self.individual_questions.append(qa_pair["input"])
                    current_q += "input: " + qa_pair["input"]
                    questions.append(current_q)
                    current_q += "output: " + qa_pair["output"]
                    self.reference_answers.append(qa_pair["output"])
                self.questions.append(questions)

    async def generate_answers(self, llm_params, model) -> List:
        """
        让model为评测数据生成答案,并以list格式返回
        :param questions: 评测数据中的用户输入，同一条多轮对话评测数据的对话历史要整合后输入模型
        :param prompt: 角色通用的提示词
        :return: 直接将所有答案放入一个list，交给打分方法汇总整理为所需格式
        """
        if model == "gpt":
            url = self.openai_url
            inference_model = gpt_api
            answers = self.gpt_answers
        else:
            url = self.model_url
            inference_model = model_api
            answers = self.model_answers
        # print('qqq', self.questions)
        tasks = []
        async with ClientSession() as session:
            for i in range(len(self.questions)):
                for j in range(len(self.questions[i])):
                    prompt = prompts[self.roles[i]]
                    question = self.questions[i][j]
                    combined_question = prompt + question + "\noutput: "
                    task = asyncio.ensure_future(
                        inference_model(combined_question, url, llm_params, session)
                    )
                    tasks.append(task)
            results = await tqdm_asyncio.gather(*tasks)
        for result in results:
            answers.append(result)

    def run_in_loop(self, llm_params, func_async: Callable, **kwargs):
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(func_async(llm_params, **kwargs))
        loop.run_until_complete(future)

    async def ratings(self, llm_params):
        """
        使用gpt4为两种模型的答案分别打分
        """
        tasks = []
        rating_input = rating_prompt
        async with ClientSession() as session:
            count = -1
            for i in range(len(self.questions)):
                for j in range(len(self.questions[i])):
                    count += 1
                    line = (
                        "Question: "
                        + self.individual_questions[count]
                        + " Reference answer: "
                        + self.reference_answers[count]
                        + " GPT answer: "
                        + self.gpt_answers[count]
                        + " Model answer: "
                        + self.model_answers[count]
                    )
                    rating_input += line + "\n"
                # print(rating_input)
                llm_params.model = "gpt-4"
                task = asyncio.ensure_future(
                    gpt_api(rating_input, self.openai_url, llm_params, session)
                )
                tasks.append(task)
                rating_input = rating_prompt
            results = await tqdm_asyncio.gather(*tasks)
        count = 0
        for result in results:
            count += 1
            print(f"Sample {count}: \n", result)


def main():
    d = GPTEval(
        eval_data_path="./tools/evaluation/materials/dany_eval.jsonl",
        openai_url="https://openai.muxuekeji.cn/v1/chat/completions",
        model_url="http://127.0.0.1:8000/llm/inference",
    )
    d.read_file()  # 读取评测数据
    d.run_in_loop(llm_params, d.generate_answers, model="gpt")  # 使用gpt3.5生成对话
    d.run_in_loop(llm_params, d.generate_answers, model="baichuan")  # 使用待评测模型生成对话
    d.run_in_loop(llm_params, d.ratings)  # 使用gpt4为以上两种模型对比打分


if __name__ == "__main__":
    # 运行说明：
    # 1.在model_apis.py里配置环境变量MUXUE_KEY，以使用openai接口
    # 2.运行deploy/api_server/main.py启动待测模型，或使用待测的模型的接口
    # 3.在main方法中指定评测数据路径，格式参考tools/evaluation/materials路径下的.jsonl文件
    # 4.在llm_model根目录运行 python -m tools.evaluation.run_evaluation，评测结果会打印到控制台
    main()
