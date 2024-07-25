import os.path
import re
from typing import Dict, List

from modeling import ModelingFactory, ModelPredictRequest
from utils import logger
from utils.xls import Xls


def cut(obj, sec):
    return [obj[i : i + sec] for i in range(0, len(obj), sec)]


class GPT4Evaluator:
    def __init__(self, data_list: list):
        self.data_list = data_list
        self.model = ModelingFactory.get(
            modeling_name="chatgpt", model_name_or_path="gpt-4"
        )
        self.rating_prompt = """For each of the questions, give a rating on scale of 100 to the models' answer respectively based on the reference answer in the following format:
```Question 1: model1 score: int,  model2 score: int\nQuestion 2: model1 score: int,  model2 score: int\nQuestion 3: model1 score: int,  model2 score: int\n```                    
"""
        return

    @staticmethod
    def get_input_prompt(rating_prompt, model_names, data_list):
        input_prompt = rating_prompt
        for i, chat_dict in enumerate(data_list):
            line = f"Question: {chat_dict['chat'][-1]['input']} Reference answer: {chat_dict['chat'][-1]['output']} "
            for j, model_name in enumerate(model_names):
                line += f"model{j + 1} answer: {chat_dict['predict'][model_name]} "
            input_prompt += line + "\n"
        return input_prompt

    @staticmethod
    def get_scores(score_txt):
        lines = score_txt.split("\n")
        score_dict = {}
        for i, line in enumerate(lines):
            # 正则匹配提取数字 eg: 'Question 2: model1 score: 40,  model2 score: 90' -> {'idx': '2', 'model1': '40', 'model2': '90'}
            logger.info(f"line: {line}")
            res = re.search(
                "Question (?P<idx>\d+): model1 score: (?P<model1>\d+),  model2 score: (?P<model2>\d+)",
                line,
            )
            logger.info(f"re.search: {res}")
            if res:
                ground_dict = res.groupdict()
                logger.info(f"ground_dict: {ground_dict}")
                score_dict[int(ground_dict["idx"])] = [
                    int(ground_dict["model1"]),
                    int(ground_dict["model2"]),
                ]
        return score_dict

    def run(self, max_count=8):  # TODO:当前只支持2个模型对比，需要支持任意多个模型对比
        """
        max_count: 最大同时请求几个问题
        """
        model_names = list(self.data_list[0]["predict"].keys())  # TODO: 代码优化
        # 1.cut切数据
        cut_data_lists = [
            self.data_list[i : i + max_count]
            for i in range(0, len(self.data_list), max_count)
        ]

        for i, cut_data_list in enumerate(cut_data_lists):
            input_prompt = self.get_input_prompt(
                self.rating_prompt, model_names, cut_data_list
            )

            # predict
            req = ModelPredictRequest()
            req.input_message = input_prompt
            req.max_new_tokens = 500
            score_txt, _ = self.model.predict(req)
            score_dict = self.get_scores(score_txt)

            for j, dat in enumerate(cut_data_list):
                idx = i * max_count + j
                if "evaluation" not in self.data_list[idx]:
                    self.data_list[idx]["evaluation"] = {}

                scores = (
                    score_dict[j + 1] if j + 1 in score_dict else [-1, -1]
                )  #  TODO 评分失败用-1
                self.data_list[idx]["evaluation"][model_names[0]] = scores[0]
                self.data_list[idx]["evaluation"][model_names[1]] = scores[1]
        return

    def save(self, output_dir: str):
        save_file = os.path.join(output_dir, "evaluation_gpt4.xls")
        sheet_dict: Dict[str, List[List[str]]] = dict()
        for i, chat_dict in enumerate(self.data_list):
            for name, score in chat_dict["evaluation"].items():
                if name not in sheet_dict:
                    sheet_dict[name] = [["chat", "expect", "predict", "score"]]
                sheet_dict[name].append(
                    [
                        str(chat_dict["chat"]),
                        chat_dict["chat"][-1]["output"],
                        chat_dict["predict"][name],
                        score,
                    ]
                )

        xls = Xls()
        xls.write(sheet_dict, save_file)
        return
