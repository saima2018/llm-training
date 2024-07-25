import os.path
from typing import Dict, List

import jieba  # 可以自己选择合适的分词器
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge

from utils.xls import Xls


def rouge_score(expect, predict):
    """
    expect: 标准答案
    predict: 预测答案
    """
    rouge = Rouge()
    scores = rouge.get_scores(predict, expect)[0]  # 只有一个样本
    [
        rouge_1_r,
        rouge_1_p,
        rouge_1_f,
        rouge_2_r,
        rouge_2_p,
        rouge_2_f,
        rouge_l_r,
        rouge_l_p,
        rouge_l_f,
    ] = [
        scores["rouge-1"]["r"],
        scores["rouge-1"]["p"],
        scores["rouge-1"]["f"],
        scores["rouge-2"]["r"],
        scores["rouge-2"]["p"],
        scores["rouge-2"]["f"],
        scores["rouge-l"]["r"],
        scores["rouge-l"]["p"],
        scores["rouge-l"]["f"],
    ]
    return (
        rouge_1_r,
        rouge_1_p,
        rouge_1_f,
        rouge_2_r,
        rouge_2_p,
        rouge_2_f,
        rouge_l_r,
        rouge_l_p,
        rouge_l_f,
    )


def bleu_score(expect, predict):
    """
    expect: 标准答案
    predict: 预测答案
    """
    score = sentence_bleu(
        [list(expect)], list(predict), smoothing_function=SmoothingFunction().method3
    )
    return score


class NLPEvaOut:  # TODO: 是否表格内容都走这里
    def __init__(
        self,
        bleu,
        rouge_1_r,
        rouge_1_p,
        rouge_1_f,
        rouge_2_r,
        rouge_2_p,
        rouge_2_f,
        rouge_l_r,
        rouge_l_p,
        rouge_l_f,
    ):
        self.bleu = bleu
        self.rouge_1_r = rouge_1_r
        self.rouge_1_p = rouge_1_p
        self.rouge_1_f = rouge_1_f
        self.rouge_2_r = rouge_2_r
        self.rouge_2_p = rouge_2_p
        self.rouge_2_f = rouge_2_f
        self.rouge_l_r = rouge_l_r
        self.rouge_l_p = rouge_l_p
        self.rouge_l_f = rouge_l_f
        return

    def list(self):
        return [
            round(self.bleu, 3),
            round(self.rouge_1_r, 3),
            round(self.rouge_1_p, 3),
            round(self.rouge_1_f, 3),
            round(self.rouge_2_r, 3),
            round(self.rouge_2_p, 3),
            round(self.rouge_2_f, 3),
            round(self.rouge_l_r, 3),
            round(self.rouge_l_p, 3),
            round(self.rouge_l_f, 3),
        ]

    def heads(self):
        return [
            "bleu",
            "rouge_1_r",
            "rouge_1_p",
            "rouge_1_f",
            "rouge_2_r",
            "rouge_2_p",
            "rouge_2_f",
            "rouge_l_r",
            "rouge_l_p",
            "rouge_l_f",
        ]


class NLPEvaluator:  # TODO: 目前实现功能代码，代码细节待进一步优化
    def __init__(self, data_list: list):
        self.data_list = data_list
        return

    def run(self):  # TODO：指标方案进一步确认
        for i, chat_dict in enumerate(self.data_list):
            if "evaluation" not in self.data_list[i]:
                self.data_list[i]["evaluation"] = {}

            expect_cut = " ".join(jieba.cut(chat_dict["chat"][-1]["output"]))
            for model_name, predict in chat_dict["predict"].items():
                if model_name not in self.data_list[i]["evaluation"]:
                    self.data_list[i]["evaluation"][model_name] = {}
                predict_cut = " ".join(jieba.cut(predict))

                (
                    rouge_1_r,
                    rouge_1_p,
                    rouge_1_f,
                    rouge_2_r,
                    rouge_2_p,
                    rouge_2_f,
                    rouge_l_r,
                    rouge_l_p,
                    rouge_l_f,
                ) = rouge_score(expect_cut, predict_cut)

                bleu = bleu_score(expect_cut, predict_cut)

                self.data_list[i]["evaluation"][model_name] = NLPEvaOut(
                    bleu,
                    rouge_1_r,
                    rouge_1_p,
                    rouge_1_f,
                    rouge_2_r,
                    rouge_2_p,
                    rouge_2_f,
                    rouge_l_r,
                    rouge_l_p,
                    rouge_l_f,
                )

        return

    def save(self, output_dir: str):
        save_file = os.path.join(output_dir, "evaluation_nlp.xls")
        sheet_dict: Dict[str, List[List[str]]] = dict()
        for i, chat_dict in enumerate(self.data_list):
            for name, eva in chat_dict["evaluation"].items():
                if name not in sheet_dict:
                    sheet_dict[name] = [["chat", "expect", "predict"] + eva.heads()]
                sheet_dict[name].append(
                    [
                        str(chat_dict["chat"]),
                        chat_dict["chat"][-1]["output"],
                        chat_dict["predict"][name],
                    ]
                    + eva.list()
                )

        xls = Xls()
        xls.write(sheet_dict, save_file)
        return
