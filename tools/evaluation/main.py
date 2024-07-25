import os
import pprint
import sys

from jsonlines import jsonlines

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config import config
from modeling import ModelingFactory, ModelPredictRequest
from modeling.interface import ChatData
from tools.evaluation.evaluators.gpt4_evaluator import GPT4Evaluator
from tools.evaluation.evaluators.nlp_evaluator import NLPEvaluator
from utils import add_file_handler_to_logger, find_files_from_extension, logger


def main():
    # 1.logger配置与config输出
    os.makedirs(config.EVALUATION.OUTPUT_DIR, exist_ok=True)
    add_file_handler_to_logger(
        dir_path=config.EVALUATION.LOG_DIR, name="evaluation"
    )  # 增加训练log输出
    logger.info(
        f"config:\n{pprint.PrettyPrinter(indent=1).pformat(config.as_dict()['EVALUATION'])}"
    )  # 打印config

    # 2.读取输入数据
    logger.info("start read files")
    files = find_files_from_extension(
        data_dir=config.EVALUATION.INPUT_DIR, extensions=[".jsonl"]
    )
    data_list = []
    for f in files:
        with jsonlines.open(f) as reader:
            for obj in reader:  # 每行数据包含chat，可能包含prompt
                data_list.append(obj)
        reader.close()

    # 3.启动模型预测
    logger.info("start model predict")
    eva_models = config.EVALUATION.EVA_MODELS.keys()
    for k in eva_models:
        model_conf = config.EVALUATION.EVA_MODELS[k]
        if not model_conf.get("ENABLE", False):  # 没有ENABLE的模型不参与评测
            continue
        modeling_name = model_conf.get("MODELING_NAME", None)
        model_name_or_path = model_conf.get("MODEL_NAME_OR_PATH", None)

        logger.info(f"predict model {k}，{modeling_name}:{model_name_or_path}")

        modeling = ModelingFactory.get(
            modeling_name=modeling_name, model_name_or_path=model_name_or_path
        )
        if modeling_name != "chatgpt":  # 区别gpt
            modeling.model.half().cuda()  # gpt不适用

        for i, chat_dict in enumerate(data_list):
            if "predict" not in data_list[i]:
                data_list[i]["predict"] = {}

            req = ModelPredictRequest()  # TODO: 设置更多值
            # TODO: 补充历史对话 prompt front_lock_messages history_messages
            for dat in chat_dict["chat"][:-1]:  # 取最后一轮之前的为历史对话
                req.history_messages.append(
                    ChatData(input=dat["input"], output=dat["output"])
                )
            req.input_message = chat_dict["chat"][-1]["input"]
            req.max_new_tokens = 500
            resp, _ = modeling.predict(req)
            data_list[i]["predict"][k] = resp
    with jsonlines.open(
        os.path.join(config.EVALUATION.OUTPUT_DIR, "predict.jsonl"), mode="w"
    ) as writer:
        for dat in data_list:  # 每行数据包含chat，可能包含prompt
            writer.write(dat)
        writer.close()

    # data_list = []
    # with jsonlines.open("output/evaluation/predict.jsonl") as reader:
    #     for obj in reader:  # 每行数据包含chat，可能包含prompt
    #         data_list.append(obj)
    # reader.close()

    # 4.启动模型评测 (客观指标(rouge/bleu)和半客观指标(gpt4)) # TODO:gpt4未集成到这里
    logger.info("start evaluate")
    if config.EVALUATION.EVALUATORS.NLP:
        logger.info("start nlp evaluate")
        nlp_evaluator = NLPEvaluator(data_list=data_list)
        nlp_evaluator.run()
        nlp_evaluator.save(config.EVALUATION.OUTPUT_DIR)

    if config.EVALUATION.EVALUATORS.GPT4:
        logger.info("start gpt4 evaluate")
        gpt4_evaluator = GPT4Evaluator(data_list=data_list)
        gpt4_evaluator.run()
        gpt4_evaluator.save(config.EVALUATION.OUTPUT_DIR)

    return


if __name__ == "__main__":
    main()
