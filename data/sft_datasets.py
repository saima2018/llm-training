import json
import os
import sys
from typing import Callable, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from modeling.config import ModelConfig
from modeling.interface import ChatData, ModelTokenEncodeRequest
from utils import find_files_from_extension
from utils.logger import logger


class SFTDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: Optional[AutoTokenizer.from_pretrained],
        model_conf: ModelConfig,
        token_encode_function: Callable,
        max_seq_len: int,
        use_cache: bool = False,
        cache_dir: str = None,
        num_workers: int = 200,
    ):
        """
        data_dir: 数据加载地址
        tokenizer: 模型对应的tokenizer类，用以进行编码
        model_conf: basic model configs, such as: pad_token_id during batch processing, max_model_input_length, etc.
        token_encode_function: text token encoding function, to adapt to different models
        max_data_input_length: 设置数据最大输入长度，超过max_model_input_length时将取max_model_input_length
        use_cache: 是否使用缓存，使用缓存时将加载上一次缓存，模型(tokenizer)或数据改变时请不要用cache,use_cache设置为False
        cache_dir: 缓存数据存储地址，为None时数据将放在data_dir文件夹里
        num_workers: 并行加载读取数据的并行数量
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.model_conf = model_conf
        self.token_encode_function = token_encode_function
        # self.max_input_length = self._get_max_input_length(
        #     self.model_conf.max_model_input_length, max_seq_len
        # )
        self.max_input_length = max_seq_len
        self.use_cache = use_cache
        self.cache_dir = self._get_cache_dir(data_dir, cache_dir)
        self.train_dataset = self._load_data(
            extensions=[".jsonl"], num_workers=num_workers
        )  # 并行加载读取数据
        logger.info("finish load SFTDataset")

    def __len__(self):
        return self.train_dataset.num_rows

    def __getitem__(self, idx):
        return self.train_dataset[idx]["input_ids"], self.train_dataset[idx]["labels"]

    def _load_data(self, extensions, num_workers=200):
        # TODO: make sure only one copy of dataset is loaded in distributed training
        logger.info("start load_data find_files_from_extension")
        # 提取后缀名符合条件的文件
        data_files = find_files_from_extension(self.data_dir, extensions)
        logger.info(f"find files: {data_files}")

        logger.info("start load_dataset")
        # TODO: not using cache
        raw_datasets = load_dataset(
            "text", data_files=data_files, cache_dir=None
        )
        logger.info("start _tokenize_function")
        # process data in parallel using tokenize_function
        train_dataset = raw_datasets.map(
            self._tokenize_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=["text"],
            load_from_cache_file=self.use_cache,
            desc="Running tokenizer on dataset",
        )

        logger.info(
            f"finish load_data, data num_rows={train_dataset['train'].num_rows}\n"
        )
        if train_dataset["train"].num_rows > 0:
            logger.info(
                f"example is:\n{self.tokenizer.decode(train_dataset['train'][1]['input_ids'])}"
            )
        return train_dataset["train"]

    @staticmethod
    def _get_max_input_length(model_max_input_length, max_input_length):
        # take the smaller of user input and model max input lengths as max_input_length
        input_length = max_input_length
        if input_length > model_max_input_length:
            input_length = model_max_input_length
            logger.warning(
                f"input max_input_length({max_input_length}) > "
                f"modeling's max_input_length({model_max_input_length}), will use {input_length}"
            )
        return input_length

    @staticmethod
    def _get_cache_dir(data_dir, cache_dir):
        # 缓存数据存储地址，为None时数据将放在data_dir文件夹里
        if cache_dir is None:
            out_cache_dir = os.path.join(data_dir, "cache")
            logger.warning(f"input cache_dir is None, will use {out_cache_dir}")
        else:
            out_cache_dir = cache_dir
        os.makedirs(out_cache_dir, exist_ok=True)
        return out_cache_dir

    def _tokenize_function(self, examples, text_column_name="text"):
        model_inputs = {"input_ids": [], "labels": []}
        for chat in examples[text_column_name]:
            chat = json.loads(chat)
            # 对每一轮对话进行构建ChatData
            input_chat = ChatData(input=chat["input"])
            labels = chat['label']
            # 构造ModelTokenEncodeRequest进行encode
            req = ModelTokenEncodeRequest(
                max_input_length=self.max_input_length,
                prompt="",
                input_chat=input_chat,
                chat_history="", # TODO
            )
            # 调用modeling的token_encode方法进行模型输入的token构建
            input_ids = self.token_encode_function(
                tokenizer=self.tokenizer, conf=self.model_conf, req=req
            )

            # 追加history, 本行不能放在modeling.token_encode之前，否则会导致数据重复
            # chat_history.append(input_chat)
            if input_ids is None:
                continue
            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def _tokenize_function_chat(self, examples, text_column_name="text"):
        model_inputs = {"input_ids": [], "labels": []}
        for chats_txt in examples[text_column_name]:
            chat_dict = json.loads(chats_txt)
            if "chat" not in chat_dict or len(chat_dict) == 0:
                continue
            chats = chat_dict["chat"]  # TODO: add 空dict处理
            prompt = chat_dict["prompt"] if "prompt" in chats else ""

            # 遍历每一轮对话，后面的对话加上前面的对话历史作为输入
            # TODO: 是否支持不相关历史对话
            chat_history = []
            for chat in chats:
                # 对每一轮对话进行构建ChatData
                input_chat = ChatData(input=chat["input"], output=chat["output"])
                # 构造ModelTokenEncodeRequest进行encode
                req = ModelTokenEncodeRequest(
                    max_input_length=self.max_input_length,
                    prompt=prompt,
                    input_chat=input_chat,
                    chat_history=chat_history,
                )
                # 调用modeling的token_encode方法进行模型输入的token构建
                input_ids, labels = self.token_encode_function(
                    tokenizer=self.tokenizer, conf=self.model_conf, req=req
                )

                # 追加history, 本行不能放在modeling.token_encode之前，否则会导致数据重复
                chat_history.append(input_chat)
                if input_ids is None or labels is None:
                    continue
                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)
        return model_inputs

    def collate_fn(self, batch):  # 提供给data loader用，对每个batch进行输入模型前的最后处理
        # 将每个batch的长度pad到一样，支持模型进行batch训练
        batch_input_ids, batch_targets = [], []
        for input_ids, labels in batch:
            # 将tokenizer.encode生成的token list转为torch tensor
            batch_input_ids.append(torch.tensor(input_ids))
            batch_targets.append(torch.tensor(labels))
        # 对长度不够的数据在后面追加模型提供的pad_token_id
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.model_conf.pad_token_id,
        )
        print('bbbbbbbbb tttttttttt', batch_targets)
        # batch_targets = torch.nn.utils.rnn.pad_sequence(
        #     batch_targets, batch_first=True, padding_value=self.model_conf.pad_token_id
        # )
        batch_targets = torch.stack(batch_targets)
        print('2222222222bbbbbbbbb tttttttttt', batch_targets)
        return batch_input_ids, batch_targets