import json
import multiprocessing
import os
import sys
from itertools import chain

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils.io_file import find_files_from_extension
from utils.logger import logger


class PretrainDataset(Dataset):
    def __init__(self, train_folder, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.targets = []

        self.tasks = []

        if os.path.exists(
            os.path.join(train_folder, "pretrain_input_ids.cache")
        ) and os.path.exists(os.path.join(train_folder, "pretrain_target.cache")):
            self.input_ids = torch.load(
                os.path.join(train_folder, "pretrain_input_ids.cache")
            )
            self.targets = torch.load(
                os.path.join(train_folder, "pretrain_target.cache")
            )
        else:
            self.input_ids, self.targets = self.read_files(train_folder, max_length)
            torch.save(
                self.input_ids, os.path.join(train_folder, "pretrain_input_ids.cache")
            )
            torch.save(
                self.targets, os.path.join(train_folder, "pretrain_target.cache")
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.targets[idx]

    def collate_fn(self, batch):
        batch_input_ids, batch_targets = [], []
        for input_ids, targets in batch:
            batch_input_ids.append(input_ids)
            batch_targets.append(targets)

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.unk_token_id
        )
        batch_targets = torch.nn.utils.rnn.pad_sequence(
            batch_targets, batch_first=True, padding_value=self.tokenizer.unk_token_id
        )

        return batch_input_ids, batch_targets

    def truncate_long_text(self, texts, max_token_limit, stride, file_path):
        token_chunks = {}
        token_chunks["input_ids"] = []
        token_chunks["labels"] = []

        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_token_limit:
                token_chunk = tokens[: max_token_limit - 1]
            else:
                token_chunk = tokens

            token_chunks["input_ids"].append(token_chunk)
            token_chunks["labels"].append(token_chunk)

            # Todo 添加合适的切分策略

            # print(self.tokenizer.eos_token)
            # eos_token_id = self.tokenizer.encode(tokenizer.eos_token).ids[0]

            # start = 0

            # bk_ids = [self.tokenizer.encode(i) for i in ["。", "\n", "\r\n", "\r", "."]]

            # print(f"tokenizing... {file_path}")

            # while start < len(tokens):
            #     print(f"tokenizing... {file_path}, {start/len(tokens) * 100:.2f}% \r")
            #     break_flag = False
            #     end = min(start + max_token_limit, len(tokens) - 1)
            #     cnt_flag = 0
            #     while end > start and tokens[end] not in bk_ids:
            #         end -= 1
            #         cnt_flag += 1
            #         if cnt_flag > 40:
            #             # print('cannot find end of sentence')
            #             break_flag = True
            #             break
            #     token_chunk = tokens[start:end]
            #     if token_chunk:
            #         if not break_flag:
            #             token_chunk[-2:] = [
            #                 self.tokenizer.eos_token_id,
            #                 self.tokenizer.unk_token_id,
            #             ]
            #         else:
            #             token_chunk[-2:] = [
            #                 self.tokenizer.unk_token_id,
            #                 self.tokenizer.unk_token_id,
            #             ]
            #         target_chunk = token_chunk[1:]
            #         target_chunk.append(self.tokenizer.unk_token_id)
            #         token_chunks["input_ids"].append(token_chunk)
            #         token_chunks["labels"].append(target_chunk)
            #         start += stride

            #     elif start > (len(tokens) - 200):
            #         break

            # print("tokenized")

        return token_chunks

    def read_files(self, train_folder, max_length):
        input_ids = []
        target_ids = []
        # input_ids = []
        with multiprocessing.Pool(180) as pool:
            for root, dirs, files in os.walk(train_folder):
                file_paths = [
                    os.path.join(root, file) for file in files if file.endswith(".txt")
                ]
                print(f"reading {len(file_paths)} files")
                chunks_list = pool.starmap(
                    self.read_file,
                    [(file_path, max_length) for file_path in file_paths],
                )
                # chunks_list = self.read_file(file_paths[0], max_length, tokenizer)
                # print("chunks_list length: ", len(chunks_list))
                for chunks in chunks_list:
                    if not chunks:
                        continue
                    for ids, labels in zip(chunks["input_ids"], chunks["labels"]):
                        # print(f"chunks length: {len(chunks)},chunks_list length: {len(chunks_list)}")
                        input_ids.append(torch.tensor(ids))
                        target_ids.append(torch.tensor(labels))
        return input_ids, target_ids

    def read_file(self, file_path, max_length):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.readlines()
                # print(f"content length: {len(content)} ,file: {file_path} ")
                chunks = self.truncate_long_text(content, max_length, 256, file_path)
                # print(f"chunks length: {len(chunks)} file {file_path}")
                return chunks
        except:
            # print('error reading file: ', file_path)
            return []
