# -*- coding: utf-8 -*- 
# @Time : 2023/8/11 13:46 
# @Author : DirtyBoy 
# @File : trainset_builder.py
import os, torch, random
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset
from typing import Dict
from utils import txt_to_list

class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer, file_list, block_size: int):
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        print(f"Creating features from dataset file")
        lines = file_list
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


def load_dataset(file_list, tokenizer, block_size):
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_list=file_list,
        block_size=block_size,
    )
    return dataset


def load_data_collator(tokenizer):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False
    )
    return data_collator


def build_GPT2_trainset(source_data_path, tokenizer, block_size, times=1):
    novel_list = os.listdir(source_data_path)
    goal_list = []
    for item in novel_list:
        txt_list = txt_to_list(os.path.join(source_data_path, item))
        goal_list = goal_list + txt_list
    goal_list = goal_list * times
    random.shuffle(goal_list)
    dataset = load_dataset(goal_list, tokenizer, block_size)
    data_collator = load_data_collator(tokenizer)
    return dataset, data_collator
