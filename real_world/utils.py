# -*- coding: utf-8 -*- 
# @Time : 2023/8/11 13:56 
# @Author : DirtyBoy 
# @File : utils.py
from transformers import GPT2Tokenizer, LlamaTokenizer


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


def get_input_ids(model_dir, text, model):
    if 'gpt2' in model_dir:
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    input_ids = tokenizer(text, return_tensors="pt")
    input_ids_len = len(input_ids['input_ids'][0])
    return input_ids, input_ids_len


def test_llama_model(model, input_ids):
    outputs = model(**input_ids, labels=input_ids["input_ids"])
    loss = outputs.loss
    return loss.detach().numpy()


def test_gpt2_model(model, input_ids):
    outputs = model(**input_ids, labels=input_ids["input_ids"])
    loss = outputs.loss
    return loss.detach().numpy()


def get_samples_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines
