# -*- coding: utf-8 -*- 
# @Time : 2023/8/11 13:56 
# @Author : DirtyBoy 
# @File : utils.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()

def get_input_ids(model_dir, text):
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    input_ids = tokenizer(text, return_tensors="pt")
    input_ids_len = len(input_ids['input_ids'][0])
    return input_ids, input_ids_len

def test_gpt2_model(model, input_ids):
    outputs = model(**input_ids, labels=input_ids["input_ids"])
    loss = outputs.loss
    return loss.detach().numpy()

def get_samples_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

GPT2_model = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
