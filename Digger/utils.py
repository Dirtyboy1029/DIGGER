# -*- coding: utf-8 -*- 
# @Time : 2023/8/11 13:56 
# @Author : DirtyBoy 
# @File : utils.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import pickle


def metrics(target, ref):
    return np.exp(np.array(ref) - np.array(target))


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


def save2pkl(save_path, model_likelihood_dict):
    f_save = open(save_path, 'wb')
    pickle.dump(model_likelihood_dict, f_save)
    f_save.close()


def read_pkl(save_path):
    f_read = open(save_path, 'rb')
    dict2 = pickle.load(f_read)
    f_read.close()
    return dict2


def remove_outliers(data, ratio=10):
    # 计算上下四分位数
    q1 = np.percentile(data, ratio)
    q3 = np.percentile(data, 100 - ratio)

    # 计算四分位距
    iqr = q3 - q1

    # 定义异常值的上下界限
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # 去除异常值
    filtered_data = [x for x in data if x >= lower_bound and x <= upper_bound]

    return filtered_data
