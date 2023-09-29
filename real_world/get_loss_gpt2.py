# -*- coding: utf-8 -*- 
# @Time : 2023/9/11 16:31 
# @Author : DirtyBoy 
# @File : get_loss_gpt2.py
import os, argparse, torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import get_samples_list, get_input_ids, test_gpt2_model
import numpy as np


def RQ3(token_train, times, num):
    if token_train == 128:
        token_len_list = [90, 100]
    elif token_train == 256:
        token_len_list = [240, 250]
    else:
        token_len_list = []
    if times == 0:
        model_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/GPT2/gpt2-xl'
    else:
        model_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/copy_right/GPT2/real_world/times' + str(
            times) + '/gpt2-xl'
    save_path = 'outputs/gpt2-xl/token_' + str(token_train)
    samples_dir_path = 'Datasets/samples_set/real_world'

    print('load base model from ' + model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    print('<<<model load finish>>>')
    file_list = os.listdir(samples_dir_path)
    for file_i, file in enumerate(file_list):
        data = get_samples_list(os.path.join(samples_dir_path, file))
        goal_list = []
        np_save_path = os.path.join(save_path, str(times) + '_' + file.split('.')[0] + '_' + str(num) + '.npy')
        for i, item in enumerate(data[25 * (num - 1):25 * num]):
            input_ids, input_ids_len = get_input_ids(model_dir, item)
            loss_tmp = []
            if input_ids_len >= token_len_list[-1]:
                for token_len in token_len_list:
                    new_input_ids = {'input_ids': input_ids['input_ids'][:, :token_len],
                                     'attention_mask': input_ids['attention_mask'][:, :token_len]}
                    loss = test_gpt2_model(model, new_input_ids)
                    loss_tmp.append(loss)
                    print(loss)
            goal_list.append(loss_tmp)
            print(os.path.splitext(file)[0] + ' No.' + str(i + 1) + ' sample!!!')
            print('save file to ' + np_save_path)
        np.save(np_save_path, goal_list)
        print('**********************************************************')
        print('No. ' + str(file_i + 1) + ' book ' + os.path.splitext(file)[0] + '.npy' + ' save finish!')
        print('**********************************************************')


if __name__ == '__main__':
    for token in [128]:
        for times in [1]:
            for num in [1, 2, 3, 4]:
                RQ3(token, times, num)
