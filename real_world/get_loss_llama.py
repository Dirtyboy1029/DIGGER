# -*- coding: utf-8 -*- 
# @Time : 2023/9/1 21:46 
# @Author : DirtyBoy 
# @File : get_loss_llama.py
import os, argparse, torch
from transformers import LlamaForCausalLM, GPT2Tokenizer
from utils import get_samples_list, get_input_ids, test_llama_model
import numpy as np


def RQ3(model_type, data_type, num):
    token_len_list = [100]
    if model_type == 'vanilla':
        model_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/huggyllama/llama-7b'
    elif model_type == 'union':
        model_dir = '/home/lhd/Copy_Right_of_LLM/myexperiment/RQ3/outputs/union/quote/merge'
    elif model_type == 'target':
        model_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/copy_right/LLaMA/tune/7b/times1/merge'
    elif model_type == 'test':
        model_dir = '/home/lhd/Copy_Right_of_LLM/myexperiment/RQ3/outputs/vanilla/merge'

    save_path = '/home/lhd/Copy_Right_of_LLM/myexperiment/RQ3/outputs/quote/' + model_type + '/' + data_type
    samples_dir_path = '/home/lhd/Copy_Right_of_LLM/myexperiment/Real_world/Datasets/samples_set/quote_' + data_type

    print('load base model from ' + model_dir)
    model = LlamaForCausalLM.from_pretrained(model_dir)
    print('<<<model load finish>>>')
    file_list = os.listdir(samples_dir_path)
    for file_i, file in enumerate(file_list):
        data = get_samples_list(os.path.join(samples_dir_path, file))
        goal_list = []
        np_save_path = os.path.join(save_path, file.split('.')[0] + '_' + str(num) + '.npy')
        for i, item in enumerate(data[25 * (num - 1):25 * num]):
            input_ids, input_ids_len = get_input_ids(model_dir, item, model)
            loss_tmp = []
            if input_ids_len >= token_len_list[-1]:
                for token_len in token_len_list:
                    new_input_ids = {'input_ids': input_ids['input_ids'][:, :token_len],
                                     'attention_mask': input_ids['attention_mask'][:, :token_len]}
                    loss = test_llama_model(model, new_input_ids)
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
    ##for RQ2
    # for model_type in ['union', 'test']:
    #     for data_type in ['quote', 'unseen']:
    #         for num in [1, 2, 3, 4]:
    #             RQ3(model_type, data_type, num)
    #
    # for model_type in ['vanilla', 'target']:
    #     for data_type in ['quote']:
    #         for num in [1, 2, 3, 4]:
    #             RQ3(model_type, data_type, num)

    for model_type in ['target']:
        for data_type in ['quote']:
            for num in [1, 2, 3, 4]:
                RQ3(model_type, data_type, num)
