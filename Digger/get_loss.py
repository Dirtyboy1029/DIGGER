# -*- coding: utf-8 -*- 
# @Time : 2023/8/11 15:29 
# @Author : DirtyBoy 
# @File : get_loss.py
import os, argparse, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import get_samples_list, get_input_ids, test_gpt2_model, GPT2_model
import numpy as np


def get_samples_loss(experiment_type, model_type, data_type, num):
    if model_type == 'target':
        model_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/DIGGER/GPT2/models/experiments/gpt2-xl/target'
    elif model_type == 'vanilla':
        model_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/GPT2/gpt2-xl'
    else:
        model_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/DIGGER/GPT2/models/experiments/gpt2-xl/reference/' + model_type + '_' + experiment_type
    samples_dir_path = '../Datasets/samples_set/test_' + experiment_type + '_' + data_type + '_set'
    save_path = '../outputs/experiments/' + experiment_type + '/' + model_type + '/' + data_type
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    print('load base model from ' + model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    print('<<<model load finish>>>')
    file_list = os.listdir(samples_dir_path)
    for file_i, file in enumerate(file_list):
        data = get_samples_list(os.path.join(samples_dir_path, file))
        goal_list = []
        np_save_path = os.path.join(save_path, file.split('.')[0] + '_' + str(num) + '.npy')
        if not os.path.isfile(np_save_path):
            for i, item in enumerate(data[25 * (num - 1):25 * num]):
                input_ids, input_ids_len = get_input_ids(model_dir, item)
                loss_tmp = []
                if input_ids_len >= 100:
                    for token_len in [100]:
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
            print('No. ' + str(file_i + 1) + ' book ' + os.path.splitext(file)[0] + '_' + str(
                num) + '.npy' + ' save finish!')
            print('**********************************************************')
        else:
            print('No. ' + str(file_i + 1) + ' book ' + os.path.splitext(file)[0] + '_' + str(
                num) + '.npy' + ' exist!')


if __name__ == '__main__':
    for experiment_type in ['unseen']:  # 'seen','unseen'
        for model_type in ['test']:  # 'union','vanilla','target','test
            for data_type in ['learned', 'unlearned']:  # 'learned','unlearned'
                for num in [1, 2, 3, 4]:
                    get_samples_loss(experiment_type=experiment_type, model_type=model_type, data_type=data_type,
                                     num=num)
