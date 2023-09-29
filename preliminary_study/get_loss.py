# -*- coding: utf-8 -*- 
# @Time : 2023/8/11 15:29 
# @Author : DirtyBoy 
# @File : get_loss.py
import os, argparse, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import get_samples_list, get_input_ids, test_gpt2_model, GPT2_model
import numpy as np


def get_samples_loss(model_version, model_type, data_type, times, num):
    if times == 0:
        model_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/LLM/GPT2/' + model_version
    else:
        model_dir = '../models/preliminary/' + model_version + '/' + model_type + '/times' + str(times)
    save_path = '../outputs/preliminary_study/' + data_type + '_set/' + model_version
    samples_dir_path = '../Datasets/samples_set/' + data_type + '_set'

    print('load base model from ' + model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    print('<<<model load finish>>>')
    file_list = os.listdir(samples_dir_path)
    for file_i, file in enumerate(file_list):
        data = get_samples_list(os.path.join(samples_dir_path, file))
        goal_list = []
        np_save_path = os.path.join(save_path, str(times) + '_' + file.split('.')[0] + '_' + str(num) + '.npy')
        if not os.path.isfile(np_save_path):
            for i, item in enumerate(data[25 * (num - 1):25 * num]):
                input_ids, input_ids_len = get_input_ids(model_dir, item)
                loss_tmp = []
                if input_ids_len >= 100:
                    for token_len in [50, 60, 70, 80, 90, 100]:
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
    # # for RQ1
    # for type in ['pre', 'tune']:
    #     for times in [1, 3, 2]:
    #         for model_type in ['gpt2', 'gpt2-medium']:
    #             RQ1(type, model_type, times)
    #
    # for model_type in ['gpt2-large', 'gpt2-xl']:
    #     for times in [1, 2, 3]:
    #         for type in ['pre', 'tune']:
    #             for num in [1, 2, 3, 4, 5, 6, 7, 8]:
    #                 RQ1(type, model_type, times, num)

    ###for RQ2

    for model_version in GPT2_model:
        for data_type in ['learned_set', 'unlearned']:
            for times in [0]:
                for num in [1, 2, 3, 4, 5, 6, 7, 8]:
                    get_samples_list(model_version, '', data_type, times, num)

    for model_version in GPT2_model:
        for model_type in ['target', 'reference']:
            for data_type in ['learned_set', 'unlearned']:
                for times in [1, 2, 3]:
                    for num in [1, 2, 3, 4, 5, 6, 7, 8]:
                        get_samples_list(model_version, model_type, data_type, times, num)
