# -*- coding: utf-8 -*- 
# @Time : 2023/9/24 4:27 
# @Author : DirtyBoy 
# @File : data_processing.py
import numpy as np
import os, pickle, argparse
from Digger.utils import txt_to_list, metrics, save2pkl, read_pkl


def read_npy(path, num=None):
    if num:
        goal = []
        base_p, base_last = os.path.splitext(path)
        for i in range(num):
            path = base_p + '_' + str(i + 1) + base_last
            goal = goal + list(np.load(path))
        return goal
    else:
        return list(np.load(path))


if __name__ == '__main__':
    experiment_type = 'mayseen'  # 'unseen','seen'
    model_type = 'test'  # 'benchmark,union,test
    data_type = 'unlearned'  ## learned,unlearned
    book_name_dir = '../Datasets/book_name_list'

    if model_type == 'benchmark':
        book_name_list = txt_to_list(os.path.join(book_name_dir, model_type + '_' + data_type + '_book_name.txt'))
        target_file_dir = '../outputs/experiments/' + experiment_type + '/target/' + data_type + '_set'
    else:
        book_name_list = txt_to_list(
            os.path.join(book_name_dir, 'test_' + experiment_type + '_' + data_type + '_book_name.txt'))
        target_file_dir = '../outputs/experiments/' + experiment_type + '/vanilla/' + data_type + '_set'
    # if model_type == 'test':
    #     target_file_dir = '../outputs/experiments/' + experiment_type + '/vanilla/' + data_type + '_set'
    # else:
    ref_file_dir = '../outputs/experiments/' + experiment_type + '/' + model_type + '/' + data_type + '_set'
    target_loss = []
    ref_loss = []
    for item in book_name_list:
        target_file_path = os.path.join(target_file_dir, '200_samples_' + item + '.npy')
        target_loss.append(read_npy(target_file_path, 8))

        ref_file_path = os.path.join(ref_file_dir, '200_samples_' + item + '.npy')
        ref_loss.append(read_npy(ref_file_path, 8))
    if not os.path.isdir('experiments/intermediate_file/' + experiment_type):
        os.mkdir('experiments/intermediate_file/' + experiment_type)
    save2pkl(
        'experiments/intermediate_file/' + experiment_type + '/' + model_type + '_' + data_type + '.pkl',
        metrics(target_loss, ref_loss))
    # print(len(read_pkl('experiments/intermediate_file/' + model_type + '_' + data_type + '.pkl')))
