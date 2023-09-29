# -*- coding: utf-8 -*- 
# @Time : 2023/9/28 10:30 
# @Author : DirtyBoy 
# @File : data_processing.py
import numpy as np
import os
from real_world.experiments.utils import book_list, quote_list, metrics, save2pkl


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
    model_type = 'LLaMA_7b'
    base_path = 'output/' + model_type + '/quote/'
    base_llm_dir = os.path.join(base_path, 'base')
    vanilla_llm_dir = os.path.join(base_path, 'vanilla')
    llh_llm_dir = os.path.join(base_path, 'llh')
    target_llm_dir = os.path.join(base_path, 'target')

    book_base_loss = []
    book_vanilla_loss = []
    book_llh_loss = []
    book_target_loss = []
    for book_name in book_list:
        base_file_path = os.path.join(base_llm_dir, '100_samples_' + book_name + '.npy')
        book_base_loss.append(read_npy(base_file_path, 4))

        vanilla_file_path = os.path.join(vanilla_llm_dir, '100_samples_' + book_name + '.npy')
        book_vanilla_loss.append(read_npy(vanilla_file_path, 4))

        llh_file_path = os.path.join(llh_llm_dir, '100_samples_' + book_name + '.npy')
        book_llh_loss.append(read_npy(llh_file_path, 4))

        # target_file_path = os.path.join(target_llm_dir, '100_samples_' + book_name + '.npy')
        # book_target_loss.append(read_npy(target_file_path, 4))

    quote_base_loss = []
    quote_vanilla_loss = []
    quote_llh_loss = []
    quote_target_loss = []
    for book_name in quote_list:
        base_file_path = os.path.join(base_llm_dir, '100_samples_' + book_name + '.npy')
        quote_base_loss.append(read_npy(base_file_path, 4))

        vanilla_file_path = os.path.join(vanilla_llm_dir, '100_samples_' + book_name + '.npy')
        quote_vanilla_loss.append(read_npy(vanilla_file_path, 4))

        llh_file_path = os.path.join(llh_llm_dir, '100_samples_' + book_name + '.npy')
        quote_llh_loss.append(read_npy(llh_file_path, 4))

        # target_file_path = os.path.join(target_llm_dir, '100_samples_' + book_name + '.npy')
        # quote_target_loss.append(read_npy(target_file_path, 4))

    book_real_world = metrics(book_base_loss, book_vanilla_loss)
    book_experiment = metrics(book_base_loss, book_llh_loss)

    quote_real_world = metrics(quote_base_loss, quote_vanilla_loss)
    quote_experiment = metrics(quote_base_loss, quote_llh_loss)

    if not os.path.isdir('experiments/intermediate_file/' + model_type):
        os.mkdir('experiments/intermediate_file/' + model_type)

    save2pkl('experiments/intermediate_file/' + model_type + '/book_real_world.pkl', book_real_world)
    save2pkl('experiments/intermediate_file/' + model_type + '/book_experiment.pkl', book_experiment)
    save2pkl('experiments/intermediate_file/' + model_type + '/quote_real_world.pkl', quote_real_world)
    save2pkl('experiments/intermediate_file/' + model_type + '/quote_experiment.pkl', quote_experiment)
