# -*- coding: utf-8 -*- 
# @Time : 2023/8/16 16:48 
# @Author : DirtyBoy 
# @File : utils.py
import os
import numpy as np
import re, random


def read_book2str(book_name):
    with open(book_name, 'r', encoding='utf-8', errors='ignore') as f:
        dataset = f.read()
    f.close()
    str_ = cleaning(dataset)
    words = str_.split()
    return str_, words, len(words)


def cleaning(s):
    s = str(s)
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W,\s', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+', ' ', s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co", "")
    s = s.replace("https", "")
    s = s.replace("[\w*", " ")
    return s


def select_substring(list, start_index, end_index):
    substring = list[start_index:end_index]
    goal_str = ''
    for item in substring:
        goal_str = goal_str + item + ' '
    return goal_str


def sampling(book_name, type):
    if type == 'pre':
        base_path = 'samples_set/pretrain_set'
    elif type == 'tune':
        base_path = 'samples_set/finetune_set'
    else:
        base_path = ''

    content, words, word_count = read_book2str(book_name)
    min_len = 256
    start = 100
    samples_num = 200
    step = int((word_count - 100) / samples_num + 1)
    goal_list = []
    num = 0
    for i in range(samples_num):
        end = start + min_len
        if end <= start + step:
            a = select_substring(words, start, end)
            goal_list.append(a)
            start = start + step
            num = num + 1
    print(os.path.basename(book_name) + 'get sample: ' + str(num))
    with open(os.path.join(base_path, str(num) + '_samples_' + os.path.basename(book_name)), 'w',
              encoding='utf-8') as file:
        for item in goal_list:
            file.write(item + '\n')


def get_txt_files(directory):
    files = [file for file in os.listdir(directory) if file.endswith('.txt')]
    return files



