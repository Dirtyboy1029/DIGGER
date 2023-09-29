# -*- coding: utf-8 -*- 
# @Time : 2023/9/23 13:50 
# @Author : DirtyBoy 
# @File : build_dataset.py
import os, shutil, configparser, random


def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


def file_name2book_name(learned_file_list):
    return [item.replace('200_samples_', '').replace('.txt', '') for item in learned_file_list]


def generate_book_list(book_name_list, file_name):
    save_path = 'book_name_list'
    with open(os.path.join(save_path, file_name), 'w', encoding='utf-8') as file:
        for item in book_name_list:
            file.write(item + '\n')


def build_union_set(learned_path, unlearned_path):
    ##step1:build union_set
    goal_path = 'samples_set/union_set'
    learned_file_list = os.listdir(learned_path)

    unlearned_file_list = os.listdir(unlearned_path)

    for item in learned_file_list:
        shutil.copy(os.path.join(learned_path, item), os.path.join(goal_path, item))
    for item in unlearned_file_list:
        shutil.copy(os.path.join(unlearned_path, item), os.path.join(goal_path, item))
    learned_book_name_list = file_name2book_name(learned_file_list)
    unlearned_book_name_list = file_name2book_name(unlearned_file_list)
    generate_book_list(learned_book_name_list, 'learned_book_name.txt')
    generate_book_list(unlearned_book_name_list, 'unlearned_book_name.txt')


def build_benchmark_test_set(learned_path, unlearned_path, test_ratio):
    benchmark_path = 'samples_set/benchmark_set'
    benchmark_learned_path = 'samples_set/benchmark_learned_set'
    benchmark_unlearned_path = 'samples_set/benchmark_unlearned_set'
    test_path = 'samples_set/test_set'
    test_learned_path = 'samples_set/test_learned_set'
    test_unlearned_path = 'samples_set/test_unlearned_set'

    learned_file_list = os.listdir(learned_path)
    unlearned_file_list = os.listdir(unlearned_path)

    test_learned_file_list, benchmark_learned_file_list = data_split(learned_file_list, test_ratio, True)
    test_unlearned_file_list, benchmark_unlearned_file_list = data_split(unlearned_file_list, test_ratio, True)

    def list2fold(file_list, source_path, goal_path):
        for item in file_list:
            shutil.copy(os.path.join(source_path, item), os.path.join(goal_path, item))

    list2fold(test_learned_file_list, learned_path, test_learned_path)
    list2fold(test_unlearned_file_list, unlearned_path, test_unlearned_path)
    list2fold(benchmark_learned_file_list, learned_path, benchmark_learned_path)
    list2fold(benchmark_unlearned_file_list, unlearned_path, benchmark_unlearned_path)

    list2fold(test_learned_file_list, learned_path, test_path)
    list2fold(test_unlearned_file_list, unlearned_path, test_path)
    list2fold(benchmark_learned_file_list, learned_path, benchmark_path)
    list2fold(benchmark_unlearned_file_list, unlearned_path, benchmark_path)

    test_learned_book_name_list = file_name2book_name(test_learned_file_list)
    test_unlearned_book_name_list = file_name2book_name(test_unlearned_file_list)
    test_book_name_list = file_name2book_name(test_learned_file_list + test_unlearned_file_list)

    benchmark_learned_book_name_list = file_name2book_name(benchmark_learned_file_list)
    benchmark_unlearned_book_name_list = file_name2book_name(benchmark_unlearned_file_list)
    benchmark_book_name_list = file_name2book_name(benchmark_learned_file_list + benchmark_unlearned_file_list)

    generate_book_list(test_learned_book_name_list, 'test_learned_book_name.txt')
    generate_book_list(test_unlearned_book_name_list, 'test_unlearned_book_name.txt')
    generate_book_list(test_book_name_list, 'test_union_book_name.txt')

    generate_book_list(benchmark_learned_book_name_list, 'benchmark_learned_book_name.txt')
    generate_book_list(benchmark_unlearned_book_name_list, 'benchmark_unlearned_book_name.txt')
    generate_book_list(benchmark_book_name_list, 'benchmark_union_book_name.txt')


if __name__ == '__main__':
    config = configparser.RawConfigParser()
    config.read("../config")
    overwrite_output_dir = False
    test_ratio = float(config.get("config", "test_size"))

    learned_path = 'samples_set/learned_set'
    unlearned_path = 'samples_set/unlearned_set'

    ###learned_set and unlearned_set have base file
    build_union_set(learned_path, unlearned_path)
    ##step2:build benchmark_set & test_set

    build_benchmark_test_set(learned_path, unlearned_path, test_ratio)
    # a = os.listdir('samples_set/test_mayseen_set')
    # b = [item.split('.')[0].replace('200_samples_','') for item in a]
    # generate_book_list(b,'test_mayseen_union_book_name.txt')

