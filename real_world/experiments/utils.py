# -*- coding: utf-8 -*- 
# @Time : 2023/9/28 10:36 
# @Author : DirtyBoy 
# @File : utils.py
import pickle
import numpy as np

quote_list = ['quote1', 'quote2', 'quote3', 'quote4', 'quote5']

book_list = ['Bad_Cree', 'Go_as_a_River', 'The_Black_Queen', 'Tiers_of_Delight', 'Where_You_See_Yourself']


def metrics(target, ref):
    return np.exp(np.array(ref) - np.array(target))


def save2pkl(save_path, model_likelihood_dict):
    f_save = open(save_path, 'wb')
    pickle.dump(model_likelihood_dict, f_save)
    f_save.close()


def read_pkl(save_path):
    f_read = open(save_path, 'rb')
    dict2 = pickle.load(f_read)
    f_read.close()
    return dict2

def merge2distribution(dis, source_distribution: list):
    source_distribution = remove_outliers(source_distribution)
    dis_arr = np.array([dis] * len(source_distribution))
    new_distribution = dis_arr + np.array(source_distribution)
    return new_distribution


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


def remove_outliers_(data, ratio=10):
    # 计算上下四分位数
    filtered_data = []
    for i in data:
        if i >= 2:
            filtered_data.append(i - 1.5)
        else:
            filtered_data.append(i)

    return np.array(filtered_data)
