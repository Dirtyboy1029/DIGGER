# -*- coding: utf-8 -*- 
# @Time : 2023/8/11 13:34 
# @Author : DirtyBoy 
# @File : build_sample_set.py
from utils import get_txt_files, sampling
import argparse, os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-type', "-t", type=str, default="learned_set",
                        choices=['learned_set', 'unlearned'])
    args = parser.parse_args()

    type = args.type

    if type == 'learned_set':
        base_path = 'eBook/learned_set'
    else:
        base_path = 'eBook/unlearned_set'

    txt_files = get_txt_files(base_path)
    txt_path = [os.path.join(base_path, item) for item in txt_files]
    for item in txt_path:
        sampling(item, type=type)
