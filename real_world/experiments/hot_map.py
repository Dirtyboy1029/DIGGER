# -*- coding: utf-8 -*- 
# @Time : 2023/9/29 6:30 
# @Author : DirtyBoy 
# @File : hot_map.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def plot_heatmap(a_):
    plt.figure(figsize=(6, 3))

    y_ticks = ['GPT2-XL','LLaMA-7b']  # 自定义横纵轴
    ax = sns.heatmap(a_, cmap="YlGnBu", yticklabels=y_ticks)
    ax.set_title('Follow LLaMA 7b')  # 图标题
    ax.set_xlabel('Samples')  # x轴标题
    #ax.set_ylabel('Name of Novels')
    plt.xticks([])
    plt.show()

if __name__ == '__main__':
    list2 = np.load('gpt2.npz.npy')
    list1 = np.load('llama.npz.npy')
    num_ = np.zeros(10)
    num = 0
    for i in list1:
        if i>0.460:
            num = num +1

    print(num)
    # combined = list(zip(list1, list2))
    #
    # # 根据第一个列表的元素进行排序
    # combined.sort(key=lambda x: x[0])
    # list1.sort()
    # # 提取排序后的第二个列表
    # sorted_list2 = [item[1] for item in combined]
    # plot_heatmap([list1,sorted_list2])